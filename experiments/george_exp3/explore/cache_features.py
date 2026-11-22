"""Cache frozen-backbone global features for the idea-A pilot.

Adapted from Shayan's users/shayan/pilot_a/cache_features.py (2026-10-10). Changes: an
opt-in --share-gpu flag; everything else, including the row selection, is unchanged so
the caches line up row for row.

    python cache_features.py --out features/dinov2_vitb14_reg_448 --limit 256   # smoke test
    python cache_features.py --out features/dinov2_vitb14_reg_448               # full cache

One vector per image: [CLS token, mean patch token] of a frozen DINOv2 ViT-B/14 (with
registers), computed on the image squashed to `--size` x `--size`, the same geometry the
ResNet baseline uses.

Which images. Only cameras that have labelled images in the TRAIN or VAL fold of
`official_ood` (248 cameras): their labelled images and their empty frames. Cameras that
only ever produced empty frames are left out, and so is every test camera. Test-fold
images are never opened, labelled or empty: the pilot selects hypotheses on validation
cameras only, and this script enforces that.

The row selection comes from `iwildcam.data.load_task`, so the class set and the folds are
exactly the ones the baseline trains on. Run it with the repo importable (from the repo
root, with the repo's virtualenv).

What is written to `--out`:
    index.csv.gz     one row per image, fixing the row order
    features.npy     float16, (n, 2 * embed_dim); rows that failed to decode are NaN
    status.npy       int8 per row: 0 not done, 1 ok, 2 file could not be decoded
    failures.jsonl   one line per undecodable file, with the error
    config.json      everything that defines the cache, checked on resume
    progress.json    how many rows are written; replaced atomically
    manifest.json    written once the cache is complete
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from iwildcam.data import (IMAGENET_MEAN, IMAGENET_STD, image_dir_for, load_task,
                           official_test_cameras)

# The DINOv2 code is fetched from GitHub by torch.hub. Pinning the commit keeps the model
# definition identical between the smoke test, the full cache and any later re-run.
DINOV2_REPO = "facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8"
SPLIT = "official_ood"
FOLDS = ("train", "val")          # never "test"
KEEP = ["file_name", "location", "category", "is_empty", "burst_id", "datetime", "hour",
        SPLIT]
FEATURES = "[x_norm_clstoken, mean(x_norm_patchtokens)]"
NOT_DONE, OK, UNDECODABLE = 0, 1, 2

# Exit codes the launcher tells apart from a crash.
EXIT_BUSY, EXIT_TOO_SLOW, EXIT_YIELD = 2, 3, 4


def select_rows() -> pd.DataFrame:
    """The images to embed: labelled train/val rows, then the empties of those cameras.

    Returns:
        One row per image with `file_name`, `location`, `category`, `is_empty`,
        `burst_id`, `datetime`, `hour`, the fold, and `y` (-1 for empty frames). Sorted by
        `file_name`, so the order is the same on every run and a resumed run lines up.
    """
    task = load_task(require_files=False)
    lab = task.df[task.df[SPLIT].isin(FOLDS)][KEEP + ["y"]]
    cameras = set(lab["location"])
    emp = task.empty[task.empty[SPLIT].isin(FOLDS) & task.empty["location"].isin(cameras)]
    rows = pd.concat([lab, emp[KEEP].assign(y=-1)], ignore_index=True)

    # Three independent statements of "no test data": the fold column, the official list
    # of held-out cameras, and each camera sitting in exactly one fold.
    assert not rows[SPLIT].eq("test").any(), "test-fold rows must never be selected"
    assert not set(rows["location"]) & set(official_test_cameras()), "a test camera was selected"
    assert rows.groupby("location")[SPLIT].nunique().eq(1).all(), "a camera spans two folds"
    assert set(rows["location"]) == cameras and rows["file_name"].is_unique
    return rows.sort_values("file_name").reset_index(drop=True)


def select_test_rows() -> pd.DataFrame:
    """The labelled images of the official test cameras, for a one-off final evaluation.

    Only `--test-only` uses this. Empty frames are not included. Kept apart from
    `select_rows` so the train/val caches stay free of test rows.
    """
    task = load_task(require_files=False)
    rows = task.df[task.df[SPLIT].eq("test")][KEEP + ["y"]]
    assert set(rows["location"]) <= set(official_test_cameras()), "a non-test camera was selected"
    assert rows["file_name"].is_unique and len(rows)
    return rows.sort_values("file_name").reset_index(drop=True)


# Crop rule, identical to Paul's linprobe.py so crop results are comparable.
MD_JSON = Path("/zfs/ai4good/groups/camera-traps/pdelap-background-20261018/data/md-v5a.json")
ORIGINALS = Path("/zfs/ai4good/groups/camera-traps/raw/train")   # full-resolution JPEGs, read only
MIN_CONF, MARGIN = 0.2, 0.15
CROP_RULE = ("most confident MegaDetector v5a animal box if conf >= 0.2, squared in pixels, "
             "padded 15% per side, shifted inside the frame, cut from the original; else whole frame")


def best_detections(names: list[str]) -> np.ndarray:
    """(conf, x, y, w, h) of each image's most confident animal box; conf 0 if none."""
    best = {}
    for im in json.load(MD_JSON.open())["images"]:
        boxes = [d for d in im.get("detections") or [] if d["category"] == "1"]
        if boxes:
            d = max(boxes, key=lambda d: d["conf"])
            best[im["file"]] = (d["conf"], *d["bbox"])
    return np.array([best.get(f, (0.0, 0.0, 0.0, 1.0, 1.0)) for f in names], dtype=np.float64)


def crop_box(det, width: float, height: float) -> tuple[float, float, float, float]:
    """Paul's crop_box: fractional (left, top, right, bottom), or the whole frame."""
    conf, x, y, w, h = det
    if conf < MIN_CONF:
        return 0.0, 0.0, 1.0, 1.0
    cx, cy = (x + w / 2) * width, (y + h / 2) * height
    side = min(max(w * width, h * height) * (1 + 2 * MARGIN), width, height)
    left = min(max(cx - side / 2, 0), width - side)
    top = min(max(cy - side / 2, 0), height - side)
    return left / width, top / height, (left + side) / width, (top + side) / height


def crop_boxes(names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Crop box of every image in `names` order, and its detection confidence."""
    task = load_task(require_files=False)
    meta = pd.concat([task.df[["file_name", "width", "height"]],
                      task.empty[["file_name", "width", "height"]]]).drop_duplicates("file_name")
    size = meta.set_index("file_name").loc[names, ["width", "height"]].to_numpy(dtype=np.float64)
    det = best_detections(names)
    boxes = np.array([crop_box(d, w, h) for d, (w, h) in zip(det, size)], dtype=np.float64)
    return boxes, det[:, 0]


class Images(Dataset):
    """JPEGs by name. Returns `(tensor, ok, error)`.

    Only a file PIL cannot decode is reported as a failure. A missing file, or anything
    going wrong in the transform, raises: those are bugs or a wrong path, not bad data.
    """

    def __init__(self, image_dir: Path, names: list[str], size: int, boxes: np.ndarray | None = None,
                 flip: bool = False):
        self.image_dir, self.names, self.size, self.boxes, self.flip = image_dir, names, size, boxes, flip
        self.transform = v2.Compose([
            v2.Resize((size, size), interpolation=v2.InterpolationMode.BILINEAR,
                      antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, i):
        try:
            im = Image.open(self.image_dir / self.names[i]).convert("RGB")
        except (FileNotFoundError, PermissionError):
            raise
        except OSError as e:                   # truncated or unidentifiable JPEG
            return torch.zeros(3, self.size, self.size), False, f"{type(e).__name__}: {e}"
        if self.boxes is not None:
            w, h = im.size
            l, t, r, b = self.boxes[i]
            im = im.crop((round(l * w), round(t * h), round(r * w), round(b * h)))
        if self.flip:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        return self.transform(im), True, ""


def gpu_memory_used_mib() -> int:
    """Memory in use on GPU 0 right now, from nvidia-smi."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True,
                         check=True)
    return int(out.stdout.strip().splitlines()[0])


def other_gpu_pids() -> list[int]:
    """PIDs of compute processes on the GPU that are not this one."""
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True,
                         check=True)
    pids = [int(p) for p in out.stdout.split() if p.strip().isdigit()]
    return [p for p in pids if p != os.getpid()]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(index: pd.DataFrame) -> str:
    """Hash of what each row is: file, camera, fold and label, in row order."""
    cols = index[["file_name", "location", SPLIT, "y"]].astype(str)
    return hashlib.sha256("\n".join(cols.agg(",".join, axis=1)).encode()).hexdigest()


def repair_failures(path: Path, done: int, status: np.ndarray, names: list[str]) -> None:
    """Make `failures.jsonl` agree with the committed rows, after an interrupted run.

    Failure records are appended as they happen, before `progress.json` commits the rows
    they belong to. An interruption can therefore leave records for rows that will be
    processed again, or a half-written last line. This keeps exactly one record per
    committed undecodable row and drops the rest, then replaces the file atomically.

    Raises:
        ValueError: if a committed undecodable row has no record, two records for a row
            disagree, a record names the wrong file, or a line other than the last is
            not valid JSON. Those mean the cache cannot be trusted.
    """
    lines = path.read_text().splitlines() if path.exists() else []
    kept: dict[int, dict] = {}
    for i, line in enumerate(lines):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1:            # the write that was interrupted
                continue
            raise ValueError(f"{path}: line {i + 1} is not valid JSON")
        row = rec["row"]
        if row >= done:                        # never committed; the row is redone
            continue
        if rec["file_name"] != names[row]:
            raise ValueError(f"{path}: row {row} is {names[row]}, record says {rec['file_name']}")
        if row in kept and kept[row] != rec:
            raise ValueError(f"{path}: conflicting records for row {row}")
        kept[row] = rec
    expected = set(np.flatnonzero(np.asarray(status[:done]) == UNDECODABLE).tolist())
    if set(kept) != expected:
        raise ValueError(f"{path}: records for rows {sorted(set(kept) ^ expected)[:10]} do not "
                         f"match status.npy")
    if kept or path.exists():
        write_atomic(path, "".join(json.dumps(kept[r]) + "\n" for r in sorted(kept)))


def write_atomic(path: Path, text: str) -> None:
    """Replace `path` in one step, so a crash never leaves it half written."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output folder for this feature set")
    ap.add_argument("--model", default="dinov2_vitb14_reg")
    ap.add_argument("--size", type=int, default=448, help="square input side, multiple of 14")
    ap.add_argument("--height", type=int, default=448, help="which prepared image folder")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--limit", type=int, help="embed only this many random rows (smoke test)")
    ap.add_argument("--max-minutes", type=float, default=60.0,
                    help="checkpoint and stop if the whole run is projected to take longer")
    ap.add_argument("--warmup-images", type=int, default=4096,
                    help="images to process before the time projection is trusted")
    ap.add_argument("--dry-run", action="store_true", help="print the selection and stop")
    ap.add_argument("--crop", action="store_true",
                    help="embed MegaDetector crops from the originals (Paul's rule) instead of full frames")
    ap.add_argument("--margin", type=float, default=MARGIN, help="crop padding per side")
    ap.add_argument("--flip", action="store_true", help="mirror every image")
    ap.add_argument("--test-only", action="store_true",
                    help="embed only the labelled test-camera images (final evaluation)")
    ap.add_argument("--share-gpu", action="store_true",
                    help="run even if another job is on the GPU, and do not yield to one")
    a = ap.parse_args()
    assert a.size % 14 == 0, "ViT-/14 needs an input side divisible by 14"

    globals()["MARGIN"] = a.margin
    rows = select_test_rows() if a.test_only else select_rows()
    if a.limit:
        rows = rows.sample(a.limit, random_state=0).sort_values("file_name").reset_index(drop=True)
    n, names = len(rows), list(rows["file_name"])
    image_dir = ORIGINALS if a.crop else image_dir_for(a.height)
    counts = {f"{fold}/{'empty' if empty else 'labelled'}": int(k)
              for (fold, empty), k in rows.groupby([SPLIT, "is_empty"]).size().items()}
    print(f"{n:,} images from {rows['location'].nunique()} cameras  {counts}")
    print(f"images: {image_dir}")
    missing = [f for f in names if not (image_dir / f).exists()]
    assert not missing, f"{len(missing)} selected files are missing, e.g. {missing[:3]}"
    boxes = conf = None
    if a.crop:
        boxes, conf = crop_boxes(names)
        cropped = conf >= MIN_CONF
        print(f"crops: {cropped.sum():,} of {n:,} images have a confident animal box; "
              f"the rest use the whole frame. Median box side {np.median((boxes[cropped, 2] - boxes[cropped, 0])):.2f} of the width")
    a.boxes = boxes
    if a.dry_run:
        return

    # The GPU is shared by the group: never start on top of someone else's job.
    used, others = gpu_memory_used_mib(), other_gpu_pids()
    if (used > 2000 or others) and not a.share_gpu:
        print(f"GPU is in use ({used} MiB, pids {others}); not starting")
        sys.exit(EXIT_BUSY)

    out = Path(a.out + ("_smoke" if a.limit else ""))
    out.mkdir(parents=True, exist_ok=True)
    index_path, feat_path, status_path = out / "index.csv.gz", out / "features.npy", out / "status.npy"
    config_path, prog_path, fail_path = out / "config.json", out / "progress.json", out / "failures.jsonl"

    # One writer per cache. The lock is removed on any normal or failed exit; only a killed
    # process leaves it behind, and then it has to be looked at and deleted by hand.
    lock = out / "LOCK"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        sys.exit(f"{lock} exists ({lock.read_text().strip()}): another writer, or a crashed "
                 f"run. Delete it only once you are sure nothing is writing here.")
    os.write(fd, f"{socket.gethostname()} pid {os.getpid()} {time.strftime('%F %T')}\n".encode())
    os.close(fd)
    try:
        code = extract(a, rows, names, image_dir, counts, index_path, feat_path, status_path,
                       config_path, prog_path, fail_path, out)
    finally:
        lock.unlink(missing_ok=True)
    sys.exit(code)


def extract(a, rows, names, image_dir, counts, index_path, feat_path, status_path,
            config_path, prog_path, fail_path, out) -> int:
    """Embed every row not yet written. Returns the process exit code."""
    n = len(rows)
    torch.backends.cudnn.benchmark = True
    resnet = a.model == "resnet50"
    if resnet:
        import torchvision.models as tvm
        model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Identity()
        model = model.cuda().eval()
        dim = 2048
        weights = sorted((Path(torch.hub.get_dir()) / "checkpoints").glob("resnet50-*"))
    else:
        model = torch.hub.load(DINOV2_REPO, a.model, trust_repo=True, skip_validation=True)
        model = model.cuda().eval()
        dim = 2 * model.embed_dim
        weights = sorted((Path(torch.hub.get_dir()) / "checkpoints").glob(f"{a.model.replace('_reg', '')}*"))
        weights = [w for w in weights if ("reg" in w.name) == ("reg" in a.model)]
    assert len(weights) == 1, f"expected one checkpoint for {a.model}, found {weights}"

    # Everything that defines what a row of features.npy means. A resumed run must match
    # it exactly, otherwise two different caches would be mixed in one file.
    config = {
        "model": a.model, "repo": "torchvision IMAGENET1K_V1" if resnet else DINOV2_REPO, "weights_file": weights[0].name,
        "weights_sha256": sha256(weights[0]), "size": a.size, "height": a.height,
        "interpolation": "bilinear, antialias", "normalisation": "imagenet", "amp": "float16",
        "features": "global average pool" if resnet else FEATURES, "dim": dim, "dtype": "float16", "split": SPLIT,
        "folds": ["test"] if a.test_only else list(FOLDS),
        "selection": "labelled test-camera images" if a.test_only else "cameras with labelled train/val images",
        "image_dir": str(image_dir), "n": n, "limit": a.limit,
        "index_sha256": fingerprint(rows), "counts": counts,
    }
    if a.flip or a.margin != 0.15:
        config.update(flip=a.flip, margin=a.margin)
    if a.boxes is not None:
        config.update(view="crop", crop_rule=CROP_RULE, detections=str(MD_JSON),
                      boxes_sha256=hashlib.sha256(np.ascontiguousarray(a.boxes).tobytes()).hexdigest())

    done = 0
    if config_path.exists():
        old = json.loads(config_path.read_text())
        diff = {k: (old.get(k), v) for k, v in config.items() if old.get(k) != v}
        assert not diff, f"this folder holds a different cache; use a new --out. Differences: {diff}"
        assert fingerprint(pd.read_csv(index_path)) == config["index_sha256"], \
            "index.csv.gz does not match config.json"
        done = json.loads(prog_path.read_text())["done"]
        feats = np.load(feat_path, mmap_mode="r+")
        status = np.load(status_path, mmap_mode="r+")
        assert feats.shape == (n, dim) and feats.dtype == np.float16, (feats.shape, feats.dtype)
        assert status.shape == (n,) and status.dtype == np.int8, (status.shape, status.dtype)
        assert 0 <= done <= n, done
        assert (status[:done] != NOT_DONE).all(), "progress.json is ahead of status.npy"
        repair_failures(fail_path, done, status, names)
        print(f"resuming at row {done:,} of {n:,}")
    else:
        assert not any(p.exists() for p in (index_path, feat_path, status_path, prog_path,
                                            fail_path)), \
            f"{out} has cache files but no config.json; use a new --out"
        rows.to_csv(index_path, index=False)
        feats = np.lib.format.open_memmap(feat_path, mode="w+", dtype=np.float16, shape=(n, dim))
        status = np.lib.format.open_memmap(status_path, mode="w+", dtype=np.int8, shape=(n,))
        write_atomic(prog_path, json.dumps({"done": 0}))
        write_atomic(config_path, json.dumps(config, indent=2))

    def checkpoint(pos: int) -> None:
        """Features and status reach the disk before progress says they are there."""
        feats.flush()
        status.flush()
        write_atomic(prog_path, json.dumps({"done": pos}))

    loader = DataLoader(Images(image_dir, names[done:], a.size,
                               None if a.boxes is None else a.boxes[done:], a.flip), batch_size=a.batch_size,
                        num_workers=a.num_workers, pin_memory=True, shuffle=False)
    started, pos, code = time.time(), done, 0
    with torch.inference_mode():
        for step, (xb, ok, err) in enumerate(loader):
            with torch.autocast("cuda", dtype=torch.float16):
                if resnet:
                    f = model(xb.cuda(non_blocking=True))
                else:
                    o = model.forward_features(xb.cuda(non_blocking=True))
                    f = torch.cat([o["x_norm_clstoken"], o["x_norm_patchtokens"].mean(1)], dim=1)
            f = f.float().cpu().numpy().astype(np.float16)
            ok = ok.numpy()
            # A decoded image must give finite numbers, also after the cast to float16.
            # If it does not, that is a numerical problem and the run stops here.
            if not np.isfinite(f[ok].astype(np.float32)).all():
                checkpoint(pos)
                raise FloatingPointError(f"non-finite features in the batch starting at row {pos}")
            f[~ok] = np.nan                    # an undecodable file must not look like data
            feats[pos:pos + len(f)] = f
            status[pos:pos + len(f)] = np.where(ok, OK, UNDECODABLE)
            if not ok.all():
                with fail_path.open("a") as fh:
                    for i in np.flatnonzero(~ok):
                        fh.write(json.dumps({"row": pos + int(i), "file_name": names[pos + i],
                                             "error": err[i]}) + "\n")
            pos += len(f)

            if step % 50 == 49 or pos == n:
                checkpoint(pos)
                elapsed = time.time() - started
                rate = (pos - done) / max(elapsed, 1e-6)
                eta = (n - pos) / max(rate, 1e-6)
                print(f"  {pos:>7,}/{n:,}  {rate:6.1f} img/s  elapsed {elapsed / 60:5.1f} min  "
                      f"eta {eta / 60:5.1f} min  "
                      f"gpu peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
                if pos == n:
                    break
                if other_gpu_pids() and not a.share_gpu:   # someone else started: give the GPU back
                    print(f"another job is on the GPU (pids {other_gpu_pids()}); "
                          f"checkpointed at row {pos:,} and stopping. Re-run to resume.")
                    code = EXIT_YIELD
                    break
                if pos - done >= a.warmup_images and (elapsed + eta) / 60 > a.max_minutes:
                    print(f"projected {(elapsed + eta) / 60:.0f} min exceeds --max-minutes "
                          f"{a.max_minutes:.0f}; checkpointed at row {pos:,} and stopping.")
                    code = EXIT_TOO_SLOW
                    break

    checkpoint(pos)
    if code:
        return code

    seconds = time.time() - started
    assert pos == n and (status != NOT_DONE).all()
    good = status == OK
    assert np.isfinite(np.asarray(feats[good], dtype=np.float32)).all(), "non-finite features in an ok row"
    failures = [json.loads(line) for line in fail_path.read_text().splitlines()] if fail_path.exists() else []
    assert len(failures) == int((status == UNDECODABLE).sum()), "failures.jsonl and status.npy disagree"
    manifest = dict(config, n_ok=int(good.sum()), n_undecodable=len(failures), failures=failures,
                    seconds_this_session=round(seconds, 1),
                    images_per_second=round((n - done) / max(seconds, 1e-6), 1),
                    gpu_peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                    batch_size=a.batch_size, num_workers=a.num_workers,
                    torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                    host=socket.gethostname(), finished=time.strftime("%Y-%m-%d %H:%M:%S"),
                    argv=sys.argv)
    write_atomic(out / "manifest.json", json.dumps(manifest, indent=2))
    norms = np.linalg.norm(np.asarray(feats[good][:2000], dtype=np.float32), axis=1)
    print(f"done: {n:,} images in {seconds / 60:.1f} min ({manifest['images_per_second']} img/s), "
          f"{len(failures)} undecodable, feature norm mean {norms.mean():.1f} -> {out}")
    return 0


if __name__ == "__main__":
    main()
