"""Run a frozen backbone over the protocol's crops and write a feature-store folder.

    uv run python scripts/extract_features.py --backbone bioclip2 --cameras trainval
    uv run python scripts/extract_features.py --backbone bioclip2 --cameras testempty
    uv run python scripts/extract_features.py --backbone bioclip2 --cameras test
    uv run python scripts/extract_features.py --backbone bioclip2 --cameras trainval --sample 512 --out /tmp/x

Resumable: rerun the same command after an interruption and it continues where it stopped
(config.json must match). Writes manifest.json when complete; readers ignore folders without it.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from iwildcam import store  # noqa: E402
from iwildcam.backbones import REGISTRY  # noqa: E402
from iwildcam.crops import Crops  # noqa: E402
from iwildcam.protocol import CROP_RULE, DETECTIONS, ORIGINALS  # noqa: E402


class Safe(Dataset):
    """Crops, but an undecodable file gives a zero image and ok=False instead of a crash."""

    def __init__(self, crops: Crops, size: int):
        self.crops, self.size = crops, size

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, i):
        try:
            image, _ = self.crops[i]
            return image, i, True
        except (OSError, ValueError):
            return torch.zeros(3, self.size, self.size), i, False


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True, choices=list(REGISTRY))
    p.add_argument("--cameras", default="trainval", choices=["trainval", "test", "testempty"])
    p.add_argument("--view", default="crop", choices=["crop", "whole"])
    p.add_argument("--sample", type=int, help="only this many random rows (to check the script)")
    p.add_argument("--out", type=Path, help="default: the store folder")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()

    spec = REGISTRY[a.backbone]
    out = a.out or store.folder(a.backbone, a.cameras, a.view)
    if (out / "manifest.json").exists():
        raise SystemExit(f"{out} is complete already")

    rows = store.select_rows(a.cameras)
    if a.sample:
        rows = rows.sample(a.sample, random_state=0).sort_values("file_name").reset_index(drop=True)
    n = len(rows)
    config = {
        "backbone": spec.name, "store_name": spec.store_name, "size": spec.size, "dim": spec.dim,
        "normalisation": list(map(list, spec.normalisation)), "features": spec.features,
        "dtype": "float16", "amp": "float16", "view": a.view, "cameras": a.cameras, "n": n,
        "crop_rule": CROP_RULE if a.view == "crop" else "whole frame",
        "detections": str(DETECTIONS), "image_dir": str(ORIGINALS), "sample": a.sample,
        "counts": {f"{fold}/{'empty' if e else 'labelled'}": int(k) for (fold, e), k in
                   rows.groupby(["official_ood", rows.is_empty.astype(str).str.lower().eq("true")]).size().items()},
    }
    paths = {k: out / f for k, f in [("index", "index.csv.gz"), ("features", "features.npy"),
                                      ("status", "status.npy"), ("config", "config.json"),
                                      ("progress", "progress.json"), ("manifest", "manifest.json")]}

    if paths["config"].exists():                       # resume
        old = json.loads(paths["config"].read_text())
        old.pop("boxes_sha256", None)
        if old != config:
            raise SystemExit(f"{out}/config.json differs from this run; use a new --out")
        features = np.load(paths["features"], mmap_mode="r+")
        status = np.load(paths["status"], mmap_mode="r+")
        done = json.loads(paths["progress"].read_text())["done"]
        print(f"resuming {out} at {done:,}/{n:,}", flush=True)
    else:
        out.mkdir(parents=True, exist_ok=False)
        config["boxes_sha256"] = sha256(DETECTIONS)
        rows.to_csv(paths["index"], index=False)
        paths["config"].write_text(json.dumps(config, indent=2))
        features = np.lib.format.open_memmap(paths["features"], mode="w+", dtype=np.float16, shape=(n, spec.dim))
        status = np.lib.format.open_memmap(paths["status"], mode="w+", dtype=np.int8, shape=(n,))
        done = 0
        paths["progress"].write_text(json.dumps({"done": 0}))
        print(f"{n:,} rows from {rows.location.nunique()} cameras -> {out}", flush=True)

    model, transform = spec.load()
    model = model.cuda()
    todo = rows.file_name.to_numpy()[done:]
    loader = DataLoader(Safe(Crops(todo, transform, a.view), spec.size), batch_size=a.batch_size,
                        num_workers=a.workers, pin_memory=True, prefetch_factor=4 if a.workers else None)
    t0, seen = time.time(), 0
    with torch.inference_mode():
        for b, (images, index, ok) in enumerate(loader):
            with torch.autocast("cuda", dtype=torch.float16):
                x = model(images.cuda(non_blocking=True)).float().cpu().numpy()
            rows_here = done + index.numpy()
            assert np.isfinite(x[ok.numpy()]).all(), "non-finite features"
            features[rows_here] = x.astype(np.float16)
            status[rows_here] = np.where(ok.numpy(), store.OK, store.UNDECODABLE)
            seen += len(index)
            if b % 50 == 0 or done + seen == n:
                features.flush(); status.flush()
                paths["progress"].write_text(json.dumps({"done": done + seen}))
                rate = seen / (time.time() - t0)
                print(f"  {done + seen:,}/{n:,}  {rate:.0f} img/s  ~{(n - done - seen) / rate / 60:.0f} min left",
                      flush=True)
    features.flush(); status.flush()
    paths["progress"].write_text(json.dumps({"done": n}))
    assert (np.asarray(status) != store.NOT_DONE).all()
    paths["manifest"].write_text(json.dumps({
        "finished": dt.datetime.now().isoformat(timespec="seconds"),
        "ok": int((np.asarray(status) == store.OK).sum()),
        "undecodable": int((np.asarray(status) == store.UNDECODABLE).sum()),
        "index_sha256": sha256(paths["index"])}, indent=2))
    print(f"done: {out}")


if __name__ == "__main__":
    main()
