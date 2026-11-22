"""Linear probe on frozen features: does a biology-pretrained backbone (BioCLIP) shrink the
camera gap compared with the ImageNet ResNet50 the baseline starts from?

The backbone never trains. Each labelled image is embedded once and cached, then only a
linear 60-species layer is fitted on the train split, so both folds reuse one extraction.

    python linprobe.py extract --backbone bioclip      # writes feats/bioclip.npz
    python linprobe.py extract --backbone resnet50     # writes feats/resnet50.npz
    python linprobe.py probe --backbone bioclip        # both folds, appends to results/linprobe.csv
    python linprobe.py zeroshot                        # BioCLIP with no training at all

Scores are macro F1 over the species present in the test split, as in `iwildcam.train`.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

from iwildcam.data import FOLDS, build_transform, image_dir_for, load_fold, load_task

FEATS = Path("feats")
OUT = Path("results/linprobe.csv")
BIOCLIP = "hf-hub:imageomics/bioclip"


class Files(Dataset):
    """Every labelled image of the task, in `task.df` order, optionally cropped to a box."""

    def __init__(self, task, transform, boxes=None, image_dir=None):
        self.names, self.transform = task.df["file_name"].to_numpy(), transform
        self.dir = Path(image_dir or task.image_dir)
        self.boxes = boxes   # (n, 4) fractional boxes from `crop_boxes`, or None for the full frame

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        im = Image.open(self.dir / self.names[i]).convert("RGB")
        if self.boxes is not None:
            w, h = im.size
            l, t, r, b = self.boxes[i]
            im = im.crop((round(l * w), round(t * h), round(r * w), round(b * h)))
        return self.transform(im)


MIN_CONF, MARGIN = 0.2, 0.15   # MegaDetector's typical threshold; context kept around the box
ORIGINALS = Path("/zfs/ai4good/groups/camera-traps/raw/train")   # full-resolution JPEGs, read only


def crop_box(det_row, width, height):
    """Box around the most confident animal as fractions of the frame `(left, top, right, bottom)`:
    squared up in pixels, padded by MARGIN on each side, shifted to stay inside the frame.
    The whole frame when there is no confident detection.

    Squaring keeps the animal's proportions when the crop is squashed to the model's input.
    Fractions make the box valid at any resolution with the frame's aspect ratio.
    """
    if det_row.conf < MIN_CONF:
        return 0.0, 0.0, 1.0, 1.0
    cx, cy = (det_row.x + det_row.w / 2) * width, (det_row.y + det_row.h / 2) * height
    side = min(max(det_row.w * width, det_row.h * height) * (1 + 2 * MARGIN), width, height)
    left = min(max(cx - side / 2, 0), width - side)
    top = min(max(cy - side / 2, 0), height - side)
    return left / width, top / height, (left + side) / width, (top + side) / height


def crop_boxes(task):
    """`crop_box` for every labelled image, in `task.df` order."""
    det = detections(task)
    size = task.df[["width", "height"]].to_numpy()   # original sizes, for the aspect ratio
    return np.array([crop_box(r, w, h) for r, (w, h) in zip(det.itertuples(), size)])


def load_bioclip(device):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(BIOCLIP)
    return model.eval().to(device), preprocess, open_clip.get_tokenizer(BIOCLIP)


def backbone(name: str, device: str):
    """Frozen encoder and its own eval transform: `(images -> features, transform)`."""
    if name == "bioclip":
        model, preprocess, _ = load_bioclip(device)
        # BioCLIP's own preprocess centre-crops, which drops ~44% of a 796x448 frame;
        # squash the whole frame instead, as the resnet50 path does, keeping CLIP's normalisation
        norm = next(t for t in preprocess.transforms if isinstance(t, T.Normalize))
        return model.encode_image, T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(), norm])
    if name == "resnet50":
        from iwildcam.models import build_model
        model = build_model("resnet50", 2).eval().to(device)   # head is unused
        return model.features, build_transform(448)            # baseline resolution
    raise ValueError(f"unknown backbone {name!r}")


@torch.no_grad()
def extract(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    task = load_task(image_dir=image_dir_for(448))
    print(task.summary())
    encode, transform = backbone(args.backbone, device)
    files = (Files(task, transform, crop_boxes(task), ORIGINALS) if args.view == "crop"
             else Files(task, transform))
    loader = DataLoader(files, batch_size=args.batch_size,
                        num_workers=args.num_workers, pin_memory=True)
    out, t0 = [], time.time()
    for i, xb in enumerate(loader):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            out.append(encode(xb.to(device, non_blocking=True)).float().cpu())
        if i % 50 == 0:
            done = (i + 1) * args.batch_size
            print(f"  {min(done, len(task.df)):,}/{len(task.df):,} images, "
                  f"{done / (time.time() - t0):.0f} img/s", flush=True)
    FEATS.mkdir(exist_ok=True)
    path = FEATS / f"{feat_name(args)}.npz"
    np.savez(path, x=torch.cat(out).numpy().astype(np.float16),
             file_name=task.df["file_name"].to_numpy())
    print(f"wrote {path} in {time.time() - t0:.0f}s")


def feat_name(args):
    """Features file stem, e.g. `bioclip` or `bioclip-crop`."""
    return args.backbone if args.view == "full" else f"{args.backbone}-{args.view}"


def load_feats(name, task):
    d = np.load(FEATS / f"{name}.npz", allow_pickle=True)
    assert (d["file_name"] == task.df["file_name"].to_numpy()).all(), \
        "features are out of sync with the task, re-run extract"
    return torch.from_numpy(d["x"].astype(np.float32))


def scores(pred, true, n_classes):
    present = [c for c in range(n_classes) if (true == c).any()]
    return {"macro_f1_present": f1_score(true, pred, average="macro", labels=present,
                                         zero_division=0),
            "accuracy": accuracy_score(true, pred)}


def fit(x, y, n_classes, wd, balanced, device, epochs=30, lr=1e-3, seed=0):
    """Softmax regression with AdamW on cached features."""
    torch.manual_seed(seed)
    head = torch.nn.Linear(x.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    weight = None
    if balanced:   # inverse class frequency, since macro F1 weighs every species equally
        counts = torch.bincount(y, minlength=n_classes).float().clamp(min=1)
        weight = (counts.sum() / counts / n_classes).to(device)
    x, y = x.to(device), y.to(device)
    for _ in range(epochs):
        for idx in torch.randperm(len(y), device=device).split(1024):
            loss = F.cross_entropy(head(x[idx]), y[idx], weight=weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return head


def best_head(task, x_all, y_all, fold, fold_name, device, verbose=True):
    """Fit the head for every (wd, balanced) setting and keep the best on val.

    Returns `(head, x, val, test, wd, balanced)`, where `x` is every image's features
    standardised with the train statistics, ready to feed to `head`.
    """
    tr = fold["train"]
    mu, sd = x_all[tr].mean(0), x_all[tr].std(0) + 1e-6   # train statistics only
    x = (x_all - mu) / sd
    best = None
    for wd in (1e-4, 1e-2, 1e-1):
        for balanced in (False, True):
            head = fit(x[tr], y_all[tr], task.n_classes, wd, balanced, device)
            with torch.no_grad():
                pred = {s: head(x[i].to(device)).argmax(1).cpu().numpy()
                        for s, i in fold.items() if s != "train"}
            val = scores(pred["val"], task.y[fold["val"]], task.n_classes)
            if verbose:
                print(f"{fold_name:13s} wd={wd:<6g} balanced={balanced!s:5s} "
                      f"val F1 {val['macro_f1_present']:.3f}", flush=True)
            if best is None or val["macro_f1_present"] > best[2]["macro_f1_present"]:
                test = scores(pred["test"], task.y[fold["test"]], task.n_classes)
                best = (head, x, val, test, wd, balanced)
    return best


def probe(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    task = load_task(image_dir=image_dir_for(448))
    x_all, y_all = load_feats(feat_name(args), task), torch.from_numpy(task.y)
    rows = []
    for fold_name in FOLDS:
        fold = load_fold(task, fold_name)
        _, _, val, test, wd, balanced = best_head(task, x_all, y_all, fold, fold_name, device)
        print(f"==> {fold_name}: test F1 {test['macro_f1_present']:.3f}, "
              f"acc {test['accuracy']:.3f} (wd={wd}, balanced={balanced})\n")
        rows.append({"method": f"linprobe-{feat_name(args)}", "split": fold_name, "wd": wd,
                     "balanced": balanced, "val_macro_f1_present": round(val["macro_f1_present"], 4),
                     "test_macro_f1_present": round(test["macro_f1_present"], 4),
                     "test_accuracy": round(test["accuracy"], 4)})
    write(rows)


def prompt(row):
    """BioCLIP's taxonomic + common name text, e.g. 'a photo of Animalia Chordata
    Mammalia Artiodactyla Bovidae Aepyceros melampus with common name impala.'"""
    ranks = ["kingdom", "phylum", "class", "order", "family", "genus"]
    words = [row[r].capitalize() for r in ranks if isinstance(row[r], str)]
    if row["taxonomy_level"] == "species":
        words.append(row["scientific_name"].split()[-1])
    return f"a photo of {' '.join(words)} with common name {row['common_name']}."


@torch.no_grad()
def zeroshot(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    task = load_task(image_dir=image_dir_for(448))
    model, _, tokenizer = load_bioclip(device)
    tax = pd.read_csv("data/taxonomy.csv").set_index("query")
    texts = [prompt(tax.loc[s]) for s in task.classes]
    print("example prompt:", texts[task.classes.index("mazama pandora")])
    t = model.encode_text(tokenizer(texts).to(device)).float()
    t = F.normalize(t, dim=-1).cpu()
    x = F.normalize(load_feats("bioclip", task), dim=-1)
    rows = []
    for fold_name in FOLDS:
        te = load_fold(task, fold_name)["test"]
        s = scores((x[te] @ t.T).argmax(1).numpy(), task.y[te], task.n_classes)
        print(f"{fold_name}: zero-shot test F1 {s['macro_f1_present']:.3f}, "
              f"acc {s['accuracy']:.3f}")
        rows.append({"method": "zeroshot-bioclip", "split": fold_name,
                     "test_macro_f1_present": round(s["macro_f1_present"], 4),
                     "test_accuracy": round(s["accuracy"], 4)})
    write(rows)


class GrayFlag(Dataset):
    """Whether each labelled image is black-and-white (infrared), in `task.df` order.

    Colour channels of an infrared frame are near-identical: the mean channel difference
    is below 1 for them and above 8 for colour frames, with almost nothing in between.
    """

    def __init__(self, task, threshold=3.0):
        self.names, self.dir, self.threshold = task.df["file_name"].to_numpy(), task.image_dir, threshold

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        im = Image.open(self.dir / self.names[i])
        im.draft("RGB", (200, 112))   # decode at reduced size, enough for a colour check
        a = np.asarray(im.convert("RGB")).astype(np.int16)
        diff = (np.abs(a[..., 0] - a[..., 1]).mean() + np.abs(a[..., 1] - a[..., 2]).mean()) / 2
        return diff < self.threshold


def gray_flags(task, num_workers):
    """Black-and-white flag per labelled image, computed once and cached in feats/."""
    path = FEATS / "gray.npz"
    if path.exists():
        d = np.load(path, allow_pickle=True)
        if (d["file_name"] == task.df["file_name"].to_numpy()).all():
            return d["gray"]
    t0 = time.time()
    gray = np.concatenate([b.numpy() for b in DataLoader(GrayFlag(task), batch_size=512,
                                                         num_workers=num_workers)])
    FEATS.mkdir(exist_ok=True)
    np.savez(path, gray=gray, file_name=task.df["file_name"].to_numpy())
    print(f"flagged {gray.mean():.0%} of images as black-and-white in {time.time() - t0:.0f}s")
    return gray


WATCH = ["penelope purpurascens", "mazama pandora", "aepyceros melampus"]

# Pia's MegaDetector v5a run over every frame (read only): normalised XYWH boxes, category "1" = animal
MD_JSON = Path("/zfs/ai4good/groups/camera-traps/pdelap-background-20261018/data/md-v5a.json")


def detections(task):
    """Most confident animal box per labelled image, in `task.df` order.

    Columns `conf, x, y, w, h` (normalised); images with no animal box get conf 0 and the
    whole frame as their box. Cached in feats/ since the JSON takes a while to parse.
    """
    path = FEATS / "md.csv"
    if path.exists():
        df = pd.read_csv(path)
        if (df["file_name"].to_numpy() == task.df["file_name"].to_numpy()).all():
            return df
    best = {}
    for im in json.load(MD_JSON.open())["images"]:
        boxes = [d for d in im.get("detections") or [] if d["category"] == "1"]
        if boxes:
            d = max(boxes, key=lambda d: d["conf"])
            best[im["file"]] = (d["conf"], *d["bbox"])
    rows = [best.get(f, (0.0, 0.0, 0.0, 1.0, 1.0)) for f in task.df["file_name"]]
    df = pd.DataFrame(rows, columns=["conf", "x", "y", "w", "h"])
    df.insert(0, "file_name", task.df["file_name"].to_numpy())
    FEATS.mkdir(exist_ok=True)
    df.to_csv(path, index=False)
    return df


def burst_average(prob, burst, weight):
    """Weighted mean of the class probabilities over each burst, given back to every frame.

    A burst whose weights are all zero falls back to the plain mean.
    """
    w = pd.Series(weight, dtype=float)
    total = w.groupby(burst).transform("sum").to_numpy()
    w = np.where(total > 0, w.to_numpy(), 1.0)
    num = pd.DataFrame(prob * w[:, None]).groupby(burst).transform("sum").to_numpy()
    return num / pd.Series(w).groupby(burst).transform("sum").to_numpy()[:, None]


def burst_weights(prob, burst, det, top_k=3):
    """Each frame's say in its burst's vote, per method. None of them looks at labels."""
    conf = det["conf"].to_numpy()
    rank = pd.Series(conf).groupby(burst).rank(ascending=False, method="first").to_numpy()
    return {
        "A equal": np.ones(len(prob)),
        "B model confidence": prob.max(1),
        "C detector confidence": conf,
        f"D detector top-{top_k}": (rank <= top_k) * (conf > 0),
        "E detector conf x box size": conf * np.sqrt(det["w"].to_numpy() * det["h"].to_numpy()),
    }


def analyse(args):
    """Per-image vs burst-averaged F1, broken down by colour and by day/night."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    task = load_task(image_dir=image_dir_for(448))
    gray = gray_flags(task, args.num_workers)
    night = ~task.df["hour"].between(7, 17).to_numpy()   # taken before 7:00 or from 18:00
    burst = task.df[task.burst_col].to_numpy()
    x_all, y_all = load_feats(feat_name(args), task), torch.from_numpy(task.y)
    rows = []
    for fold_name in FOLDS:
        fold = load_fold(task, fold_name)
        head, x, *_ = best_head(task, x_all, y_all, fold, fold_name, device, verbose=False)
        te = fold["test"]
        with torch.no_grad():
            prob = head(x[te].to(device)).softmax(1).cpu().numpy()
        # every frame of a burst gets the burst's mean probabilities, so scores stay per image
        burst_prob = pd.DataFrame(prob).groupby(burst[te]).transform("mean").to_numpy()
        preds = {"image": prob.argmax(1), "burst": burst_prob.argmax(1)}
        subsets = {"all": np.ones(len(te), bool), "colour": ~gray[te], "black_white": gray[te],
                   "day": ~night[te], "night": night[te]}
        for agg, pred in preds.items():
            for sub, m in subsets.items():
                s = scores(pred[m], task.y[te][m], task.n_classes)
                rows.append({"backbone": feat_name(args), "split": fold_name, "aggregation": agg,
                             "subset": sub, "n_images": int(m.sum()),
                             "macro_f1_present": round(s["macro_f1_present"], 4),
                             "accuracy": round(s["accuracy"], 4)})
            per_class = f1_score(task.y[te], pred, labels=range(task.n_classes), average=None,
                                 zero_division=0)
            for name in WATCH:
                c = task.classes.index(name)
                rows.append({"backbone": feat_name(args), "split": fold_name, "aggregation": agg,
                             "subset": f"species: {name}",
                             "n_images": int((task.y[te] == c).sum()),
                             "macro_f1_present": round(per_class[c], 4)})
    df = pd.DataFrame(rows)
    print(df.pivot_table(index=["split", "subset"], columns="aggregation",
                         values="macro_f1_present", sort=False).round(3).to_string())
    path = Path(f"results/analysis-{feat_name(args)}.csv")
    path.parent.mkdir(exist_ok=True)
    df.to_csv(path, index=False)
    print(f"wrote {path}")
    compare_bursts(args, task, x_all, y_all, burst, device)


def compare_bursts(args, task, x_all, y_all, burst, device):
    """Score every burst-weighting method on val and test; the method is chosen on val."""
    det = detections(task)
    rows = []
    for fold_name in FOLDS:
        fold = load_fold(task, fold_name)
        head, x, *_ = best_head(task, x_all, y_all, fold, fold_name, device, verbose=False)
        for split in ("val", "test"):
            idx = fold[split]
            with torch.no_grad():
                prob = head(x[idx].to(device)).softmax(1).cpu().numpy()
            rows.append({"split": fold_name, "eval": split, "method": "per image",
                         "f1": scores(prob.argmax(1), task.y[idx], task.n_classes)["macro_f1_present"]})
            for method, w in burst_weights(prob, burst[idx], det.iloc[idx]).items():
                pred = burst_average(prob, burst[idx], w).argmax(1)
                rows.append({"split": fold_name, "eval": split, "method": method,
                             "f1": scores(pred, task.y[idx], task.n_classes)["macro_f1_present"]})
    df = pd.DataFrame(rows)
    table = df.pivot_table(index="method", columns=["split", "eval"], values="f1", sort=False)
    print("\nburst weighting, macro F1 over present species (choose on val, report test):")
    print(table.round(3).to_string())
    for fold_name in FOLDS:
        best = table[(fold_name, "val")].idxmax()
        print(f"  {fold_name}: best on val is {best!r} -> test {table.loc[best, (fold_name, 'test')]:.3f}")
    path = Path(f"results/bursts-{feat_name(args)}.csv")
    df.to_csv(path, index=False)
    print(f"wrote {path}")


# the fine-tuned resnet50 baseline's saved test predictions (read only); they hold only the
# predicted class per image, so bursts can only be combined by vote, not by probabilities
BASELINE_PREDS = Path("/zfs/ai4good/groups/camera-traps/camera-trap/results/preds")


def burst_vote(pred, burst, weight, n_classes):
    """Each frame votes for its predicted class with `weight`; every frame gets its burst's winner."""
    return burst_average(np.eye(n_classes)[pred], burst, weight).argmax(1)


def votes(args):
    """Per-image vs burst-vote F1 for the fine-tuned baseline and the linear probes, on test.

    Voting is the only burst rule the baseline's saved predictions allow, so the probes are
    scored with the same rule here to keep the comparison fair.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    task = load_task(image_dir=image_dir_for(448))
    burst, conf = task.df[task.burst_col].to_numpy(), detections(task)["conf"].to_numpy()
    preds = {}   # (method, fold) -> [(test index, predicted class), ...], one entry per seed
    for fold_name in FOLDS:
        for f in sorted(BASELINE_PREDS.glob(f"{fold_name}_resnet50_s*_sz448_pre_e12.npz")):
            d = np.load(f)
            assert (d["true"] == task.y[d["index"]]).all(), f"{f.name} is out of sync with the task"
            preds.setdefault(("resnet50 fine-tuned", fold_name), []).append((d["index"], d["pred"]))
    for name in args.feats:
        x_all, y_all = load_feats(name, task), torch.from_numpy(task.y)
        for fold_name in FOLDS:
            fold = load_fold(task, fold_name)
            head, x, *_ = best_head(task, x_all, y_all, fold, fold_name, device, verbose=False)
            with torch.no_grad():
                pred = head(x[fold["test"]].to(device)).argmax(1).cpu().numpy()
            preds[(f"{name} probe", fold_name)] = [(fold["test"], pred)]
    rows = []
    for (method, fold_name), runs in preds.items():
        for idx, pred in runs:
            for rule, p in {"per image": pred,
                            "equal vote": burst_vote(pred, burst[idx], np.ones(len(idx)), task.n_classes),
                            "detector-weighted vote": burst_vote(pred, burst[idx], conf[idx], task.n_classes),
                            }.items():
                rows.append({"method": method, "split": fold_name, "rule": rule,
                             "f1": scores(p, task.y[idx], task.n_classes)["macro_f1_present"]})
    df = pd.DataFrame(rows)
    table = df.groupby(["method", "rule", "split"], sort=False)["f1"].agg(["mean", "std", "count"])
    table = table.unstack("split").round(3)
    print(table.to_string())
    df.to_csv("results/votes.csv", index=False)
    print("wrote results/votes.csv")


def write(rows):
    fields = ["method", "split", "wd", "balanced", "val_macro_f1_present",
              "test_macro_f1_present", "test_accuracy"]
    new = not OUT.exists()
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("a", newline="") as f:
        w = csv.DictWriter(f, fields)
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"appended to {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["extract", "probe", "zeroshot", "analyse", "votes"])
    ap.add_argument("--backbone", choices=["bioclip", "resnet50"], default="bioclip")
    ap.add_argument("--view", choices=["full", "crop"], default="full",
                    help="crop: the most confident MegaDetector box, from the full-resolution originals")
    ap.add_argument("--feats", nargs="+", default=["bioclip", "resnet50"],
                    help="votes: which cached features to probe alongside the baseline")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()
    {"extract": extract, "probe": probe, "zeroshot": zeroshot,
     "analyse": analyse, "votes": votes}[args.command](args)
