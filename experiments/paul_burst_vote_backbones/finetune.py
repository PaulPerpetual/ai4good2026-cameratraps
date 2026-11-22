"""LP-FT of BioCLIP on MegaDetector crops: start from the linear probe's head, then fine-tune
the whole image encoder gently, and score per image and with detector-weighted burst averaging.

    python finetune.py --split official_ood --seed 0
    python finetune.py --split official_ood --seed 0 --limit 512 --epochs 1   # quick smoke test

Each run appends one line to results/finetune.jsonl, saves its best checkpoint (fp16) under
checkpoints/ and the val/test class probabilities under results/finetune-preds/, so any burst
rule or per-species breakdown can be recomputed later without the GPU.

WiSE-FT is evaluated at the end for free: the weights are interpolated between the start
(BioCLIP + probe head) and the fine-tuned model, with the mixing weight chosen on val.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

import linprobe as lp
from iwildcam.data import image_dir_for, load_fold, load_task

OUT = Path("results/finetune.jsonl")
PREDS = Path("results/finetune-preds")
CKPT = Path("checkpoints")


class Labelled(Dataset):
    """Images at `idx` with their class index, for training."""

    def __init__(self, files, idx, y):
        self.files, self.idx, self.y = files, idx, y

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return self.files[self.idx[i]], self.y[self.idx[i]]


class Classifier(nn.Module):
    """BioCLIP's image encoder followed by a linear species head."""

    def __init__(self, visual, head):
        super().__init__()
        self.visual, self.head = visual, head

    def forward(self, x):
        return self.head(self.visual(x))


def probe_head(task, fold, fold_name):
    """The linear probe's best head, rewritten to take raw (unstandardised) features.

    The probe standardises features with the train mean `mu` and std `sd`, so
    `W (f - mu) / sd + b` becomes `(W / sd) f + (b - (W / sd) mu)`.
    """
    x_all, y_all = lp.load_feats("bioclip-crop", task), torch.from_numpy(task.y)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    head, _, val, _, wd, balanced = lp.best_head(task, x_all, y_all, fold, fold_name, device,
                                                 verbose=False)
    tr = fold["train"]
    mu, sd = x_all[tr].mean(0), x_all[tr].std(0) + 1e-6
    w = head.weight.detach().cpu() / sd
    out = nn.Linear(w.shape[1], w.shape[0])
    with torch.no_grad():
        out.weight.copy_(w)
        out.bias.copy_(head.bias.detach().cpu() - w @ mu)
    print(f"probe head: val F1 {val['macro_f1_present']:.3f} (wd={wd}, balanced={balanced})")
    return out


def transforms(norm):
    """Train-time augmentation on the crop, and the plain eval transform the probe used."""
    train = T.Compose([
        T.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.8, 1.25),
                            interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.RandomGrayscale(p=0.25),   # day frames that look like infrared ones
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        T.ToTensor(), norm])
    evaluate = T.Compose([T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
                          T.ToTensor(), norm])
    return train, evaluate


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    out = []
    for xb in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            out.append(model(xb.to(device, non_blocking=True)).float().softmax(1).cpu())
    return torch.cat(out).numpy()


def evaluate(prob, idx, task, burst, conf):
    """Macro F1 over present species, per image and with detector-weighted burst averaging."""
    y = task.y[idx]
    per_image = lp.scores(prob.argmax(1), y, task.n_classes)
    bursty = lp.burst_average(prob, burst[idx], conf[idx]).argmax(1)
    return {"f1_image": per_image["macro_f1_present"], "acc_image": per_image["accuracy"],
            "f1_burst": lp.scores(bursty, y, task.n_classes)["macro_f1_present"]}


def interpolate(start, end, alpha):
    """WiSE-FT weights: (1 - alpha) * start + alpha * end."""
    return {k: (1 - alpha) * start[k] + alpha * end[k] if start[k].is_floating_point() else end[k]
            for k in end}


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    task = load_task(image_dir=image_dir_for(448))
    full_fold = load_fold(task, args.split)
    fold = full_fold
    if args.limit:   # smoke test: a small random slice of every split
        rng = np.random.default_rng(args.seed)
        fold = {s: np.sort(rng.choice(i, min(args.limit, len(i)), replace=False))
                for s, i in full_fold.items()}
    burst, conf = task.df[task.burst_col].to_numpy(), lp.detections(task)["conf"].to_numpy()
    print(task.summary())
    print(f"{args.split} seed {args.seed}: " + ", ".join(f"{s}={len(i):,}" for s, i in fold.items()))

    # start: BioCLIP's image encoder + the probe head (fitted on the full train split)
    clip, preprocess, _ = lp.load_bioclip("cpu")
    norm = next(t for t in preprocess.transforms if isinstance(t, T.Normalize))
    model = Classifier(clip.visual, probe_head(task, full_fold, args.split)).to(device)
    del clip   # the text tower is not needed
    start = copy.deepcopy(model.state_dict())

    train_tf, eval_tf = transforms(norm)
    boxes = lp.crop_boxes(task)
    files = {"train": lp.Files(task, train_tf, boxes, lp.ORIGINALS),
             "eval": lp.Files(task, eval_tf, boxes, lp.ORIGINALS)}
    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(Labelled(files["train"], fold["train"], task.y), batch_size=args.batch_size,
                              shuffle=True, drop_last=True, generator=g, pin_memory=True,
                              num_workers=args.num_workers, persistent_workers=True)
    eval_loader = {s: DataLoader(Subset(files["eval"], fold[s]), batch_size=256, pin_memory=True,
                                 num_workers=args.num_workers) for s in ("val", "test")}

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total = args.epochs * len(train_loader)
    warmup = min(args.warmup_steps, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warmup if s < warmup else
                                              0.5 * (1 + math.cos(math.pi * (s - warmup) / (total - warmup))))

    def val_score():
        return evaluate(predict(model, eval_loader["val"], device), fold["val"], task, burst, conf)

    history, t_start = [], time.time()
    r = val_score()   # epoch 0 must reproduce the probe's val score (up to bf16 noise)
    print(f"epoch 0  val F1 image {r['f1_image']:.3f}  burst {r['f1_burst']:.3f}", flush=True)
    history.append({"epoch": 0, **r})
    best = (r["f1_burst"], 0, copy.deepcopy(model.state_dict()))
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, seen, running = time.time(), 0, 0.0
        for xb, yb in train_loader:
            yb = yb.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = F.cross_entropy(model(xb.to(device, non_blocking=True)).float(), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            seen += len(xb)
            running = 0.98 * running + 0.02 * loss.item() if running else loss.item()
            if step % 100 == 0:
                rate = seen / (time.time() - t0)
                left = (total - step) * args.batch_size / rate
                print(f"  step {step:,}/{total:,}  loss {running:.3f}  {rate:.0f} img/s  "
                      f"lr {sched.get_last_lr()[0]:.1e}  ~{left / 60:.0f} min left", flush=True)
        r = val_score()
        history.append({"epoch": epoch, "train_loss": running, **r})
        print(f"epoch {epoch}  val F1 image {r['f1_image']:.3f}  burst {r['f1_burst']:.3f}  "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        if r["f1_burst"] > best[0]:
            best = (r["f1_burst"], epoch, copy.deepcopy(model.state_dict()))

    # WiSE-FT between the start and the best epoch; alpha 1 is plain fine-tuning, 0 the probe
    tag = f"{args.split}_s{args.seed}" + (f"_limit{args.limit}" if args.limit else "")
    results = {}
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        model.load_state_dict(interpolate(start, best[2], alpha))
        prob = {s: predict(model, eval_loader[s], device) for s in ("val", "test")}
        results[alpha] = {s: evaluate(prob[s], fold[s], task, burst, conf) for s in prob}
        print(f"alpha {alpha:<4}  val burst {results[alpha]['val']['f1_burst']:.3f}  "
              f"test image {results[alpha]['test']['f1_image']:.3f}  "
              f"burst {results[alpha]['test']['f1_burst']:.3f}", flush=True)
        PREDS.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(PREDS / f"{tag}_alpha{alpha}.npz",
                            **{f"{s}_prob": prob[s].astype(np.float16) for s in prob},
                            **{f"{s}_index": fold[s] for s in prob})
    chosen = max(results, key=lambda a: results[a]["val"]["f1_burst"])
    print(f"==> best epoch {best[1]}, WiSE alpha {chosen} (on val): test F1 image "
          f"{results[chosen]['test']['f1_image']:.3f}, burst {results[chosen]['test']['f1_burst']:.3f}")

    CKPT.mkdir(exist_ok=True)
    torch.save({k: v.half() if v.is_floating_point() else v for k, v in best[2].items()},
               CKPT / f"bioclip-crop-lpft_{tag}.pt")
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("a") as f:
        f.write(json.dumps({"split": args.split, "seed": args.seed, "limit": args.limit,
                            "config": vars(args), "best_epoch": best[1], "wise_alpha": chosen,
                            "test": results[chosen]["test"], "val": results[chosen]["val"],
                            "by_alpha": {str(a): v for a, v in results.items()},
                            "history": history, "minutes": (time.time() - t_start) / 60}) + "\n")
    print(f"appended to {OUT} after {(time.time() - t_start) / 60:.0f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["official_ood", "random_burst"], default="official_ood")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="use only this many images per split")
    main(ap.parse_args())
