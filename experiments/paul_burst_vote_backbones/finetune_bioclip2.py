"""LP-FT of BioCLIP 2 on the crops: finetune.py's method, without WiSE-FT since it gave no gain.

    python finetune_bioclip2.py --epochs 6 --max-minutes 55
    python finetune_bioclip2.py --limit 512 --epochs 1     # quick smoke test

The best epoch by val burst F1 is saved after every improvement. Test probabilities are saved
for the ensemble but not scored here, so the test set stays closed until `ensemble.py test`.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import nn
from torch.utils.data import DataLoader, Subset

import linprobe as lp
from finetune import Classifier, Labelled, evaluate, predict, transforms
from iwildcam.data import image_dir_for, load_fold, load_task

MODEL = "hf-hub:imageomics/bioclip-2"
PREDS = Path("results/finetune-bioclip2-preds")
CKPT = Path("checkpoints")
OUT = Path("results/finetune-bioclip2.jsonl")


def probe_head(task, fold):
    """As finetune.probe_head, on the BioCLIP 2 crop features."""
    x_all, y_all = lp.load_feats("bioclip2-crop", task), torch.from_numpy(task.y)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    head, _, val, _, wd, balanced = lp.best_head(task, x_all, y_all, fold, "official_ood", device, verbose=False)
    tr = fold["train"]
    mu, sd = x_all[tr].mean(0), x_all[tr].std(0) + 1e-6
    w = head.weight.detach().cpu() / sd
    out = nn.Linear(w.shape[1], w.shape[0])
    with torch.no_grad():
        out.weight.copy_(w)
        out.bias.copy_(head.bias.detach().cpu() - w @ mu)
    print(f"probe head: val F1 {val['macro_f1_present']:.3f} (wd={wd}, balanced={balanced})", flush=True)
    return out


def main(args):
    device = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    task = load_task(image_dir=image_dir_for(448))
    full_fold = load_fold(task, "official_ood")
    fold = full_fold
    if args.limit:
        rng = np.random.default_rng(args.seed)
        fold = {s: np.sort(rng.choice(i, min(args.limit, len(i)), replace=False)) for s, i in full_fold.items()}
    burst, conf = task.df[task.burst_col].to_numpy(), lp.detections(task)["conf"].to_numpy()
    tag = f"official_ood_s{args.seed}" + (f"_limit{args.limit}" if args.limit else "")

    clip, _, preprocess = open_clip.create_model_and_transforms(MODEL)
    norm = next(t for t in preprocess.transforms if isinstance(t, T.Normalize))
    model = Classifier(clip.visual, probe_head(task, full_fold)).to(device)
    del clip

    train_tf, eval_tf = transforms(norm)
    boxes = lp.crop_boxes(task)
    train_files, eval_files = lp.Files(task, train_tf, boxes, lp.ORIGINALS), lp.Files(task, eval_tf, boxes, lp.ORIGINALS)
    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(Labelled(train_files, fold["train"], task.y), batch_size=args.batch_size, shuffle=True,
                              drop_last=True, generator=g, pin_memory=True, num_workers=args.num_workers,
                              persistent_workers=True)
    eval_loader = {s: DataLoader(Subset(eval_files, fold[s]), batch_size=256, pin_memory=True,
                                 num_workers=args.num_workers) for s in ("val", "test")}

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total = args.epochs * len(train_loader)
    warmup = max(1, min(args.warmup_steps, total // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warmup if s < warmup else
                                              0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup))))

    def val_score():
        return evaluate(predict(model, eval_loader["val"], device), fold["val"], task, burst, conf)

    CKPT.mkdir(exist_ok=True)
    PREDS.mkdir(parents=True, exist_ok=True)
    ckpt = CKPT / f"bioclip2-crop-lpft_{tag}.pt"
    history, t_start = [], time.time()
    r = val_score()
    print(f"epoch 0  val F1 image {r['f1_image']:.3f}  burst {r['f1_burst']:.3f}", flush=True)
    history.append({"epoch": 0, **r})
    best = (r["f1_burst"], 0, copy.deepcopy(model.state_dict()))
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, seen, running = time.time(), 0, 0.0
        for xb, yb in train_loader:
            yb = yb.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
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
                print(f"  step {step:,}/{total:,}  loss {running:.3f}  {rate:.0f} img/s  "
                      f"~{(total - step) * args.batch_size / rate / 60:.0f} min left", flush=True)
        r = val_score()
        history.append({"epoch": epoch, "train_loss": running, **r})
        print(f"epoch {epoch}  val F1 image {r['f1_image']:.3f}  burst {r['f1_burst']:.3f}  "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        if r["f1_burst"] > best[0]:
            best = (r["f1_burst"], epoch, copy.deepcopy(model.state_dict()))
            torch.save({k: v.half() if v.is_floating_point() else v for k, v in best[2].items()}, ckpt)
        if (time.time() - t_start) / 60 > args.max_minutes:
            print(f"stopping after epoch {epoch}: past {args.max_minutes} min", flush=True)
            break

    model.load_state_dict(best[2])
    prob = {s: predict(model, eval_loader[s], device) for s in ("val", "test")}
    final = evaluate(prob["val"], fold["val"], task, burst, conf)
    np.savez_compressed(PREDS / f"{tag}.npz", **{f"{s}_prob": prob[s].astype(np.float16) for s in prob},
                        **{f"{s}_index": fold[s] for s in prob})
    with OUT.open("a") as f:
        f.write(json.dumps({"seed": args.seed, "limit": args.limit, "config": vars(args), "best_epoch": best[1],
                            "val": final, "history": history, "minutes": (time.time() - t_start) / 60}) + "\n")
    print(f"==> best epoch {best[1]}: val F1 image {final['f1_image']:.3f}, burst {final['f1_burst']:.3f}; "
          f"test probabilities saved, not scored; {(time.time() - t_start) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--max-minutes", type=float, default=55)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    main(ap.parse_args())
