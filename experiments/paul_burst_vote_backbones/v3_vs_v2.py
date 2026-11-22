"""Is DINOv3 really behind DINOv2 on official_ood val? Test is never read.

Paired bootstrap over val cameras at George's settings, then each model again at the shrinkage its
own held-out training cameras choose (same folds and rule as val_compare.py), since 0.3 was chosen
on DINOv2 features.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import linprobe as lp
from iwildcam.data import image_dir_for, load_task

G = "/zfs/ai4good/groups/camera-traps"
sys.path.insert(0, f"{G}/users/george/scripts")
from score_test import burst_weights, lda_parts, lda_scores, load_labelled, macro_f1, pool  # noqa: E402

C = 60
SHRINK = (1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001)
CACHES = {"v2L_448": f"{G}/features/dinov2_vitl14_reg_448_crop", "v3L_256": "feats/dinov3_vitl16_crop",
          "v3L_448": "feats/dinov3_vitl16_448_crop"}
task = load_task(image_dir=image_dir_for(448))
conf = pd.Series(lp.detections(task)["conf"].to_numpy(), index=task.df["file_name"].to_numpy())
folds = pd.read_csv(f"{G}/users/george/results/val_compare_vitl_crop/train_camera_predictions.csv.gz")
cam_fold = dict(zip(folds["location"], folds["fold"]))


def fit(x, y, w):
    mu, sd = x.mean(0), x.std(0) + 1e-6
    present = np.bincount(y, weights=w, minlength=C) > 0
    return (mu, sd, *lda_parts((x - mu) / sd, y, w, present, C), present)


def own_shrinkage(x, y, df):
    cv, bt = df["location"].map(cam_fold).to_numpy(), df["burst_id"].to_numpy()
    w = burst_weights(df)[1]
    pred = {s: np.empty(len(y), dtype=np.int64) for s in SHRINK}
    for f in np.unique(cv):
        tr, q = cv != f, cv == f
        mu, sd, M, S, present = fit(x[tr], y[tr], w[tr])
        codes, _ = pd.factorize(bt[q])
        for s in SHRINK:
            pred[s][q] = pool(lda_scores((x[q] - mu) / sd, M, S, s, present), codes, codes.max() + 1).argmax(1)[codes]
    scores = {s: macro_f1(y, pred[s], C) for s in SHRINK}
    print("  held-out F1 by shrinkage:", {s: round(v, 4) for s, v in scores.items()}, flush=True)
    return max(SHRINK, key=scores.get)


preds, order = {}, None
for name, cache in CACHES.items():
    df, x = load_labelled(Path(cache))
    order = df["file_name"].to_numpy() if order is None else order
    assert (df["file_name"].to_numpy() == order).all(), "row order"
    y = df["y"].to_numpy().astype(np.int64)
    tr, va = df["official_ood"].eq("train").to_numpy(), df["official_ood"].eq("val").to_numpy()
    mu, sd, M, S, present = fit(x[tr], y[tr], burst_weights(df[tr])[1])
    print(name, flush=True)
    own = own_shrinkage(x[tr], y[tr], df[tr].reset_index(drop=True))
    for label, s in (("s=0.3", 0.3), (f"own s={own}", own)):
        preds[name, label.split("=")[0].strip()] = lda_scores((x[va] - mu) / sd, M, S, s, present).argmax(1)
        print(f"  {label}: image {macro_f1(y[va], preds[name, label.split('=')[0].strip()], C):.4f}", flush=True)

yv, cams = y[va], df["location"].to_numpy()[va]
burst, c = df["burst_id"].to_numpy()[va], conf.loc[df["file_name"][va]].to_numpy()
vote = {k: lp.burst_vote(p, burst, c, C) for k, p in preds.items()}
for k in preds:
    print(k, f"image {macro_f1(yv, preds[k], C):.4f} vote {macro_f1(yv, vote[k], C):.4f}")

where = {k: np.flatnonzero(cams == k) for k in np.unique(cams)}
rng, keys = np.random.default_rng(0), list(where)
draws = [np.concatenate([where[k] for k in rng.choice(keys, len(keys))]) for _ in range(2000)]
for a, b in ((("v3L_448", "s"), ("v2L_448", "s")), (("v3L_256", "s"), ("v2L_448", "s")),
             (("v3L_448", "own s"), ("v2L_448", "own s")), (("v3L_256", "own s"), ("v2L_448", "own s"))):
    for label, pa, pb in (("image", preds[a], preds[b]), ("vote", vote[a], vote[b])):
        d = [macro_f1(yv[s], pa[s], C) - macro_f1(yv[s], pb[s], C) for s in draws]
        print(f"{a} minus {b} {label}: {macro_f1(yv, pa, C) - macro_f1(yv, pb, C):+.4f} "
              f"CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}] share above 0 {np.mean(np.array(d) > 0):.3f}", flush=True)
