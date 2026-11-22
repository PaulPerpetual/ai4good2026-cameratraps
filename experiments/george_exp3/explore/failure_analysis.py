"""Where do frozen crop pipelines fail? Held-out training cameras and validation only; no test data.

Uses the camera-wise cross-validation predictions val_compare.py saved (every photo predicted by
a model that never saw its camera) and refits the linear discriminant for the validation cameras.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from cache_features import best_detections, crop_boxes  # noqa: E402
from score_random_burst import burst_average  # noqa: E402
from score_test import lda_parts, lda_scores, load_labelled, macro_f1  # noqa: E402
from iwildcam.data import load_task  # noqa: E402

G = "/zfs/ai4good/groups/camera-traps"
C = 60
task = load_task(require_files=False)
cls = np.array(task.classes)
meta = task.df.set_index("file_name")
tax = pd.read_csv(f"{G}/users/george/data/taxonomy.csv")
print("taxonomy columns:", tax.columns.tolist())
key = next(c for c in tax.columns if c.lower() in ("species", "category", "name"))
tax = tax.set_index(tax[key].str.lower())
ytrain = task.y[task.df["official_ood"].eq("train").to_numpy()]
n_train = np.bincount(ytrain, minlength=C)


def f1s(t, p):
    tp = np.bincount(t[t == p], minlength=C).astype(float)
    nt = np.bincount(t, minlength=C).astype(float)
    np_ = np.bincount(p, minlength=C).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(nt > 0, 2 * tp / np.maximum(nt + np_, 1), np.nan)


tax = tax[~tax.index.duplicated()]


def rel(a, b):
    a, b = cls[a], cls[b]
    if a.split()[0] == b.split()[0]:
        return "genus"
    for lvl in ("family", "order"):
        if a in tax.index and b in tax.index and pd.notna(tax.loc[a, lvl]) and tax.loc[a, lvl] == tax.loc[b, lvl]:
            return lvl
    return "unrelated"


def report(name, df):
    t, p = df.y.to_numpy(), df.p.to_numpy()
    names = df.file_name.tolist()
    conf = best_detections(names)[:, 0]
    boxes = crop_boxes(names)[0]
    side = boxes[:, 2] - boxes[:, 0]
    hour = meta.loc[names, "hour"].to_numpy()
    night = (hour < 6) | (hour >= 19)
    burst = df.burst_id.to_numpy()
    vote = burst_average(np.eye(C)[p], burst, conf).argmax(1)
    err = t != p
    print(f"\n===== {name}: {len(t):,} photos, {df.location.nunique()} cameras, {len(np.unique(t))} species")
    print(f"macro F1 per photo {macro_f1(t, p, C):.3f}, vote {macro_f1(t, vote, C):.3f}; accuracy {1 - err.mean():.3f}, vote {np.mean(vote == t):.3f}")

    print("\n-- by what the photo looks like (accuracy, share of all errors)")
    groups = {"no animal box": conf < .2, "box, small (<15% width)": (conf >= .2) & (side < .15),
              "box, medium (15-35%)": (conf >= .2) & (side >= .15) & (side < .35), "box, large (>35%)": (conf >= .2) & (side >= .35),
              "night": night, "day": ~night}
    for k, m in groups.items():
        print(f"  {k:26s} {m.mean():5.1%} of photos  acc {1 - err[m].mean():.3f}  vote acc {np.mean(vote[m] == t[m]):.3f}  {err[m].sum() / err.sum():5.1%} of errors")

    print("\n-- by how many training photos the species has (mean F1)")
    f, fv = f1s(t, p), f1s(t, vote)
    bins = [(0, 100), (100, 500), (500, 2000), (2000, 10**7)]
    for lo, hi in bins:
        s = (n_train >= lo) & (n_train < hi) & ~np.isnan(f)
        print(f"  {lo:>5}-{hi if hi < 10**7 else 'up':<6} {s.sum():2d} species  F1 {np.nanmean(f[s]):.3f}  vote F1 {np.nanmean(fv[s]):.3f}")

    pred_n, true_n = np.bincount(p, minlength=C), np.bincount(t, minlength=C)
    print("\n-- where errors go")
    c = pd.Series(zip(t[err], p[err])).value_counts()
    rels = pd.Series([rel(a, b) for a, b in zip(t[err], p[err])]).value_counts(normalize=True)
    print("  relation of true and predicted species:", rels.round(3).to_dict())
    common = np.argsort(-n_train)[:10]
    print(f"  errors going to the 10 most common training species: {np.isin(p[err], common).mean():.1%}")
    print(f"  predicted more than 1.5x as often as it occurs: {[str(cls[i]) for i in np.argsort(-pred_n / np.maximum(true_n, 1)) if true_n[i] >= 30 and pred_n[i] / true_n[i] > 1.5][:6]}")
    print("  top confusions (true -> predicted, photos):")
    for (a, b), n in c.head(8).items():
        print(f"    {cls[a]} -> {cls[b]}: {n} ({n / true_n[a]:.0%} of that species, {rel(a, b)})")

    print("\n-- worst species with at least 30 photos (F1, vote F1, photos, training photos)")
    idx = [i for i in np.argsort(f) if true_n[i] >= 30 and not np.isnan(f[i])][:8]
    for i in idx:
        print(f"    {cls[i]:28s} {f[i]:.2f}  {fv[i]:.2f}  {true_n[i]:5d}  {n_train[i]:6d}")

    per = []
    for cam, g in df.assign(e=err).groupby("location"):
        per.append((cam, len(g), g.e.mean(), g.y.nunique()))
    per = pd.DataFrame(per, columns=["cam", "n", "err", "species"]).sort_values("err", ascending=False)
    w = per[per.n >= 50]
    print(f"\n-- cameras (at least 50 photos): {len(w)}; error rate median {w.err.median():.2f}, 10% worst above {w.err.quantile(.9):.2f}")
    worst = per.assign(e=per.n * per.err).sort_values("e", ascending=False)
    print(f"  worst 10% of cameras hold {worst.e.head(max(1, len(per) // 10)).sum() / worst.e.sum():.0%} of all errors")


def heldout_preds(cache, folds, shrink):
    """Per-photo predictions at held-out training cameras, one refit per val_compare fold."""
    df, X = load_labelled(Path(cache))
    df["row"] = np.arange(len(df))
    tr = df[df["official_ood"].eq("train")].merge(folds[["file_name", "fold"]], on="file_name")
    Xt = X[tr.row.to_numpy()]
    y = tr.y.to_numpy().astype(np.int64)
    codes, _ = pd.factorize(tr.burst_id)
    w = (1.0 / np.bincount(codes)[codes]).astype(np.float32)
    p = np.empty(len(tr), dtype=int)
    for f in sorted(tr.fold.unique()):
        fit, q = tr.fold.to_numpy() != f, tr.fold.to_numpy() == f
        mu, sd = Xt[fit].mean(0), Xt[fit].std(0) + 1e-6
        present = np.bincount(y[fit], weights=w[fit], minlength=C) > 0
        M, S = lda_parts((Xt[fit] - mu) / sd, y[fit], w[fit], present, C)
        p[q] = lda_scores((Xt[q] - mu) / sd, M, S, shrink, present).argmax(1)
    return tr.assign(p=p)[["file_name", "location", "burst_id", "y", "p"]]


def val_preds(cache, shrink):
    df, X = load_labelled(Path(cache))
    tr = df["official_ood"].eq("train").to_numpy()
    va = df["official_ood"].eq("val").to_numpy()
    y = df.y.to_numpy().astype(np.int64)
    codes, _ = pd.factorize(df.burst_id)
    w = (1.0 / np.bincount(codes)[codes]).astype(np.float32)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xs = (X - mu) / sd
    present = np.bincount(y[tr], weights=w[tr], minlength=C) > 0
    M, S = lda_parts(Xs[tr], y[tr], w[tr], present, C)
    d = df[va].reset_index(drop=True)
    d["p"] = lda_scores(Xs[va], M, S, shrink, present).argmax(1)
    return d[["file_name", "location", "burst_id", "y", "p"]]


R = f"{G}/users/george/results"
for label, sel, cache in (("ResNet-50 crop", "val_compare_resnet50_crop", f"{G}/features/resnet50_448_crop"),
                          ("DINOv2-L crop", "val_compare_vitl_crop", f"{G}/features/dinov2_vitl14_reg_448_crop")):
    shrink = json.load(open(f"{R}/{sel}/selection.json"))["chosen"]["linear discriminant"]["shrinkage"]
    d = pd.read_csv(f"{R}/{sel}/train_camera_predictions.csv.gz")
    report(f"{label}, held-out training cameras", heldout_preds(cache, d, shrink))
    report(f"{label}, validation cameras", val_preds(cache, shrink))
