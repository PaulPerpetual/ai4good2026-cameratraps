"""Recompute the nested cross-validation for the linear discriminant with separate loop and
index logic, and compare with what val_compare.py saved. Synthetic data only."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import val_compare as vc
from pilot_a import Data, macro_f1, pool
cache, repo, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
C = 60
d = Data(cache, repo, C)
rng = np.random.default_rng(0)
order = sorted(d.train_cams, key=lambda c: (d.region.get(c, "none"), rng.random()))
fold_of_cam = {c: i % 5 for i, c in enumerate(order)}
tr = np.flatnonzero(d.fold == "train")
fold = np.array([fold_of_cam[c] for c in d.cam[tr]])
X, y, w, burst = d.X[tr], d.y[tr], d.w[tr], d.burst[tr]

def predict(train, query, shrinkage):
    """Pooled linear-discriminant predictions for `query` rows from a model fitted on `train` rows."""
    mu, sd = X[train].mean(0), X[train].std(0) + 1e-6
    Xt, Xq = (X[train] - mu) / sd, (X[query] - mu) / sd
    present = np.bincount(y[train], weights=w[train], minlength=C) > 0
    M, S = vc.lda_parts(Xt, y[train], w[train], present, C)
    z = vc.lda_scores(Xq, M, S, shrinkage, present)
    uniq, local = np.unique(burst[query], return_inverse=True)
    return pool(z, local, len(uniq)).argmax(1)[local]

nested, choice = np.full(len(tr), -1), []
for f in range(5):
    score = {}
    for s in vc.SHRINK_GRID:
        p = np.full(len(tr), -1)
        for g in range(5):
            if g != f:
                p[fold == g] = predict(np.flatnonzero((fold != f) & (fold != g)), np.flatnonzero(fold == g), s)
        score[s] = macro_f1(y[fold != f], p[fold != f], C)
    best = vc.first_max(vc.SHRINK_GRID, score)
    choice.append(best)
    nested[fold == f] = predict(np.flatnonzero(fold != f), np.flatnonzero(fold == f), best)
saved = pd.read_csv(out / "train_camera_predictions.csv.gz")
res = json.loads((out / "results.json").read_text()) if (out / "results.json").exists() else None
assert (saved["y"].to_numpy() == y).all() and (saved["fold"].to_numpy() == fold).all()
assert (saved["nested_lda"].to_numpy() == nested).all(), "nested predictions differ"
if res:
    assert [c["shrinkage"] for c in res["held_out_training_cameras"]["settings_chosen_per_fold"]["linear discriminant"]] == choice
print("nested cross-validation for the linear discriminant recomputed independently: ok; shrinkage per fold", choice)
