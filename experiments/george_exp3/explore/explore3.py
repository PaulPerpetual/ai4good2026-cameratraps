"""Do averaged crop views help? Held-out training cameras and validation only.

usage: explore3.py resnet50|dinov2l  (fixed head from explore2: its best shrink and lambda)
Each config averages the named views' features, for fitting and for scoring alike.
"""
import sys

import numpy as np
import pandas as pd

NAME = sys.argv[1]
sys.argv = ["x", NAME]
exec(open("scripts/explore_heads.py").read().split("s0 = json.load")[0])

train_mask = ix.official_ood.eq("train").to_numpy()
SPLITS = {"heldout": [(ix.cv.to_numpy() != f, ix.cv.to_numpy() == f) for f in sorted(set(ix.cv) - {-1})],
          "val": [(train_mask, ix.official_ood.eq("val").to_numpy())]}
M = {"resnet50": "resnet50", "dinov2l": "dinov2_vitl14_reg"}[NAME]
V = f"{G}/users/george/features_views/{M}_448_crop_"
views = {"crop": X_crop}
for v in ("flip", "pad05", "pad40"):
    ixv, views[v] = load(V + v)
    assert (ixv.file_name == ix.file_name).all()
HEAD = {"resnet50": (0.03, 4), "dinov2l": (0.1, 8)}[NAME]
CFG = {"crop": ["crop"], "crop+flip": ["crop", "flip"], "crop+pad40": ["crop", "pad40"], "crop+pad05": ["crop", "pad05"],
       "crop+flip+pad40": ["crop", "flip", "pad40"], "all4": ["crop", "flip", "pad05", "pad40"]}
res = {}
for sn, splits in SPLITS.items():
    store = {k: [] for k in CFG}
    truth, cams = [], []
    for fit, q in splits:
        fit = fit & train_mask
        q = q & lab
        for k, vs in CFG.items():
            X = np.mean([views[v] for v in vs], axis=0)
            F = Fit(X, fit)
            ab = F.weights(*HEAD, 0)
            p = F.score(X[q], ab).argmax(1)
            v = burst_average(np.eye(C)[p], burst[q], conf[q]).argmax(1)
            store[k].append((p, v))
        truth.append(y[q]); cams.append(ix.location.to_numpy()[q])
        print(f"  {sn} fold done", flush=True)
    res[sn] = (np.concatenate(truth), np.concatenate(cams),
               {k: [np.concatenate([x[i] for x in v]) for i in range(2)] for k, v in store.items()})
rows = []
for k in CFG:
    h, v = res["heldout"], res["val"]
    rows.append({"views": k, "h_photo": macro_f1(h[0], h[2][k][0], C), "h_vote": macro_f1(h[0], h[2][k][1], C),
                 "v_photo": macro_f1(v[0], v[2][k][0], C), "v_vote": macro_f1(v[0], v[2][k][1], C)})
d = pd.DataFrame(rows).round(4)
d.to_csv(f"{R}/explore3_{NAME}.csv", index=False)
print(d.to_string(index=False))


def boot(sn, k, i, n=400):
    t, cam, st = res[sn]
    a, b = st[k][i], st["crop"][i]
    cs = np.unique(cam)
    idx = {c: np.flatnonzero(cam == c) for c in cs}
    rng = np.random.default_rng(0)
    ds = []
    for _ in range(n):
        s = np.concatenate([idx[c] for c in rng.choice(cs, len(cs))])
        ds.append(macro_f1(t[s], a[s], C) - macro_f1(t[s], b[s], C))
    return np.percentile(ds, 2.5), np.percentile(ds, 97.5), np.mean(np.array(ds) > 0)


best = d.iloc[1:].assign(m=lambda x: x.h_photo + x.h_vote).sort_values("m", ascending=False).iloc[0].views
print("best by held-out:", best)
for sn in ("heldout", "val"):
    for i, name in enumerate(("per photo", "vote")):
        lo, hi, pos = boot(sn, best, i)
        print(f"{sn:8s} {name:10s} {best} minus crop: 95% [{lo:+.4f}, {hi:+.4f}]  better in {pos:.0%}")
