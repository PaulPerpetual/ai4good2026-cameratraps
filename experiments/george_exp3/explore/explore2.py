"""Second pass: camera-noise head grid, feature pooling over bursts, error budget, error bars.

Held-out training cameras and validation only; no test-camera data is read.
"""
import importlib.util
import sys

import numpy as np
import pandas as pd

NAME = sys.argv[1]
sys.argv = ["x", NAME]
src = open("scripts/explore_heads.py").read().split("s0 = json.load")[0]
exec(src)   # data loading, Fit, burst_average, macro_f1

train_mask = ix.official_ood.eq("train").to_numpy()
SPLITS = {"heldout": [(ix.cv.to_numpy() != f, ix.cv.to_numpy() == f) for f in sorted(set(ix.cv) - {-1})],
          "val": [(train_mask, ix.official_ood.eq("val").to_numpy())]}
SHR = [0.3, 0.1, 0.03, 0.01]
LAM = [0, 2, 4, 8, 16, 32]


def pool_features(Xq, b, c):
    """Detector-weighted mean feature per burst, given back to every photo of the burst."""
    codes, _ = pd.factorize(b)
    w = pd.Series(c).groupby(codes).transform("sum").to_numpy()
    w = np.where(w > 0, c, 1.0)
    tot = np.bincount(codes, weights=w)
    out = np.zeros((codes.max() + 1, Xq.shape[1]))
    np.add.at(out, codes, Xq * w[:, None])
    return (out / tot[:, None])[codes].astype(np.float32)


stores = {}
for set_name, splits in SPLITS.items():
    t, cam, bu, cf, store = [], [], [], [], {}
    for fit, q in splits:
        fit = fit & train_mask
        q = q & lab
        F = Fit(X_crop, fit)
        Xq, b, c = X_crop[q], burst[q], conf[q]
        Xp = pool_features(Xq, b, c)
        for s in SHR:
            for lam in LAM:
                ab = F.weights(s, lam, 0)
                p = F.score(Xq, ab).argmax(1)
                v = burst_average(np.eye(C)[p], b, c).argmax(1)
                fp = F.score(Xp, ab).argmax(1)
                store.setdefault((s, lam), []).append((p, v, fp))
        t.append(y[q]); cam.append(ix.location.to_numpy()[q]); bu.append(b); cf.append(c)
        print(f"  {set_name} fold done", flush=True)
    t, cam, bu, cf = map(np.concatenate, (t, cam, bu, cf))
    stores[set_name] = (t, cam, bu, cf, {k: [np.concatenate([x[i] for x in v]) for i in range(3)] for k, v in store.items()})

rows = []
for k in stores["val"][4]:
    h = stores["heldout"]; v = stores["val"]
    rows.append({"s": k[0], "lam": k[1],
                 "h_photo": macro_f1(h[0], h[4][k][0], C), "h_vote": macro_f1(h[0], h[4][k][1], C), "h_feat": macro_f1(h[0], h[4][k][2], C),
                 "v_photo": macro_f1(v[0], v[4][k][0], C), "v_vote": macro_f1(v[0], v[4][k][1], C), "v_feat": macro_f1(v[0], v[4][k][2], C)})
d = pd.DataFrame(rows)
d["h_mean"] = d[["h_photo", "h_vote"]].mean(1)
d.round(4).to_csv(f"{R}/explore2_{NAME}.csv", index=False)
print(d.sort_values("h_mean", ascending=False).head(10).round(4).to_string(index=False))
best = d.sort_values("h_mean", ascending=False).iloc[0]
kb = (best.s, int(best.lam))
s0 = d[d.lam.eq(0)].sort_values("h_mean", ascending=False).iloc[0].s
k0 = (s0, 0)
print(f"\nbest by held-out mean: shrink {kb[0]}, lambda {kb[1]}; best plain: shrink {k0[0]}")


def boot(set_name, i, n=400):
    t, cam, _, _, st = stores[set_name]
    a, b = st[kb][i], st[k0][i]
    cams = np.unique(cam)
    idx = {c: np.flatnonzero(cam == c) for c in cams}
    rng = np.random.default_rng(0)
    ds = []
    for _ in range(n):
        s = np.concatenate([idx[c] for c in rng.choice(cams, len(cams))])
        ds.append(macro_f1(t[s], a[s], C) - macro_f1(t[s], b[s], C))
    return np.mean(ds), np.percentile(ds, 2.5), np.percentile(ds, 97.5), np.mean(np.array(ds) > 0)


for sn in ("heldout", "val"):
    for i, lab_ in enumerate(("per photo", "vote", "feature pool")):
        m, lo, hi, pos = boot(sn, i)
        print(f"{sn:8s} {lab_:13s} camera-noise minus plain: {m:+.4f}  95% [{lo:+.4f}, {hi:+.4f}]  better in {pos:.0%}")

print("\n-- burst error budget, best camera-noise head")
for sn in ("heldout", "val"):
    t, cam, bu, cf, st = stores[sn]
    p, v, _ = st[kb]
    g = pd.DataFrame({"b": bu, "t": t, "v": v, "p": p, "box": cf >= 0.2}).groupby("b")
    a = g.agg(n=("t", "size"), boxed=("box", "sum"), species=("t", "nunique"), right=("v", lambda s: 0), ).reset_index()
    per = pd.DataFrame({"b": bu, "ok": v == t}).groupby("b").ok.mean().to_numpy()
    a["ok"] = per
    a["kind"] = np.select([a.species > 1, a.boxed == 0, a.boxed == 1], ["mixed species", "no boxed photo", "one boxed photo"], "2+ boxed photos")
    s = a.groupby("kind").agg(bursts=("ok", "size"), photos=("n", "sum"), correct=("ok", "mean"))
    s["share_of_burst_errors"] = a.assign(e=1 - a.ok).groupby("kind").e.sum() / (1 - a.ok).sum()
    print(sn); print(s.round(3).to_string())
    nb = cf < .2
    print(f"  no-box photos: {nb.mean():.1%} of photos; per-photo macro F1 if those were all right: {macro_f1(t, np.where(nb, t, p), C):.3f} (now {macro_f1(t, p, C):.3f})")
    print(f"  photos with a box only: macro F1 {macro_f1(t[~nb], p[~nb], C):.3f}")
