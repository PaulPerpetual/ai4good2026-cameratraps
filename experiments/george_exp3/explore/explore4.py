"""Should photos with no detected animal train the head? Held-out and validation only.

usage: explore4.py resnet50|dinov2l
Scoring always covers every labelled photo; only the fitting data changes. Burst weights are
recomputed on the photos kept, so every burst still counts once.
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
HEAD = {"resnet50": (0.03, 4), "dinov2l": (0.1, 8)}[NAME]
PLAIN = {"resnet50": (0.03, 0), "dinov2l": (0.3, 0)}[NAME]
CFG = {"all photos": 0.0, "conf>=0.05": 0.05, "conf>=0.2": 0.2, "conf>=0.5": 0.5, "conf>=0.8": 0.8}
HEADS = {"plain": PLAIN, "camera": HEAD}


def predict(F, ab, Xq, b, c):
    p = F.score(Xq, ab).argmax(1)
    return p, burst_average(np.eye(C)[p], b, c).argmax(1)


res = {}
n_species = {}
for sn, splits in SPLITS.items():
    store, truth, cams = {}, [], []
    for fit, q in splits:
        fit = fit & train_mask
        q = q & lab
        for k, thr in CFG.items():
            keep = fit & (~lab | (conf >= thr))
            F = Fit(X_crop, keep)
            n_species[k] = min(n_species.get(k, 60), int(F.present.sum()))
            for h, hp in HEADS.items():
                store.setdefault((k, h), []).append(predict(F, F.weights(hp[0], hp[1], 0), X_crop[q], burst[q], conf[q]))
        truth.append(y[q]); cams.append(ix.location.to_numpy()[q])
        print(f"  {sn} fold done", flush=True)
    res[sn] = (np.concatenate(truth), np.concatenate(cams),
               {k: [np.concatenate([x[i] for x in v]) for i in range(2)] for k, v in store.items()})
rows = []
for (k, h) in res["val"][2]:
    H, V = res["heldout"], res["val"]
    rows.append({"train on": k, "head": h, "species kept": n_species[k], "h_photo": macro_f1(H[0], H[2][(k, h)][0], C), "h_vote": macro_f1(H[0], H[2][(k, h)][1], C),
                 "v_photo": macro_f1(V[0], V[2][(k, h)][0], C), "v_vote": macro_f1(V[0], V[2][(k, h)][1], C)})
d = pd.DataFrame(rows).round(4)
d.to_csv(f"{R}/explore4_{NAME}.csv", index=False)
print(d.to_string(index=False))


def boot(sn, a, b, i, n=400):
    t, cam, st = res[sn]
    A, B = st[a][i], st[b][i]
    cs = np.unique(cam)
    idx = {c: np.flatnonzero(cam == c) for c in cs}
    rng = np.random.default_rng(0)
    ds = []
    for _ in range(n):
        s = np.concatenate([idx[c] for c in rng.choice(cs, len(cs))])
        ds.append(macro_f1(t[s], A[s], C) - macro_f1(t[s], B[s], C))
    return np.percentile(ds, 2.5), np.percentile(ds, 97.5), np.mean(np.array(ds) > 0)


x = d.assign(m=lambda z: z.h_photo + z.h_vote).sort_values("m", ascending=False).iloc[0]
a, b = (x["train on"], x["head"]), ("all photos", "plain")
print("best by held-out:", a)
for sn in ("heldout", "val"):
    for i, name in enumerate(("per photo", "vote")):
        lo, hi, pos = boot(sn, a, b, i)
        print(f"{sn:8s} {name:10s} best minus plain all-photos: 95% [{lo:+.4f}, {hi:+.4f}]  better in {pos:.0%}")
