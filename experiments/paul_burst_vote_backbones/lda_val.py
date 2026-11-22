"""Crop caches on official_ood val, per image and with the detector-weighted vote. Test is never read.

    python lda_val.py name=cache_dir [name=cache_dir ...]

Each cache gets George's two heads at his settings, unchanged: the discriminant (shrinkage 0.3)
and the discriminant with his camera-noise term (shrinkage 0.1, lambda 8).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import linprobe as lp
from iwildcam.data import image_dir_for, load_task

G = "/zfs/ai4good/groups/camera-traps"
sys.path.insert(0, f"{G}/users/george/scripts")
from score_test import burst_weights, lda_parts, load_labelled, macro_f1  # noqa: E402

C = 60
task = load_task(image_dir=image_dir_for(448))
conf = pd.Series(lp.detections(task)["conf"].to_numpy(), index=task.df["file_name"].to_numpy())


def head(xq, M, S, scam, shrink, lam, present):
    d = len(S)
    A = np.linalg.solve((1 - shrink) * S + shrink * (np.trace(S) / d) * np.eye(d) + lam * scam, M.T)
    z = xq @ A - 0.5 * np.einsum("cf,fc->c", M, A)
    z[:, ~present] = -np.inf
    return z


def camera_cov(xs, y, cam, w, M):
    """Covariance of the species-by-camera mean offsets, as in explore_heads.py."""
    key = pd.factorize(pd.Series(y.astype(str)) + "_" + pd.Series(cam.astype(str)))[0]
    n = np.bincount(key, weights=w)
    off = np.zeros((len(n), xs.shape[1]))
    np.add.at(off, key, (xs - M[y]) * w[:, None])
    off /= n[:, None]
    return (off * n[:, None]).T @ off / n.sum()


rows = []
for arg in sys.argv[1:]:
    name, cache = arg.split("=")
    df, x = load_labelled(Path(cache))
    y = df["y"].to_numpy().astype(np.int64)
    tr, va = df["official_ood"].eq("train").to_numpy(), df["official_ood"].eq("val").to_numpy()
    w = burst_weights(df[tr])[1]
    mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-6
    xt, xv = (x[tr] - mu) / sd, (x[va] - mu) / sd
    present = np.bincount(y[tr], weights=w, minlength=C) > 0
    M, S = lda_parts(xt, y[tr], w, present, C)
    scam = camera_cov(xt, y[tr], df["location"].to_numpy()[tr], w, M)
    burst, c = df["burst_id"].to_numpy()[va], conf.loc[df["file_name"][va]].to_numpy()
    for label, shrink, lam in (("LDA", 0.3, 0.0), ("LDA + camera noise", 0.1, 8.0)):
        pred = head(xv, M, S, scam, shrink, lam, present).argmax(1)
        rows.append({"model": name, "head": label, "image": macro_f1(y[va], pred, C),
                     "vote": macro_f1(y[va], lp.burst_vote(pred, burst, c, C), C)})
        print(rows[-1], flush=True)
print(pd.DataFrame(rows).round(4).to_string(index=False))
pd.DataFrame(rows).to_csv("results/lda_val.csv", index=False)
