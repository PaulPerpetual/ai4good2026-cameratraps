"""Compare frozen-feature heads that use training data only, for ResNet-50 and DINOv2-L.

Scores at held-out training cameras (val_compare.py's camera folds) and on the validation
cameras. No test-camera data is read: the caches hold train and val cameras only.

Each config is one linear discriminant on crop features, optionally with:
  camera    add lambda * (covariance of species-by-camera mean offsets) to the within-species
            covariance, so directions that move with the camera count as noise
  empty     add beta * (covariance of the training cameras' empty frames), the same for
            directions that background varies along
  fusion    add alpha * the whole-photo model's scores to the crop model's
  prior     add tau * log(class frequency) to the scores
Each is reported per photo and with three burst rules: detector vote, detector-weighted mean
score, detector-weighted mean feature.
"""
import json
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from cache_features import best_detections  # noqa: E402
from score_random_burst import burst_average  # noqa: E402
from score_test import macro_f1  # noqa: E402

G = "/zfs/ai4good/groups/camera-traps"
R = f"{G}/users/george/results"
C = 60
BACKBONES = {"resnet50": ("resnet50_448", "resnet50_crop"), "dinov2l": ("dinov2_vitl14_reg_448", "vitl_crop")}
NAME = sys.argv[1]
whole_dir, sel_name = BACKBONES[NAME]


def load(cache):
    ix = pd.read_csv(f"{cache}/index.csv.gz")
    ok = np.load(f"{cache}/status.npy") == 1
    assert ok.all()
    return ix, np.asarray(np.load(f"{cache}/features.npy", mmap_mode="r"), dtype=np.float32)


ix, X_crop = load(f"{G}/features/{whole_dir}_crop")
ix2, X_whole = load(f"{G}/features/{whole_dir}")
assert (ix.file_name == ix2.file_name).all()
assert not ix.official_ood.eq("test").any(), "no test rows may be read"
ix["is_empty"] = ix["is_empty"].astype(str).str.lower().eq("true")
conf = best_detections(ix.file_name.tolist())[:, 0]
folds = pd.read_csv(f"{R}/val_compare_{sel_name}/train_camera_predictions.csv.gz")[["location", "fold"]].drop_duplicates()
cam_fold = dict(zip(folds.location, folds.fold))
ix["cv"] = ix.location.map(cam_fold).fillna(-1).astype(int)
lab = ~ix.is_empty.to_numpy()
y = ix.y.to_numpy().astype(np.int64)
burst = ix.burst_id.to_numpy()


class Fit:
    """Everything fitted on one set of source cameras, for one view."""

    def __init__(self, X, fit):           # fit: boolean over all rows (labelled and empty)
        tl = fit & lab
        self.mu, self.sd = X[tl].mean(0), X[tl].std(0) + 1e-6
        Xs = (X[tl] - self.mu) / self.sd
        yt, cam, bu = y[tl], ix.location.to_numpy()[tl], burst[tl]
        codes, _ = pd.factorize(bu)
        w = 1.0 / np.bincount(codes)[codes]
        self.present = np.bincount(yt, weights=w, minlength=C) > 0
        self.M = np.zeros((C, X.shape[1]))
        for c in np.flatnonzero(self.present):
            m = yt == c
            self.M[c] = np.average(Xs[m], axis=0, weights=w[m])
        R_ = (Xs - self.M[yt]) * np.sqrt(w)[:, None]
        self.S = (R_.T @ R_) / w.sum()
        # species-by-camera offsets
        key = pd.factorize(pd.Series(yt.astype(str)) + "_" + pd.Series(cam.astype(str)))[0]
        n = np.bincount(key, weights=w)
        off = np.zeros((len(n), X.shape[1]))
        np.add.at(off, key, (Xs - self.M[yt]) * w[:, None])
        off /= n[:, None]
        self.Scam = (off * n[:, None]).T @ off / n.sum()
        te = fit & ~lab
        E = (X[te] - self.mu) / self.sd
        E = E - E.mean(0)
        self.Se = (E.T @ E) / max(len(E), 1)
        mass = np.bincount(yt, weights=w, minlength=C)
        self.logf = np.log(np.maximum(mass, 1e-9) / mass.sum())

    def weights(self, shrink, lam=0.0, beta=0.0):
        d = len(self.S)
        Ss = (1 - shrink) * self.S + shrink * (np.trace(self.S) / d) * np.eye(d) + lam * self.Scam + beta * self.Se
        A = np.linalg.solve(Ss, self.M.T)
        return A, -0.5 * np.einsum("cf,fc->c", self.M, A)

    def score(self, Xq, ab):
        z = ((Xq - self.mu) / self.sd) @ ab[0] + ab[1]
        z[:, ~self.present] = -np.inf
        return z


def rules(z_fn, Xq_mean_fn, q):
    """Per photo and burst predictions for the rows in q."""
    z = z_fn()
    p = z.argmax(1)
    b, c = burst[q], conf[q]
    finite = np.where(np.isfinite(z), z, -1e6)
    vote = burst_average(np.eye(C)[p], b, c).argmax(1)
    mean_score = burst_average(finite, b, c).argmax(1)
    return p, vote, mean_score


def evaluate(cfgs):
    q_sets = [("heldout", [(ix.cv.to_numpy() != f, ix.cv.to_numpy() == f) for f in sorted(set(ix.cv) - {-1})]),
              ("val", [(ix.official_ood.eq("train").to_numpy(), ix.official_ood.eq("val").to_numpy())])]
    rows = {k: {} for k in cfgs}
    for set_name, splits in q_sets:
        preds = {k: [] for k in cfgs}
        truth, order = [], []
        for fit, q in splits:
            fit = fit & ix.official_ood.eq("train").to_numpy()
            q = q & lab
            t0 = time.time()
            Fc, Fw = Fit(X_crop, fit), Fit(X_whole, fit)
            print(f"  {set_name} fit {time.time() - t0:.0f}s", flush=True)
            cache = {}
            for k, c in cfgs.items():
                kc = (c["s"], c.get("lam", 0), c.get("beta", 0))
                if kc not in cache:
                    cache[kc] = Fc.weights(*kc)
                z = Fc.score(X_crop[q], cache[kc])
                if c.get("alpha"):
                    kw = ("w", c["s"])
                    if kw not in cache:
                        cache[kw] = Fw.weights(c["s"])
                    z = z + c["alpha"] * Fw.score(X_whole[q], cache[kw])
                if c.get("tau"):
                    z = z + c["tau"] * Fc.logf
                preds[k].append(rules(lambda: z, None, q))
            truth.append(y[q])
        t = np.concatenate(truth)
        for k in cfgs:
            p = [np.concatenate([x[i] for x in preds[k]]) for i in range(3)]
            rows[k][set_name] = [macro_f1(t, p[0], C), macro_f1(t, p[1], C), macro_f1(t, p[2], C)]
    return rows


s0 = json.load(open(f"{R}/val_compare_{sel_name}/selection.json"))["chosen"]["linear discriminant"]["shrinkage"]
cfgs = {"base": {"s": s0}}
for s in (0.3, 0.1, 0.03, 0.01):
    cfgs[f"shrink={s}"] = {"s": s}
for lam in (0.5, 1, 2, 4, 8):
    cfgs[f"camera lam={lam}"] = {"s": s0, "lam": lam}
for beta in (0.1, 0.3, 1):
    cfgs[f"empty beta={beta}"] = {"s": s0, "beta": beta}
for a in (0.1, 0.25, 0.5, 1):
    cfgs[f"fusion alpha={a}"] = {"s": s0, "alpha": a}
for tau in (0.25, 0.5, 1):
    cfgs[f"prior tau={tau}"] = {"s": s0, "tau": tau}

res = evaluate(cfgs)
out = []
for k, v in res.items():
    out.append({"backbone": NAME, "config": k, "hp": v["heldout"][0], "hv": v["heldout"][1], "hs": v["heldout"][2],
                "vp": v["val"][0], "vv": v["val"][1], "vs": v["val"][2]})
d = pd.DataFrame(out).round(4)
d.to_csv(f"{R}/explore_{NAME}.csv", index=False)
print(d.to_string(index=False))
