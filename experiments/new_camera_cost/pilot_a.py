"""Idea-A pilot on cached features. Validation cameras only, CPU only.

    python pilot_a.py --cache features/dinov2_vitb14_reg_448 --out results/pilot_a_v1

Three questions. Each has ONE primary comparison, chosen from the training cameras before
the validation cameras are scored, a margin fixed in advance, and three possible outcomes:
supported, not supported, or inconclusive when the interval spans the margin.

  K1  Is a camera's species mix worth knowing?  The primary head with a label-assisted
      reference mix, against an ordinary pooled classifier tuned on its own. Margin 2
      macro-F1 points. The reference is the camera's mix estimated from half of its
      bursts with their labels. It is NOT the true prior and not an upper bound, so a
      failure here stops this implementation, not the idea.
  K2  Do a camera's empty frames add anything beyond its GPS position?  Margin 1 point.
  K3  Does training the mix into the loss beat adding the same prior to an ordinary
      head afterwards?  Margin 1 point, on one deployable prior.

Method. Two linear heads are trained on frozen features of the TRAINING cameras. The
ordinary head minimises burst-weighted cross-entropy. The prior-in-loss head minimises the
same loss with `log pi_c` added to its scores, where `pi_c` is the camera's burst-level
species mix estimated from the half of the camera's bursts the image is not in. At an
unseen camera a prior is plugged in: the global mix, the regional mix, a mix predicted
from GPS neighbours, a mix predicted from the similarity of empty-frame embeddings, both
combined, or any of these adjusted on the camera's own unlabelled bursts by an EM-style
soft-count update. Evidence is pooled over each burst before the prior is applied.

Selection. Everything is chosen by camera-wise cross-validation inside the training
cameras, and every step that is fitted (feature scaling, the centring of empty-frame
descriptors, prior hyperparameters, the heads) is refitted without the held-out cameras.
Each method is tuned with its own deployable prior; no labelled reference is used to tune
a deployable method. The validation cameras are scored once. Test cameras are not in the
cache.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

FLOOR = 1e-3            # share of uniform mixed into every prior: no species gets zero
SMOOTH = 2.0            # pseudo-bursts pulling a camera's mix toward its base prior
SPLIT = "official_ood"
KINDS = ["global", "region", "gps", "empties", "gps+empties"]
REGIME = {"global": "source only", "region": "+ camera GPS", "gps": "+ camera GPS",
          "empties": "+ camera empties", "gps+empties": "+ camera GPS and empties"}
TEMPS = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
GAMMAS = [0.0, 0.25, 0.5, 1.0]
ALPHAS = [1.0, 3.0, 10.0, 30.0, 100.0]
MARGIN = {"K1": 0.02, "K2": 0.01, "K3": 0.01}
HEADS = {"ord": "ordinary", "pil": "prior-in-loss"}


# --- small numeric helpers --------------------------------------------------------------

def floored(p: np.ndarray) -> np.ndarray:
    """Normalise and mix in a little uniform mass, along the last axis."""
    p = p / p.sum(-1, keepdims=True)
    return (1 - FLOOR) * p + FLOOR / p.shape[-1]


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def macro_f1(true: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
    """Macro F1 over the classes present in `true`, as the repo's headline metric."""
    tp = np.bincount(true[true == pred], minlength=n_classes).astype(float)
    n_true = np.bincount(true, minlength=n_classes).astype(float)
    n_pred = np.bincount(pred, minlength=n_classes).astype(float)
    present = n_true > 0
    return float((2 * tp[present] / np.maximum(n_true[present] + n_pred[present], 1)).mean())


def haversine_km(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Great-circle distance between every row of `a` and every row of `b` (lat, lon)."""
    la1, lo1 = np.radians(a[:, None, 0]), np.radians(a[:, None, 1])
    la2, lo2 = np.radians(b[None, :, 0]), np.radians(b[None, :, 1])
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(h))


# --- data ---------------------------------------------------------------------------------

class Data:
    """The cache, reshaped into images, bursts and cameras. Nothing is fitted here."""

    def __init__(self, cache: Path, repo: Path, n_classes: int):
        ix = pd.read_csv(cache / "index.csv.gz")
        ok = np.load(cache / "status.npy") == 1
        X = np.asarray(np.load(cache / "features.npy", mmap_mode="r")[ok], dtype=np.float32)
        ix = ix[ok].reset_index(drop=True)
        assert not ix[SPLIT].eq("test").any(), "the pilot must not see test rows"
        assert ix.groupby("location")[SPLIT].nunique().eq(1).all(), "a camera spans two folds"
        self.C = n_classes
        lab = ~ix["is_empty"].to_numpy()

        self.X = X[lab]                                       # raw; scaled per source set
        self.img = ix[lab].reset_index(drop=True)
        self.y = self.img["y"].to_numpy().astype(np.int64)
        self.cam = self.img["location"].to_numpy()
        self.fold = self.img[SPLIT].to_numpy()

        # Bursts: one row per burst, labelled with its majority species.
        per_burst = self.img.groupby("burst_id")
        assert per_burst["location"].nunique().eq(1).all(), "a burst spans two cameras"
        assert per_burst[SPLIT].nunique().eq(1).all(), "a burst spans two folds"
        codes, _ = pd.factorize(self.img["burst_id"])
        self.burst = codes                                    # burst index of every image
        g = self.img.groupby(codes)
        self.b_cam = g["location"].first().to_numpy()
        self.b_fold = g[SPLIT].first().to_numpy()
        self.b_n = g.size().to_numpy()
        self.b_y = g["y"].agg(lambda s: s.value_counts().index[0]).to_numpy().astype(np.int64)
        self.w = (1.0 / self.b_n[self.burst]).astype(np.float32)   # each burst weighs 1

        self.train_cams = np.unique(self.cam[self.fold == "train"])
        self.val_cams = np.unique(self.cam[self.fold == "val"])
        cams = np.concatenate([self.train_cams, self.val_cams])
        self.counts = {c: np.bincount(self.b_y[self.b_cam == c], minlength=self.C).astype(float)
                       for c in cams}

        # Per camera, the mean unit-norm feature of its empty frames. It uses that camera's
        # frames only; the centring that makes cameras comparable is fitted per source set.
        e = ix["is_empty"].to_numpy()
        E = X[e] / np.linalg.norm(X[e], axis=1, keepdims=True)
        e_cam = ix["location"].to_numpy()[e]
        self.empty_mean = {c: E[e_cam == c].mean(0) for c in cams if (e_cam == c).any()}

        gps = json.loads((repo / "data" / "gps_locations.json").read_text())
        self.gps = {int(k): (v["latitude"], v["longitude"]) for k, v in gps.items()
                    if int(k) in set(cams.tolist())}
        self.region = {c: ("americas" if lo < -30 else "africa" if lo < 60 else "asia")
                       for c, (_, lo) in self.gps.items()}

    def summary(self) -> dict:
        v, t = self.val_cams, self.train_cams
        return {
            "train_images": int((self.fold == "train").sum()), "val_images": int((self.fold == "val").sum()),
            "train_cameras": len(t), "val_cameras": len(v),
            "train_bursts": int((self.b_fold == "train").sum()), "val_bursts": int((self.b_fold == "val").sum()),
            "val_species_present": int(len(np.unique(self.y[self.fold == "val"]))),
            "val_cameras_with_gps": int(sum(c in self.gps for c in v)),
            "val_cameras_with_empties": int(sum(c in self.empty_mean for c in v)),
            "train_cameras_with_gps": int(sum(c in self.gps for c in t)),
            "train_cameras_with_empties": int(sum(c in self.empty_mean for c in t)),
        }


# --- everything fitted on a set of source cameras ---------------------------------------

class Source:
    """What can be learned from a set of source cameras without looking anywhere else.

    Feature scaling, the centre of the empty-frame descriptors, the global and regional
    species mixes, and the species-mix estimators for a camera. The source set is all
    training cameras, or the training cameras of one cross-validation fold; a held-out
    camera never contributes to anything here.
    """

    def __init__(self, d: Data, source: np.ndarray):
        self.d, self.source = d, np.asarray(source)
        self.in_source = set(self.source.tolist())
        rows = np.isin(d.cam, self.source)
        self.mu, self.sd = d.X[rows].mean(0), d.X[rows].std(0) + 1e-6

        self._tot = sum(d.counts[c] for c in self.source)
        self.total = floored(self._tot)           # the mix an ordinary head absorbs
        self._reg, self._n_reg = {}, {}
        for r in ("americas", "africa", "asia"):
            cams = [c for c in self.source if d.region.get(c) == r]
            if cams:
                self._reg[r], self._n_reg[r] = sum(d.counts[c] for c in cams), len(cams)
        self.mix = {c: d.counts[c] / d.counts[c].sum() for c in self.source}
        self.g_cams = np.array([c for c in self.source if c in d.gps])
        self.g_xy = np.array([d.gps[c] for c in self.g_cams])
        self.e_cams = np.array([c for c in self.source if c in d.empty_mean])
        self._centre = np.mean([d.empty_mean[c] for c in self.e_cams], axis=0)
        self.e_vec = np.array([self.descriptor(c) for c in self.e_cams])
        self.hp = None

    def scale(self, rows: np.ndarray) -> np.ndarray:
        """Standardised features of the given images, with this source set's statistics."""
        return (self.d.X[rows] - self.mu) / self.sd

    def descriptor(self, cam) -> np.ndarray:
        """A camera's empty-frame descriptor, centred on this source set's cameras."""
        v = self.d.empty_mean[cam] - self._centre
        return v / np.linalg.norm(v)

    def global_for(self, cam) -> np.ndarray:
        """The global mix of the source cameras, leaving `cam` itself out if it is one."""
        own = self.d.counts[cam] if cam in self.in_source else 0.0
        return floored(self._tot - own)

    def has_region(self, cam) -> bool:
        """True if at least three source cameras other than `cam` share its region."""
        r = self.d.region.get(cam)
        return r in self._reg and self._n_reg[r] - (cam in self.in_source) >= 3

    def base(self, cam) -> np.ndarray:
        """Regional mix if the camera's region is known, else the global mix.

        A source camera never contributes to its own base prior.
        """
        if not self.has_region(cam):
            return self.global_for(cam)
        own = self.d.counts[cam] if cam in self.in_source else 0.0
        return floored(self._reg[self.d.region[cam]] - own)

    def gps(self, cam, k: int, shrink: float) -> np.ndarray | None:
        """Mean mix of the `k` nearest source cameras, shrunk toward the base prior."""
        if cam not in self.d.gps:
            return None
        keep = self.g_cams != cam
        if not keep.any():
            return None
        dist = haversine_km(np.array([self.d.gps[cam]]), self.g_xy[keep])[0]
        near = self.g_cams[keep][np.argsort(dist)[:k]]
        p = np.mean([self.mix[c] for c in near], axis=0)
        return floored((1 - shrink) * p + shrink * self.base(cam))

    def empties(self, cam, k: int, temp: float, shrink: float) -> np.ndarray | None:
        """Mix of the source cameras whose empty frames look most like this camera's."""
        if cam not in self.d.empty_mean:
            return None
        keep = self.e_cams != cam
        if not keep.any():
            return None
        sim = self.e_vec[keep] @ self.descriptor(cam)
        top = np.argsort(-sim)[:k]
        wgt = np.exp((sim[top] - sim[top].max()) / temp)
        p = np.sum([w * self.mix[c] for w, c in zip(wgt, self.e_cams[keep][top])], axis=0) / wgt.sum()
        return floored((1 - shrink) * p + shrink * self.base(cam))

    def estimate(self, cam, kind: str) -> tuple[np.ndarray, str]:
        """The prior of one kind for a camera, and which estimator actually produced it.

        A camera without GPS or without empties falls back: empties -> gps -> base.
        """
        if kind == "global":
            return self.global_for(cam), "global"
        if kind == "region":
            return self.base(cam), "region" if self.has_region(cam) else "global"
        g = self.gps(cam, **self.hp["gps"])
        e = self.empties(cam, **self.hp["empties"])
        if kind == "gps":
            return (g, "gps") if g is not None else (self.base(cam), "base")
        if kind == "empties":
            if e is not None:
                return e, "empties"
            return (g, "gps") if g is not None else (self.base(cam), "base")
        if kind == "gps+empties":
            if g is not None and e is not None:
                w = self.hp["combo"]
                return floored(np.exp(w * np.log(e) + (1 - w) * np.log(g))), "gps+empties"
            return self.estimate(cam, "empties" if e is not None else "gps")
        raise ValueError(kind)

    def tune(self) -> dict:
        """Choose the estimators' hyperparameters by leave-one-camera-out log score.

        Each source camera's mix is predicted from the other source cameras and scored on
        its own burst labels. These scores select hyperparameters and are not evidence.
        """
        d = self.d

        def score(fn) -> float:
            tot, n = 0.0, 0
            for c in self.source:
                p = fn(c)
                if p is not None:
                    yb = d.b_y[d.b_cam == c]
                    tot += np.log(p[yb]).sum()
                    n += len(yb)
            return tot / max(n, 1)

        hp, scores = {}, {}
        grid = [(k, s) for k in (3, 5, 10, 20) for s in (0.05, 0.2, 0.5)]
        sc = [score(lambda c, k=k, s=s: self.gps(c, k, s)) for k, s in grid]
        hp["gps"], scores["gps"] = dict(zip(("k", "shrink"), grid[int(np.argmax(sc))])), max(sc)
        grid = [(k, t, s) for k in (5, 10, 20, 50) for t in (0.02, 0.05, 0.1, 0.2) for s in (0.05, 0.2, 0.5)]
        sc = [score(lambda c, k=k, t=t, s=s: self.empties(c, k, t, s)) for k, t, s in grid]
        hp["empties"], scores["empties"] = dict(zip(("k", "temp", "shrink"), grid[int(np.argmax(sc))])), max(sc)

        def combo(c, w):
            g, e = self.gps(c, **hp["gps"]), self.empties(c, **hp["empties"])
            return None if g is None or e is None else floored(np.exp(w * np.log(e) + (1 - w) * np.log(g)))
        ws = (0.25, 0.5, 0.75)
        sc = [score(lambda c, w=w: combo(c, w)) for w in ws]
        hp["combo"], scores["gps+empties"] = ws[int(np.argmax(sc))], max(sc)
        self.hp = hp
        return {"hp": hp, "tuning_log_score": {k: round(float(v), 4) for k, v in scores.items()}}


def crossfit_mix(burst_y: np.ndarray, base: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A camera's mix for each of its bursts, estimated from the other half of its bursts.

    The bursts are split in two at random. A burst never contributes to its own prior, so
    the prior says nothing about that burst's label beyond what the camera's other bursts
    say. With one burst there is nothing to learn from and the base is used.
    """
    n = len(burst_y)
    block = np.zeros(n, dtype=int)
    block[rng.permutation(n)[n // 2:]] = 1
    out = np.empty((n, len(base)))
    for k in (0, 1):
        cnt = np.bincount(burst_y[block != k], minlength=len(base)).astype(float)
        out[block == k] = (cnt + SMOOTH * base) / (cnt.sum() + SMOOTH)
    return floored(out)


def em_adjust(ev: np.ndarray, prior0: np.ndarray, alpha: float, iters: int = 100) -> np.ndarray:
    """Adjust a camera's mix on its own unlabelled bursts.

    `ev` is (bursts, classes): pooled, scaled evidence. Each step recomputes the bursts'
    soft assignments under the current mix and replaces the mix by the soft counts plus
    `alpha` pseudo-bursts of `prior0`. This is an EM-style soft-count update, a regularised
    approximation and not an exact posterior mean; it keeps the estimate inside the
    simplex, and a camera with two bursts barely moves.
    """
    pi = prior0.copy()
    for _ in range(iters):
        r = np.exp(log_softmax(ev + np.log(pi)))
        new = (r.sum(0) + alpha * prior0) / (len(ev) + alpha)
        moved = np.abs(new - pi).max()
        pi = new
        if moved < 1e-7:
            break
    return pi


# --- heads --------------------------------------------------------------------------------

_gpu_is_ours = False


def gpu_is_free() -> bool:
    """True if no other compute process is on GPU 0.

    Before this process has touched the GPU it must also be nearly empty (under 2 GB in
    use). Afterwards its own memory shows up in that number, so only other processes count.
    """
    global _gpu_is_ours
    q = lambda what: subprocess.run(["nvidia-smi", f"--query-{what}", "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True, check=True).stdout
    used = int(q("gpu=memory.used").strip().splitlines()[0])
    others = [int(p) for p in q("compute-apps=pid").split() if p.strip().isdigit() and int(p) != os.getpid()]
    free = not others and (_gpu_is_ours or used < 2000)
    _gpu_is_ours = _gpu_is_ours or free
    return free


def fit_head(X: np.ndarray, y: np.ndarray, w: np.ndarray, offset: np.ndarray | None,
             l2: float, n_classes: int, max_iter: int,
             device: str = "cpu") -> tuple[np.ndarray, np.ndarray, dict]:
    """Weighted multinomial logistic regression by L-BFGS.

    The objective is convex, so there is no initialisation to vary; whether the optimiser
    actually got there is recorded in the returned diagnostics.

    Args:
        offset: fixed (n, classes) term added to the scores inside the loss. None gives
            the ordinary head; `log pi_c` gives the prior-in-loss head.

    Returns:
        Weights, bias, and `{"iterations", "loss", "grad_max", "hit_cap", "seconds"}`.
    """
    started = time.time()
    Xt, yt = torch.from_numpy(X).to(device), torch.from_numpy(y).to(device)
    wt = torch.from_numpy(w / w.sum()).to(device)
    off = None if offset is None else torch.from_numpy(offset.astype(np.float32)).to(device)
    W = torch.zeros(X.shape[1], n_classes, device=device, requires_grad=True)
    b = torch.zeros(n_classes, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=max_iter, history_size=20,
                            line_search_fn="strong_wolfe", tolerance_grad=1e-6,
                            tolerance_change=1e-10)

    def closure():
        opt.zero_grad()
        z = Xt @ W + b
        if off is not None:
            z = z + off
        loss = (wt * F.cross_entropy(z, yt, reduction="none")).sum() + 0.5 * l2 * (W ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    loss = closure()
    n_iter = opt.state[opt._params[0]]["n_iter"]
    diag = {"iterations": int(n_iter), "loss": float(loss),
            "grad_max": float(max(W.grad.abs().max(), b.grad.abs().max())),
            "hit_cap": bool(n_iter >= max_iter), "seconds": round(time.time() - started, 1)}
    return W.detach().cpu().numpy(), b.detach().cpu().numpy(), diag


def pool(scores: np.ndarray, burst: np.ndarray, n_bursts: int) -> np.ndarray:
    """Mean score per burst. `burst` holds each image's burst index in 0..n_bursts-1."""
    out = np.zeros((n_bursts, scores.shape[1]))
    np.add.at(out, burst, scores)
    return out / np.bincount(burst, minlength=n_bursts)[:, None]


def scaled(pooled: np.ndarray, n: np.ndarray, temp: float, gamma: float) -> np.ndarray:
    """Pooled evidence counted as n**gamma frames and divided by the temperature."""
    return (n[:, None] ** gamma) * pooled / temp


# --- the experiment -----------------------------------------------------------------------

def run(a) -> dict:
    t0 = time.time()
    torch.set_num_threads(a.threads)
    if a.device == "cuda" and not gpu_is_free():      # the GPU is shared: never pile on
        print("GPU is in use; not starting")
        sys.exit(2)
    d = Data(Path(a.cache), Path(a.repo), a.n_classes)
    C, info = d.C, d.summary()
    print(json.dumps(info), flush=True)
    rng = np.random.default_rng(a.seed)
    tr_img, va_img = d.fold == "train", d.fold == "val"
    n_b = len(d.b_y)
    l2_grid = [float(v) for v in a.l2.split(",")]
    fits = []

    def mix_rows(cams, src_of, seed) -> np.ndarray:
        """Cross-fitted mix for every burst of the given cameras; other rows stay NaN."""
        r = np.random.default_rng(seed)
        out = np.full((n_b, C), np.nan)
        for c in cams:
            idx = np.flatnonzero(d.b_cam == c)
            out[idx] = crossfit_mix(d.b_y[idx], src_of(c).base(c), r)
        return out

    def fit_both(src: Source, l2: float, seed: int, tag: str) -> dict:
        """Fit the two heads on the source cameras' images."""
        rows = np.isin(d.cam, src.source)
        X, y, w = src.scale(rows), d.y[rows], d.w[rows]
        off = np.log(mix_rows(src.source, lambda c: src, seed))[d.burst[rows]]
        out = {}
        for h, o in (("ord", None), ("pil", off)):
            if a.device == "cuda" and not gpu_is_free():
                print("another job is on the GPU; stopping. Nothing is saved, re-run later.")
                sys.exit(4)
            W, b, diag = fit_head(X, y, w, o, l2, C, a.max_iter, a.device)
            fits.append(dict(diag, head=h, l2=l2, fit=tag))
            out[h] = (W, b)
        return out

    def evidence(h: str, wb, src: Source, rows: np.ndarray) -> np.ndarray:
        """Per-image scores with the prior the head learned taken out.

        The ordinary head has absorbed the source cameras' global mix into its scores, so
        that mix is divided out. The prior-in-loss head was handed its priors explicitly
        and its scores are already prior-free.
        """
        s = src.scale(rows) @ wb[0] + wb[1]
        return (s - np.log(src.total) if h == "ord" else s).astype(np.float32)

    # 1. Camera-wise cross-validation inside the training cameras. Per fold: a source set
    #    fitted without the held-out cameras, both heads per l2, and out-of-camera evidence
    #    and priors for the held-out cameras.
    order = sorted(d.train_cams, key=lambda c: (d.region.get(c, "none"), rng.random()))
    fold_of = {c: i % a.folds for i, c in enumerate(order)}
    tr_pos = np.flatnonzero(tr_img)
    img_fold = np.array([fold_of[c] for c in d.cam[tr_img]])
    tr_b = np.flatnonzero(d.b_fold == "train")
    b_local = {b: i for i, b in enumerate(tr_b)}
    tr_burst_local = np.array([b_local[b] for b in d.burst[tr_img]])
    y_tr, bn_tr, by_tr, bcam_tr = d.y[tr_img], d.b_n[tr_b], d.b_y[tr_b], d.b_cam[tr_b]

    oof = {(h, l2): np.zeros((len(tr_pos), C), dtype=np.float32) for h in HEADS for l2 in l2_grid}
    raw_oof = {l2: np.zeros(len(tr_pos), dtype=np.int64) for l2 in l2_grid}
    prior_tr = {k: np.full((len(tr_b), C), np.nan) for k in KINDS}
    fold_src, fold_tuning = {}, []
    for f in range(a.folds):
        src = fold_src[f] = Source(d, np.array([c for c in d.train_cams if fold_of[c] != f]))
        fold_tuning.append(src.tune())
        held = np.zeros(len(d.y), dtype=bool)
        held[tr_pos[img_fold == f]] = True
        for l2 in l2_grid:
            heads = fit_both(src, l2, a.seed + f, f"fold{f}")
            for h in HEADS:
                oof[(h, l2)][img_fold == f] = evidence(h, heads[h], src, held)
            raw_oof[l2][img_fold == f] = (src.scale(held) @ heads["ord"][0] + heads["ord"][1]).argmax(1)
        for c in (c for c in d.train_cams if fold_of[c] == f):
            for k in KINDS:
                prior_tr[k][bcam_tr == c] = src.estimate(c, k)[0]
        last = fits[-2 * len(l2_grid):]
        print(f"  fold {f + 1}/{a.folds} fitted  {time.time() - t0:6.0f}s  iterations "
              f"{[x['iterations'] for x in last]}  hit cap {sum(x['hit_cap'] for x in last)}/{len(last)}  "
              f"max |grad| {max(x['grad_max'] for x in last):.1e}", flush=True)
        # The remaining folds, plus about one fold's worth for the final fits and scoring.
        projected = (time.time() - t0) / (f + 1) * (a.folds + 1)
        if a.max_minutes and projected > a.max_minutes * 60:
            print(f"projected {projected / 60:.0f} min exceeds --max-minutes {a.max_minutes:.0f}; stopping.")
            sys.exit(3)
    ref_tr = mix_rows(d.train_cams, lambda c: fold_src[fold_of[c]], a.seed + 100)[tr_b]
    pooled_tr = {key: pool(v, tr_burst_local, len(tr_b)) for key, v in oof.items()}

    # 2. Tune each method on the training cameras with ITS OWN prior. Criterion: held-out
    #    image-level macro-F1, ties broken by burst log-loss.
    def score_tr(logpost: np.ndarray) -> tuple[float, float]:
        f1 = macro_f1(y_tr, logpost.argmax(1)[tr_burst_local], C)
        return f1, float(-logpost[np.arange(len(tr_b)), by_tr].mean())

    def tune(h: str, prior_rows: np.ndarray) -> dict:
        best = None
        for l2 in l2_grid:
            for t in TEMPS:
                for g in GAMMAS:
                    f1, nll = score_tr(log_softmax(scaled(pooled_tr[(h, l2)], bn_tr, t, g) + np.log(prior_rows)))
                    if best is None or (f1, -nll) > (best["cv_macro_f1"], -best["cv_burst_nll"]):
                        best = {"l2": l2, "temp": t, "gamma": g, "cv_macro_f1": f1, "cv_burst_nll": nll}
        return best

    def em_rows(ev: np.ndarray, prior_rows: np.ndarray, cams_of_bursts: np.ndarray, alpha: float) -> np.ndarray:
        out = np.empty_like(prior_rows)
        for c in np.unique(cams_of_bursts):
            sel = cams_of_bursts == c
            out[sel] = em_adjust(ev[sel], prior_rows[sel][0], alpha)
        return out

    cfg = {}
    for h in HEADS:
        for k in KINDS:
            cfg[(h, k)] = tune(h, prior_tr[k])
            c0 = cfg[(h, k)]
            ev = scaled(pooled_tr[(h, c0["l2"])], bn_tr, c0["temp"], c0["gamma"])
            best = None
            for al in ALPHAS:
                f1, nll = score_tr(log_softmax(ev + np.log(em_rows(ev, prior_tr[k], bcam_tr, al))))
                if best is None or (f1, -nll) > (best["cv_macro_f1"], -best["cv_burst_nll"]):
                    best = dict(c0, alpha=al, cv_macro_f1=f1, cv_burst_nll=nll)
            cfg[(h, k + " + EM")] = best
        cfg[(h, "reference")] = tune(h, ref_tr)
    raw_l2 = cfg[("ord", "global")]["l2"]
    raw_cv_f1 = macro_f1(y_tr, raw_oof[raw_l2], C)
    print(f"tuned {len(cfg)} methods on the training cameras  {time.time() - t0:6.0f}s", flush=True)

    # Fixed now, from the training cameras only, before any validation camera is scored.
    cv = {key: v["cv_macro_f1"] for key, v in cfg.items()}
    primary_head = max(HEADS, key=lambda h: cv[(h, "reference")])
    deployable = [k for k in KINDS if k != "global"] + [k + " + EM" for k in KINDS]
    primary_prior = max(deployable, key=lambda k: np.mean([cv[(h, k)] for h in HEADS]))
    em_tag = " + EM" if primary_prior.endswith(" + EM") else ""
    empties_variant = max(("empties", "gps+empties"), key=lambda k: cv[(primary_head, k + em_tag)])
    plan = {"primary_head": HEADS[primary_head], "primary_prior_for_K3": primary_prior,
            "K2_contrast": [f"{HEADS[primary_head]} + {empties_variant}{em_tag}",
                            f"{HEADS[primary_head]} + gps{em_tag}"]}
    print("prespecified from training cameras:", json.dumps(plan), flush=True)

    # 3. Final source set and heads on all training cameras.
    full = Source(d, d.train_cams)
    final_tuning = full.tune()
    final = {l2: fit_both(full, l2, a.seed + 7, "final") for l2 in sorted({c["l2"] for c in cfg.values()})}
    print(f"final heads fitted  {time.time() - t0:6.0f}s", flush=True)

    # 4. Score the validation cameras once.
    va_b = np.flatnonzero(d.b_fold == "val")
    v_local = {b: i for i, b in enumerate(va_b)}
    va_burst_local = np.array([v_local[b] for b in d.burst[va_img]])
    yv, cam_v = d.y[va_img], d.cam[va_img]
    bn, by, bcam = d.b_n[va_b], d.b_y[va_b], d.b_cam[va_b]
    balance = np.log(full.total)          # subtracting this favours rare species
    pooled_va = {(h, l2): pool(evidence(h, final[l2][h], full, va_img), va_burst_local, len(va_b))
                 for l2 in final for h in HEADS}
    est = {k: {c: full.estimate(c, k) for c in d.val_cams} for k in KINDS}
    prior_va = {k: np.array([est[k][c][0] for c in bcam]) for k in KINDS}
    used = {k: {s: int(n) for s, n in pd.Series([est[k][c][1] for c in d.val_cams]).value_counts().items()}
            for k in KINDS}

    preds, rows = {}, []        # preds[name] is a list of prediction arrays (one per repeat)

    def add(name, head, prior, regime, logposts, c=None):
        """Record one method from its burst log posteriors (one array per repeat)."""
        p = [lp.argmax(1)[va_burst_local] for lp in logposts]
        pb = [(lp - balance).argmax(1)[va_burst_local] for lp in logposts]
        preds[name] = p
        rows.append({
            "method": name, "head": head, "prior": prior, "information": regime,
            "macro_f1": round(float(np.mean([macro_f1(yv, x, C) for x in p])), 4),
            "macro_f1_balanced_rule": round(float(np.mean([macro_f1(yv, x, C) for x in pb])), 4),
            "accuracy": round(float(np.mean([(x == yv).mean() for x in p])), 4),
            "burst_nll": round(float(np.mean([-lp[np.arange(len(by)), by].mean() for lp in logposts])), 4),
            "repeats": len(p), "cv_macro_f1_train_cameras": round(c["cv_macro_f1"], 4) if c else None,
            "tuned": {k: c[k] for k in ("l2", "temp", "gamma", "alpha") if k in c} if c else None,
        })

    # The ordinary head as it comes: per image, no pooling, no calibration, no prior change.
    wb = final[raw_l2]["ord"]
    raw_pred = (full.scale(va_img) @ wb[0] + wb[1]).argmax(1)
    preds["ordinary, raw argmax per image"] = [raw_pred]
    rows.append({"method": "ordinary, raw argmax per image", "head": "ordinary", "prior": "as trained",
                 "information": "source only", "macro_f1": round(macro_f1(yv, raw_pred, C), 4),
                 "macro_f1_balanced_rule": None, "accuracy": round(float((raw_pred == yv).mean()), 4),
                 "burst_nll": None, "repeats": 1, "cv_macro_f1_train_cameras": round(raw_cv_f1, 4),
                 "tuned": {"l2": raw_l2}})

    for h, label in HEADS.items():
        for k in KINDS:
            c = cfg[(h, k)]
            ev = scaled(pooled_va[(h, c["l2"])], bn, c["temp"], c["gamma"])
            add(f"{label} + {k}", label, k, REGIME[k], [log_softmax(ev + np.log(prior_va[k]))], c)
            c = cfg[(h, k + " + EM")]
            ev = scaled(pooled_va[(h, c["l2"])], bn, c["temp"], c["gamma"])
            add(f"{label} + {k} + EM", label, k + " + EM", REGIME[k] + ", + its unlabelled bursts",
                [log_softmax(ev + np.log(em_rows(ev, prior_va[k], bcam, c["alpha"])))], c)
        # Label-assisted reference: the camera's mix from the other half of its own bursts.
        c = cfg[(h, "reference")]
        ev = scaled(pooled_va[(h, c["l2"])], bn, c["temp"], c["gamma"])
        add(f"{label} + reference mix", label, "half of the camera's own labelled bursts", "uses labels",
            [log_softmax(ev + np.log(mix_rows(d.val_cams, lambda _: full, a.seed + 1000 + r)[va_b]))
             for r in range(a.oracle_repeats)], c)

    # 5. Paired bootstrap over validation cameras. A method with several repeats is
    #    averaged over them inside every resample, so its interval and its point estimate
    #    describe the same quantity.
    by_cam = {c: np.flatnonzero(cam_v == c) for c in d.val_cams}
    brng = np.random.default_rng(a.seed + 5)
    draws = [np.concatenate([by_cam[c] for c in brng.choice(d.val_cams, len(d.val_cams))])
             for _ in range(a.bootstrap)]

    def mean_f1(name: str, idx=None) -> float:
        return float(np.mean([macro_f1(yv if idx is None else yv[idx], p if idx is None else p[idx], C)
                              for p in preds[name]]))

    def contrast(m1: str, m0: str, margin: float | None = None) -> dict:
        diff = np.array([mean_f1(m1, i) - mean_f1(m0, i) for i in draws])
        lo, hi = float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))
        out = {"a": m1, "b": m0, "diff": round(mean_f1(m1) - mean_f1(m0), 4),
               "ci95": [round(lo, 4), round(hi, 4)], "share_of_resamples_a_better": round(float((diff > 0).mean()), 3)}
        if margin is not None:
            out["margin"] = margin
            out["outcome"] = ("supported" if out["diff"] >= margin and lo > 0
                              else "not supported" if hi < margin else "inconclusive")
        return out

    ph = HEADS[primary_head]
    k1 = contrast(f"{ph} + reference mix", "ordinary + global", MARGIN["K1"])
    k1_matched = {label: contrast(f"{label} + reference mix", f"{label} + global") for label in HEADS.values()}
    k2 = contrast(*plan["K2_contrast"], MARGIN["K2"])
    k3 = contrast(f"prior-in-loss + {primary_prior}", f"ordinary + {primary_prior}", MARGIN["K3"])
    diagnostics = {
        "pooling_gain": contrast("ordinary + global", "ordinary, raw argmax per image"),
        "prior_in_loss_vs_ordinary_by_prior": [contrast(f"prior-in-loss + {k}", f"ordinary + {k}")
                                               for k in deployable + ["global", "reference mix"]],
        "empties_vs_gps": [contrast(f"{label} + {x}{e}", f"{label} + gps{e}") for label in HEADS.values()
                           for x in ("empties", "gps+empties") for e in ("", " + EM")],
        "each_prior_vs_global_same_head": [contrast(f"{label} + {k}", f"{label} + global")
                                           for label in HEADS.values() for k in deployable],
    }
    hit = [f for f in fits if f["hit_cap"]]
    return {
        "data": info, "prespecified_from_training_cameras": plan, "margins": MARGIN,
        "K1": k1, "K1_matched_head_gain": k1_matched, "K2": k2, "K3": k3,
        "results": rows, "diagnostic_contrasts": diagnostics,
        "val_prior_source": used,
        "prior_tuning": {"final": final_tuning, "per_fold": fold_tuning},
        "optimisation": {"fits": len(fits), "hit_iteration_cap": len(hit),
                         "max_iterations": max(f["iterations"] for f in fits),
                         "max_grad": max(f["grad_max"] for f in fits), "detail": fits},
        "notes": [
            "The reference mix uses labels of half of each validation camera's bursts. It is "
            "label-assisted, smoothed, and not the true prior or an upper bound.",
            "Bootstrap resamples cameras, so the set of species in a resample varies; the "
            "intervals do not hold the 45-species macro-F1 fixed.",
            "cv_macro_f1_train_cameras is the score each method was tuned on and is "
            "optimistic; prior tuning log scores select hyperparameters and are not evidence.",
            "The head objective is convex, so there is no initialisation noise, but the "
            "random halving of bursts and the folds are not varied here.",
            "Whole-image features from one backbone. A negative result does not show the "
            "idea fails with crop features.",
        ],
        "seconds": round(time.time() - t0, 1), "args": vars(a),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="folder written by cache_features.py")
    ap.add_argument("--repo", default=".", help="repo root, for data/gps_locations.json")
    ap.add_argument("--out", required=True, help="folder for results.json and results.md")
    ap.add_argument("--n-classes", type=int, default=60)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--l2", default="1e-4,1e-3,1e-2", help="comma-separated grid")
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument("--oracle-repeats", type=int, default=20)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--max-minutes", type=float, default=0,
                    help="stop after a fold if the whole run is projected to take longer; 0 = no limit")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="where the head fits run; everything else is CPU either way")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    res = run(a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(
        json.dumps(res, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    lines = ["| method | information used | macro-F1 | balanced rule | accuracy | burst NLL | tuning score |",
             "|---|---|---|---|---|---|---|"]
    fmt = lambda v: "" if v is None else f"{v:.3f}"
    for r in res["results"]:
        lines.append(f"| {r['method']} | {r['information']} | {fmt(r['macro_f1'])} | "
                     f"{fmt(r['macro_f1_balanced_rule'])} | {fmt(r['accuracy'])} | {fmt(r['burst_nll'])} | "
                     f"{fmt(r['cv_macro_f1_train_cameras'])} |")
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("prespecified", json.dumps(res["prespecified_from_training_cameras"]))
    for key in ("K1", "K2", "K3"):
        print(key, json.dumps(res[key]))
    print("K1 matched-head gain", json.dumps(res["K1_matched_head_gain"]))
    o = res["optimisation"]
    print(f"optimisation: {o['fits']} fits, {o['hit_iteration_cap']} hit the cap, "
          f"max iterations {o['max_iterations']}, max |grad| {o['max_grad']:.2e}")
    print(f"wrote {out}/results.json and results.md in {res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
