"""Do the classifier changes from the rare-species checks hold on the validation cameras?

    python val_compare.py --cache features/dinov2_vitb14_reg_448 --out results/val_compare_v1 --device cuda

Three classifiers on the same frozen features, with the same pooling: a burst's prediction
is the argmax of its mean score, and every image of the burst gets it.

  ordinary             linear head, burst-weighted cross-entropy. Setting: l2.
  frequency-adjusted   the same head with tau * log(class frequency) subtracted from its
                       scores afterwards. Settings: l2 and tau.
  linear discriminant  species means and one shared covariance, no class prior. Setting:
                       the covariance shrinkage.

Stage 1 uses the TRAINING cameras only. Every setting on the grids below is scored at
held-out training cameras with the pilot's five camera-wise folds, and each classifier
gets the setting with the highest pooled macro-F1. The choice is written to
`selection.json` before any validation image is scored. If a chosen setting sits on an
arbitrary end of its grid, or a fit at a chosen l2 reached the optimiser's iteration or
evaluation cap, the run stops there, with nothing scored on validation.

Stage 1 also gives the comparison at held-out training cameras by nested
cross-validation: the setting used for an outer fold's cameras is chosen by a four-fold
inner cross-validation over the other four folds, with models that saw neither the outer
fold nor the inner held-out fold. No model or score behind that choice has seen the outer
fold.

Stage 2 fits the three chosen classifiers on all training cameras and scores the
validation cameras once. A marker file is written just before the first validation
prediction, and the script refuses to run if the marker or the results exist.

Fixed in this file, before validation is scored: the grids, the pooling, the selection
rule, the three contrasts with their margins, and which contrast is primary.

  V1 (primary)  linear discriminant - ordinary            margin 2 macro-F1 points
  V2            linear discriminant - frequency-adjusted  margin 1 point
  V3            frequency-adjusted - ordinary             margin 1 point

"Supported" means the difference meets the margin and its 95% interval (paired bootstrap
over validation cameras) is above zero; "not supported" means the interval lies below
the margin; anything else is inconclusive.

Also reported, outside the contrasts: the pilot's calibrated ordinary pipeline (l2 0.01,
temperature 1.5, burst-size exponent 0.5), refitted here to check that this script
reproduces the pilot's 0.824.

Per-fit diagnostics are written to `fit_diagnostics.json` after stage 1 and again after
the final fits, before any exit that depends on them.

Class frequency means the weight a species carries in the loss: the sum of image weights,
where every burst weighs 1 in total. A species with no training image cannot be predicted
and is masked for every classifier. Test cameras are not in the cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from iwildcam.data import load_task
from pilot_a import Data, Source, gpu_is_free, log_softmax, macro_f1, pool, scaled
from rare_checks import BIN_NAMES, BINS, class_scores

# Grids. Ordered so that the first of several equal scores is the more regularised, or the
# less adjusted, setting.
L2_GRID = [0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]
TAU_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
SHRINK_GRID = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]
# Ends of a grid that are arbitrary rather than natural limits. tau = 0 (no adjustment)
# and shrinkage = 1 (a spherical covariance) are natural limits and may be chosen.
ARBITRARY_EDGES = {"l2": {L2_GRID[0], L2_GRID[-1]}, "tau": {TAU_GRID[-1]}, "shrinkage": {SHRINK_GRID[-1]}}
HEADS = ["ordinary", "frequency-adjusted", "linear discriminant"]
SLUG = {"ordinary": "ordinary", "frequency-adjusted": "freq_adjusted", "linear discriminant": "lda"}
CONTRASTS = {"V1": ("linear discriminant", "ordinary", 0.02),
             "V2": ("linear discriminant", "frequency-adjusted", 0.01),
             "V3": ("frequency-adjusted", "ordinary", 0.01)}
PRIMARY = "V1"
PILOT = {"l2": 0.01, "temp": 1.5, "gamma": 0.5, "reported_macro_f1": 0.8240}
PILOT_NAME = "pilot's calibrated ordinary pipeline"
STARTED = "stage2_started.txt"      # written just before the first validation prediction
EXIT_BUSY, EXIT_TOO_SLOW, EXIT_YIELD, EXIT_STOP, EXIT_EXISTS, EXIT_CHANGED = 2, 3, 4, 5, 6, 7


def fit_linear(X: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float, n_classes: int, max_iter: int,
               device: str = "cpu") -> tuple[np.ndarray, np.ndarray, dict]:
    """The pilot's `fit_head` for an ordinary head, with the reason it stopped recorded.

    Same objective, optimiser and settings as `pilot_a.fit_head`, so the weights are the
    same. torch's L-BFGS stops at `max_iter` iterations and also at `max_iter * 5 // 4`
    loss evaluations; `fit_head` reports only the first. Here either counts as the cap.

    Returns:
        Weights, bias, and `{"iterations", "evaluations", "loss", "grad_max", "grad_norm",
        "hit_cap", "seconds"}`.
    """
    started = time.time()
    Xt, yt = torch.from_numpy(X).to(device), torch.from_numpy(y).to(device)
    wt = torch.from_numpy(w / w.sum()).to(device)
    W = torch.zeros(X.shape[1], n_classes, device=device, requires_grad=True)
    b = torch.zeros(n_classes, device=device, requires_grad=True)
    max_eval = max_iter * 5 // 4                                  # torch's default, made explicit
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=max_iter, max_eval=max_eval, history_size=20,
                            line_search_fn="strong_wolfe", tolerance_grad=1e-6, tolerance_change=1e-10)

    def closure():
        opt.zero_grad()
        z = Xt @ W + b
        loss = (wt * F.cross_entropy(z, yt, reduction="none")).sum() + 0.5 * l2 * (W ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    loss = closure()
    state = opt.state[opt._params[0]]
    g = torch.cat([W.grad.flatten(), b.grad.flatten()])
    diag = {"iterations": int(state["n_iter"]), "evaluations": int(state["func_evals"]),
            "loss": float(loss.detach()), "grad_max": float(g.abs().max()), "grad_norm": float(g.norm()),
            "hit_cap": bool(state["n_iter"] >= max_iter or state["func_evals"] >= max_eval),
            "seconds": round(time.time() - started, 1)}
    return W.detach().cpu().numpy(), b.detach().cpu().numpy(), diag


def masked(scores: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Scores with species absent from training set to -inf. Present ones must be finite."""
    s = np.array(scores, dtype=np.float64)
    assert np.isfinite(s[:, present]).all(), "non-finite score for a species that was trained on"
    s[:, ~present] = -np.inf
    return s


def safe_argmax(scores: np.ndarray) -> np.ndarray:
    """Row-wise argmax. -inf marks an absent species; anything else non-finite is an error."""
    assert not np.isnan(scores).any() and not np.isposinf(scores).any(), "NaN or +inf score"
    return scores.argmax(1)


def lda_parts(Xt: np.ndarray, y: np.ndarray, w: np.ndarray, present: np.ndarray, C: int):
    """Species means and the pooled within-species covariance, bursts weighted equally."""
    M = np.zeros((C, Xt.shape[1]), dtype=np.float64)
    for c in np.flatnonzero(present):
        m = y == c
        M[c] = np.average(Xt[m], axis=0, weights=w[m])
    R = (Xt - M[y]) * np.sqrt(w)[:, None]
    return M, (R.T @ R) / w.sum()


def lda_scores(Xq: np.ndarray, M: np.ndarray, S: np.ndarray, shrinkage: float, present: np.ndarray) -> np.ndarray:
    """Linear discriminant scores with the covariance shrunk toward a scaled identity."""
    Ss = (1 - shrinkage) * S + shrinkage * (np.trace(S) / len(S)) * np.eye(len(S))
    A = np.linalg.solve(Ss, M.T)                                  # (features, classes)
    return masked(Xq @ A - 0.5 * np.einsum("cf,fc->c", M, A), present)


def first_max(keys: list, score: dict):
    """The first key, in the given order, with the highest score."""
    best = max(score[k] for k in keys)
    return next(k for k in keys if score[k] == best)


def describe(key) -> dict:
    kind, k = key
    return {"l2": k[0], "tau": k[1]} if kind == "lin" else {"shrinkage": k}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-iter", type=int, default=2000)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--max-minutes", type=float, default=30)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--select-only", action="store_true", help="stop after stage 1; validation is not scored")
    a = ap.parse_args()

    t0 = time.time()
    out = Path(a.out)
    for done in ("results.json", STARTED):                # validation is scored once
        if (out / done).exists():
            print(f"{out}/{done} exists; scoring of the validation cameras has been started before. Not running again.")
            sys.exit(EXIT_EXISTS)
    torch.set_num_threads(a.threads)
    if a.device == "cuda" and not gpu_is_free():          # the GPU is shared: never pile on
        print("GPU is in use; not starting")
        sys.exit(EXIT_BUSY)
    classes = load_task(require_files=False).classes
    C = len(classes)
    d = Data(Path(a.cache), Path(a.repo), C)
    info = d.summary()
    print(json.dumps(info), flush=True)
    rng = np.random.default_rng(a.seed)
    fits = []

    n = len(d.y)
    tr_img, va_img = d.fold == "train", d.fold == "val"
    tr_b = d.b_fold == "train"
    assert (tr_img | va_img).all() and not (tr_img & va_img).any()
    n_bursts = np.bincount(d.b_y[tr_b], minlength=C)      # per species, over all training cameras
    rarity = np.digitize(n_bursts, BINS[1:-1])            # 0..3, index into BIN_NAMES

    # The pilot's folds.
    order_c = sorted(d.train_cams, key=lambda c: (d.region.get(c, "none"), rng.random()))
    fold_of = {c: i % a.folds for i, c in enumerate(order_c)}
    img_fold = np.array([fold_of.get(c, -1) for c in d.cam])
    b_foldid = np.array([fold_of.get(c, -1) for c in d.b_cam])
    assert (img_fold[tr_img] >= 0).all() and (img_fold[va_img] == -1).all()

    def guard() -> None:
        if a.device == "cuda" and not gpu_is_free():
            print("another job is on the GPU; stopping. Nothing is saved, re-run later.")
            sys.exit(EXIT_YIELD)

    def prepare(train_b: np.ndarray, query_img: np.ndarray):
        """Scaled training and query features, labels, weights, and each species' log frequency.

        Scaling and class frequencies come from the training bursts only.
        """
        ti = train_b[d.burst]
        assert not (ti & query_img).any(), "a query image is in the training set"
        mu, sd = d.X[ti].mean(0), d.X[ti].std(0) + 1e-6
        y, w = d.y[ti], d.w[ti]
        mass = np.bincount(y, weights=w, minlength=C)     # what each species weighs in the loss
        present = mass > 0
        logf = np.zeros(C)                                 # absent species are masked anyway
        logf[present] = np.log(mass[present] / mass.sum())
        return (d.X[ti] - mu) / sd, (d.X[query_img] - mu) / sd, y, w, present, logf

    def linear(Xt, y, w, l2: float, tag: str) -> tuple[np.ndarray, np.ndarray]:
        guard()
        W, b, diag = fit_linear(Xt, y, w, l2, C, a.max_iter, a.device)
        fits.append(dict(diag, head="linear", l2=l2, fit=tag))
        return W, b

    def bursts_of(query_img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The query's bursts, and each query image's position among them."""
        return np.unique(d.burst[query_img], return_inverse=True)

    # ===== Stage 1: choose every setting at held-out TRAINING cameras ======================
    lin_keys = [("lin", (l2, tau)) for l2 in L2_GRID for tau in TAU_GRID]
    lda_keys = [("lda", s) for s in SHRINK_GRID]
    pairs = [(i, j) for i in range(a.folds) for j in range(i + 1, a.folds)]
    sets_done = 0

    def grid_predictions(train_b: np.ndarray, query_img: np.ndarray, tag: str) -> dict:
        """Predictions of every grid setting for the query images, from one training set.

        Returns:
            `{setting: (per-image argmax, pooled argmax)}`, both in the order of the query images.
        """
        nonlocal sets_done
        Xt, Xq, y, w, present, logf = prepare(train_b, query_img)
        uniq, local = bursts_of(query_img)
        res = {}
        for l2 in L2_GRID:
            W, b = linear(Xt, y, w, l2, tag)
            z = masked(Xq @ W + b, present)
            zb = pool(z, local, len(uniq))
            for tau in TAU_GRID:      # the mean of adjusted scores is the adjusted mean score
                res[("lin", (l2, tau))] = (safe_argmax(z - tau * logf), safe_argmax(zb - tau * logf)[local])
        M, S = lda_parts(Xt, y, w, present, C)
        for s in SHRINK_GRID:
            z = lda_scores(Xq, M, S, s, present)
            res[("lda", s)] = (safe_argmax(z), safe_argmax(pool(z, local, len(uniq)))[local])
        sets_done += 1
        last = [x for x in fits if x["fit"] == tag]
        print(f"  {tag:18s} {sets_done:2d}/{a.folds + len(pairs)}  {time.time() - t0:5.0f}s  {len(last)} fits, "
              f"{sum(x['hit_cap'] for x in last)} reached a cap, most evaluations {max(x['evaluations'] for x in last)}, "
              f"max |grad| {max(x['grad_max'] for x in last):.1e}", flush=True)
        projected = (time.time() - t0) / sets_done * (a.folds + len(pairs) + 2)
        if a.max_minutes and projected > a.max_minutes * 60:
            print(f"projected {projected / 60:.0f} min exceeds --max-minutes {a.max_minutes:.0f}; stopping.")
            sys.exit(EXIT_TOO_SLOW)
        return res

    # Outer folds: every training image predicted by models that did not see its camera.
    P = {key: {"image": np.full(n, -1), "pooled": np.full(n, -1)} for key in lin_keys + lda_keys}
    for f in range(a.folds):
        q = tr_img & (img_fold == f)
        for key, (per_image, pooled) in grid_predictions(tr_b & (b_foldid != f), q, f"fold {f}").items():
            P[key]["image"][q], P[key]["pooled"][q] = per_image, pooled
    assert all((v[k][tr_img] >= 0).all() and (v[k][va_img] == -1).all() for v in P.values() for k in v), \
        "stage 1 must predict every training image and no validation image"

    # Inner folds for the nested estimate. A model trained without folds i and j predicts
    # both: its predictions for fold i belong to the inner cross-validation that leaves
    # outer fold j out entirely, and the other way round.
    inner = {f: {key: np.full(n, -1, dtype=np.int16) for key in P} for f in range(a.folds)}
    for i, j in pairs:
        q = tr_img & ((img_fold == i) | (img_fold == j))
        rows_q, fold_q = np.flatnonzero(q), img_fold[q]
        for key, (_, pooled) in grid_predictions(tr_b & (b_foldid != i) & (b_foldid != j), q, f"without {i} and {j}").items():
            inner[j][key][rows_q[fold_q == i]] = pooled[fold_q == i]
            inner[i][key][rows_q[fold_q == j]] = pooled[fold_q == j]
    for f in range(a.folds):
        rest = tr_img & (img_fold != f)
        assert all((v[rest] >= 0).all() and (v[~rest] == -1).all() for v in inner[f].values()), \
            "the inner cross-validation of an outer fold must cover the other folds and nothing else"

    yt, cam_t, fold_t = d.y[tr_img], d.cam[tr_img], img_fold[tr_img]
    Pt = {key: v["pooled"][tr_img] for key, v in P.items()}
    cv = {key: macro_f1(yt, p, C) for key, p in Pt.items()}
    cv_table = pd.DataFrame([{
        "classifier": "linear head" if key[0] == "lin" else "linear discriminant", **describe(key),
        "cv_macro_f1_pooled": round(cv[key], 4),
        "cv_macro_f1_per_image": round(macro_f1(yt, P[key]["image"][tr_img], C), 4),
        "cv_accuracy_pooled": round(float((Pt[key] == yt).mean()), 4)} for key in P],
        columns=["classifier", "l2", "tau", "shrinkage", "cv_macro_f1_pooled", "cv_macro_f1_per_image", "cv_accuracy_pooled"])

    # Candidate settings per classifier, in tie-break order.
    cand = {"ordinary": [("lin", (l2, 0.0)) for l2 in L2_GRID], "frequency-adjusted": lin_keys,
            "linear discriminant": lda_keys}
    chosen = {h: first_max(cand[h], cv) for h in HEADS}
    # Reasons not to go on to the validation cameras with this choice.
    stops = [f"{h}: {name} = {describe(chosen[h])[name]} is an arbitrary end of its grid"
             for h in HEADS for name, bad in ARBITRARY_EDGES.items() if describe(chosen[h]).get(name) in bad]
    capped = {x["l2"] for x in fits if x["hit_cap"]}
    stops += [f"{h}: a fit at its l2 = {describe(chosen[h])['l2']} reached the iteration or evaluation cap"
              for h in HEADS[:2] if describe(chosen[h])["l2"] in capped]
    selection = {
        "chosen": {h: dict(describe(chosen[h]), cv_macro_f1_pooled=round(cv[chosen[h]], 4)) for h in HEADS},
        "reasons_to_stop_before_validation": stops,
        "grids": {"l2": L2_GRID, "tau": TAU_GRID, "shrinkage": SHRINK_GRID},
        "rule": "highest pooled macro-F1 over all held-out training images; ties go to the earlier grid entry",
        "pooling": "argmax of the mean score over a burst's frames, given to every frame",
        "contrasts": {k: {"a": v[0], "b": v[1], "margin": v[2]} for k, v in CONTRASTS.items()}, "primary": PRIMARY,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "folds": a.folds, "seed": a.seed,
    }
    sel_path = out / "selection.json"
    if sel_path.exists():                                 # an earlier stage-1 run: the settings must agree
        earlier = {h: {k: v for k, v in s.items() if k != "cv_macro_f1_pooled"}
                   for h, s in json.loads(sel_path.read_text())["chosen"].items()}
        if earlier != {h: describe(chosen[h]) for h in HEADS}:
            print("selection.json from an earlier stage-1 run chose different settings; "
                  "stopping before the validation cameras are scored.")
            sys.exit(EXIT_CHANGED)
    out.mkdir(parents=True, exist_ok=True)
    cv_table.to_csv(out / "cv_scores.csv", index=False)
    sel_path.write_text(json.dumps(selection, indent=2))
    (out / "fit_diagnostics.json").write_text(json.dumps(fits, indent=1))
    print("chosen on the training cameras:", json.dumps(selection["chosen"]))
    print("selection.json sha256", hashlib.sha256(sel_path.read_bytes()).hexdigest(), flush=True)

    # The same comparison at held-out training cameras by nested cross-validation: the
    # cameras of outer fold f are predicted by the outer model (trained on the other four
    # folds) with the setting that wins the inner cross-validation over those four folds.
    nested, nested_choice = {}, {}
    for h in HEADS:
        nested[h], nested_choice[h] = np.full(len(yt), -1), []
        for f in range(a.folds):
            rest = fold_t != f
            key = first_max(cand[h], {key: macro_f1(yt[rest], inner[f][key][tr_img][rest], C) for key in cand[h]})
            nested[h][~rest] = Pt[key][~rest]
            nested_choice[h].append(describe(key))
        assert (nested[h] >= 0).all()

    def bin_f1(true: np.ndarray, pred: np.ndarray) -> dict:
        """Mean F1 of the species in each rarity bin that occur in `true`, and how many there are."""
        cs = class_scores(true, pred, C)
        res = {}
        for bi, name in enumerate(BIN_NAMES):
            sel = (rarity == bi) & (cs["n_true"] > 0)
            res[name] = {"species": int(sel.sum()), "mean_f1": round(float(cs["f1"][sel].mean()), 4) if sel.any() else None}
        return res

    def boot_f1(true: np.ndarray, preds: dict[str, np.ndarray], cam: np.ndarray, seed: int) -> dict[str, np.ndarray]:
        """Macro-F1 of every prediction vector in each camera resample. Resamples are shared."""
        cams = np.unique(cam)
        by_cam = {c: np.flatnonzero(cam == c) for c in cams}
        r = np.random.default_rng(seed)
        res = {k: np.empty(a.bootstrap) for k in preds}
        for j in range(a.bootstrap):
            i = np.concatenate([by_cam[c] for c in r.choice(cams, len(cams))])
            for k, p in preds.items():
                res[k][j] = macro_f1(true[i], p[i], C)
        return res

    def contrast(true, preds, boot, m1: str, m0: str, margin: float | None = None) -> dict:
        diff = boot[m1] - boot[m0]
        lo, hi = float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))
        res = {"a": m1, "b": m0, "diff": round(macro_f1(true, preds[m1], C) - macro_f1(true, preds[m0], C), 4),
               "ci95": [round(lo, 4), round(hi, 4)], "share_of_resamples_a_better": round(float((diff > 0).mean()), 3)}
        if margin is not None:
            res["margin"] = margin
            res["outcome"] = ("supported" if res["diff"] >= margin and lo > 0
                              else "not supported" if hi < margin else "inconclusive")
        return res

    boot_t = boot_f1(yt, nested, cam_t, a.seed + 5)
    train_side = {
        "macro_f1_pooled": {h: round(macro_f1(yt, nested[h], C), 4) for h in HEADS},
        "accuracy_pooled": {h: round(float((nested[h] == yt).mean()), 4) for h in HEADS},
        "mean_f1_by_training_bursts": {h: bin_f1(yt, nested[h]) for h in HEADS},
        "contrasts": {k: contrast(yt, nested, boot_t, v[0], v[1]) for k, v in CONTRASTS.items()},
        "settings_chosen_per_fold": nested_choice,
    }
    print("held-out training cameras, nested cross-validation:",
          json.dumps({"macro_f1_pooled": train_side["macro_f1_pooled"], "contrasts": train_side["contrasts"]}), flush=True)
    d.img.loc[tr_img, ["file_name", "location", "burst_id"]].assign(
        fold=fold_t, y=yt, **{f"nested_{SLUG[h]}": nested[h] for h in HEADS},
        **{f"allfolds_{SLUG[h]}": Pt[chosen[h]] for h in HEADS}).to_csv(out / "train_camera_predictions.csv.gz", index=False)

    if stops:
        print("stopping before the validation cameras are scored:", "; ".join(stops))
        print("Change the grid or the iteration cap and get the run approved again.")
        sys.exit(EXIT_STOP)
    if a.select_only:
        print(f"--select-only: stopping before the validation cameras are scored  {time.time() - t0:5.0f}s")
        return

    # ===== Stage 2: fit on all training cameras, score the validation cameras once =========
    Xt, Xq, y, w, present, logf = prepare(tr_b, va_img)
    uniq, local = bursts_of(va_img)
    yv, cam_v = d.y[va_img], d.cam[va_img]
    heads_wb = {l2: linear(Xt, y, w, l2, "final") for l2 in
                dict.fromkeys([chosen["ordinary"][1][0], chosen["frequency-adjusted"][1][0], PILOT["l2"]])}

    (out / "fit_diagnostics.json").write_text(json.dumps(fits, indent=1))
    if any(x["hit_cap"] for x in fits if x["fit"] == "final"):
        print("a final fit reached the iteration or evaluation cap; stopping before the validation cameras are scored.")
        sys.exit(EXIT_STOP)
    # From here on the validation cameras are scored. The marker makes a second attempt
    # impossible even if this one fails before results.json is written.
    (out / STARTED).write_text(f"{time.strftime('%F %T')}  selection.json sha256 "
                               f"{hashlib.sha256(sel_path.read_bytes()).hexdigest()}\n")

    def lin_preds(l2: float, tau: float) -> tuple[np.ndarray, np.ndarray]:
        W, b = heads_wb[l2]
        z = masked(Xq @ W + b, present)
        return safe_argmax(z - tau * logf), safe_argmax(pool(z, local, len(uniq)) - tau * logf)[local]

    per_image, pv = {}, {}
    for h in ("ordinary", "frequency-adjusted"):
        per_image[h], pv[h] = lin_preds(*chosen[h][1])
    M, S = lda_parts(Xt, y, w, present, C)
    z = lda_scores(Xq, M, S, chosen["linear discriminant"][1], present)
    per_image["linear discriminant"], pv["linear discriminant"] = safe_argmax(z), safe_argmax(pool(z, local, len(uniq)))[local]

    # The pilot's pipeline, as in pilot_a.py: the source cameras' global mix is taken out of
    # the scores, the pooled evidence is scaled, and the mix is added back. Its feature
    # scaling is the one used above; it masks no species.
    full = Source(d, d.train_cams)
    assert np.allclose(full.scale(va_img), Xq, atol=1e-5), "the pilot's feature scaling differs"
    W, b = heads_wb[PILOT["l2"]]
    ev = scaled(pool((Xq @ W + b - np.log(full.total)).astype(np.float32), local, len(uniq)),
                d.b_n[uniq], PILOT["temp"], PILOT["gamma"])
    pv[PILOT_NAME] = log_softmax(ev + np.log(full.total)).argmax(1)[local]

    absent_in_val = int((np.bincount(yv, minlength=C)[~present] > 0).sum())
    rows = []
    for h in HEADS:
        cs = class_scores(yv, pv[h], C)
        rows.append({"classifier": h, "setting": describe(chosen[h]),
                     "macro_f1_pooled": round(macro_f1(yv, pv[h], C), 4),
                     "macro_f1_per_image": round(macro_f1(yv, per_image[h], C), 4),
                     "accuracy_pooled": round(float((pv[h] == yv).mean()), 4),
                     "validation_species_never_predicted": int(((cs["n_true"] > 0) & (cs["n_pred"] == 0)).sum()),
                     "mean_f1_by_training_bursts": bin_f1(yv, pv[h]),
                     "cv_macro_f1_pooled_train_cameras": round(cv[chosen[h]], 4)})
    pilot_f1 = macro_f1(yv, pv[PILOT_NAME], C)
    rows.append({"classifier": PILOT_NAME, "setting": {k: PILOT[k] for k in ("l2", "temp", "gamma")},
                 "macro_f1_pooled": round(pilot_f1, 4), "macro_f1_per_image": None,
                 "accuracy_pooled": round(float((pv[PILOT_NAME] == yv).mean()), 4),
                 "validation_species_never_predicted": None,
                 "mean_f1_by_training_bursts": bin_f1(yv, pv[PILOT_NAME]), "cv_macro_f1_pooled_train_cameras": None})
    boot_v = boot_f1(yv, pv, cam_v, a.seed + 5)
    contrasts = {k: contrast(yv, pv, boot_v, *v) for k, v in CONTRASTS.items()}
    versus_pilot = {h: contrast(yv, pv, boot_v, h, PILOT_NAME) for h in HEADS}

    per_class = pd.DataFrame({"species": classes, "train_bursts": n_bursts, "bin": [BIN_NAMES[r] for r in rarity],
                              "val_images": np.bincount(yv, minlength=C)})
    for h in HEADS:
        cs = class_scores(yv, pv[h], C)
        for m in ("f1", "precision", "recall"):
            per_class[f"{m}_{SLUG[h]}"] = np.round(cs[m], 3)
        per_class[f"predicted_{SLUG[h]}"] = cs["n_pred"].astype(int)
    per_class[per_class["val_images"] > 0].to_csv(out / "per_class_val.csv", index=False)
    d.img.loc[va_img, ["file_name", "location", "burst_id"]].assign(
        y=yv, **{f"pooled_{SLUG[h]}": pv[h] for h in HEADS}, **{f"per_image_{SLUG[h]}": per_image[h] for h in HEADS},
        pooled_pilot_calibrated=pv[PILOT_NAME]).to_csv(out / "val_predictions.csv.gz", index=False)

    hit = [x for x in fits if x["hit_cap"]]
    res = {
        "data": dict(info, classes=C, species_per_bin={name: int((rarity == i).sum()) for i, name in enumerate(BIN_NAMES)},
                     species_without_training_images=int((~present).sum()),
                     validation_species_without_training_images=absent_in_val),
        "selection": selection,
        "validation": {"rows": rows, "contrasts": contrasts, "primary": PRIMARY, "versus_pilot_pipeline": versus_pilot,
                       "pilot_reproduction": {"reported": PILOT["reported_macro_f1"], "here": round(pilot_f1, 4),
                                              "agrees_within_0.003": bool(abs(pilot_f1 - PILOT["reported_macro_f1"]) <= 0.003)}},
        "held_out_training_cameras": train_side,
        "optimisation": {"fits": len(fits), "reached_a_cap": len(hit), "max_iterations": max(x["iterations"] for x in fits),
                         "max_evaluations": max(x["evaluations"] for x in fits),
                         "max_grad": max(x["grad_max"] for x in fits), "detail": fits},
        "notes": [
            "Grids, pooling, selection rule, contrasts, margins and the primary contrast are constants of the script "
            "and were fixed before the validation cameras were scored. The validation cameras were scored once.",
            "Not a first look for everything: the pilot scored the ordinary head on these validation cameras, and its "
            "'balanced rule' column (0.837) is close to the frequency-adjusted classifier at tau = 1. The linear "
            "discriminant had not been scored on them before.",
            "32 validation cameras and one split. The bootstrap resamples cameras, so the set of species in a resample "
            "varies; intervals are wide and do not hold the species set fixed.",
            "The pilot's calibrated pipeline uses a different pooling rule and is shown for continuity; it is not part "
            "of V1 to V3.",
            "Held-out training cameras: nested cross-validation. The setting for an outer fold comes from a four-fold "
            "inner cross-validation whose models and scores exclude that fold. This removes selection within the grids. "
            "It does not remove that the three classifiers and the grid ranges were fixed after Run 4, which used the "
            "same cameras and folds, and it is one fold assignment.",
            "cv_scores.csv is the table the settings were chosen from; its best values are optimistic.",
            "Whole-image features from one backbone. Fits that stop before either cap ended on the optimiser's "
            "no-progress rule in single precision; they are not certified converged.",
        ],
        "seconds": round(time.time() - t0, 1), "args": vars(a),
    }
    (out / "results.json").write_text(json.dumps(res, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))

    fmt = lambda v: "" if v is None else f"{v:.3f}"
    print("\nValidation cameras")
    print("| classifier | setting | macro-F1 pooled | per image | accuracy | "
          + " | ".join(f"F1 {name} (species)" for name in BIN_NAMES) + " | tuning score |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        bins = " | ".join(f"{fmt(v['mean_f1'])} ({v['species']})" for v in r["mean_f1_by_training_bursts"].values())
        print(f"| {r['classifier']} | {json.dumps(r['setting'])} | {fmt(r['macro_f1_pooled'])} | {fmt(r['macro_f1_per_image'])} | "
              f"{fmt(r['accuracy_pooled'])} | {bins} | {fmt(r['cv_macro_f1_pooled_train_cameras'])} |")
    for k, v in contrasts.items():
        print(f"{k}{' (primary)' if k == PRIMARY else ''}: {json.dumps(v)}")
    print("versus the pilot's pipeline:", json.dumps(versus_pilot))
    print("pilot reproduction:", json.dumps(res["validation"]["pilot_reproduction"]))
    print(f"species without training images: {int((~present).sum())}, of which present in validation: {absent_in_val}")
    o = res["optimisation"]
    print(f"optimisation: {o['fits']} fits, {o['reached_a_cap']} reached a cap, most iterations {o['max_iterations']}, "
          f"most evaluations {o['max_evaluations']}, max |grad| {o['max_grad']:.2e}")
    print(f"wrote {out}/ in {res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
