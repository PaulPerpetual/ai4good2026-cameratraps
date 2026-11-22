"""Why do rare species fail at cameras the model has not seen?

    python rare_checks.py --cache features/dinov2_vitb14_reg_448 --out results/rare_v1 --device cuda

Diagnostics and controls on the frozen-feature pipeline of the idea-A pilot. Nothing here
is a new method. Three parts, all with the pilot's camera-wise folds over the TRAINING
cameras, so every prediction is for a camera the model did not train on. Validation and
test cameras are not used.

  A  Where the loss is. Per-species scores of the ordinary head, how often each species
     is predicted compared with how often it occurs, where its errors go (same family,
     same order, same region, a common species), and burst recall against how many of the
     species' bursts the fold could train on.
  B  Simple heads. The ordinary head against untuned standard alternatives: subtracting
     the log class frequency after training, the same inside the loss, class-balanced
     weights, class prototypes (cosine), and a shared-covariance linear discriminant.
  C  Reduced support. Half of the common species at a time are cut down to k training
     bursts, taken from few cameras, and scored at held-out cameras. This is the
     within-species response to having less labelled support. It changes the number of
     bursts and the number of cameras together, and common species are not the same
     animals as rare ones, so it does not by itself say why naturally rare species fail.
     It is set beside the naturally rare species at the same fold-specific support.

Every setting is a fixed default (l2 0.01 as in the pilot, adjustment strength 1,
shrinkage 0.1). Nothing is tuned, so differences between heads are indications, and the
species bins are small.

Class frequency always means the weight a species carries in the loss: the sum of image
weights, where every burst weighs 1 in total. A species with no training image in a fold
cannot be predicted there and is masked for every head.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from iwildcam.data import load_task, load_taxonomy
from pilot_a import Data, fit_head, gpu_is_free, macro_f1, pool

BINS = [0, 20, 50, 150, 10**9]
BIN_NAMES = ["under 20", "20 to 49", "50 to 149", "150 or more"]
SUPPORT_BINS = [-1, 4, 9, 19, 49, 149, 10**9]
SUPPORT_NAMES = ["0 to 4", "5 to 9", "10 to 19", "20 to 49", "50 to 149", "150 or more"]
HEADS = ["ordinary", "ordinary - 0.5 log freq", "ordinary - log freq", "log freq in the loss",
         "class-balanced loss", "prototypes (cosine)", "linear discriminant"]


def class_scores(true: np.ndarray, pred: np.ndarray, C: int) -> dict[str, np.ndarray]:
    """Per-class precision, recall, F1 and counts. Undefined values are NaN."""
    tp = np.bincount(true[true == pred], minlength=C).astype(float)
    n_true = np.bincount(true, minlength=C).astype(float)
    n_pred = np.bincount(pred, minlength=C).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"f1": np.where(n_true > 0, 2 * tp / np.maximum(n_true + n_pred, 1), np.nan),
                "precision": np.where(n_pred > 0, tp / np.maximum(n_pred, 1), np.nan),
                "recall": np.where(n_true > 0, tp / np.maximum(n_true, 1), np.nan),
                "n_true": n_true, "n_pred": n_pred}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--l2", type=float, default=0.01)
    ap.add_argument("--shrinkage", type=float, default=0.1, help="linear discriminant covariance shrinkage")
    ap.add_argument("--rare-k", default="5,10,20,50", help="burst counts for reduced support")
    ap.add_argument("--per-camera", type=int, default=2, help="bursts kept per camera when cutting a species")
    ap.add_argument("--cut-min", type=int, default=150,
                    help="species with at least this many training bursts are cut in part C (150 = the common bin)")
    ap.add_argument("--max-iter", type=int, default=2000)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--max-minutes", type=float, default=30)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    t0 = time.time()
    torch.set_num_threads(a.threads)
    if a.device == "cuda" and not gpu_is_free():          # the GPU is shared: never pile on
        print("GPU is in use; not starting")
        sys.exit(2)
    classes = load_task(require_files=False).classes
    C = len(classes)
    d = Data(Path(a.cache), Path(a.repo), C)
    rng = np.random.default_rng(a.seed)
    fits = []

    tr_img = d.fold == "train"
    tr_b = d.b_fold == "train"
    n_all_bursts = len(d.b_y)
    # A burst is "pure" when all its images carry one label. d.b_y is the majority label.
    b_pure = pd.Series(d.y).groupby(d.burst).nunique().to_numpy() == 1
    # Species facts from the training cameras: burst and camera counts, region, taxonomy.
    n_bursts = np.bincount(d.b_y[tr_b], minlength=C)
    n_cams = np.array([len(np.unique(d.b_cam[tr_b & (d.b_y == y)])) for y in range(C)])
    rarity = np.digitize(n_bursts, BINS[1:-1])            # 0..3, index into BIN_NAMES
    tax = load_taxonomy().reindex(classes)
    fam, order = tax["family"].fillna("?").to_numpy(), tax["order"].fillna("?").to_numpy()
    sp_region = []
    for y in range(C):
        r = pd.Series([d.region.get(c) for c in d.b_cam[tr_b & (d.b_y == y)]]).dropna()
        sp_region.append(r.mode().iat[0] if len(r) else "?")
    sp_region = np.array(sp_region)

    # The pilot's folds.
    order_c = sorted(d.train_cams, key=lambda c: (d.region.get(c, "none"), rng.random()))
    fold_of = {c: i % a.folds for i, c in enumerate(order_c)}
    img_fold = np.array([fold_of.get(c, -1) for c in d.cam])
    b_foldid = np.array([fold_of.get(c, -1) for c in d.b_cam])

    def guard() -> None:
        if a.device == "cuda" and not gpu_is_free():
            print("another job is on the GPU; stopping. Nothing is saved, re-run later.")
            sys.exit(4)

    def prepare(train_b: np.ndarray, query_img: np.ndarray):
        """Scaled training and query features, labels, weights and the loss mass per species."""
        ti = train_b[d.burst]
        mu, sd = d.X[ti].mean(0), d.X[ti].std(0) + 1e-6
        y, w = d.y[ti], d.w[ti]
        mass = np.bincount(y, weights=w, minlength=C)     # what each species weighs in the loss
        return (d.X[ti] - mu) / sd, (d.X[query_img] - mu) / sd, y, w, mass

    def masked(scores: np.ndarray, present: np.ndarray) -> np.ndarray:
        """Scores with species absent from training set to -inf. Present ones must be finite."""
        s = np.array(scores, dtype=np.float64)
        assert np.isfinite(s[:, present]).all(), "non-finite score for a species that was trained on"
        s[:, ~present] = -np.inf
        return s

    def ordinary_scores(train_b: np.ndarray, query_img: np.ndarray, tag: str) -> np.ndarray:
        Xt, Xq, y, w, mass = prepare(train_b, query_img)
        guard()
        W, b, diag = fit_head(Xt, y, w, None, a.l2, C, a.max_iter, a.device)
        fits.append(dict(diag, head="ordinary", fit=tag))
        return masked(Xq @ W + b, mass > 0)

    def heads_predict(train_b: np.ndarray, query_img: np.ndarray, tag: str) -> dict[str, np.ndarray]:
        """Fit every head on the given bursts and score the query images.

        Returns:
            Per head, the (query images, classes) score matrix, -inf for absent species.
            Scaling, class mass, prototypes and covariance come from the training bursts only.
        """
        Xt, Xq, y, w, mass = prepare(train_b, query_img)
        present = mass > 0
        # log share of the loss; absent species get a very negative finite value so the
        # in-loss head can be fitted, and they are masked at prediction anyway
        logf = np.where(present, np.log(np.maximum(mass, 1e-12) / mass.sum()), -30.0)
        out = {}

        def linear(name, weights, offset):
            guard()
            W, b, diag = fit_head(Xt, y, weights, offset, a.l2, C, a.max_iter, a.device)
            fits.append(dict(diag, head=name, fit=tag))
            return Xq @ W + b

        z = linear("ordinary", w, None)
        out["ordinary"] = masked(z, present)
        out["ordinary - 0.5 log freq"] = masked(z - 0.5 * logf, present)
        out["ordinary - log freq"] = masked(z - logf, present)
        # trained with the log frequency added to the scores; predictions use the scores alone
        out["log freq in the loss"] = masked(linear("log freq in the loss", w, np.tile(logf, (len(y), 1))), present)
        # each species carries the same total weight; a burst still counts once within it
        out["class-balanced loss"] = masked(linear("class-balanced loss", (w / mass[y]).astype(np.float32), None), present)

        members = [np.flatnonzero(y == c) for c in range(C)]
        # Prototypes: mean unit-norm feature per species (bursts weighted equally), cosine score.
        tn, qn = np.linalg.norm(Xt, axis=1, keepdims=True), np.linalg.norm(Xq, axis=1, keepdims=True)
        assert (tn > 0).all() and (qn > 0).all(), "zero-norm feature vector"
        Tn, Qn = Xt / tn, Xq / qn
        P = np.zeros((C, Tn.shape[1]), dtype=np.float64)
        for c in np.flatnonzero(present):
            P[c] = (Tn[members[c]] * w[members[c], None]).sum(0)
        pn = np.linalg.norm(P[present], axis=1, keepdims=True)
        assert (pn > 0).all(), "zero-norm prototype"
        P[present] /= pn
        out["prototypes (cosine)"] = masked(Qn @ P.T, present)

        # Linear discriminant: species means, one shared covariance with shrinkage, no prior.
        M = np.zeros((C, Xt.shape[1]), dtype=np.float64)
        for c in np.flatnonzero(present):
            M[c] = np.average(Xt[members[c]], axis=0, weights=w[members[c]])
        R = (Xt - M[y]) * np.sqrt(w)[:, None]
        S = (R.T @ R) / w.sum()
        S = (1 - a.shrinkage) * S + a.shrinkage * (np.trace(S) / len(S)) * np.eye(len(S))
        A = np.linalg.solve(S, M.T)                              # (features, classes)
        out["linear discriminant"] = masked(Xq @ A - 0.5 * np.einsum("cf,fc->c", M, A), present)
        return out

    def to_preds(scores: np.ndarray, query_img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-image argmax, and the argmax of each burst's mean score given to its images.

        A species masked with -inf stays -inf in the burst mean. Anything else non-finite
        is an error and stops the run.
        """
        assert not np.isnan(scores).any() and not np.isposinf(scores).any(), "NaN or +inf score"
        b = d.burst[query_img]
        uniq, local = np.unique(b, return_inverse=True)
        with np.errstate(invalid="ignore"):
            pooled = pool(scores, local, len(uniq))
        assert not np.isnan(pooled).any(), "NaN after pooling"
        return scores.argmax(1), pooled.argmax(1)[local]

    # --- held-out-camera predictions for every training image ---------------------------------
    n = len(d.y)
    pred = {h: {"image": np.full(n, -1), "pooled": np.full(n, -1)} for h in HEADS}
    ks = sorted(int(k) for k in a.rare_k.split(","))
    common = np.flatnonzero(n_bursts >= a.cut_min)
    halves = np.array_split(rng.permutation(common), 2) if len(common) >= 2 else []
    cut_pred = {(h, k): np.full(n, -1) for h in range(len(halves)) for k in ks}
    ref_pred = np.full(n, -1)                              # part C reference: pure bursts, nothing cut
    support_rows = []
    for f in range(a.folds):
        train_b = tr_b & (b_foldid != f)
        q_img = tr_img & (img_fold == f)
        for h, s in heads_predict(train_b, q_img, f"fold{f}").items():
            pred[h]["image"][q_img], pred[h]["pooled"][q_img] = to_preds(s, q_img)

        # C: training sets use pure bursts only, so a cut species has no stray images left
        # inside other species' bursts. Queries are untouched.
        pure_b = train_b & b_pure
        if halves:
            ref_pred[q_img] = to_preds(ordinary_scores(pure_b, q_img, f"fold{f}/pure reference"), q_img)[1]
        for hi, half in enumerate(halves):
            # One ordering per species: cameras at random, at most `per_camera` bursts from
            # each. Every k takes a prefix of it, so the subsets are nested.
            r = np.random.default_rng(a.seed + 100 * f + 10 * hi)
            seq = {}
            for y in half:
                idx = np.flatnonzero(pure_b & (d.b_y == y))
                seq[int(y)] = np.array([b for c in r.permutation(np.unique(d.b_cam[idx]))
                                        for b in r.permutation(idx[d.b_cam[idx] == c])[:a.per_camera]], dtype=int)
            for k in ks:
                tb = pure_b.copy()
                for y in half:
                    keep = seq[int(y)][:k]
                    tb[pure_b & (d.b_y == y)] = False
                    tb[keep] = True
                    support_rows.append({"half": hi, "k": k, "fold": f, "species": classes[y],
                                         "bursts_kept": int(len(keep)), "cameras_kept": int(len(np.unique(d.b_cam[keep]))),
                                         "pure_bursts_available": int((pure_b & (d.b_y == y)).sum()),
                                         "cameras_available": int(len(np.unique(d.b_cam[pure_b & (d.b_y == y)]))),
                                         "reached_k": bool(len(keep) == k)})
                cut_pred[(hi, k)][q_img] = to_preds(ordinary_scores(tb, q_img, f"fold{f}/cut half {hi} k={k}"), q_img)[1]
        last = [x for x in fits if x["fit"].startswith(f"fold{f}")]
        print(f"  fold {f + 1}/{a.folds} done  {time.time() - t0:5.0f}s  {len(last)} fits, "
              f"{sum(x['hit_cap'] for x in last)} hit the cap, max |grad| {max(x['grad_max'] for x in last):.1e}", flush=True)
        projected = (time.time() - t0) / (f + 1) * (a.folds + 1)
        if a.max_minutes and projected > a.max_minutes * 60:
            print(f"projected {projected / 60:.0f} min exceeds --max-minutes {a.max_minutes:.0f}; stopping.")
            sys.exit(3)

    yt, cam_t = d.y[tr_img], d.cam[tr_img]
    assert all((pred[h][k][tr_img] >= 0).all() for h in HEADS for k in ("image", "pooled"))
    brng = np.random.default_rng(a.seed + 5)
    cams = np.unique(cam_t)
    by_cam = {c: np.flatnonzero(cam_t == c) for c in cams}
    draws = [np.concatenate([by_cam[c] for c in brng.choice(cams, len(cams))]) for _ in range(a.bootstrap)]

    def summarise(y, p_img, p_pool) -> dict:
        cs = class_scores(y, p_pool, C)
        row = {"macro_f1_per_image": round(macro_f1(y, p_img, C), 4), "macro_f1_pooled": round(macro_f1(y, p_pool, C), 4),
               "accuracy_pooled": round(float((p_pool == y).mean()), 4)}
        for bi, name in enumerate(BIN_NAMES):
            sel = (rarity == bi) & (cs["n_true"] > 0)
            row[f"mean_f1 {name}"] = round(float(np.nanmean(cs["f1"][sel])), 4) if sel.any() else None
        return row

    def rare_f1(y, p) -> float:
        cs = class_scores(y, p, C)
        sel = (rarity == 0) & (cs["n_true"] > 0)
        return float(np.nanmean(cs["f1"][sel])) if sel.any() else np.nan

    def versus_ordinary(h: str) -> dict:
        p1, p0 = pred[h]["pooled"][tr_img], pred["ordinary"]["pooled"][tr_img]
        dm = np.array([macro_f1(yt[i], p1[i], C) - macro_f1(yt[i], p0[i], C) for i in draws])
        dr = np.array([rare_f1(yt[i], p1[i]) - rare_f1(yt[i], p0[i]) for i in draws])
        ci = lambda v: [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]
        return {"macro_f1_diff": round(macro_f1(yt, p1, C) - macro_f1(yt, p0, C), 4), "macro_f1_ci95": ci(dm),
                "rarest_bin_f1_diff": round(rare_f1(yt, p1) - rare_f1(yt, p0), 4), "rarest_bin_ci95": ci(dr)}

    part_b = {h: summarise(yt, pred[h]["image"][tr_img], pred[h]["pooled"][tr_img]) for h in HEADS}
    for h in HEADS[1:]:
        part_b[h]["versus_ordinary"] = versus_ordinary(h)

    # --- A: where the ordinary head loses ---------------------------------------------------
    p0 = pred["ordinary"]["pooled"][tr_img]
    cs = class_scores(yt, p0, C)
    wrong = p0 != yt
    conf = pd.crosstab(pd.Series(yt[wrong], name="true"), pd.Series(p0[wrong], name="pred"))
    per_class = []
    for y in range(C):
        e = conf.loc[y] if y in conf.index else pd.Series(dtype=float)
        e = e[e > 0].sort_values(ascending=False)
        tot = float(e.sum())
        share = lambda mask: round(float(e[[c for c in e.index if mask(c)]].sum() / tot), 3) if tot else np.nan
        per_class.append({
            "species": classes[y], "train_bursts": int(n_bursts[y]), "train_cameras": int(n_cams[y]),
            "bin": BIN_NAMES[rarity[y]], "region": sp_region[y], "family": fam[y], "order": order[y],
            "images": int(cs["n_true"][y]), "f1": round(float(cs["f1"][y]), 3),
            "precision": round(float(cs["precision"][y]), 3), "recall": round(float(cs["recall"][y]), 3),
            "predicted_per_true_image": round(float(cs["n_pred"][y] / max(cs["n_true"][y], 1)), 3),
            "errors_to_same_family": share(lambda c: fam[c] == fam[y] and fam[y] != "?"),
            "errors_to_same_order": share(lambda c: order[c] == order[y] and order[y] != "?"),
            "errors_to_same_region": share(lambda c: sp_region[c] == sp_region[y] and sp_region[y] != "?"),
            "errors_to_common_species": share(lambda c: rarity[c] == 3),
            "top_confusions": "; ".join(f"{classes[c]} ({int(v)})" for c, v in e.head(3).items()),
        })
    per_class = pd.DataFrame(per_class)
    shortfall = (1 - per_class["f1"]) / C
    part_a = {}
    for name in BIN_NAMES:
        g = per_class[per_class["bin"] == name]
        part_a[name] = {
            "species": int(len(g)), "mean_f1": round(float(g["f1"].mean()), 3),
            "mean_recall": round(float(g["recall"].mean()), 3),
            # precision is undefined for a species that is never predicted; say how many count
            "mean_precision_where_defined": round(float(g["precision"].mean()), 3),
            "species_with_defined_precision": int(g["precision"].notna().sum()),
            "median_predicted_per_true_image": round(float(g["predicted_per_true_image"].median()), 3),
            "share_of_macro_f1_shortfall": round(float(shortfall[g.index].sum() / shortfall.sum()), 3),
            "errors_to_same_family": round(float(g["errors_to_same_family"].mean()), 3),
            "errors_to_same_order": round(float(g["errors_to_same_order"].mean()), 3),
            "errors_to_same_region": round(float(g["errors_to_same_region"].mean()), 3),
            "errors_to_common_species": round(float(g["errors_to_common_species"].mean()), 3),
        }
    part_a["macro_f1_pooled_all_60"] = round(float(per_class["f1"].mean()), 4)

    # Burst recall against the support the fold actually had. The pooled prediction of a
    # burst is compared with its majority label; mixed-label bursts are counted here too.
    b_idx = np.flatnonzero(tr_b)
    bt, bfold = d.b_y[tr_b], b_foldid[tr_b]

    def burst_level(p_img_level: np.ndarray) -> np.ndarray:
        """One pooled prediction per training burst (all its images share it)."""
        return pd.Series(p_img_level[tr_img]).groupby(d.burst[tr_img]).first().reindex(b_idx).to_numpy()

    def recall_by_support(p_burst: np.ndarray, species: np.ndarray, support_of) -> tuple[pd.DataFrame, dict]:
        rows = []
        for y in species:
            for f in range(a.folds):
                q = (bt == y) & (bfold == f)
                if q.any():
                    rows.append({"species": classes[y], "fold": f, "query_bursts": int(q.sum()),
                                 "support_bursts": int(support_of(y, f)),
                                 "burst_recall": round(float((p_burst[q] == y).mean()), 3)})
        t = pd.DataFrame(rows, columns=["species", "fold", "query_bursts", "support_bursts", "burst_recall"])
        t["support"] = pd.cut(t["support_bursts"], SUPPORT_BINS, labels=SUPPORT_NAMES)
        summary = {str(k): {"species_fold_pairs": int(len(g)), "species": int(g["species"].nunique()),
                            "query_bursts": int(g["query_bursts"].sum()),
                            "recall": round(float(np.average(g["burst_recall"], weights=g["query_bursts"])), 3)}
                   for k, g in t.groupby("support", observed=True)}
        return t, summary

    natural, part_a["burst_recall_by_fold_support"] = recall_by_support(
        burst_level(pred["ordinary"]["pooled"]), np.arange(C), lambda y, f: ((bt == y) & (bfold != f)).sum())
    part_a["mixed_label_bursts"] = {"training_cameras": int((~b_pure[tr_b]).sum()), "of": int(tr_b.sum())}

    # --- C: reduced support -----------------------------------------------------------------
    part_c, cut_tables = {}, []
    if halves:
        support = pd.DataFrame(support_rows)
        ref_b = burst_level(ref_pred)
        cs_ref = class_scores(yt, ref_pred[tr_img], C)
        rows = []
        for hi, half in enumerate(halves):
            for k in ks:
                cs_k = class_scores(yt, cut_pred[(hi, k)][tr_img], C)
                sup = support[(support["half"] == hi) & (support["k"] == k)]
                tab, _ = recall_by_support(burst_level(cut_pred[(hi, k)]), half,
                                           lambda y, f, sup=sup: sup[(sup["species"] == classes[y]) & (sup["fold"] == f)]["bursts_kept"].iat[0])
                cut_tables.append(tab.assign(k=k))
                for y in half:
                    s_y = sup[sup["species"] == classes[y]]
                    rows.append({"species": classes[y], "k": k, "mean_bursts_kept": round(float(s_y["bursts_kept"].mean()), 1),
                                 "mean_cameras_kept": round(float(s_y["cameras_kept"].mean()), 1),
                                 "folds_reaching_k": int(s_y["reached_k"].sum()),
                                 "f1": round(float(cs_k["f1"][y]), 3), "precision": round(float(cs_k["precision"][y]), 3),
                                 "recall": round(float(cs_k["recall"][y]), 3),
                                 "f1_uncut": round(float(cs_ref["f1"][y]), 3), "recall_uncut": round(float(cs_ref["recall"][y]), 3),
                                 "all_train_bursts": int(n_bursts[y])})
        art = pd.DataFrame(rows)
        for k in ks:
            g = art[art["k"] == k]
            part_c[f"cut to {k} bursts"] = {
                "species": int(len(g)), "mean_bursts_kept": round(float(g["mean_bursts_kept"].mean()), 1),
                "mean_cameras_kept": round(float(g["mean_cameras_kept"].mean()), 1),
                "species_fold_pairs_below_k": int((support[support["k"] == k]["reached_k"] == False).sum()),
                "mean_f1": round(float(g["f1"].mean()), 3), "mean_recall": round(float(g["recall"].mean()), 3),
                "mean_precision_where_defined": round(float(g["precision"].mean()), 3),
                "species_with_defined_precision": int(g["precision"].notna().sum()),
                "f1_range_over_species": [round(float(g["f1"].min()), 3), round(float(g["f1"].max()), 3)]}
        u = art.drop_duplicates("species")
        part_c["same species, nothing cut (pure bursts only)"] = {"mean_f1": round(float(u["f1_uncut"].mean()), 3),
                                                                 "mean_recall": round(float(u["recall_uncut"].mean()), 3),
                                                                 "mean_train_bursts": round(float(u["all_train_bursts"].mean()), 1)}
        cut = pd.concat(cut_tables)
        part_c["burst_recall_by_fold_support, common species cut down"] = {
            str(k): {"species_fold_pairs": int(len(g)), "species": int(g["species"].nunique()), "query_bursts": int(g["query_bursts"].sum()),
                     "recall": round(float(np.average(g["burst_recall"], weights=g["query_bursts"])), 3)}
            for k, g in cut.groupby("support", observed=True)}
        nat_rare = natural[natural["species"].isin([classes[y] for y in np.flatnonzero(rarity <= 1)])]
        part_c["burst_recall_by_fold_support, naturally rare species (under 50 bursts)"] = {
            str(k): {"species_fold_pairs": int(len(g)), "species": int(g["species"].nunique()), "query_bursts": int(g["query_bursts"].sum()),
                     "recall": round(float(np.average(g["burst_recall"], weights=g["query_bursts"])), 3)}
            for k, g in nat_rare.groupby("support", observed=True)}

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    per_class.to_csv(out / "per_class.csv", index=False)
    natural.to_csv(out / "species_by_fold.csv", index=False)
    if halves:
        art.to_csv(out / "reduced_support.csv", index=False)
        support.to_csv(out / "reduced_support_by_fold.csv", index=False)
        cut.to_csv(out / "reduced_support_recall_by_fold.csv", index=False)
    hit = [x for x in fits if x["hit_cap"]]
    res = {
        "data": {"train_cameras": len(d.train_cams), "train_images": int(tr_img.sum()), "train_bursts": int(tr_b.sum()),
                 "species_per_bin": {name: int((rarity == i).sum()) for i, name in enumerate(BIN_NAMES)}},
        "A_where_the_ordinary_head_loses": part_a, "B_heads": part_b, "C_reduced_support": part_c,
        "optimisation": {"fits": len(fits), "hit_iteration_cap": len(hit), "max_iterations": max(x["iterations"] for x in fits),
                         "max_grad": max(x["grad_max"] for x in fits), "detail": fits},
        "notes": [
            "Held-out-camera predictions from five camera-wise folds over the training cameras. No validation or test camera.",
            "Pooled means the argmax of a burst's mean scores, without the calibration behind the pilot's 0.824.",
            "No setting was tuned. Head differences are indications; the rarest bin has 10 species.",
            "Species are binned by how many training bursts they have over all training cameras; tables by "
            "support use the bursts each fold could actually train on.",
            "Part C trains on pure bursts only and cuts bursts and cameras together. It is the within-species "
            "response to less labelled support, not an explanation of why naturally rare species fail. A cut "
            "species' precision also changes because the classifier predicts it less.",
            "Burst recall compares a burst's pooled prediction with its majority label.",
            "Fits that stop before the cap ended on the optimiser's own rule; they are not certified converged.",
        ],
        "seconds": round(time.time() - t0, 1), "args": vars(a),
    }
    (out / "results.json").write_text(json.dumps(res, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))

    fmt = lambda v: "" if v is None else f"{v:.3f}"
    print("\nA. Ordinary head at held-out training cameras, by how many training bursts a species has")
    print("| bursts | species | mean F1 | recall | precision (species where defined) | predicted per true image | share of macro-F1 shortfall | errors to same family | to a common species |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name in BIN_NAMES:
        r = part_a[name]
        print(f"| {name} | {r['species']} | {r['mean_f1']:.3f} | {r['mean_recall']:.3f} | {r['mean_precision_where_defined']:.3f} "
              f"({r['species_with_defined_precision']}) | {r['median_predicted_per_true_image']:.2f} | {r['share_of_macro_f1_shortfall']:.2f} | "
              f"{r['errors_to_same_family']:.2f} | {r['errors_to_common_species']:.2f} |")
    print("burst recall by fold support, all species:", json.dumps(part_a["burst_recall_by_fold_support"]))
    print("\nB. Heads at held-out training cameras")
    print("| head | macro-F1 per image | pooled | accuracy | " + " | ".join(f"F1 {b}" for b in BIN_NAMES) + " |")
    print("|---|---|---|---|---|---|---|---|")
    for h in HEADS:
        r = part_b[h]
        print(f"| {h} | {r['macro_f1_per_image']:.3f} | {r['macro_f1_pooled']:.3f} | {r['accuracy_pooled']:.3f} | "
              + " | ".join(fmt(r[f'mean_f1 {b}']) for b in BIN_NAMES) + " |")
    print("difference from the ordinary head (pooled macro-F1; rarest-bin mean F1), 95% intervals over cameras:")
    for h in HEADS[1:]:
        v = part_b[h]["versus_ordinary"]
        print(f"  {h:26s} {v['macro_f1_diff']:+.3f} {v['macro_f1_ci95']}   rarest bin {v['rarest_bin_f1_diff']:+.3f} {v['rarest_bin_ci95']}")
    print("\nC. Reduced support:")
    for key, val in part_c.items():
        print(f"  {key}: {json.dumps(val)}")
    o = res["optimisation"]
    print(f"\noptimisation: {o['fits']} fits, {o['hit_iteration_cap']} hit the cap, max |grad| {o['max_grad']:.2e}")
    print(f"wrote {out}/ in {res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
