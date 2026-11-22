"""How much does it still cost to meet a camera for the first time?

    python gap_measure.py --cache features/dinov2_vitb14_reg_448 --out results/gap_v1 --device cuda

A measurement, not a method. For the frozen-feature pipeline of the idea-A pilot (DINOv2
features, ordinary linear head, optional pooling over bursts) it compares, on the SAME
query bursts:

  unseen   the query camera contributes nothing to training
  seen     other bursts of the query camera are in training

Only the training cameras of `official_ood` are used (camera-wise folds, as in the pilot).
Validation and test cameras are not touched.

How the comparison is kept fair.
- Matched queries: both models are scored on exactly the same bursts, so the classes and
  the difficulty are the same on both sides.
- Matched budget: the seen model gets the query cameras' other bursts ("support") and
  gives up the same number of bursts, drawn at random, from the remaining cameras. Both
  models train on the same number of bursts; only where they come from differs.
- Two ways to pick the queries of a camera. "random": a random half of its bursts, which
  is what the repo's `random_burst` split does and lets neighbouring bursts of one visit
  fall on both sides. "later": the later half in time, with only bursts that END before
  the first query burst STARTS as support, which is closer to a camera that has been
  running for a while. Bursts that straddle that boundary, or have no usable timestamp,
  are used on neither side and counted.
- Class support: a species can be missing from a model's training images (its cameras all
  sit in the held-out fold, or the budget swap removed its last bursts). Scores are given
  over all query images and again over the query images whose species occurs in the
  training IMAGES of both models.

Cameras with a single burst have no support and are left out of the queries.

What the number is. Both models train on the same number of bursts, not the same species
mix or image count. The gap is the value of replacing bursts from other cameras by bursts
from the query camera, including what those say about the camera's species mix. It is not
a pure background effect. "Pooled" is the plain argmax of a burst's mean scores, without
the calibration and prior of the pilot's 0.824 pipeline.
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

from pilot_a import Data, gpu_is_free, macro_f1, pool
from val_compare import fit_linear   # pilot_a.fit_head with both optimiser caps recorded

SPLITS = ("random", "later")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="folder written by cache_features.py")
    ap.add_argument("--repo", default=".", help="repo root, for data/gps_locations.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-classes", type=int, default=60)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--l2", default="0.01,0.03,0.1",
                    help="comma-separated; the FIRST value is the primary one (the pilot's choice)")
    ap.add_argument("--repeats", type=int, default=3, help="random query/support splits")
    ap.add_argument("--max-iter", type=int, default=2000)
    ap.add_argument("--bootstrap", type=int, default=2000)
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
    d = Data(Path(a.cache), Path(a.repo), a.n_classes)
    C = d.C
    l2_grid = [float(v) for v in a.l2.split(",")]
    rng = np.random.default_rng(a.seed)

    # Training cameras only. Everything below indexes bursts of the training fold.
    tr_b = np.flatnonzero(d.b_fold == "train")
    tr_set = np.zeros(len(d.b_y), dtype=bool)
    tr_set[tr_b] = True
    img_in_train = tr_set[d.burst]
    times = d.img.assign(t=pd.to_datetime(d.img["datetime"], errors="coerce")).groupby(d.burst)["t"]
    b_time, b_end = times.min().to_numpy(), times.max().to_numpy()
    b_timed = ~(np.isnat(b_time) | np.isnat(b_end)) & ~times.apply(lambda t: t.isna().any()).to_numpy()
    b_hour = d.img.groupby(d.burst)["hour"].first().to_numpy()
    b_id = d.img.groupby(d.burst)["burst_id"].first().to_numpy()

    # The same fold construction as the pilot: balanced by region, then at random.
    order = sorted(d.train_cams, key=lambda c: (d.region.get(c, "none"), rng.random()))
    fold_of = {c: i % a.folds for i, c in enumerate(order)}
    b_foldid = np.array([fold_of.get(c, -1) for c in d.b_cam])

    def split_bursts(split: str, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
        """Query and support bursts of every training camera that can supply both.

        Returns:
            Two booleans over bursts and the counts of what was left out. For "later",
            every support burst of a camera ends strictly before its first query burst
            starts; this is asserted.
        """
        r = np.random.default_rng(seed)
        q, sup = np.zeros(len(d.b_y), dtype=bool), np.zeros(len(d.b_y), dtype=bool)
        left = {"cameras_with_under_2_usable_bursts": 0, "bursts_without_timestamp": 0,
                "support_bursts_straddling_the_boundary": 0, "cameras_left_without_support": 0}
        for c in d.train_cams:
            idx = np.flatnonzero(d.b_cam == c)
            if split == "later":
                left["bursts_without_timestamp"] += int((~b_timed[idx]).sum())
                idx = idx[b_timed[idx]]
                idx = idx[np.lexsort((b_end[idx], b_time[idx]))]
            else:
                idx = r.permutation(idx)
            if len(idx) < 2:
                left["cameras_with_under_2_usable_bursts"] += 1
                continue
            cut = len(idx) - len(idx) // 2
            s_idx, q_idx = idx[:cut], idx[cut:]
            if split == "later":
                clean = b_end[s_idx] < b_time[q_idx].min()
                left["support_bursts_straddling_the_boundary"] += int((~clean).sum())
                s_idx = s_idx[clean]
                if len(s_idx) == 0:
                    left["cameras_left_without_support"] += 1
                    continue
                assert b_end[s_idx].max() < b_time[q_idx].min()
            q[q_idx], sup[s_idx] = True, True
        return q, sup, left

    def fit_predict(train_b: np.ndarray, query_b: np.ndarray, l2: float) -> tuple[np.ndarray, np.ndarray, dict]:
        """Train an ordinary head on the given bursts; predict the query bursts.

        Scaling is fitted on the training images of this model only.

        Returns:
            Per-image predictions for the query images (in image order), the per-burst
            predictions from pooled scores (aligned with `np.flatnonzero(query_b)`), and
            the fit diagnostics.
        """
        if a.device == "cuda" and not gpu_is_free():
            print("another job is on the GPU; stopping. Nothing is saved, re-run later.")
            sys.exit(4)
        tr_img, q_img = train_b[d.burst], query_b[d.burst]
        mu, sd = d.X[tr_img].mean(0), d.X[tr_img].std(0) + 1e-6
        W, b, diag = fit_linear((d.X[tr_img] - mu) / sd, d.y[tr_img], d.w[tr_img], l2, C,
                                a.max_iter, a.device)
        s = ((d.X[q_img] - mu) / sd) @ W + b
        q_idx = np.flatnonzero(query_b)
        local = np.searchsorted(q_idx, d.burst[q_img])
        return s.argmax(1), pool(s, local, len(q_idx)).argmax(1), diag

    # One entry per (split, repeat): which bursts are queries, and predictions per model.
    runs = [("later", 0)] + [("random", r) for r in range(a.repeats)]
    store = {}
    for split, rep in runs:
        q, sup, left = split_bursts(split, a.seed + 10 * rep + 1)
        store[(split, rep)] = {"query": q, "support": sup, "left_out": left,
                               "pred": {(m, k, l2): None for m in ("unseen", "seen")
                                        for k in ("image", "burst") for l2 in l2_grid},
                               # per query IMAGE: its species is in both models' training images
                               "supported": np.zeros(len(d.y), dtype=bool),
                               "moved": {}}
    fits = []
    for f in range(a.folds):
        other = tr_set & (b_foldid != f)                  # bursts of the other folds' cameras
        held = tr_set & (b_foldid == f)
        unseen_cache = {}
        for split, rep in runs:
            st = store[(split, rep)]
            q = st["query"] & held
            support = st["support"] & held
            # Budget: the seen model gives up as many bursts of the other cameras as it gains.
            r = np.random.default_rng(a.seed + 1000 * f + 10 * rep + (0 if split == "later" else 5))
            drop = r.choice(np.flatnonzero(other), size=int(support.sum()), replace=False)
            seen_train = other.copy()
            seen_train[drop] = False
            seen_train |= support
            assert seen_train.sum() == other.sum() and not (seen_train & q).any() and not (other & q).any()
            st["moved"][f] = int(support.sum())
            q_img_mask = q[d.burst]
            in_both = np.intersect1d(np.unique(d.y[other[d.burst]]), np.unique(d.y[seen_train[d.burst]]))
            st["supported"][q_img_mask] = np.isin(d.y[q_img_mask], in_both)
            for l2 in l2_grid:
                # The unseen model does not depend on the split; its fit is reused, but it is
                # scored on each split's own queries.
                if l2 not in unseen_cache:
                    tr_img = other[d.burst]
                    mu, sd = d.X[tr_img].mean(0), d.X[tr_img].std(0) + 1e-6
                    if a.device == "cuda" and not gpu_is_free():
                        print("another job is on the GPU; stopping. Nothing is saved, re-run later.")
                        sys.exit(4)
                    W, b, diag = fit_linear((d.X[tr_img] - mu) / sd, d.y[tr_img], d.w[tr_img], l2, C,
                                            a.max_iter, a.device)
                    fits.append(dict(diag, model="unseen", fold=f, l2=l2))
                    unseen_cache[l2] = (mu, sd, W, b)
                mu, sd, W, b = unseen_cache[l2]
                s = ((d.X[q_img_mask] - mu) / sd) @ W + b
                q_idx = np.flatnonzero(q)
                local = np.searchsorted(q_idx, d.burst[q_img_mask])
                out = {"unseen": (s.argmax(1), pool(s, local, len(q_idx)).argmax(1))}
                pi, pb, diag = fit_predict(seen_train, q, l2)
                fits.append(dict(diag, model="seen", fold=f, l2=l2, split=split, repeat=rep))
                out["seen"] = (pi, pb)
                for m, (p_img, p_burst) in out.items():
                    for k, val, where in (("image", p_img, q_img_mask), ("burst", p_burst, q)):
                        full = st["pred"][(m, k, l2)]
                        if full is None:
                            full = st["pred"][(m, k, l2)] = np.full(len(where), -1, dtype=np.int64)
                        full[where] = val
        last = [x for x in fits if x["fold"] == f]
        print(f"  fold {f + 1}/{a.folds} done  {time.time() - t0:5.0f}s  {len(last)} fits, "
              f"{sum(x['hit_cap'] for x in last)} reached a cap, most evaluations {max(x['evaluations'] for x in last)}, "
              f"max |grad| {max(x['grad_max'] for x in last):.1e}",
              flush=True)
        projected = (time.time() - t0) / (f + 1) * (a.folds + 0.5)
        if a.max_minutes and projected > a.max_minutes * 60:
            print(f"projected {projected / 60:.0f} min exceeds --max-minutes {a.max_minutes:.0f}; stopping.")
            sys.exit(3)

    # --- scoring --------------------------------------------------------------------------
    brng = np.random.default_rng(a.seed + 5)

    def scores(split: str, l2: float, restrict: bool) -> dict:
        """Scores of both models on one split type, averaged over its repeats."""
        reps = [rep for s_, rep in runs if s_ == split]
        per_rep, cam_sets = [], []
        for rep in reps:
            st = store[(split, rep)]
            q = st["query"] & tr_set
            qi = q[d.burst] & img_in_train
            if restrict:
                qi = qi & st["supported"]
            item = {"y_img": d.y[qi], "cam_img": d.cam[qi], "n_bursts": int(len(np.unique(d.burst[qi]))),
                    "n_images": int(qi.sum()), "n_cameras": int(len(np.unique(d.cam[qi]))),
                    "n_classes": int(len(np.unique(d.y[qi])))}
            for m in ("unseen", "seen"):
                item[(m, "image")] = st["pred"][(m, "image", l2)][qi]
                item[(m, "burst")] = st["pred"][(m, "burst", l2)][d.burst[qi]]
                assert (item[(m, "image")] >= 0).all() and (item[(m, "burst")] >= 0).all(), "a query has no prediction"
            per_rep.append(item)
            cam_sets.append(np.unique(item["cam_img"]))

        def f1(item, m, k, idx=None):
            y, p = item["y_img"], item[(m, k)]
            return macro_f1(y if idx is None else y[idx], p if idx is None else p[idx], C)

        out = {"repeats": len(reps), **{k: float(np.mean([it[k] for it in per_rep]))
                                        for k in ("n_bursts", "n_images", "n_cameras", "n_classes")}}
        # The primary l2 gets the full bootstrap; the sensitivity values a tenth of it.
        n_boot = a.bootstrap if l2 == l2_grid[0] else max(200, a.bootstrap // 10)
        cams = np.unique(np.concatenate(cam_sets))
        by_cam = [{c: np.flatnonzero(it["cam_img"] == c) for c in cams} for it in per_rep]
        out["bootstrap_resamples"] = n_boot
        for k, name in (("image", "per_image"), ("burst", "pooled")):
            u = float(np.mean([f1(it, "unseen", k) for it in per_rep]))
            s = float(np.mean([f1(it, "seen", k) for it in per_rep]))
            acc_u = float(np.mean([(it[("unseen", k)] == it["y_img"]).mean() for it in per_rep]))
            acc_s = float(np.mean([(it[("seen", k)] == it["y_img"]).mean() for it in per_rep]))
            # Paired bootstrap over cameras; repeats are averaged inside every resample.
            diffs = []
            for _ in range(n_boot):
                draw = brng.choice(cams, len(cams))
                dd = []
                for it, bc in zip(per_rep, by_cam):
                    idx = np.concatenate([bc[c] for c in draw])
                    dd.append(f1(it, "seen", k, idx) - f1(it, "unseen", k, idx))
                diffs.append(np.mean(dd))
            out[name] = {"macro_f1_unseen": round(u, 4), "macro_f1_seen": round(s, 4), "gap": round(s - u, 4),
                         "gap_ci95": [round(float(np.percentile(diffs, 2.5)), 4),
                                      round(float(np.percentile(diffs, 97.5)), 4)],
                         "accuracy_unseen": round(acc_u, 4), "accuracy_seen": round(acc_s, 4)}
        return out

    results = {f"{split} | l2={l2:g} | {'species in both models training images' if r else 'all query images'}":
               scores(split, l2, r) for split in SPLITS for l2 in l2_grid for r in (False, True)}

    # Burst-level predictions at the primary l2, for the error analysis that follows.
    l2 = l2_grid[0]
    frames = []
    for split, rep in (("later", 0), ("random", 0)):
        st = store[(split, rep)]
        q = np.flatnonzero(st["query"] & tr_set)
        qi = st["query"][d.burst] & img_in_train
        frame_ok = {m: pd.Series(st["pred"][(m, "image", l2)][qi] == d.y[qi]).groupby(d.burst[qi]).mean()
                    for m in ("unseen", "seen")}
        frames.append(pd.DataFrame({
            "split": split, "burst_id": b_id[q], "camera": d.b_cam[q], "fold": b_foldid[q], "y": d.b_y[q],
            "n_frames": d.b_n[q], "start": b_time[q], "hour": b_hour[q],
            "species_in_both_models_training": pd.Series(st["supported"][qi]).groupby(d.burst[qi]).all().reindex(q).to_numpy(),
            "pred_unseen_pooled": st["pred"][("unseen", "burst", l2)][q],
            "pred_seen_pooled": st["pred"][("seen", "burst", l2)][q],
            "frames_right_unseen": frame_ok["unseen"].reindex(q).to_numpy(),
            "frames_right_seen": frame_ok["seen"].reindex(q).to_numpy(),
        }))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(frames).to_csv(out / "query_bursts.csv.gz", index=False)

    # Every query image of every split and repeat at the primary l2, so that runs on
    # different features (same index, same seeds, hence the same queries) can be compared
    # pair by pair afterwards.
    rows = []
    for (split, rep), st in store.items():
        qi = st["query"][d.burst] & img_in_train
        rows.append(pd.DataFrame({
            "split": split, "repeat": rep, "file_name": d.img["file_name"].to_numpy()[qi],
            "camera": d.cam[qi], "burst_id": b_id[d.burst[qi]], "y": d.y[qi], "supported": st["supported"][qi],
            **{f"pred_{m}_{k}": st["pred"][(m, k, l2)][qi] if k == "image" else st["pred"][(m, k, l2)][d.burst[qi]]
               for m in ("unseen", "seen") for k in ("image", "burst")}}))
    pd.concat(rows).to_csv(out / "query_images.csv.gz", index=False)

    hit = [x for x in fits if x["hit_cap"]]
    res = {
        "data": {"train_cameras": len(d.train_cams), "train_bursts": int(tr_set.sum()),
                 "train_images": int(img_in_train.sum()),
                 "cameras_with_one_burst_excluded": int(sum((d.b_cam[tr_b] == c).sum() < 2 for c in d.train_cams))},
        "primary_l2": l2, "results": results,
        "bursts_moved_per_fold": {f"{s}/{r}": store[(s, r)]["moved"] for s, r in runs},
        "left_out_of_queries": {f"{s}/{r}": store[(s, r)]["left_out"] for s, r in runs},
        "optimisation": {"fits": len(fits), "reached_a_cap": len(hit),
                         "max_iterations": max(x["iterations"] for x in fits),
                         "max_evaluations": max(x["evaluations"] for x in fits),
                         "max_grad": max(x["grad_max"] for x in fits), "detail": fits},
        "notes": [
            "Training cameras only; cross-validated over cameras. Not validation, not test.",
            "Both models train on the same number of bursts and are scored on the same query bursts. "
            "Species mix and image counts are not matched, so the gap includes what the camera's own "
            "bursts say about its species mix; it is not a pure background effect.",
            "Pooled means the argmax of a burst's mean scores: the uncalibrated pipeline, not the "
            "calibrated and prior-adjusted one that scored 0.824 on the validation cameras.",
            "The l2 values were not tuned for this measurement; the first is the pilot's choice.",
            "Bootstrap resamples cameras, so the set of species in a resample varies.",
            "A fit counts as capped if it reached the iteration cap or the loss-evaluation cap (1.25 times the iteration "
            "cap). Fits that stop before both ended on the optimiser's no-progress rule; they are not "
            "certified converged.",
        ],
        "seconds": round(time.time() - t0, 1), "args": vars(a),
    }
    (out / "results.json").write_text(json.dumps(res, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print("| queries | l2 | scored on | bursts | pooled: unseen | seen | gap (95% interval) | per image: unseen | seen | gap |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for key, r in results.items():
        split, l2s, scope = [x.strip() for x in key.split("|")]
        p, i = r["pooled"], r["per_image"]
        print(f"| {split} | {l2s[3:]} | {scope} | {r['n_bursts']:.0f} | {p['macro_f1_unseen']:.3f} | {p['macro_f1_seen']:.3f} | "
              f"{p['gap']:+.3f} ({p['gap_ci95'][0]:+.3f} to {p['gap_ci95'][1]:+.3f}) | {i['macro_f1_unseen']:.3f} | "
              f"{i['macro_f1_seen']:.3f} | {i['gap']:+.3f} |")
    o = res["optimisation"]
    print(f"optimisation: {o['fits']} fits, {o['reached_a_cap']} reached a cap, most iterations {o['max_iterations']}, "
          f"most evaluations {o['max_evaluations']}, max |grad| {o['max_grad']:.2e}")
    print(f"wrote {out}/results.json, query_bursts.csv.gz and query_images.csv.gz in {res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
