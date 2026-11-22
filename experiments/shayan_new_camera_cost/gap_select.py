"""Which bursts should be labelled at a new camera? Random vs spread-out vs most uncertain.

    python gap_select.py --cache vitl_crop=features/dinov2_vitl14_reg_448_crop --cache vitl=... \
        --primary vitl_crop --check-budget results/budget_v1/query_images.csv.gz \
        --repo . --out results/select_v1 --device cuda

A measurement on the training cameras of `official_ood` only, with gap_budget.py's folds,
chronological queries, 24-hour buffer and "up to k annotated bursts per camera". Validation
and test cameras are not used. Only the choice of WHICH eligible bursts get labelled changes:

  random     gap_budget's random order (same seeds), so ADD_random_k must reproduce its ADD_k.
  spread     farthest-point (k-center greedy) order over the camera's eligible bursts: start
             from random's first burst, then always add the burst farthest (Euclidean, burst-mean
             of features standardised as for U) from those already chosen. Uses no labels.
  uncertain  highest entropy first of U's burst-pooled prediction (U trains on the other folds
             only, so no label of the new camera is used). Deterministic: fitted once per fold.

Every arm adds its s_k to the other folds' bursts (ADD); U is the other folds alone. Contrasts on
the same query images for every k and selector (cameras with at least one eligible burst; paired
bootstrap over cameras, repeats averaged inside every resample):

  spread_k - random_k, uncertain_k - random_k, and each selector's gain over U.

Cameras with at most k eligible bursts get the same bursts under every selector, so differences
come from the cameras with more; the share of cameras where the choice matters is reported.
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
from val_compare import fit_linear

BUDGETS = (1, 2, 4, 8, 16)
SELECTORS = ("random", "spread", "uncertain")
MARGIN = 0.01


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="append", required=True, help="name=folder; repeat for each feature set")
    ap.add_argument("--primary", required=True)
    ap.add_argument("--check-budget", required=True, help="gap_budget query_images.csv.gz: U and random ADD_k must match it")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-classes", type=int, default=60)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--l2", type=float, default=0.01)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--far-hours", type=float, default=24.0)
    ap.add_argument("--primary-k", type=int, default=8)
    ap.add_argument("--max-iter", type=int, default=2000)
    ap.add_argument("--bootstrap", type=int, default=2000, help="for the primary table; the others get a fifth")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--max-minutes", type=float, default=60)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    caches = dict(c.split("=", 1) for c in a.cache)
    names = list(caches)
    assert a.primary in caches and a.primary_k in BUDGETS
    out = Path(a.out)
    if out.exists():
        print(f"{out} exists; not overwriting")
        sys.exit(6)
    t0 = time.time()
    torch.set_num_threads(a.threads)
    if a.device == "cuda" and not gpu_is_free():
        print("GPU is in use; not starting")
        sys.exit(2)
    d = Data(Path(caches[names[0]]), Path(a.repo), a.n_classes)
    C = d.C
    rng = np.random.default_rng(a.seed)

    # --- gap_budget.py's structure, unchanged ----------------------------------------------------
    tr_b = np.flatnonzero(d.b_fold == "train")
    tr_set = np.zeros(len(d.b_y), dtype=bool)
    tr_set[tr_b] = True
    img_in_train = tr_set[d.burst]
    times = d.img.assign(t=pd.to_datetime(d.img["datetime"], errors="coerce")).groupby(d.burst)["t"]
    b_time, b_end = times.min().to_numpy(), times.max().to_numpy()
    b_timed = ~(np.isnat(b_time) | np.isnat(b_end)) & ~times.apply(lambda t: t.isna().any()).to_numpy()
    order = sorted(d.train_cams, key=lambda c: (d.region.get(c, "none"), rng.random()))
    fold_of = {c: i % a.folds for i, c in enumerate(order)}
    b_foldid = np.array([fold_of.get(c, -1) for c in d.b_cam])
    b_id = d.img.groupby(d.burst)["burst_id"].first().to_numpy()

    q = np.zeros(len(d.b_y), dtype=bool)
    elig = np.zeros(len(d.b_y), dtype=bool)
    h = np.timedelta64(int(a.far_hours * 3600), "s")
    for c in d.train_cams:
        idx = np.flatnonzero(d.b_cam == c)
        idx = idx[b_timed[idx]]
        idx = idx[np.lexsort((b_end[idx], b_time[idx]))]
        if len(idx) < 2:
            continue
        cut = len(idx) - len(idx) // 2
        s_idx, q_idx = idx[:cut], idx[cut:]
        s_idx = s_idx[b_end[s_idx] < b_time[q_idx].min()]
        if len(s_idx) == 0:
            continue
        q[q_idx] = True
        elig[s_idx[b_end[s_idx] <= b_time[q_idx].min() - h]] = True
    n_elig = {c: int((elig & (d.b_cam == c)).sum()) for c in d.train_cams}
    # gap_budget's random orders: same seeds, same per-camera permutations
    ranked = {rep: {} for rep in range(a.repeats)}
    for rep in range(a.repeats):
        r_order = np.random.default_rng(a.seed + 500 + rep)
        ranked[rep] = {c: r_order.permutation(np.flatnonzero(elig & (d.b_cam == c))) for c in d.train_cams}

    def spread_order(first: int, cand: np.ndarray, F: np.ndarray) -> np.ndarray:
        """k-center greedy over the rows of F (one per candidate burst), starting from `first`."""
        chosen = [int(np.flatnonzero(cand == first)[0])]
        dmin = np.linalg.norm(F - F[chosen[0]], axis=1)
        for _ in range(min(len(cand), max(BUDGETS)) - 1):
            dmin[chosen] = -1
            nxt = int(dmin.argmax())
            chosen.append(nxt)
            dmin = np.minimum(dmin, np.linalg.norm(F - F[nxt], axis=1))
        return cand[np.array(chosen)]

    # --- fits ------------------------------------------------------------------------------------
    arms = ["U"] + [f"{s}_{k}" for s in SELECTORS for k in BUDGETS]
    preds, fits, plan_rows = {}, [], []
    qi = q[d.burst] & img_in_train
    ref = {"file_name": d.img["file_name"].to_numpy(), "burst": d.burst.copy(), "y": d.y.copy()}

    def fit(X, train_b, dd, tag):
        if a.device == "cuda" and not gpu_is_free():
            print("another job is on the GPU; stopping. Nothing is saved, re-run later.")
            sys.exit(4)
        tr_img = train_b[dd.burst]
        mu, sd = X[tr_img].mean(0), X[tr_img].std(0) + 1e-6
        W, b, diag = fit_linear((X[tr_img] - mu) / sd, dd.y[tr_img], dd.w[tr_img], a.l2, C, a.max_iter, a.device)
        assert np.isfinite(W).all() and np.isfinite(b).all(), f"non-finite weights {tag}"
        fits.append(dict(diag, **tag))
        return mu, sd, W, b

    for ci, name in enumerate(names):
        dd = d if ci == 0 else Data(Path(caches[name]), Path(a.repo), a.n_classes)
        assert (dd.img["file_name"].to_numpy() == ref["file_name"]).all() and (dd.burst == ref["burst"]).all() \
            and (dd.y == ref["y"]).all(), f"{name}: images, bursts or labels differ"
        X = dd.X
        for f in range(a.folds):
            other, held = tr_set & (b_foldid != f), tr_set & (b_foldid == f)
            qf = q & held
            qfi = qf[dd.burst]
            q_idx = np.flatnonzero(qf)
            local = np.searchsorted(q_idx, dd.burst[qfi])
            mu, sd, W, b = fit(X, other, dd, {"set": name, "arm": "U", "fold": f})
            s_u = ((X[qfi] - mu) / sd) @ W + b
            # U's view of the eligible bursts: burst-mean standardised features and pooled entropy
            el = elig & held
            el_idx = np.flatnonzero(el)
            eli = el[dd.burst]
            Z = (X[eli] - mu) / sd
            loc_e = np.searchsorted(el_idx, dd.burst[eli])
            F = pool(Z, loc_e, len(el_idx))
            pl = pool(Z @ W + b, loc_e, len(el_idx))
            pz = np.exp(pl - pl.max(1, keepdims=True))
            pz /= pz.sum(1, keepdims=True)
            ent = -(pz * np.log(np.maximum(pz, 1e-12))).sum(1)
            pos_of = {bi: j for j, bi in enumerate(el_idx)}
            cams_f = [c for c in d.train_cams if fold_of.get(c) == f]
            orders = {}
            for rep in range(a.repeats):
                for c in cams_f:
                    rk = ranked[rep][c]
                    if len(rk) == 0:
                        continue
                    rows = np.array([pos_of[bi] for bi in rk])
                    orders[("random", rep, c)] = rk
                    orders[("spread", rep, c)] = spread_order(rk[0], rk, F[rows])
                    if rep == 0:
                        unc = rk[np.lexsort((np.arange(len(rk)), -ent[rows]))]   # ties keep random order
                        orders[("uncertain", rep, c)] = unc
            outs = {(rep, "U"): s_u for rep in range(a.repeats)}
            for sel in SELECTORS:
                for k in BUDGETS:
                    for rep in range(a.repeats):
                        if sel == "uncertain" and rep > 0:
                            outs[(rep, f"{sel}_{k}")] = outs[(0, f"{sel}_{k}")]
                            continue
                        s_k = np.zeros_like(q)
                        for c in cams_f:
                            o = orders.get((sel, rep, c))
                            if o is not None:
                                s_k[o[:k]] = True
                        assert not (s_k & ~el).any() and not ((other | s_k) & qf).any()
                        plan_rows.append(pd.DataFrame({"set": name, "selector": sel, "k": k, "repeat": rep, "fold": f,
                                                       "burst_id": b_id[s_k]}))
                        m2, s2, W2, b2 = fit(X, other | s_k, dd, {"set": name, "arm": f"{sel}_{k}", "fold": f, "repeat": rep})
                        outs[(rep, f"{sel}_{k}")] = ((X[qfi] - m2) / s2) @ W2 + b2
            for (rep, arm), s in outs.items():
                assert np.isfinite(s).all(), f"non-finite scores {name} {arm}"
                for kind, p in (("image", s.argmax(1)), ("burst", pool(s, local, len(q_idx)).argmax(1)[local])):
                    key = (name, rep, arm, kind)
                    if key not in preds:
                        preds[key] = np.full(len(d.y), -1, dtype=np.int16)
                    preds[key][qfi] = p
            last = [x for x in fits if x["set"] == name and x["fold"] == f]
            print(f"  {name} fold {f + 1}/{a.folds} done  {time.time() - t0:5.0f}s  {len(last)} fits, "
                  f"{sum(x['hit_cap'] for x in last)} reached a cap", flush=True)
            projected = (time.time() - t0) / (ci * a.folds + f + 1) * len(names) * a.folds * 1.15
            if a.max_minutes and projected > a.max_minutes * 60:
                print(f"projected {projected / 60:.0f} min exceeds --max-minutes {a.max_minutes:.0f}; stopping.")
                sys.exit(3)
        if ci:
            del dd, X

    # U and random ADD_k must be gap_budget's, image for image; spread_1 must equal random_1.
    old = pd.read_csv(a.check_budget)
    pos = pd.Index(ref["file_name"]).get_indexer(old["file_name"])
    assert (pos >= 0).all() and set(ref["file_name"][qi]) == set(old["file_name"]), "queries differ from gap_budget"
    assert (ref["y"][pos] == old["y"].to_numpy()).all(), "labels differ from gap_budget"
    checks, bad = {}, {}
    for name in names:
        for rep in range(a.repeats):
            pairs = [("U", f"{name}|{rep}|U|image")] + [(f"random_{k}", f"{name}|{rep}|ADD_{k}|image") for k in BUDGETS]
            for arm, col in pairs:
                assert col in old.columns, f"gap_budget has no {col} to check against"
                n = int((preds[(name, rep, arm, "image")][pos] != old[col].to_numpy()).sum())
                checks[f"{name} {rep} {arm} vs budget"] = n
                if n:
                    bad[f"{name} {rep} {arm}"] = n
            n = int((preds[(name, rep, "spread_1", "image")] != preds[(name, rep, "random_1", "image")]).sum())
            checks[f"{name} {rep} spread_1 vs random_1"] = n
            if n:
                bad[f"{name} {rep} spread_1"] = n
    if bad:
        out.mkdir(parents=True)
        (out / "reproduction_failure.json").write_text(json.dumps({"differing_images": bad, "all": checks}, indent=2))
        print(f"reproduction checks failed: {bad}. Stopping.")
        sys.exit(7)

    # --- scoring ---------------------------------------------------------------------------------
    brng = np.random.default_rng(a.seed + 5)
    cams_any = np.array([c for c in d.train_cams if n_elig[c] >= 1])

    def table(kind: str, n_boot: int) -> dict:
        m = qi & np.isin(d.cam, cams_any)
        y, cam = d.y[m], d.cam[m]
        keys = [(n, arm) for n in names for arm in arms]
        P = {(n, rep, arm): preds[(n, rep, arm, kind)][m] for n in names for rep in range(a.repeats) for arm in arms}
        assert all((v >= 0).all() for v in P.values()), "a scored image has no prediction"
        cams = np.unique(cam)
        by_cam = {c: np.flatnonzero(cam == c) for c in cams}

        def f1s(ix):
            return np.array([np.mean([macro_f1(y[ix], P[(n, rep, arm)][ix], C) for rep in range(a.repeats)])
                             for n, arm in keys])

        point = f1s(np.arange(len(y)))
        boot = np.empty((n_boot, len(keys)))
        for i in range(n_boot):
            boot[i] = f1s(np.concatenate([by_cam[c] for c in brng.choice(cams, len(cams))]))
        col = {k_: j for j, k_ in enumerate(keys)}

        def diff(n, x, y_):
            v = point[col[(n, x)]] - point[col[(n, y_)]]
            bv = boot[:, col[(n, x)]] - boot[:, col[(n, y_)]]
            return {"value": round(float(v), 4), "ci95": [round(float(np.percentile(bv, 2.5)), 4),
                                                          round(float(np.percentile(bv, 97.5)), 4)]}

        res = {"cameras": int(len(cams)), "images": int(len(y)), "bootstrap_resamples": n_boot, "sets": {}}
        for n in names:
            r_ = {"macro_f1": {arm: round(float(point[col[(n, arm)]]), 4) for arm in arms}}
            for k in BUDGETS:
                for sel in SELECTORS:
                    r_[f"gain_{sel}_{k}"] = diff(n, f"{sel}_{k}", "U")
                for sel in ("spread", "uncertain"):
                    r_[f"{sel}_minus_random_{k}"] = diff(n, f"{sel}_{k}", f"random_{k}")
            res["sets"][n] = r_
        return res

    results = {"pooled": table("burst", a.bootstrap), "per image": table("image", a.bootstrap // 5)}
    g = results["pooled"]["sets"][a.primary][f"spread_minus_random_{a.primary_k}"]
    lo, hi = g["ci95"]
    verdict = (f"supported: spread-out choice beats random at up to {a.primary_k} labels, estimate >= one point, interval above zero"
               if g["value"] >= MARGIN and lo > 0 else
               f"not supported: a one-point advantage of spread-out over random choice is not supported" if hi < MARGIN
               else "inconclusive")
    matters = {k: round(float(np.mean([n_elig[c] > k for c in cams_any])), 3) for k in BUDGETS}

    # --- outputs ---------------------------------------------------------------------------------
    out.mkdir(parents=True)
    pd.DataFrame({"file_name": ref["file_name"][qi], "camera": d.cam[qi], "y": d.y[qi],
                  **{f"{n}|{rep}|{arm}|{k}": preds[(n, rep, arm, k)][qi]
                     for n in names for rep in range(a.repeats) for arm in arms for k in ("image", "burst")}}
                 ).to_csv(out / "query_images.csv.gz", index=False)
    pd.concat(plan_rows).to_csv(out / "plans.csv.gz", index=False)
    (out / "fits.json").write_text(json.dumps(fits, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    res = {"primary": {"set": a.primary, "contrast": f"spread_{a.primary_k} - random_{a.primary_k}, pooled",
                       "margin": MARGIN, "value": g["value"], "ci95": g["ci95"], "verdict": verdict},
           "results": results, "share_of_cameras_with_more_than_k_candidates": matters, "checks": checks,
           "optimisation": {"fits": len(fits), "reached_a_cap": sum(x["hit_cap"] for x in fits),
                            "max_grad": max(x["grad_max"] for x in fits), "all_fits": "fits.json"},
           "notes": ["Training cameras only; chronological queries; labels end >= far-hours before the first query.",
                     "Selectors use no labels of the new camera: spread uses features, uncertain uses U's predictions.",
                     "Cameras with at most k eligible bursts get identical labels under every selector."],
           "seconds": round(time.time() - t0, 1), "args": vars(a)}
    (out / "results.json").write_text(json.dumps(res, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"primary ({a.primary}, spread - random at k={a.primary_k}): {g['value']:+.3f} ({lo:+.3f} to {hi:+.3f}) -> {verdict}")
    print("share of cameras with more than k candidates:", matters)
    for key, t in results.items():
        print(f"\n{key}: {t['cameras']} cameras")
        for n, s in t["sets"].items():
            print(f"  {n}: U {s['macro_f1']['U']:.3f}")
            for k in BUDGETS:
                print(f"    k={k:2d} " + "  ".join(f"{sel} {s[f'gain_{sel}_{k}']['value']:+.3f}" for sel in SELECTORS)
                      + f"  spread-random {s[f'spread_minus_random_{k}']['value']:+.3f} {s[f'spread_minus_random_{k}']['ci95']}"
                      + f"  uncertain-random {s[f'uncertain_minus_random_{k}']['value']:+.3f} {s[f'uncertain_minus_random_{k}']['ci95']}")
    print(f"\noptimisation: {res['optimisation']['fits']} fits, {res['optimisation']['reached_a_cap']} reached a cap")


if __name__ == "__main__":
    main()
