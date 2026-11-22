"""How many labelled bursts from a new camera does it take to close the gap?

    python gap_budget.py --cache vitl_crop=features/dinov2_vitl14_reg_448_crop --cache vitl=... \
        --primary vitl_crop --check-decompose results/decompose_v1/query_images.csv.gz \
        --repo . --out results/budget_v1 --device cuda

A measurement on the training cameras of `official_ood` only, with gap_decompose.py's folds
and its chronological "later_far" protocol: each camera's later half of bursts are queries, and
its eligible support is every burst that ends at least --far-hours before the camera's first
query burst. Validation and test cameras are not used.

Each repeat puts every held camera's eligible bursts in a random order. "Up to k labels per
camera": s_k holds the first k bursts of every held camera, or all it has if it has fewer
(effective counts reported). Every burst in s_k counts as one annotation, mixed bursts too.

  U        the other folds' bursts ("other") only; the same model as gap_decompose's U.
  ADD_k    other + s_k: the labels are added, as in deployment.
  SWAP_k   other - e_k + s'_k: s'_k is the pure part of s_k that has a same-species pure donor
           burst in other, e_k those donors, so SWAP_k has U's class loss mass (asserted). Its
           matched size is saved; it can be smaller than s_k.
  REF      other + every eligible support burst: the all-eligible-support reference. More
           support is not guaranteed to help, so this is not a ceiling.

Contrasts on the same query images for every k (cameras with at least one eligible burst;
paired bootstrap over cameras, repeats averaged inside every resample, one set of draws):

  gain_k       ADD_k - U       what up to k labels per camera add
  local_k      SWAP_k - U      the same with class mass held fixed
  remaining_k  REF - ADD_k
  half_8       gain_8 - 0.5 gain_REF   only meaningful when gain_REF is clearly positive

ADD_k - SWAP_k also differs in training size and donor removal, so it is not a mechanism
split. A second population, the cameras with at least --fixed-min eligible bursts, is fixed
across k; curves over cameras that change with k are not reported.
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

BUDGETS = (1, 2, 4, 8, 16, 32)
MARGIN = 0.02


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="append", required=True, help="name=folder; repeat for each feature set")
    ap.add_argument("--primary", required=True)
    ap.add_argument("--check-decompose", required=True,
                    help="gap_decompose query_images.csv.gz: U must reproduce its later_far U for every feature set")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-classes", type=int, default=60)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--l2", type=float, default=0.01)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--far-hours", type=float, default=24.0)
    ap.add_argument("--fixed-min", type=int, default=8)
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

    # --- gap_measure.py's structure and "later" split, gap_decompose.py's 24-hour buffer ----------
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
    b_pure = pd.Series(d.y == d.b_y[d.burst]).groupby(d.burst).all().to_numpy()
    b_id = d.img.groupby(d.burst)["burst_id"].first().to_numpy()

    q = np.zeros(len(d.b_y), dtype=bool)
    elig = np.zeros(len(d.b_y), dtype=bool)
    left = {"cameras_with_under_2_usable_bursts": 0, "support_bursts_straddling_the_boundary": 0,
            "support_bursts_too_close": 0, "cameras_left_without_support": 0}
    h = np.timedelta64(int(a.far_hours * 3600), "s")
    for c in d.train_cams:
        idx = np.flatnonzero(d.b_cam == c)
        idx = idx[b_timed[idx]]
        idx = idx[np.lexsort((b_end[idx], b_time[idx]))]
        if len(idx) < 2:
            left["cameras_with_under_2_usable_bursts"] += 1
            continue
        cut = len(idx) - len(idx) // 2
        s_idx, q_idx = idx[:cut], idx[cut:]
        clean = b_end[s_idx] < b_time[q_idx].min()
        left["support_bursts_straddling_the_boundary"] += int((~clean).sum())
        s_idx = s_idx[clean]
        if len(s_idx) == 0:
            left["cameras_left_without_support"] += 1
            continue
        q[q_idx] = True
        far = b_end[s_idx] <= b_time[q_idx].min() - h
        left["support_bursts_too_close"] += int((~far).sum())
        elig[s_idx[far]] = True
    n_elig = {c: int((elig & (d.b_cam == c)).sum()) for c in d.train_cams}

    def mass(train_b: np.ndarray) -> np.ndarray:
        m = train_b[d.burst]
        return np.bincount(d.y[m], weights=d.w[m].astype(np.float64), minlength=C)

    # --- plans: labels only, the same for every feature set ---------------------------------------
    plans, plan_rows, stats = {}, [], {}
    for rep in range(a.repeats):
        r_order = np.random.default_rng(a.seed + 500 + rep)
        ranked = {c: r_order.permutation(np.flatnonzero(elig & (d.b_cam == c))) for c in d.train_cams}
        for f in range(a.folds):
            other, held = tr_set & (b_foldid != f), tr_set & (b_foldid == f)
            oth_pure = np.flatnonzero(other & b_pure)
            mass_u = mass(other)
            cams_f = [c for c in d.train_cams if fold_of.get(c) == f]
            sets = {"U": other, "REF": other | (elig & held)}
            for k in BUDGETS:
                s_k = np.zeros_like(q)
                for c in cams_f:
                    s_k[ranked[c][:k]] = True
                sets[f"ADD_{k}"] = other | s_k
                r = np.random.default_rng(a.seed + 9000 + 1000 * f + 10 * rep + k)
                pure_idx = np.flatnonzero(s_k & b_pure)
                s_prime, e = np.zeros_like(q), np.zeros_like(q)
                for sp in np.unique(d.b_y[pure_idx]):
                    sk, ok = pure_idx[d.b_y[pure_idx] == sp], oth_pure[d.b_y[oth_pure] == sp]
                    n = min(len(sk), len(ok))
                    if n:
                        s_prime[r.choice(sk, n, replace=False)] = True
                        e[r.choice(ok, n, replace=False)] = True
                sets[f"SWAP_{k}"] = (other & ~e) | s_prime
                assert np.allclose(mass(sets[f"SWAP_{k}"]), mass_u, rtol=0, atol=1e-6), "SWAP changes class mass"
                stats[f"{rep}/{f}/{k}"] = {"labelled_bursts": int(s_k.sum()), "pure": int(len(pure_idx)),
                                           "swapped": int(s_prime.sum())}
                for role, m in ((f"labelled_{k}", s_k), (f"swapped_{k}", s_prime), (f"donor_{k}", e)):
                    plan_rows.append(pd.DataFrame({"repeat": rep, "fold": f, "role": role, "burst_id": b_id[m]}))
            for arm, m in sets.items():
                assert not (m & q & held).any(), arm      # the fold's own queries never train
            plans[(rep, f)] = sets
    for f in range(a.folds):
        plan_rows.append(pd.DataFrame({"repeat": -1, "fold": f, "role": "query", "burst_id": b_id[q & (b_foldid == f)]}))
    arms = ["U", "REF"] + [f"{x}_{k}" for k in BUDGETS for x in ("ADD", "SWAP")]

    # --- fits ------------------------------------------------------------------------------------
    preds, fits = {}, []
    qi = q[d.burst] & img_in_train
    ref = {"file_name": d.img["file_name"].to_numpy(), "burst": d.burst.copy(), "y": d.y.copy()}
    for ci, name in enumerate(names):
        dd = d if ci == 0 else Data(Path(caches[name]), Path(a.repo), a.n_classes)
        assert (dd.img["file_name"].to_numpy() == ref["file_name"]).all() and (dd.burst == ref["burst"]).all() \
            and (dd.y == ref["y"]).all(), f"{name}: images, bursts or labels differ"
        X = dd.X
        for f in range(a.folds):
            qf = q & (b_foldid == f)
            qfi = qf[dd.burst]
            q_idx = np.flatnonzero(qf)
            local = np.searchsorted(q_idx, dd.burst[qfi])
            fixed = {}                                     # U and REF do not depend on the repeat
            for rep in range(a.repeats):
                for arm in arms:
                    if arm in fixed:
                        s = fixed[arm]
                    else:
                        if a.device == "cuda" and not gpu_is_free():
                            print("another job is on the GPU; stopping. Nothing is saved, re-run later.")
                            sys.exit(4)
                        tr_img = plans[(rep, f)][arm][dd.burst]
                        mu, sd = X[tr_img].mean(0), X[tr_img].std(0) + 1e-6
                        W, b, diag = fit_linear((X[tr_img] - mu) / sd, dd.y[tr_img], dd.w[tr_img], a.l2, C,
                                                a.max_iter, a.device)
                        assert np.isfinite(W).all() and np.isfinite(b).all(), f"non-finite weights {name} {arm}"
                        fits.append(dict(diag, set=name, arm=arm, fold=f, repeat=rep))
                        s = ((X[qfi] - mu) / sd) @ W + b
                        assert np.isfinite(s).all(), f"non-finite scores {name} {arm}"
                        if arm in ("U", "REF"):
                            fixed[arm] = s
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

    # U must be gap_decompose's U, image for image (same folds, data, l2 and fit).
    checks = {}
    old = pd.read_csv(a.check_decompose)
    old = old[(old["split"] == "later_far") & (old["repeat"] == 0)]
    pos = pd.Index(ref["file_name"]).get_indexer(old["file_name"])
    assert (pos >= 0).all(), "a gap_decompose query image is not in this cache"
    assert (ref["y"][pos] == old["y"].to_numpy()).all(), "labels differ from gap_decompose"
    assert set(ref["file_name"][qi]) == set(old["file_name"]), "queries differ from gap_decompose's later_far"
    for name in names:
        col = f"{name}|U|image"
        assert col in old.columns, f"gap_decompose has no {col} to check against"
        n = int((preds[(name, 0, "U", "image")][pos] != old[col].to_numpy()).sum())
        checks[name] = {"U_images_differing_from_gap_decompose": n, "images_checked": int(len(pos))}
        if n:
            out.mkdir(parents=True)
            (out / "reproduction_failure.json").write_text(json.dumps(checks, indent=2))
            print(f"{name}: U does not reproduce gap_decompose ({n} images differ). Stopping.")
            sys.exit(7)

    # --- scoring ---------------------------------------------------------------------------------
    brng = np.random.default_rng(a.seed + 5)
    cams_any = np.array([c for c in d.train_cams if n_elig[c] >= 1])
    cams_fixed = np.array([c for c in d.train_cams if n_elig[c] >= a.fixed_min])

    def table(kind: str, cams_ok: np.ndarray, n_boot: int) -> dict:
        m = qi & np.isin(d.cam, cams_ok)
        y, cam = d.y[m], d.cam[m]
        keys = [(n, arm) for n in names for arm in arms]
        P = {(n, rep, arm): preds[(n, rep, arm, kind)][m] for n in names for rep in range(a.repeats) for arm in arms}
        assert all((v >= 0).all() for v in P.values()), "a scored image has no prediction"
        cams = np.unique(cam)
        by_cam = {c: np.flatnonzero(cam == c) for c in cams}

        def f1s(ix) -> np.ndarray:
            return np.array([np.mean([macro_f1(y[ix], P[(n, rep, arm)][ix], C) for rep in range(a.repeats)])
                             for n, arm in keys])

        point = f1s(np.arange(len(y)))
        boot = np.empty((n_boot, len(keys)))
        for i in range(n_boot):
            boot[i] = f1s(np.concatenate([by_cam[c] for c in brng.choice(cams, len(cams))]))
        col = {k: j for j, k in enumerate(keys)}

        def contrast(n, plus, minus, w=None):
            w = w or [1.0] * len(plus)
            v = sum(wi * point[col[(n, x)]] for wi, x in zip(w, plus)) - sum(point[col[(n, x)]] for x in minus)
            bv = sum(wi * boot[:, col[(n, x)]] for wi, x in zip(w, plus)) - sum(boot[:, col[(n, x)]] for x in minus)
            return {"value": round(float(v), 4),
                    "ci95": [round(float(np.percentile(bv, 2.5)), 4), round(float(np.percentile(bv, 97.5)), 4)]}

        res = {"cameras": int(len(cams)), "images": int(len(y)), "species": int(len(np.unique(y))),
               "bootstrap_resamples": n_boot, "sets": {}}
        for n in names:
            r_ = {"macro_f1": {arm: round(float(point[col[(n, arm)]]), 4) for arm in arms},
                  "gain_REF": contrast(n, ["REF"], ["U"])}
            for k in BUDGETS:
                r_[f"gain_{k}"] = contrast(n, [f"ADD_{k}"], ["U"])
                r_[f"local_{k}"] = contrast(n, [f"SWAP_{k}"], ["U"])
                r_[f"remaining_{k}"] = contrast(n, ["REF"], [f"ADD_{k}"])
            # half_k = ADD_k - U - 0.5 (REF - U) = ADD_k - 0.5 REF - 0.5 U
            r_[f"half_{a.primary_k}"] = contrast(n, [f"ADD_{a.primary_k}", "REF", "U"], [], w=[1.0, -0.5, -0.5])
            res["sets"][n] = r_
        return res

    results = {"up to k labels, cameras with >= 1 eligible burst | pooled": table("burst", cams_any, a.bootstrap),
               "up to k labels, cameras with >= 1 eligible burst | per image": table("image", cams_any, a.bootstrap // 5),
               f"cameras with >= {a.fixed_min} eligible bursts | pooled": table("burst", cams_fixed, a.bootstrap // 5)}
    pkey = "up to k labels, cameras with >= 1 eligible burst | pooled"
    g = results[pkey]["sets"][a.primary][f"gain_{a.primary_k}"]
    lo, hi = g["ci95"]
    verdict = (f"supported: up to {a.primary_k} labels per camera help, estimate >= two points, interval above zero"
               if g["value"] >= MARGIN and lo > 0 else
               f"not supported: a two-point gain from up to {a.primary_k} labels is not supported" if hi < MARGIN
               else "inconclusive")
    reach = {k: round(float(np.mean([n_elig[c] >= k for c in cams_any])), 3) for k in BUDGETS}

    # --- outputs ---------------------------------------------------------------------------------
    out.mkdir(parents=True)
    pd.DataFrame({"file_name": ref["file_name"][qi], "camera": d.cam[qi], "y": d.y[qi],
                  **{f"{n}|{rep}|{arm}|{k}": preds[(n, rep, arm, k)][qi]
                     for n in names for rep in range(a.repeats) for arm in arms for k in ("image", "burst")}}
                 ).to_csv(out / "query_images.csv.gz", index=False)
    pd.concat(plan_rows).to_csv(out / "plans.csv.gz", index=False)
    (out / "fits.json").write_text(json.dumps(fits, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    res = {"primary": {"table": pkey, "set": a.primary, "contrast": f"gain_{a.primary_k} = ADD_{a.primary_k} - U",
                       "margin": MARGIN, "value": g["value"], "ci95": g["ci95"], "verdict": verdict},
           "results": results, "share_of_cameras_reaching_k": reach,
           "eligible_bursts_per_camera": {"median": float(np.median([n_elig[c] for c in cams_any])),
                                          "cameras_with_any": int(len(cams_any)),
                                          f"cameras_with_at_least_{a.fixed_min}": int(len(cams_fixed))},
           "left_out": left, "per_plan": stats, "checks": checks,
           "optimisation": {"fits": len(fits), "reached_a_cap": sum(x["hit_cap"] for x in fits),
                            "max_grad": max(x["grad_max"] for x in fits), "all_fits": "fits.json"},
           "notes": ["Training cameras only; chronological queries; support ends >= far-hours before the first query.",
                     "Every labelled burst counts as one annotation, mixed bursts included; SWAP uses only the pure, matchable part.",
                     "REF is the all-eligible-support reference, not a ceiling. half_k is only meaningful when gain_REF is clearly positive.",
                     "ADD_k - SWAP_k differs in training size and donor removal; it is not a mechanism split."],
           "seconds": round(time.time() - t0, 1), "args": vars(a)}
    (out / "results.json").write_text(json.dumps(res, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"primary ({a.primary}, gain_{a.primary_k}): {g['value']:+.3f} ({lo:+.3f} to {hi:+.3f}) -> {verdict}")
    print("share of cameras reaching k:", reach)
    for key, t in results.items():
        print(f"\n{key}: {t['cameras']} cameras, {t['images']} images")
        for n, s in t["sets"].items():
            mf = s["macro_f1"]
            print(f"  {n}: U {mf['U']:.3f}  REF {mf['REF']:.3f} (gain {s['gain_REF']['value']:+.3f})  "
                  + "  ".join(f"k={k}: ADD {mf[f'ADD_{k}']:.3f} SWAP {mf[f'SWAP_{k}']:.3f}" for k in BUDGETS))
    print(f"\noptimisation: {res['optimisation']['fits']} fits, {res['optimisation']['reached_a_cap']} reached a cap")
    print(f"wrote {out}/results.json, fits.json, plans.csv.gz and query_images.csv.gz in {res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
