"""What does a camera's own support add, once species counts are held fixed?

    python gap_decompose.py --cache vitl=features/dinov2_vitl14_reg_448 --cache vitl_crop=... \
        --primary vitl --pair vitl_crop=vitl --repo . --out results/decompose_v1 --device cuda

A measurement on the training cameras of `official_ood` only, built on gap_measure.py: the
same camera-wise folds and the same query/support splits and seeds, so the queries of the
"random" repeats and of "later" are those of the gap runs, and U and S_all reproduce those
runs' unseen and seen predictions exactly (both checked with --check-queries; any mismatch
stops the run). Validation and test cameras are not used.

For every held-out fold and every feature set, these models are scored on the same query
bursts. All fitted ones train on exactly as many bursts as U.

  U        unseen: the other folds' bursts ("other") only.
  S_all    seen as in gap_measure: other minus |s| random bursts, plus the support s of the
           fold's query cameras. A reference that ties back to the gap runs.
  S_rand   other minus |s'| random bursts, plus s', the matchable part of the support.
  S_match  other minus e, plus s'. For every burst in s', e holds one burst of the same species
           from other, so S_match has exactly U's training loss mass per species (asserted).
  P<lam>   no refit: U's scores plus log(pi_cam / pi_src). pi_src is the class loss mass of U's
           training images, pi_cam the species mix of the query camera's own bursts in s',
           smoothed toward pi_src with the weight of lam bursts. The support's labels, not its
           pixels.

Matching, per fold: only pure bursts (every image has the burst's species) are swapped, on
both sides, so each swapped burst carries a loss mass of exactly 1 for its species (each burst
weighs 1 in the loss). Species k has m_k pure bursts in the fold's support and a_k pure bursts
in other; min(m_k, a_k) of each are drawn at random and paired, and the rest of the support is
not used (counted per fold). Every plan (queries, support, s', donors e, random removals) is
saved by burst id.

Splits. "random": a random half of each camera's bursts are queries, as in the gap runs, so
support and query bursts can come from the same visit. "later": the later half in time is
queried and earlier bursts are support. "later_far": the same queries, with only the support
bursts that end at least --far-hours before the camera's first query burst. That separates
support and queries in time; it does not prove that they come from different visits. The split
itself is fixed for "later" and
"later_far"; their repeats redraw the matched and random bursts only.

Contrasts, on the query images of cameras with at least one burst in s' (paired bootstrap over
cameras, repeats averaged inside every resample, one set of draws shared by every contrast and
feature set):

  cost        S_rand - U        the cost of a new camera, with the matched support
  local       S_match - U       the local-support replacement effect at equal class mass: the
                                camera's own bursts in place of other cameras' bursts of the
                                same species
  residual    S_rand - S_match  the difference between the two removal policies (random vs
                                same-species); the removed external bursts differ both in
                                species and in how they look, so it is not a species-mix effect
  prior       P5 - U            the camera's species mix from labels alone
  reference   S_all - U         the gap as in gap_measure, on the same images

cost = local + residual is arithmetic, not evidence for two separate mechanisms. These are
interventions on the training set, not a causal decomposition of macro-F1. "Local" covers
everything the camera's own bursts carry: background, viewpoint, lighting, image quality and
the animals living there. The 24-hour buffer of "later_far" gives temporal separation; it does
not guarantee different visits. Under "random", the local effect may benefit from temporally
neighbouring events.
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

FITTED = ("U", "S_all", "S_rand", "S_match")
LAMBDAS = (5.0, 1.0, 20.0)                            # prior smoothing; the first is the main one
ARMS = FITTED + tuple(f"P{lam:g}" for lam in LAMBDAS)
CONTRASTS = {"cost": ("S_rand", "U"), "local": ("S_match", "U"), "residual": ("S_rand", "S_match"),
             "prior": ("P5", "U"), "reference": ("S_all", "U")}
SPLIT_CODE = {"later": 0, "random": 5, "later_far": 7}    # 0 and 5 are gap_measure's budget seeds
MARGIN = 0.02                                          # the primary rule's margin, in macro-F1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="append", required=True, help="name=folder; repeat for each feature set")
    ap.add_argument("--primary", required=True, help="the feature set the primary contrast is about")
    ap.add_argument("--pair", action="append", default=[], help="a=b: local(a) - local(b) and cost(a) - cost(b), paired")
    ap.add_argument("--check-queries", action="append", default=[],
                    help="name=query_images.csv.gz of an earlier gap run on that feature set")
    ap.add_argument("--repo", default=".", help="repo root, for data/gps_locations.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-classes", type=int, default=60)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--l2", default="0.01,0.03", help="comma-separated; the FIRST value is the primary one")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--far-hours", type=float, default=24.0)
    ap.add_argument("--primary-split", default="later_far", choices=list(SPLIT_CODE))
    ap.add_argument("--max-iter", type=int, default=2000)
    ap.add_argument("--bootstrap", type=int, default=2000, help="for the primary table; the others get a fifth")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--max-minutes", type=float, default=120)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    caches = dict(c.split("=", 1) for c in a.cache)
    names = list(caches)
    pairs = [tuple(p.split("=", 1)) for p in a.pair]
    assert a.primary in caches and all(x in caches and y in caches for x, y in pairs), "unknown feature-set name"
    out = Path(a.out)
    if out.exists():
        print(f"{out} exists; not overwriting")
        sys.exit(6)
    t0 = time.time()
    torch.set_num_threads(a.threads)
    if a.device == "cuda" and not gpu_is_free():          # the GPU is shared: never pile on
        print("GPU is in use; not starting")
        sys.exit(2)
    d = Data(Path(caches[names[0]]), Path(a.repo), a.n_classes)
    C = d.C
    l2_grid = [float(v) for v in a.l2.split(",")]
    rng = np.random.default_rng(a.seed)

    # --- the same structure as gap_measure.py -----------------------------------------------
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

    def split_bursts(split: str, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
        """gap_measure.py's query and support bursts, unchanged."""
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

    def far_support(q: np.ndarray, sup: np.ndarray) -> tuple[np.ndarray, dict]:
        """The support bursts at least --far-hours from every query burst of their camera.

        Distances are between the bursts' [first, last] frame times. A camera with an untimed
        query burst keeps no support, since the distance cannot be shown.
        """
        far = np.zeros_like(sup)
        left = {"support_bursts_too_close": 0, "support_bursts_without_timestamp": 0,
                "cameras_dropped_for_untimed_queries": 0}
        h = np.timedelta64(int(a.far_hours * 3600), "s")
        for c in d.train_cams:
            s_idx, q_idx = np.flatnonzero(sup & (d.b_cam == c)), np.flatnonzero(q & (d.b_cam == c))
            if len(s_idx) == 0:
                continue
            if not b_timed[q_idx].all():
                left["cameras_dropped_for_untimed_queries"] += 1
                continue
            left["support_bursts_without_timestamp"] += int((~b_timed[s_idx]).sum())
            s_idx = s_idx[b_timed[s_idx]]
            gap = np.maximum(b_time[q_idx][None, :] - b_end[s_idx][:, None],
                             b_time[s_idx][:, None] - b_end[q_idx][None, :])
            keep = (gap >= h).all(1)
            left["support_bursts_too_close"] += int((~keep).sum())
            far[s_idx[keep]] = True
        return far, left

    runs = [(s, r) for s in ("later", "random", "later_far") for r in range(a.repeats)]
    store = {}
    for split, rep in runs:
        if split == "later_far":
            q = store[("later", rep)]["query"]
            sup, left = far_support(q, store[("later", rep)]["support"])
        else:
            q, sup, left = split_bursts(split, a.seed + 10 * rep + 1)
        store[(split, rep)] = {"query": q, "support": sup, "left_out": left, "folds": {}}

    # --- training sets: they depend on labels only, so they are the same for every feature set
    b_pure = pd.Series(d.y == d.b_y[d.burst]).groupby(d.burst).all().to_numpy()
    b_id = d.img.groupby(d.burst)["burst_id"].first().to_numpy()

    def mass(train_b: np.ndarray) -> np.ndarray:
        """Loss mass per class of a training set: image labels weighted as in the fit."""
        m = train_b[d.burst]
        return np.bincount(d.y[m], weights=d.w[m].astype(np.float64), minlength=C)

    plans, plan_rows = {}, []
    for f in range(a.folds):
        other, held = tr_set & (b_foldid != f), tr_set & (b_foldid == f)
        oth_idx = np.flatnonzero(other)
        oth_pure = oth_idx[b_pure[oth_idx]]
        mass_u = mass(other)
        for split, rep in runs:
            st = store[(split, rep)]
            q, support = st["query"] & held, st["support"] & held
            code = SPLIT_CODE[split]
            seed_all = a.seed + 1000 * f + 10 * rep + code             # = gap_measure's for random/later
            seed_match = a.seed + 7000 + 1000 * f + 10 * rep + code
            drop_all = np.random.default_rng(seed_all).choice(oth_idx, size=int(support.sum()), replace=False)
            r = np.random.default_rng(seed_match)
            sup_idx = np.flatnonzero(support & b_pure)
            s_prime, e = np.zeros_like(support), np.zeros_like(support)
            unmatched_species, exhausted_species = 0, 0
            for k in np.unique(d.b_y[sup_idx]):
                sk, ok = sup_idx[d.b_y[sup_idx] == k], oth_pure[d.b_y[oth_pure] == k]
                n = min(len(sk), len(ok))
                unmatched_species += int(n < len(sk))
                exhausted_species += int(0 < len(ok) <= len(sk))
                if n:
                    s_prime[r.choice(sk, n, replace=False)] = True
                    e[r.choice(ok, n, replace=False)] = True
            drop_rand = r.choice(oth_idx, size=int(s_prime.sum()), replace=False)
            sets = {"U": other}
            x = other.copy()
            x[drop_all] = False
            sets["S_all"] = x | support
            x = other.copy()
            x[drop_rand] = False
            sets["S_rand"] = x | s_prime
            sets["S_match"] = (other & ~e) | s_prime
            for arm, m in sets.items():
                assert m.sum() == other.sum() and not (m & q).any(), arm
            assert np.allclose(mass(sets["S_match"]), mass_u, rtol=0, atol=1e-6), "S_match changes the class loss mass"
            cams = np.unique(d.b_cam[s_prime])
            plans[(f, split, rep)] = {"sets": sets, "q": q, "s_prime": s_prime, "cams": cams,
                                      "pi_src": mass_u / mass_u.sum()}
            for role, m in (("query", q), ("support", support), ("matched_support", s_prime), ("donor", e)):
                plan_rows.append(pd.DataFrame({"split": split, "repeat": rep, "fold": f, "role": role,
                                               "burst_id": b_id[m]}))
            for role, idx in (("removed_for_S_all", drop_all), ("removed_for_S_rand", drop_rand)):
                plan_rows.append(pd.DataFrame({"split": split, "repeat": rep, "fold": f, "role": role,
                                               "burst_id": b_id[idx]}))
            st["folds"][f] = {"seed_S_all": seed_all, "seed_matching": seed_match,
                              "support_bursts": int(support.sum()),
                              "support_bursts_not_pure": int((support & ~b_pure).sum()),
                              "matched_bursts": int(s_prime.sum()),
                              "support_species": int(len(np.unique(d.b_y[support]))),
                              "species_not_fully_matched": unmatched_species,
                              "species_whose_external_pure_bursts_were_all_swapped": exhausted_species,
                              "query_cameras": int(len(np.unique(d.b_cam[q]))),
                              "query_cameras_with_matched_support": int(len(cams))}

    def scored_mask(split: str, rep: int) -> np.ndarray:
        """Query images of the cameras that have matched support in their fold."""
        m = np.zeros(len(d.y), dtype=bool)
        for f in range(a.folds):
            pl = plans[(f, split, rep)]
            m |= pl["q"][d.burst] & np.isin(d.cam, pl["cams"])
        return m & img_in_train

    # Queries must be those of the earlier gap runs (labels and seeds only, so checked now).
    checks = {}
    old_runs = dict(c.split("=", 1) for c in a.check_queries)
    fname = d.img["file_name"].to_numpy()
    for name, path in old_runs.items():
        old = pd.read_csv(path)
        for (split, rep), g in old.groupby(["split", "repeat"]):
            ours = fname[store[(split, rep)]["query"][d.burst] & img_in_train]
            assert set(ours) == set(g["file_name"]), f"queries differ from {path} for {split}/{rep}"
        checks[name] = {"queries_identical": True, "file": path}

    # --- fits ---------------------------------------------------------------------------------
    preds = {}                    # (name, split, rep, arm, kind, l2) -> int16 per image, -1 if not a query
    fits = []
    done, total = 0, a.folds * len(names)
    ref = {"file_name": fname, "burst": d.burst.copy(), "y": d.y.copy()}

    def fit(X, train_b: np.ndarray, dd: Data, l2: float, tag: dict):
        if a.device == "cuda" and not gpu_is_free():
            print("another job is on the GPU; stopping. Nothing is saved, re-run later.")
            sys.exit(4)
        tr_img = train_b[dd.burst]
        mu, sd = X[tr_img].mean(0), X[tr_img].std(0) + 1e-6
        W, b, diag = fit_linear((X[tr_img] - mu) / sd, dd.y[tr_img], dd.w[tr_img], l2, C, a.max_iter, a.device)
        assert np.isfinite(W).all() and np.isfinite(b).all(), f"non-finite weights: {tag}"
        fits.append(dict(diag, **tag, l2=l2))
        return mu, sd, W, b

    for ci, name in enumerate(names):
        dd = d if ci == 0 else Data(Path(caches[name]), Path(a.repo), a.n_classes)
        assert (dd.img["file_name"].to_numpy() == ref["file_name"]).all() and (dd.burst == ref["burst"]).all() \
            and (dd.y == ref["y"]).all(), f"{name}: images, bursts or labels differ from {names[0]}"
        X = dd.X
        for key in [(name, s, r, arm, k, l2) for s, r in runs for arm in ARMS for k in ("image", "burst") for l2 in l2_grid]:
            preds[key] = np.full(len(d.y), -1, dtype=np.int16)
        for f in range(a.folds):
            u = {l2: fit(X, plans[(f, "later", 0)]["sets"]["U"], dd, l2, {"set": name, "arm": "U", "fold": f})
                 for l2 in l2_grid}
            for split, rep in runs:
                pl = plans[(f, split, rep)]
                q = pl["q"]
                qi = q[dd.burst]
                q_idx = np.flatnonzero(q)
                local = np.searchsorted(q_idx, dd.burst[qi])
                cam_u, inv = np.unique(dd.cam[qi], return_inverse=True)
                counts = np.stack([np.bincount(d.b_y[pl["s_prime"] & (d.b_cam == c)], minlength=C) for c in cam_u])
                for l2 in l2_grid:
                    for arm in FITTED:
                        mu, sd, W, b = u[l2] if arm == "U" else fit(
                            X, pl["sets"][arm], dd, l2, {"set": name, "arm": arm, "fold": f, "split": split, "repeat": rep})
                        s = ((X[qi] - mu) / sd) @ W + b
                        outs = {arm: s}
                        if arm == "U":
                            pi = pl["pi_src"]
                            for lam in LAMBDAS:
                                cam_mix = (counts + lam * pi) / (counts.sum(1, keepdims=True) + lam)
                                adj = np.where(pi > 0, np.log(np.maximum(cam_mix, 1e-300)) - np.log(np.maximum(pi, 1e-300)), 0.0)
                                outs[f"P{lam:g}"] = s + adj[inv]
                        for arm_name, sc in outs.items():
                            assert np.isfinite(sc).all(), f"non-finite scores: {name} {split}/{rep} {arm_name} fold {f}"
                            preds[(name, split, rep, arm_name, "image", l2)][qi] = sc.argmax(1)
                            preds[(name, split, rep, arm_name, "burst", l2)][qi] = pool(sc, local, len(q_idx)).argmax(1)[local]
            done += 1
            last = [x for x in fits if x["set"] == name and x["fold"] == f]
            print(f"  {name} fold {f + 1}/{a.folds} done  {time.time() - t0:5.0f}s  {len(last)} fits, "
                  f"{sum(x['hit_cap'] for x in last)} reached a cap, max |grad| {max(x['grad_max'] for x in last):.1e}",
                  flush=True)
            projected = (time.time() - t0) / done * total * 1.15          # plus about 15% for the bootstrap
            if a.max_minutes and projected > a.max_minutes * 60:
                print(f"projected {projected / 60:.0f} min exceeds --max-minutes {a.max_minutes:.0f}; stopping.")
                sys.exit(3)
        if name in old_runs:      # U and S_all must reproduce the earlier run's unseen and seen predictions
            old = pd.read_csv(old_runs[name])
            pos = pd.Index(fname).get_indexer(old["file_name"])
            differ = {}
            for (split, rep), g in old.groupby(["split", "repeat"]):
                p = pos[g.index]
                for arm, col in (("U", "pred_unseen_image"), ("S_all", "pred_seen_image")):
                    for kind, oc in (("image", col), ("burst", col.replace("_image", "_burst"))):
                        ours, theirs, yy = preds[(name, split, rep, arm, kind, l2_grid[0])][p], g[oc].to_numpy(), g["y"].to_numpy()
                        n = int((ours != theirs).sum())
                        if n:
                            per_class = lambda pr: {int(c): 2 * ((yy == c) & (pr == c)).sum() / max((yy == c).sum() + (pr == c).sum(), 1)
                                                    for c in np.unique(yy)}
                            po, pt = per_class(ours), per_class(theirs)
                            differ[f"{split}/{rep} {arm} {kind}"] = {
                                "images_differing": n, "of": int(len(p)),
                                "macro_f1_ours": macro_f1(yy, ours, C), "macro_f1_earlier": macro_f1(yy, theirs, C),
                                "classes_with_f1_change": {c: [round(po[c], 4), round(pt[c], 4)] for c in po if po[c] != pt[c]}}
            checks[name]["reproduces_earlier_run_exactly"] = not differ
            if differ:
                out.mkdir(parents=True)
                (out / "reproduction_failure.json").write_text(json.dumps({"set": name, "differences": differ}, indent=2,
                                                                          default=lambda o: o.item() if hasattr(o, "item") else str(o)))
                print(f"{name}: U/S_all do not reproduce {old_runs[name]} exactly; see reproduction_failure.json. Stopping.")
                sys.exit(7)
        if ci:
            del dd, X

    # --- scoring --------------------------------------------------------------------------------
    brng = np.random.default_rng(a.seed + 5)

    def table(split: str, l2: float, kind: str, n_boot: int) -> dict:
        reps = [r for s_, r in runs if s_ == split]
        items = []
        for rep in reps:
            m = scored_mask(split, rep)
            it = {"y": d.y[m], "cam": d.cam[m],
                  "p": {(n, arm): preds[(n, split, rep, arm, kind, l2)][m] for n in names for arm in ARMS}}
            assert all((v >= 0).all() for v in it["p"].values()), "a scored image has no prediction"
            items.append(it)
        cams = np.unique(np.concatenate([it["cam"] for it in items]))
        by_cam = [{c: np.flatnonzero(it["cam"] == c) for c in cams} for it in items]
        keys = [(n, arm) for n in names for arm in ARMS]

        def f1s(idx_per_rep) -> np.ndarray:
            return np.array([np.mean([macro_f1(it["y"][ix], it["p"][k][ix], C) for it, ix in zip(items, idx_per_rep)])
                             for k in keys])

        point = f1s([np.arange(len(it["y"])) for it in items])
        boot = np.empty((n_boot, len(keys)))
        for i in range(n_boot):
            draw = brng.choice(cams, len(cams))
            boot[i] = f1s([np.concatenate([bc[c] for c in draw]) for bc in by_cam])
        col = {k: j for j, k in enumerate(keys)}

        def ci(v: np.ndarray) -> list:
            return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]

        res = {"repeats": len(reps), "cameras": int(len(cams)),
               "images": float(np.mean([len(it["y"]) for it in items])),
               "species": float(np.mean([len(np.unique(it["y"])) for it in items])),
               "per_repeat": [{"repeat": rep, "cameras": int(len(np.unique(it["cam"]))), "images": int(len(it["y"])),
                               "species": int(len(np.unique(it["y"])))} for rep, it in zip(reps, items)],
               "bootstrap_resamples": n_boot, "sets": {}}
        for n in names:
            r_ = {"macro_f1": {arm: round(float(point[col[(n, arm)]]), 4) for arm in ARMS}}
            for cname, (x, y) in CONTRASTS.items():
                r_[cname] = {"value": round(float(point[col[(n, x)]] - point[col[(n, y)]]), 4),
                             "ci95": ci(boot[:, col[(n, x)]] - boot[:, col[(n, y)]])}
            cost = boot[:, col[(n, "S_rand")]] - boot[:, col[(n, "U")]]
            loc = boot[:, col[(n, "S_match")]] - boot[:, col[(n, "U")]]
            pc = point[col[(n, "S_rand")]] - point[col[(n, "U")]]
            r_["local_share_of_cost_exploratory"] = {      # unstable near zero cost; not an attributable fraction
                "value": round(float((point[col[(n, "S_match")]] - point[col[(n, "U")]]) / pc), 3) if pc else None,
                "ci95_where_cost_positive": ci(loc[cost > 0] / cost[cost > 0]) if (cost > 0).any() else None}
            res["sets"][n] = r_
        res["pairs"] = {}
        for x, y in pairs:
            dx = (boot[:, col[(x, "S_match")]] - boot[:, col[(x, "U")]]) - (boot[:, col[(y, "S_match")]] - boot[:, col[(y, "U")]])
            dc = (boot[:, col[(x, "S_rand")]] - boot[:, col[(x, "U")]]) - (boot[:, col[(y, "S_rand")]] - boot[:, col[(y, "U")]])
            px = res["sets"][x]
            py = res["sets"][y]
            res["pairs"][f"{x} - {y}"] = {
                "local": {"value": round(px["local"]["value"] - py["local"]["value"], 4), "ci95": ci(dx)},
                "cost": {"value": round(px["cost"]["value"] - py["cost"]["value"], 4), "ci95": ci(dc)},
                "unseen": {"value": round(px["macro_f1"]["U"] - py["macro_f1"]["U"], 4),
                           "ci95": ci(boot[:, col[(x, "U")]] - boot[:, col[(y, "U")]])},
                "seen_matched": {"value": round(px["macro_f1"]["S_match"] - py["macro_f1"]["S_match"], 4),
                                 "ci95": ci(boot[:, col[(x, "S_match")]] - boot[:, col[(y, "S_match")]])}}
        return res

    results = {}
    for split in SPLIT_CODE:
        for l2 in l2_grid:
            for kind in ("burst", "image"):
                primary = split == a.primary_split and l2 == l2_grid[0] and kind == "burst"
                results[f"{split} | l2={l2:g} | {'pooled' if kind == 'burst' else 'per image'}"] = \
                    table(split, l2, kind, a.bootstrap if primary else max(200, a.bootstrap // 5))
    pkey = f"{a.primary_split} | l2={l2_grid[0]:g} | pooled"
    app = results[pkey]["sets"][a.primary]["local"]
    lo, hi = app["ci95"]
    verdict = ("supported: local-support benefit at equal class mass, estimate >= two points, interval above zero"
               if app["value"] >= MARGIN and lo > 0
               else "not supported: a two-point local-support benefit is not supported" if hi < MARGIN else "inconclusive")

    # --- outputs ----------------------------------------------------------------------------------
    out.mkdir(parents=True)
    rows = []
    for split, rep in runs:
        qi = store[(split, rep)]["query"][d.burst] & img_in_train
        m = scored_mask(split, rep)
        rows.append(pd.DataFrame({
            "split": split, "repeat": rep, "file_name": fname[qi], "camera": d.cam[qi], "y": d.y[qi],
            "scored": m[qi],
            **{f"{n}|{arm}|{k}": preds[(n, split, rep, arm, k, l2_grid[0])][qi]
               for n in names for arm in ARMS for k in ("image", "burst")}}))
    pd.concat(rows).to_csv(out / "query_images.csv.gz", index=False)
    pd.concat(plan_rows).to_csv(out / "plans.csv.gz", index=False)
    (out / "fits.json").write_text(json.dumps(fits, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    hit = [x for x in fits if x["hit_cap"]]
    res = {
        "primary": {"table": pkey, "set": a.primary, "contrast": "local = S_match - U", "margin": MARGIN,
                    "value": app["value"], "ci95": app["ci95"], "verdict": verdict},
        "results": results,
        "splits": {f"{s}/{r}": {"left_out": store[(s, r)]["left_out"], "per_fold": store[(s, r)]["folds"]}
                   for s, r in runs},
        "checks": checks,
        "data": {"train_cameras": len(d.train_cams), "train_bursts": int(tr_set.sum()),
                 "train_images": int(img_in_train.sum()), "caches": caches},
        "optimisation": {"fits": len(fits), "reached_a_cap": len(hit),
                         "max_iterations": max(x["iterations"] for x in fits),
                         "max_evaluations": max(x["evaluations"] for x in fits),
                         "max_grad": max(x["grad_max"] for x in fits), "capped": hit, "all_fits": "fits.json"},
        "notes": [
            "Training cameras only; cross-validated over cameras. Not validation, not test.",
            "Every fitted model trains on as many bursts as U; S_match also has U's class loss mass (only pure bursts swapped).",
            "Scored: query images of cameras with at least one matched support burst in their fold.",
            "Pooled = argmax of a burst's mean scores, uncalibrated. l2 not tuned for this measurement.",
            "Interventions on the training set, not a causal decomposition of macro-F1; cost = local + residual is arithmetic.",
            "The 24-hour buffer gives temporal separation, not guaranteed different visits; under random queries the "
            "local effect may benefit from temporally neighbouring events.",
        ],
        "seconds": round(time.time() - t0, 1), "args": vars(a),
    }
    (out / "results.json").write_text(json.dumps(res, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"primary ({pkey}, {a.primary}): local {app['value']:+.3f} ({lo:+.3f} to {hi:+.3f}) -> {verdict}")
    for key, r in results.items():
        print(f"\n{key}: {r['cameras']} cameras, {r['images']:.0f} images, {r['species']:.0f} species")
        print("| set | U | S_rand | S_match | P5 | cost | local | residual | prior | reference |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for n, s in r["sets"].items():
            mf = s["macro_f1"]
            cells = " | ".join(f"{s[c]['value']:+.3f} ({s[c]['ci95'][0]:+.3f} to {s[c]['ci95'][1]:+.3f})" for c in CONTRASTS)
            print(f"| {n} | {mf['U']:.3f} | {mf['S_rand']:.3f} | {mf['S_match']:.3f} | {mf['P5']:.3f} | {cells} |")
        for p, v in r["pairs"].items():
            print(f"  {p}: local {v['local']['value']:+.3f} {v['local']['ci95']}, cost {v['cost']['value']:+.3f} "
                  f"{v['cost']['ci95']}, unseen {v['unseen']['value']:+.3f}, seen (matched) {v['seen_matched']['value']:+.3f}")
    o = res["optimisation"]
    print(f"\noptimisation: {o['fits']} fits, {o['reached_a_cap']} reached a cap, max |grad| {o['max_grad']:.2e}")
    print(f"wrote {out}/results.json, fits.json, plans.csv.gz and query_images.csv.gz in {res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
