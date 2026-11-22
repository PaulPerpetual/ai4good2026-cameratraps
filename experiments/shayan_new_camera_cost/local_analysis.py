"""What does the local-support benefit of gap_decompose.py go along with? Exploratory, CPU only.

    python local_analysis.py --decompose results/decompose_v1 --cache features/dinov2_vitl14_reg_448 \
        --sets vitl,vitl_crop --repo . --out results/local_analysis_v1

Reads gap_decompose's saved per-image predictions; fits nothing. Training cameras only. Every
number here is descriptive: the bootstrap resamples cameras with the fitted models held fixed,
so it does not carry training uncertainty, and no outcome rule was set.

  1. Local effect (S_match - U, burst-pooled macro-F1) on day and night queries. "Night" is a
     time-of-day proxy (hour < 6 or >= 18 from the image timestamp), not observed darkness or
     infrared mode.
  2. By species frequency: each species' F1 is computed from all scored query predictions, then
     averaged within bins of training bursts (< 50, >= 50), so false positives from other
     species count.
  3. Across cameras: per-camera accuracy gain S_match - U against "empty-scene distance", the
     cosine distance between the camera's mean unit-norm empty-frame feature (from --cache) and
     the nearest such mean among training cameras of the other folds. Spearman, with the
     covariates that could drive both: empty count, night share of empties, eligible support
     bursts, query images, baseline accuracy. Also with cameras under --min-empties excluded.
     The primary descriptive quantity is the later_far Spearman for the first set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pilot_a import Data, macro_f1


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decompose", required=True, help="gap_decompose output folder")
    ap.add_argument("--cache", required=True, help="a whole-frame cache with empty frames, for the empty-scene features")
    ap.add_argument("--sets", default="vitl,vitl_crop")
    ap.add_argument("--splits", default="later_far,random")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-classes", type=int, default=60)
    ap.add_argument("--min-empties", type=int, default=20)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5, help="as in gap_decompose")
    ap.add_argument("--fold-seed", type=int, default=0, help="gap_decompose's --seed, which built its folds")
    a = ap.parse_args()
    out = Path(a.out)
    assert not out.exists(), f"{out} exists; not overwriting"
    sets, splits = a.sets.split(","), a.splits.split(",")
    C = a.n_classes
    rng = np.random.default_rng(a.seed)

    d = Data(Path(a.cache), Path(a.repo), C)
    qi = pd.read_csv(Path(a.decompose) / "query_images.csv.gz")
    plans = pd.read_csv(Path(a.decompose) / "plans.csv.gz")
    meta = d.img.set_index("file_name")
    qi["hour"] = meta.loc[qi["file_name"], "hour"].to_numpy()
    qi["night"] = (qi["hour"] < 6) | (qi["hour"] >= 18)
    tr_bursts = np.bincount(d.b_y[d.b_fold == "train"], minlength=C)

    # Camera-level descriptors: folds from the saved plans, empty scenes from the cache.
    ix = pd.read_csv(Path(a.cache) / "index.csv.gz")
    emp = ix[ix["is_empty"].astype(bool)]
    n_emp = emp.groupby("location").size()
    night_emp = emp.assign(n=(emp["hour"] < 6) | (emp["hour"] >= 18)).groupby("location")["n"].mean()
    burst_cam = d.img.groupby("burst_id")["location"].first()
    # The folds of all training cameras, rebuilt exactly as gap_measure/gap_decompose build them
    # (single-burst cameras train too), then checked against the saved query plans.
    frng = np.random.default_rng(a.fold_seed)
    order = sorted(d.train_cams, key=lambda c: (d.region.get(c, "none"), frng.random()))
    fold_of = {c: i % a.folds for i, c in enumerate(order)}
    qp = plans[plans["role"] == "query"].drop_duplicates("burst_id")
    for c, f in zip(burst_cam.loc[qp["burst_id"]].to_numpy(), qp["fold"].to_numpy()):
        assert fold_of[c] == f, f"camera {c}: rebuilt fold {fold_of[c]} differs from the plans' {f}"
    cams = [c for c in d.train_cams if c in d.empty_mean]
    M = np.stack([d.empty_mean[c] / np.linalg.norm(d.empty_mean[c]) for c in cams])
    fold_arr = np.array([fold_of[c] for c in cams])
    sim = M @ M.T
    dist = {c: float(1 - sim[i, fold_arr != fold_arr[i]].max()) for i, c in enumerate(cams)}
    res_bank = {"training_cameras": int(len(d.train_cams)), "with_empties": int(len(cams)),
                "with_empties_per_fold": {int(f): int((fold_arr == f).sum()) for f in range(a.folds)}}

    res = {"notes": ["Exploratory and descriptive; training cameras only; nothing fitted.",
                     "Night = hour < 6 or >= 18, a time-of-day proxy, not observed illumination.",
                     "Bootstrap over cameras with the fitted models held fixed: no training uncertainty.",
                     "A correlation with empty-scene distance does not show that background causes the benefit."],
           "by_time_of_day": {}, "by_frequency": {}, "across_cameras": {}}
    rows = []
    for split in splits:
        part = qi[(qi["split"] == split) & qi["scored"]]
        reps = sorted(part["repeat"].unique())
        for s in sets:
            u, m = f"{s}|U|burst", f"{s}|S_match|burst"
            # 1. day / night
            tod = {}
            for name, sel in (("day", ~part["night"]), ("night", part["night"])):
                p = part[sel]
                cams_p = p["camera"].unique()
                val = np.mean([macro_f1(g["y"].to_numpy(), g[m].to_numpy(), C) - macro_f1(g["y"].to_numpy(), g[u].to_numpy(), C)
                               for _, g in p.groupby("repeat")])
                by_rep = []
                for _, g in p.groupby("repeat"):
                    cam_g = g["camera"].to_numpy()
                    by_rep.append((g["y"].to_numpy(), g[m].to_numpy(), g[u].to_numpy(),
                                   {c: np.flatnonzero(cam_g == c) for c in cams_p}))
                boots = []
                for _ in range(a.bootstrap):
                    draw = rng.choice(cams_p, len(cams_p))
                    vals = []
                    for yy, pm, pu, bc in by_rep:
                        idx = np.concatenate([bc[c] for c in draw])
                        vals.append(macro_f1(yy[idx], pm[idx], C) - macro_f1(yy[idx], pu[idx], C))
                    boots.append(np.mean(vals))
                tod[name] = {"local": round(float(val), 4), "ci95_descriptive": [round(float(np.percentile(boots, 2.5)), 4),
                                                                                round(float(np.percentile(boots, 97.5)), 4)],
                             "images_per_repeat": int(len(p) / len(reps)), "cameras": int(len(cams_p))}
            res["by_time_of_day"][f"{split} | {s}"] = tod
            # 2. frequency bins: per-species F1 over all scored predictions, averaged within bins
            diffs = {"under_50": [], "50_or_more": []}
            for c_ in range(C):
                per = []
                for _, g in part.groupby("repeat"):
                    y = g["y"].to_numpy()
                    if not (y == c_).any():
                        continue
                    f = lambda pr: 2 * ((y == c_) & (pr == c_)).sum() / max((y == c_).sum() + (pr == c_).sum(), 1)
                    per.append(f(g[m].to_numpy()) - f(g[u].to_numpy()))
                if per:
                    diffs["under_50" if tr_bursts[c_] < 50 else "50_or_more"].append(float(np.mean(per)))
            res["by_frequency"][f"{split} | {s}"] = {k: {"mean_f1_change": round(float(np.mean(v)), 4) if v else None,
                                                         "species": len(v)} for k, v in diffs.items()}
            # 3. across cameras
            # Per camera and repeat first, then averaged over the repeats the camera appears in.
            g = part.assign(gain=(part[m] == part["y"]).astype(float) - (part[u] == part["y"]).astype(float),
                            base=(part[u] == part["y"]).astype(float))
            cr = g.groupby(["camera", "repeat"]).agg(gain=("gain", "mean"), base=("base", "mean"), queries=("y", "size"))
            sup = plans[(plans["split"] == split) & (plans["role"] == "matched_support")]
            n_sup = sup.assign(camera=burst_cam.loc[sup["burst_id"]].to_numpy()).groupby(["camera", "repeat"]).size()
            cr["support_bursts"] = n_sup.reindex(cr.index).fillna(0).to_numpy()
            per_cam = cr.groupby("camera").mean()
            per_cam["repeats_available"] = cr.groupby("camera").size()
            per_cam["empty_scene_distance"] = [dist.get(c, np.nan) for c in per_cam.index]
            per_cam["empties"] = n_emp.reindex(per_cam.index).fillna(0).to_numpy()
            per_cam["night_share_of_empties"] = night_emp.reindex(per_cam.index).to_numpy()
            per_cam = per_cam.dropna(subset=["empty_scene_distance"])
            entry = {}
            for label, pc in (("all cameras with empties", per_cam),
                              (f"cameras with >= {a.min_empties} empties", per_cam[per_cam["empties"] >= a.min_empties])):
                x, y = pc["empty_scene_distance"].to_numpy(), pc["gain"].to_numpy()
                bs = [spearman(x[i], y[i]) for i in (rng.integers(0, len(x), len(x)) for _ in range(a.bootstrap))]
                entry[label] = {"cameras": int(len(pc)), "spearman_distance_vs_gain": round(spearman(x, y), 3),
                                "ci95_descriptive": [round(float(np.nanpercentile(bs, 2.5)), 3), round(float(np.nanpercentile(bs, 97.5)), 3)],
                                "spearman_distance_vs": {k: round(spearman(x, pc[k].to_numpy()), 3)
                                                         for k in ("empties", "night_share_of_empties", "support_bursts", "queries", "base")
                                                         if pc[k].notna().all()},
                                "spearman_gain_vs": {k: round(spearman(y, pc[k].to_numpy()), 3)
                                                     for k in ("empties", "night_share_of_empties", "support_bursts", "queries", "base")
                                                     if pc[k].notna().all()}}
            res["across_cameras"][f"{split} | {s}"] = entry
            rows.append(per_cam.assign(split=split, set=s).reset_index())
    res["empty_scene_reference_bank"] = res_bank
    res["notes"].append("Correlations with baseline accuracy are mathematically coupled: the gain contains minus the baseline.")
    res["primary_descriptive"] = {"what": f"later_far | {sets[0]}: Spearman of empty-scene distance vs local accuracy gain",
                                  **res["across_cameras"].get(f"later_far | {sets[0]}", {}).get("all cameras with empties", {})}
    out.mkdir(parents=True)
    pd.concat(rows).to_csv(out / "per_camera.csv", index=False)
    (out / "results.json").write_text(json.dumps(res, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(json.dumps({k: res[k] for k in ("primary_descriptive", "by_time_of_day", "by_frequency")}, indent=1))


if __name__ == "__main__":
    main()
