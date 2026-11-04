"""Compare the cost of an unseen camera between two feature sets, pair by pair.

    python gap_compare.py results/gap_vitl14_v1 results/gap_vitb14_v2 --names "ViT-L/14" "ViT-B/14"

Reads `query_images.csv.gz` and `results.json` of two runs of gap_measure.py. Runs on
caches with the same index and the same seeds score exactly the same query images with
models trained on exactly the same bursts, so the two runs can be compared per image. This
is checked, not assumed. No GPU and no new predictions: it only re-scores saved ones.

Fixed before the runs it compares (recorded in APPROVALS.md):
  primary    D = gap(A) - gap(B), random queries, pooled, primary l2, all query images.
             "A shrinks the gap" is supported if D <= -0.02 and the upper end of its 95%
             interval is below 0; not supported if the lower end is above -0.02;
             inconclusive otherwise.
  secondary  the same difference for unseen and for seen macro-F1 on their own, the later
             split, per-image scoring, and the restriction to species in both models'
             training images; intervals, no outcome labels.

Intervals: paired bootstrap over cameras, the same resamples for both runs, repeats
averaged inside every resample, as in gap_measure.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pilot_a import macro_f1

KEYS = ["split", "repeat", "file_name", "camera", "burst_id", "y", "supported"]
MARGIN = 0.02


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="run folder A (the feature set being tested)")
    ap.add_argument("b", help="run folder B (the reference)")
    ap.add_argument("--names", nargs=2, default=["A", "B"])
    ap.add_argument("--n-classes", type=int, default=60)
    ap.add_argument("--bootstrap", type=int, default=2000, help="for the primary; secondary rows get a quarter")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", help="where to write the comparison as JSON (default: next to run A)")
    args = ap.parse_args()
    C = args.n_classes
    runs = [Path(args.a), Path(args.b)]
    q = [pd.read_csv(r / "query_images.csv.gz") for r in runs]
    res = [json.loads((r / "results.json").read_text()) for r in runs]
    for qa in q:
        qa.sort_values(["split", "repeat", "file_name"], inplace=True, ignore_index=True)
    assert len(q[0]) == len(q[1]) and q[0][KEYS].equals(q[1][KEYS]), \
        "the two runs do not score the same query images with the same labels and support"
    l2 = res[0]["primary_l2"]
    assert l2 == res[1]["primary_l2"], "different primary l2"

    def f1s(frame: pd.DataFrame, kind: str, idx=None) -> dict:
        y = frame["y"].to_numpy() if idx is None else frame["y"].to_numpy()[idx]
        out = {}
        for m in ("unseen", "seen"):
            p = frame[f"pred_{m}_{kind}"].to_numpy()
            out[m] = macro_f1(y, p if idx is None else p[idx], C)
        return out

    rng = np.random.default_rng(args.seed)
    table, checks = {}, []
    for split in ("random", "later"):
        for scope in ("all query images", "species in both models training images"):
            for kind, kname in (("burst", "pooled"), ("image", "per_image")):
                per_rep = []
                for rep in sorted(q[0].loc[q[0]["split"] == split, "repeat"].unique()):
                    sel = (q[0]["split"] == split) & (q[0]["repeat"] == rep)
                    if scope != "all query images":
                        sel &= q[0]["supported"]
                    per_rep.append([qa[sel.to_numpy()].reset_index(drop=True) for qa in q])
                point = [{m: float(np.mean([f1s(rep_[i], kind)[m] for rep_ in per_rep])) for m in ("unseen", "seen")}
                         for i in range(2)]
                # Each run's own numbers must match what gap_measure.py reported.
                key = f"{split} | l2={l2:g} | {scope}"
                for i in range(2):
                    rep_res = res[i]["results"][key][kname]
                    for m in ("unseen", "seen"):
                        checks.append(abs(round(point[i][m], 4) - rep_res[f"macro_f1_{m}"]) <= 1e-4)
                cams = np.unique(np.concatenate([rep_[0]["camera"].to_numpy() for rep_ in per_rep]))
                by_cam = [{c: np.flatnonzero(rep_[0]["camera"].to_numpy() == c) for c in cams} for rep_ in per_rep]
                primary = split == "random" and scope == "all query images" and kind == "burst"
                n_boot = args.bootstrap if primary else max(200, args.bootstrap // 4)
                draws = {"gap": [], "unseen": [], "seen": []}
                for _ in range(n_boot):
                    draw = rng.choice(cams, len(cams))
                    acc = {i: {"unseen": [], "seen": []} for i in range(2)}
                    for rep_, bc in zip(per_rep, by_cam):
                        idx = np.concatenate([bc[c] for c in draw])
                        for i in range(2):
                            v = f1s(rep_[i], kind, idx)
                            acc[i]["unseen"].append(v["unseen"])
                            acc[i]["seen"].append(v["seen"])
                    mean = {i: {m: np.mean(acc[i][m]) for m in ("unseen", "seen")} for i in range(2)}
                    draws["unseen"].append(mean[0]["unseen"] - mean[1]["unseen"])
                    draws["seen"].append(mean[0]["seen"] - mean[1]["seen"])
                    draws["gap"].append((mean[0]["seen"] - mean[0]["unseen"]) - (mean[1]["seen"] - mean[1]["unseen"]))
                ci = {k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
                      for k, v in draws.items()}
                row = {
                    "repeats": len(per_rep), "query_images": int(np.mean([len(r_[0]) for r_ in per_rep])),
                    "cameras": int(len(cams)), "bootstrap_resamples": n_boot,
                    **{f"{name} unseen": round(point[i]["unseen"], 4) for i, name in enumerate(args.names)},
                    **{f"{name} seen": round(point[i]["seen"], 4) for i, name in enumerate(args.names)},
                    **{f"{name} gap": round(point[i]["seen"] - point[i]["unseen"], 4) for i, name in enumerate(args.names)},
                    "difference in gap": round((point[0]["seen"] - point[0]["unseen"]) - (point[1]["seen"] - point[1]["unseen"]), 4),
                    "difference in gap ci95": ci["gap"],
                    "difference unseen": round(point[0]["unseen"] - point[1]["unseen"], 4), "difference unseen ci95": ci["unseen"],
                    "difference seen": round(point[0]["seen"] - point[1]["seen"], 4), "difference seen ci95": ci["seen"],
                }
                if primary:
                    dg, (lo, hi) = row["difference in gap"], ci["gap"]
                    row["margin"] = MARGIN
                    row["outcome"] = (f"supported: {args.names[0]} shrinks the gap" if dg <= -MARGIN and hi < 0
                                      else "not supported" if lo > -MARGIN else "inconclusive")
                table[f"{split} | {kname} | {scope}"] = row
    assert all(checks), "re-scored numbers differ from the runs' own results.json"

    out = Path(args.out) if args.out else runs[0] / f"compare_with_{runs[1].name}.json"
    out.write_text(json.dumps({"a": str(runs[0]), "b": str(runs[1]), "names": args.names, "primary_l2": l2,
                               "rows": table}, indent=2))
    a_, b_ = args.names
    print(f"A = {a_}, B = {b_}; every number re-scored from the saved predictions matches both runs' results.json")
    print(f"| queries | scoring | scored on | {a_} unseen | {a_} seen | {a_} gap | {b_} unseen | {b_} seen | {b_} gap | "
          f"gap A - B (95%) | unseen A - B (95%) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for key, r in table.items():
        split, kname, scope = key.split(" | ")
        print(f"| {split} | {kname} | {scope} | {r[f'{a_} unseen']:.3f} | {r[f'{a_} seen']:.3f} | {r[f'{a_} gap']:+.3f} | "
              f"{r[f'{b_} unseen']:.3f} | {r[f'{b_} seen']:.3f} | {r[f'{b_} gap']:+.3f} | "
              f"{r['difference in gap']:+.3f} ({r['difference in gap ci95'][0]:+.3f} to {r['difference in gap ci95'][1]:+.3f}) | "
              f"{r['difference unseen']:+.3f} ({r['difference unseen ci95'][0]:+.3f} to {r['difference unseen ci95'][1]:+.3f}) |")
    prim = table["random | pooled | all query images"]
    print(f"primary: difference in gap {prim['difference in gap']:+.3f} {prim['difference in gap ci95']}, "
          f"margin {MARGIN}: {prim['outcome']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
