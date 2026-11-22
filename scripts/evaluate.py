"""The protocol's evaluation of a frozen backbone + linear discriminant, end to end.

    # 1. choose settings on the camera-wise CV folds, report official val, freeze a final config
    uv run python scripts/evaluate.py select --backbone dinov2l --head lda_cam
    # 2. score test once (refused if this config was scored before; logged in results/test_ledger.jsonl)
    uv run python scripts/evaluate.py test configs/final/dinov2l_lda_cam.json

Heads: lda (plain linear discriminant) and lda_cam (camera-aware; George's step 2).
Selection metric: mean per-image macro F1 over the CV folds (per-burst is near its ceiling and
moves in coarse steps). Both metrics are always reported.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from iwildcam import ledger, lda, metrics, store  # noqa: E402
from iwildcam.crops import detector_confidence  # noqa: E402
from iwildcam.folds import fold_of  # noqa: E402
from iwildcam.protocol import N_CV_FOLDS, REPO  # noqa: E402

GRID = {
    "lda": {"shrinkage": [0.01, 0.03, 0.1, 0.3, 0.6], "camera_weight": [0]},
    "lda_cam": {"shrinkage": [0.01, 0.03, 0.1, 0.3], "camera_weight": [1, 2, 4, 8, 16]},
}


def columns(rows):
    return (rows.y.to_numpy().astype(int), rows.location.to_numpy(), rows.burst_id.to_numpy(),
            detector_confidence(rows.file_name))


def select(a):
    rows, x = store.load(a.backbone)
    y, cam, burst, conf = columns(rows)
    train = (rows.official_ood == "train").to_numpy()
    val = (rows.official_ood == "val").to_numpy()
    fold = fold_of(cam)
    grid = list(itertools.product(GRID[a.head]["shrinkage"], GRID[a.head]["camera_weight"]))

    cv = {g: [] for g in grid}
    for f in range(N_CV_FOLDS):
        fit_rows, held = train & (fold != f), train & (fold == f)
        stats = lda.statistics(x[fit_rows], y[fit_rows], cam[fit_rows], burst[fit_rows])
        for s, w in grid:
            pred = lda.classifier(stats, s, w).predict(x[held])
            cv[(s, w)].append(metrics.score(y[held], pred, burst[held], conf[held]))
        print(f"  fold {f + 1}/{N_CV_FOLDS} done", flush=True)
    table = {g: {k: float(np.mean([r[k] for r in v])) for k in ("per_image", "per_burst")} for g, v in cv.items()}
    for (s, w), r in sorted(table.items(), key=lambda kv: -kv[1]["per_image"]):
        print(f"  shrinkage {s:<5} camera_weight {w:<3}  CV {r['per_image']:.4f} · {r['per_burst']:.4f}")
    best = max(table, key=lambda g: table[g]["per_image"])

    classifier = lda.fit(x[train], y[train], cam[train], burst[train], *best)
    pred = classifier.predict(x[val])
    val_score = metrics.score(y[val], pred, burst[val], conf[val])
    voted = metrics.burst_vote(pred, burst[val], conf[val])
    val_ci = {"per_image": metrics.camera_bootstrap(y[val], pred, cam[val]),
              "per_burst": metrics.camera_bootstrap(y[val], voted, cam[val])}
    print(f"chosen shrinkage {best[0]}, camera_weight {best[1]}: CV {table[best]['per_image']:.4f} · "
          f"{table[best]['per_burst']:.4f}   official val {val_score['per_image']:.4f} · {val_score['per_burst']:.4f}")

    final = {"backbone": a.backbone, "head": a.head, "shrinkage": best[0], "camera_weight": best[1],
             "fit_on": "official_ood train cameras",
             "store_manifest": json.loads((store.folder(a.backbone) / "manifest.json").read_text()),
             "cv": table[best], "official_val": val_score,
             "official_val_ci95": {k: [v["low"], v["high"]] for k, v in val_ci.items()}}
    out = REPO / "configs" / "final" / f"{a.backbone}_{a.head}.json"
    if out.exists() and json.loads(out.read_text()) != final:
        out = out.with_name(out.stem + "_new.json")
        print(f"note: existing final config differs; wrote {out.name} instead, compare before replacing")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(final, indent=2))
    print(f"wrote {out.relative_to(REPO)}")


def test(a):
    config = json.loads(a.config.read_text())
    rows, x = store.load(config["backbone"])
    y, cam, burst, conf = columns(rows)
    train = (rows.official_ood == "train").to_numpy()
    val = (rows.official_ood == "val").to_numpy()
    classifier = lda.fit(x[train], y[train], cam[train], burst[train], config["shrinkage"], config["camera_weight"])
    check = metrics.score(y[val], classifier.predict(x[val]), burst[val], conf[val])
    for k, v in config["official_val"].items():
        if abs(check[k] - v) > 5e-4:
            raise SystemExit(f"official val {k} {check[k]:.4f} != {v:.4f} in the config: not scoring test")

    test_rows, test_x = ledger.load_test(config["backbone"], a.config)
    ty, tcam, tburst, tconf = columns(test_rows)
    pred = classifier.predict(test_x)
    scores = metrics.score(ty, pred, tburst, tconf)
    scores["ci95"] = {"per_image": metrics.camera_bootstrap(ty, pred, tcam),
                      "per_burst": metrics.camera_bootstrap(ty, metrics.burst_vote(pred, tburst, tconf), tcam)}
    ledger.record(a.config, scores)
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(required=True)
    s = sub.add_parser("select")
    s.add_argument("--backbone", required=True)
    s.add_argument("--head", default="lda_cam", choices=list(GRID))
    s.set_defaults(run=select)
    t = sub.add_parser("test")
    t.add_argument("config", type=Path)
    t.set_defaults(run=test)
    args = p.parse_args()
    args.run(args)
