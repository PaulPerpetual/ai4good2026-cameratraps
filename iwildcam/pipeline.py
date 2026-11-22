"""The consolidated pipeline: one entry point, one YAML config per run.

    crop -> frozen backbone -> feature store -> head -> burst vote -> score
    (MegaDetector v5a)  (dinov2l | bioclip2)   (lda | lda_cam)  (detector-weighted)  (CV · val · test once)

    uv run python -m iwildcam.pipeline configs/pipeline/dinov2l_lda_cam.yaml          # select on CV, report val
    uv run python -m iwildcam.pipeline configs/pipeline/dinov2l_lda_cam.yaml --test   # score test, once

A select run chooses the head's settings on the camera-wise CV folds (per-image macro F1),
reports the official validation cameras with camera-bootstrap intervals, freezes the result
in configs/final/<name>.json and appends a row to results/runs.jsonl (committed, so every
method lands in one table). --test refits the frozen config, checks that validation
reproduces, and scores test through the ledger (refused if this config was scored before).

Features must exist in the store first (scripts/extract_features.py). Adding a stage: give
it a config key, implement it as a function below, and record it in PROTOCOL.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import getpass
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import yaml

from iwildcam import ledger, lda, metrics, store
from iwildcam.crops import detector_confidence
from iwildcam.folds import fold_of
from iwildcam.protocol import N_CV_FOLDS, REPO

RUNS = REPO / "results" / "runs.jsonl"
HEADS = {"lda": "plain linear discriminant", "lda_cam": "camera-aware linear discriminant (George)"}
DEFAULT_GRID = {
    "lda": {"shrinkage": [0.01, 0.03, 0.1, 0.3, 0.6], "camera_weight": [0]},
    "lda_cam": {"shrinkage": [0.01, 0.03, 0.1, 0.3], "camera_weight": [1, 2, 4, 8, 16]},
}


def read_config(path: Path) -> dict:
    config = yaml.safe_load(Path(path).read_text())
    config.setdefault("view", "crop")
    config.setdefault("grid", DEFAULT_GRID[config["head"]])
    assert config["head"] in HEADS, f"head must be one of {list(HEADS)}"
    assert config.get("select_on", "per_image") == "per_image", "protocol: select on per-image CV macro F1"
    return config


def labelled(backbone: str, view: str):
    rows, x = store.load(backbone, view)
    return rows, x, (rows.y.to_numpy().astype(int), rows.location.to_numpy(), rows.burst_id.to_numpy(),
                     detector_confidence(rows.file_name))


def cross_validate(x, y, cam, burst, conf, train, grid) -> dict:
    """Mean per-image and per-burst macro F1 over the camera-wise folds, for every setting."""
    fold = fold_of(cam)
    settings = list(itertools.product(grid["shrinkage"], grid["camera_weight"]))
    scores = {s: [] for s in settings}
    for f in range(N_CV_FOLDS):
        fit, held = train & (fold != f), train & (fold == f)
        stats = lda.statistics(x[fit], y[fit], cam[fit], burst[fit])
        for s, w in settings:
            pred = lda.classifier(stats, s, w).predict(x[held])
            scores[(s, w)].append(metrics.score(y[held], pred, burst[held], conf[held]))
        print(f"  fold {f + 1}/{N_CV_FOLDS}", flush=True)
    return {s: {k: float(np.mean([r[k] for r in v])) for k in ("per_image", "per_burst")} for s, v in scores.items()}


def with_intervals(y, pred, cam, burst, conf) -> dict:
    voted = metrics.burst_vote(pred, burst, conf)
    out = metrics.score(y, pred, burst, conf)
    out["ci95"] = {"per_image": metrics.camera_bootstrap(y, pred, cam),
                   "per_burst": metrics.camera_bootstrap(y, voted, cam)}
    return out


def select(config_path: Path) -> dict:
    config = read_config(config_path)
    rows, x, (y, cam, burst, conf) = labelled(config["backbone"], config["view"])
    train = (rows.official_ood == "train").to_numpy()
    val = (rows.official_ood == "val").to_numpy()

    table = cross_validate(x, y, cam, burst, conf, train, config["grid"])
    for (s, w), r in sorted(table.items(), key=lambda kv: -kv[1]["per_image"])[:5]:
        print(f"  shrinkage {s:<5} camera_weight {w:<3} CV {r['per_image']:.4f} · {r['per_burst']:.4f}")
    shrinkage, camera_weight = max(table, key=lambda s: table[s]["per_image"])

    classifier = lda.fit(x[train], y[train], cam[train], burst[train], shrinkage, camera_weight)
    val_scores = with_intervals(y[val], classifier.predict(x[val]), cam[val], burst[val], conf[val])
    final = {"name": config["name"], "backbone": config["backbone"], "view": config["view"],
             "head": config["head"], "shrinkage": shrinkage, "camera_weight": camera_weight,
             "fit_on": "official_ood train cameras",
             "store_manifest": json.loads((store.folder(config["backbone"], view=config["view"]) / "manifest.json").read_text()),
             "cv": table[(shrinkage, camera_weight)],
             "official_val": {k: val_scores[k] for k in ("per_image", "per_burst")},
             "official_val_ci95": {k: [v["low"], v["high"]] for k, v in val_scores["ci95"].items()}}
    out = REPO / "configs" / "final" / f"{config['name']}.json"
    if out.exists() and json.loads(out.read_text()) != final and ledger.entries():
        if any(e["config_hash"] == ledger.config_hash(out) for e in ledger.entries()):
            raise SystemExit(f"{out.name} was scored on test and now differs; give the run a new name")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(final, indent=2))

    _log({"event": "select", **final, "cv_table": {f"{s}/{w}": v for (s, w), v in table.items()}})
    print(f"{config['name']}: shrinkage {shrinkage}, camera_weight {camera_weight}  "
          f"CV {final['cv']['per_image']:.4f} · {final['cv']['per_burst']:.4f}  "
          f"val {final['official_val']['per_image']:.4f} · {final['official_val']['per_burst']:.4f}  -> {out.relative_to(REPO)}")
    return final


def test(config_path: Path) -> dict:
    config = read_config(config_path)
    final_path = REPO / "configs" / "final" / f"{config['name']}.json"
    final = json.loads(final_path.read_text())
    rows, x, (y, cam, burst, conf) = labelled(final["backbone"], final["view"])
    train = (rows.official_ood == "train").to_numpy()
    val = (rows.official_ood == "val").to_numpy()
    classifier = lda.fit(x[train], y[train], cam[train], burst[train], final["shrinkage"], final["camera_weight"])
    check = metrics.score(y[val], classifier.predict(x[val]), burst[val], conf[val])
    for k, v in final["official_val"].items():
        if abs(check[k] - v) > 5e-4:
            raise SystemExit(f"official val {k} {check[k]:.4f} != {v:.4f} in {final_path.name}: test not scored")

    test_rows, test_x = ledger.load_test(final["backbone"], final_path, final["view"])
    ty, tcam, tburst, tconf = (test_rows.y.to_numpy().astype(int), test_rows.location.to_numpy(),
                               test_rows.burst_id.to_numpy(), detector_confidence(test_rows.file_name))
    scores = with_intervals(ty, classifier.predict(test_x), tcam, tburst, tconf)
    ledger.record(final_path, scores)
    _log({"event": "test", "name": final["name"], "scores": scores})
    print(json.dumps(scores, indent=2))
    return scores


def _log(entry: dict) -> None:
    entry = {"date": dt.datetime.now().isoformat(timespec="seconds"), "who": getpass.getuser(), **entry}
    entry["run_hash"] = hashlib.sha256(json.dumps(entry, sort_keys=True, default=str).encode()).hexdigest()[:12]
    RUNS.parent.mkdir(parents=True, exist_ok=True)
    with RUNS.open("a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("config", type=Path)
    p.add_argument("--test", action="store_true", help="score the frozen config on test (once)")
    a = p.parse_args()
    (test if a.test else select)(a.config)
