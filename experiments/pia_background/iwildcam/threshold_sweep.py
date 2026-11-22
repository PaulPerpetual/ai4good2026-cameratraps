"""Compare crop thresholds with fixed training settings; rank by validation F1."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

from .backgrounds import fingerprint_artifacts
from .data import load_fold, load_task, image_dir_for
from .prepare_foregrounds import animal_box
from .train import run_one
from .train_utils import load_config, tee_console


def aggregate_validation(rows):
    """Sample standard deviation across independent training seeds, not standard error."""
    groups = []
    for threshold in sorted({r["threshold"] for r in rows}):
        group = [r for r in rows if r["threshold"] == threshold]
        scores = [r["validation_macro_f1"] for r in group]
        groups.append(dict(threshold=threshold, n_seeds=len(scores),
                           seeds=[r["seed"] for r in group],
                           mean_validation_macro_f1=mean(scores),
                           std_validation_macro_f1=stdev(scores) if len(scores) > 1 else None))
    return groups


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--detections", required=True)
    ap.add_argument("--baseline-manifest", required=True)
    ap.add_argument("--thresholds", default="0.2,0.4,0.6")
    ap.add_argument("--output", default="results/threshold-sweep")
    ap.add_argument("--coverage-only", action="store_true")
    ap.add_argument("--seeds", default="0", help="Comma-separated seeds; completed runs are reused")
    ap.add_argument("--report-dir", help="Keep reports separate while reusing existing manifests")
    args = ap.parse_args()
    thresholds = sorted(set(float(x) for x in args.thresholds.split(",")))
    if not thresholds or any(not 0 <= t <= 1 for t in thresholds):
        ap.error("thresholds must be between 0 and 1")
    cfg = load_config(args.config, {})
    seeds = sorted(set(int(s) for s in args.seeds.split(",")))
    cfg["seed"] = 0
    task = load_task(cfg.get("min_per_camera", 10), cfg.get("min_cameras", 5),
                     image_dir=cfg.get("image_dir") or image_dir_for(cfg["height"]))
    fold = load_fold(task, cfg["split"])
    raw = Path(args.detections).read_bytes()
    md = json.loads(raw)
    if md.get("detection_categories", {}).get("1") != "animal":
        raise ValueError("Expected MegaDetector category 1 = animal")
    records = {}
    for item in md["images"]:
        name = Path(item["file"].replace("\\", "/")).name
        if name in records:
            raise ValueError(f"Duplicate image basename: {name}")
        records[name] = item
    baseline = json.loads(Path(args.baseline_manifest).read_text())
    if baseline["detections_sha256"] != hashlib.sha256(raw).hexdigest():
        raise ValueError("Baseline and sweep must use the same detector outputs")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    reports = Path(args.report_dir) if args.report_dir else output
    reports.mkdir(parents=True, exist_ok=True)
    configs, coverage = [], []
    for threshold in thresholds:
        images = {}
        for name in task.df.file_name:
            item = records[name]
            if item.get("failure"):
                raise ValueError(f"Failed detector output: {name}")
            box = animal_box(item, threshold)
            images[name] = dict(box=box, foreground=None,
                                status="box_only" if box is not None else "no_detection")
        counts = dict(Counter(r["status"] for r in images.values()))
        manifest = dict(format_version=1, threshold=threshold, sam_model=None,
                        detector_info=md.get("info", {}),
                        detections_sha256=hashlib.sha256(raw).hexdigest(),
                        counts=counts, images=images)
        if threshold == baseline["threshold"]:
            # Reuse the exact artifact path/config to recover the existing 0.2 run.
            for name, record in images.items():
                if record["box"] != baseline["images"][name]["box"]:
                    raise ValueError(f"Baseline box mismatch: {name}")
            path = Path(args.baseline_manifest)
        else:
            path = output / f"manifest-{threshold:g}.json"
            content = json.dumps(manifest, indent=2)
            if path.exists() and path.read_text() != content:
                raise ValueError(f"Refusing to replace different artifact: {path}")
            path.write_text(content)
        row = {"threshold": threshold}
        for split, indices in fold.items():
            accepted = sum(images[name]["box"] is not None
                           for name in task.df.iloc[indices].file_name)
            row[split] = dict(n=len(indices), cropped=accepted,
                              fallback=len(indices)-accepted,
                              crop_fraction=accepted / len(indices))
        coverage.append(row)
        run_cfg = dict(cfg, background=dict(cfg.get("background", {}),
                                           mode="crop", manifest=str(path)))
        configs.append(fingerprint_artifacts(run_cfg))
    (reports / "coverage.json").write_text(json.dumps(coverage, indent=2))
    print("Detection coverage: " + json.dumps(coverage), flush=True)
    if args.coverage_only:
        return
    summary = []
    for threshold, run_cfg in zip(thresholds, configs):
        for seed in seeds:
            seeded_cfg = dict(run_cfg, seed=seed)
            (reports / "status.txt").write_text(f"training threshold {threshold:g}, seed {seed}")
            with tee_console(seeded_cfg):
                result = run_one(seeded_cfg)
            summary.append(dict(threshold=threshold, seed=seed,
                                validation_macro_f1=result["val_macro_f1_present"],
                                run_key=result["run_key"]))
            (reports / "validation-results.json").write_text(json.dumps(summary, indent=2))
            (reports / "validation-averages.json").write_text(
                json.dumps(aggregate_validation(summary), indent=2))
    selected = max(aggregate_validation(summary), key=lambda r: r["mean_validation_macro_f1"])
    print("Validation results: " + json.dumps(summary), flush=True)
    print("Selected by validation: " + json.dumps(selected), flush=True)
    (reports / "status.txt").write_text("complete")


if __name__ == "__main__":
    main()
