"""Replicate MD crops versus SAM-masked crops at the same scale across three seeds."""
import argparse
import json
from pathlib import Path
from statistics import mean, stdev

from .backgrounds import fingerprint_artifacts
from .train import run_one
from .train_utils import load_config, tee_console


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", default="results/crop-sam-three-seeds")
    args = ap.parse_args()
    cfg = load_config(args.config, {})
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for mode in ("crop", "masked_crop"):
        for seed in (0, 1, 2):
            run_cfg = fingerprint_artifacts(dict(
                cfg, seed=seed, background=dict(cfg["background"], mode=mode)))
            (output / "status.txt").write_text(f"training {mode}, seed {seed}")
            with tee_console(run_cfg):
                result = run_one(run_cfg)
            rows.append(dict(mode=mode, seed=seed, run_key=result["run_key"],
                             validation_macro_f1=result["val_macro_f1_present"],
                             test_macro_f1=result["test_macro_f1_present"]))
            (output / "per-seed.json").write_text(json.dumps(rows, indent=2))
    summary = []
    for mode in ("crop", "masked_crop"):
        group = [r for r in rows if r["mode"] == mode]
        row = dict(mode=mode, seeds=[r["seed"] for r in group], n_seeds=len(group))
        for metric in ("validation_macro_f1", "test_macro_f1"):
            values = [r[metric] for r in group]
            row[metric + "_mean"] = mean(values)
            row[metric + "_std"] = stdev(values)
        summary.append(row)
    differences = []
    for seed in (0, 1, 2):
        baseline = next(r for r in rows if r["mode"] == "crop" and r["seed"] == seed)
        masked = next(r for r in rows if r["mode"] == "masked_crop" and r["seed"] == seed)
        differences.append(dict(seed=seed, **{
            metric: masked[metric] - baseline[metric]
            for metric in ("validation_macro_f1", "test_macro_f1")}))
    report = dict(methods=summary, paired_sam_minus_crop=differences,
                  std_definition="sample standard deviation across seeds; ddof=1")
    (output / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    (output / "status.txt").write_text("complete")


if __name__ == "__main__":
    main()
