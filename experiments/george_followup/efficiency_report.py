"""Accuracy against cost of the whole on-camera pipeline, for every combination.

    python efficiency_report.py       # after efficiency_cost.py, detector_cost.py, efficiency_accuracy.py

Pipeline cost per photo = backbone + classifier, plus the detector for crops. The best
trade-offs are picked on the validation cameras: a combination is on the frontier when no
cheaper one has a higher validation macro F1.
"""
import json

import pandas as pd

import plots
from common import RESULTS


def main():
    backbone_cost = {(r["backbone"], r["size"]): r for r in json.loads((RESULTS / "efficiency_cost.json").read_text())["rows"]}
    detector_cost = {r["detector"]: r for r in json.loads((RESULTS / "detector_cost.json").read_text())["rows"]}
    rows = json.loads((RESULTS / "efficiency_accuracy.json").read_text())

    for r in rows:
        backbone = backbone_cost[(r["backbone"], r["size"])]
        detector = detector_cost[r["detector"]] if r["view"] == "crop" else {"gflops": 0.0, "cpu_ms": 0.0}
        r["pipeline_gflops"] = backbone["gflops"] + backbone["classifier_gflops"] + detector["gflops"]
        r["pipeline_cpu_ms"] = backbone["cpu_ms"] + detector["cpu_ms"]
        r["backbone_gflops"] = backbone["gflops"]

    best_so_far = -1.0
    for r in sorted(rows, key=lambda r: r["pipeline_gflops"]):
        r["on_frontier"] = r["val_per_photo"] > best_so_far
        best_so_far = max(best_so_far, r["val_per_photo"])

    table = pd.DataFrame(rows).sort_values("pipeline_gflops")
    columns = ["name", "pipeline_gflops", "pipeline_cpu_ms", "val_per_photo", "ood_per_photo", "ood_per_burst",
               "id_per_photo", "id_per_burst", "on_frontier"]
    table[columns].to_csv(RESULTS / "efficiency_report.csv", index=False)
    print(table[columns].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    plots.accuracy_vs_cost(rows, "pipeline_gflops", "Accuracy against cost per photo (detector + backbone)",
                           RESULTS / "efficiency_accuracy_vs_gflops.png")
    by_size = [r for r in rows if r["view"] == "crop" and r["detector"] == "md5a"
               and r["backbone"] in ("resnet50", "dinov2s", "dinov2l")]
    plots.by_animal_size(sorted(by_size, key=lambda r: (r["backbone"], r["size"])),
                         "Do small animals need more pixels?", RESULTS / "efficiency_by_animal_size.png")


if __name__ == "__main__":
    main()
