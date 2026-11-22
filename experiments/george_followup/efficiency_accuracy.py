"""Accuracy of every combination in combinations.py whose features exist.

    python efficiency_accuracy.py

Every combination gets the camera-aware classifier of Experiment 3's step 2, with its own
shrinkage and camera weight chosen on held-out training cameras (5 groups of cameras, never
the validation or test cameras). Then: macro F1 on the validation cameras, the unseen test
cameras (OOD) and the random_burst split (ID), and per-photo accuracy on unseen cameras for
small, medium and large animals.
"""
import json

import numpy as np
import pandas as pd

import exp3_path  # noqa: F401
import lda
import plots
from combinations import COMBINATIONS
from common import RESULTS
from data import animal_box_area, detector_confidence, load_features, random_burst_split
from metrics import score
from settings import MIN_DETECTION_CONFIDENCE, feature_folder

SHRINKAGES = (0.01, 0.03, 0.1, 0.3)
CAMERA_WEIGHTS = (0, 2, 4, 8)
N_CAMERA_GROUPS = 5


def main():
    rows = []
    for c in COMBINATIONS:
        if not all((feature_folder(c.backbone, c.view, t, c.size, c.detector) / "features.npy").exists()
                   for t in (False, True)):
            continue
        rows.append(evaluate_combination(c))
        r = rows[-1]
        print(f"{c.name:32s} shrinkage {r['shrinkage']:<5} camera {r['camera_weight']}   "
              f"val {r['val_per_photo']:.3f} · {r['val_per_burst']:.3f}   OOD {r['ood_per_photo']:.3f} · "
              f"{r['ood_per_burst']:.3f}   ID {r['id_per_photo']:.3f} · {r['id_per_burst']:.3f}", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "efficiency_accuracy.json").write_text(json.dumps(rows, indent=2))
    pd.DataFrame([{k: v for k, v in r.items() if k != "accuracy_by_size"} for r in rows]).to_csv(
        RESULTS / "efficiency_accuracy.csv", index=False)


def evaluate_combination(c) -> dict:
    seen, seen_features = load_features(c.backbone, False, size=c.size, detector=c.detector, views=(c.view,))
    test, test_features = load_features(c.backbone, True, size=c.size, detector=c.detector, views=(c.view,))
    train = (seen.official_ood == "train").to_numpy()
    val = (seen.official_ood == "val").to_numpy()
    shrinkage, camera_weight = choose_settings(seen[train].reset_index(drop=True), seen_features[train])

    model = fit(seen[train], seen_features[train], shrinkage, camera_weight)
    val_score = evaluate(model, seen[val], seen_features[val])
    ood_predicted = model.predict(test_features)
    ood_score = score(test.y.to_numpy().astype(int), ood_predicted, test.burst_id.to_numpy(),
                      detector_confidence(test.file_name))

    everything = pd.concat([seen, test], ignore_index=True)
    every_feature = np.concatenate([seen_features, test_features])
    split = random_burst_split(everything).to_numpy()
    id_model = fit(everything[split == "train"], every_feature[split == "train"], shrinkage, camera_weight)
    id_score = evaluate(id_model, everything[split == "test"], every_feature[split == "test"])

    return {"backbone": c.backbone, "size": c.size, "view": c.view, "detector": c.detector, "name": c.name,
            "shrinkage": shrinkage, "camera_weight": camera_weight,
            "val_per_photo": val_score["per_photo"], "val_per_burst": val_score["per_burst"],
            "ood_per_photo": ood_score["per_photo"], "ood_per_burst": ood_score["per_burst"],
            "id_per_photo": id_score["per_photo"], "id_per_burst": id_score["per_burst"],
            "accuracy_by_size": accuracy_by_animal_size(test, ood_predicted, seen[train])}


def fit(photos, features, shrinkage, camera_weight):
    return lda.fit(features, photos.y.to_numpy().astype(int), photos.location.to_numpy(),
                   photos.burst_id.to_numpy(), shrinkage, camera_weight)


def evaluate(model, photos, features):
    return score(photos.y.to_numpy().astype(int), model.predict(features), photos.burst_id.to_numpy(),
                 detector_confidence(photos.file_name))


def choose_settings(photos, features):
    """The shrinkage and camera weight with the best macro F1 (per photo + per burst) on
    held-out training cameras: each group of cameras is predicted by a classifier fitted on
    the other groups."""
    cameras = np.sort(photos.location.unique())
    group_of_camera = dict(zip(np.random.default_rng(0).permutation(cameras),
                               np.arange(len(cameras)) % N_CAMERA_GROUPS))
    group = photos.location.map(group_of_camera).to_numpy()
    species = photos.y.to_numpy().astype(int)
    grid = [(s, w) for s in SHRINKAGES for w in CAMERA_WEIGHTS]
    predicted = {setting: np.zeros(len(photos), dtype=int) for setting in grid}
    for g in range(N_CAMERA_GROUPS):
        held_out = group == g
        stats = lda.statistics(features[~held_out], species[~held_out], photos.location.to_numpy()[~held_out],
                               photos.burst_id.to_numpy()[~held_out])
        for setting in grid:
            predicted[setting][held_out] = lda.classifier(stats, *setting).predict(features[held_out])
    confidence = detector_confidence(photos.file_name)
    totals = {setting: sum(score(species, p, photos.burst_id.to_numpy(), confidence).values())
              for setting, p in predicted.items()}
    return max(grid, key=lambda setting: totals[setting])


def accuracy_by_animal_size(test, predicted, train):
    """Per-photo accuracy for small, medium and large animals (thirds of the training photos'
    MegaDetector box areas), and for photos without a confident box."""
    def area(photos):
        a = animal_box_area(photos.file_name)
        a[detector_confidence(photos.file_name) < MIN_DETECTION_CONFIDENCE] = np.nan
        return a
    small_limit, large_limit = np.nanquantile(area(train), [1 / 3, 2 / 3])
    test_area = area(test)
    correct = predicted == test.y.to_numpy().astype(int)
    groups = {"small": test_area <= small_limit, "medium": (test_area > small_limit) & (test_area <= large_limit),
              "large": test_area > large_limit, "no box": np.isnan(test_area)}
    return {name: float(correct[mask].mean()) for name, mask in groups.items()}


if __name__ == "__main__":
    main()
