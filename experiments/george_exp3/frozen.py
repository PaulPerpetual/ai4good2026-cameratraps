"""Steps 1-3: a frozen backbone and a linear discriminant on its saved features.

    python frozen.py --backbone dinov2l --step 1

Step 1  plain classifier on the crop features
Step 2  camera-aware classifier
Step 3  camera-aware classifier on the average of four views (crop, mirror, tighter, looser)

Prints and saves the macro F1 on unseen cameras (OOD) and on seen cameras (ID).
"""
import argparse
import json

import numpy as np
import pandas as pd

import lda
from data import detector_confidence, load_features, random_burst_split
from metrics import score
from settings import CLASSIFIER, EXPECTED_VALIDATION, RESULTS, VIEWS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, choices=["resnet50", "dinov2l"])
    parser.add_argument("--step", required=True, type=int, choices=[1, 2, 3])
    args = parser.parse_args()

    views = tuple(VIEWS) if args.step == 3 else ("crop",)
    seen_photos, seen_features = load_features(args.backbone, test=False, views=views)
    test_photos, test_features = load_features(args.backbone, test=True, views=views)

    ood = unseen_cameras(args.backbone, args.step, seen_photos, seen_features, test_photos, test_features)
    in_distribution = seen_cameras(args.backbone, args.step,
                                   pd.concat([seen_photos, test_photos], ignore_index=True),
                                   np.concatenate([seen_features, test_features]))

    result = {"backbone": args.backbone, "step": args.step, "ood": ood, "id": in_distribution}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"frozen_{args.backbone}_step{args.step}.json").write_text(json.dumps(result, indent=2))
    print(f"{args.backbone} step {args.step}   "
          f"OOD {ood['per_photo']:.3f} · {ood['per_burst']:.3f}   "
          f"ID {in_distribution['per_photo']:.3f} · {in_distribution['per_burst']:.3f}")


def unseen_cameras(backbone, step, photos, features, test_photos, test_features):
    """Fit on the training cameras, check the validation cameras, score the test cameras."""
    train = (photos.official_ood == "train").to_numpy()
    val = (photos.official_ood == "val").to_numpy()
    classifier = fit(photos[train], features[train], *CLASSIFIER[(step, backbone, "ood")])

    validation = evaluate(classifier, photos[val], features[val])
    check_validation(validation, EXPECTED_VALIDATION[(step, backbone)])
    return evaluate(classifier, test_photos, test_features)


def seen_cameras(backbone, step, photos, features):
    """Fit and score on the random_burst split, where test photos come from training cameras."""
    split = random_burst_split(photos).to_numpy()
    train, test = split == "train", split == "test"
    classifier = fit(photos[train], features[train], *CLASSIFIER[(step, backbone, "id")])
    return evaluate(classifier, photos[test], features[test])


def fit(photos, features, shrinkage, camera_weight):
    return lda.fit(features, photos.y.to_numpy().astype(int), photos.location.to_numpy(),
                   photos.burst_id.to_numpy(), shrinkage, camera_weight)


def evaluate(classifier, photos, features):
    predicted = classifier.predict(features)
    return score(photos.y.to_numpy().astype(int), predicted, photos.burst_id.to_numpy(),
                 detector_confidence(photos.file_name))


def check_validation(got, expected):
    """Stop before test if validation differs from the score these settings were chosen with."""
    for key, value in zip(("per_photo", "per_burst"), expected):
        if abs(got[key] - value) > 0.0005:
            raise SystemExit(f"validation {key} {got[key]:.4f}, expected {value}: settings differ, test not scored")


if __name__ == "__main__":
    main()
