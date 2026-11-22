"""Can the classifier's confidence be trusted, and can it decide which photos a human checks?

    python calibration.py --backbone dinov2l          # step 2: camera-aware classifier on crops

The classifier's scores become probabilities with a softmax at a temperature fitted on the
validation cameras (for ID: on random_burst's validation part). Photos and bursts get their
own temperature: averaging the photos of a burst already evens out their probabilities, so a
burst needs less softening than a single photo. A temperature never changes which species is
predicted, only how sure the model says it is. Reported before and after, per photo and per
burst:
    ECE   expected calibration error: average gap between confidence and actual accuracy
    NLL   average -log(probability of the true species)
and how the macro F1 of the photos kept rises when the least confident ones go to a human.
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

import exp3_path  # noqa: F401
import plots
from common import RESULTS, classifier, settings, training_statistics
from metrics import macro_f1

N_BINS = 15
KEPT_SHARES = (1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, choices=["resnet50", "dinov2l"])
    parser.add_argument("--step", type=int, default=2, choices=[1, 2, 3])
    args = parser.parse_args()

    report = {"backbone": args.backbone, "step": args.step}
    for setting in settings(args.backbone, args.step):
        model = classifier(training_statistics(setting.train), args.backbone, args.step, setting.name)
        val_scores, val = model.scores(setting.val.features), setting.val
        temperatures = {
            "per_photo": fit_temperature(lambda t: negative_log_likelihood(probabilities(val_scores, t), val.species)),
            "per_burst": fit_temperature(lambda t: negative_log_likelihood(*per_burst(probabilities(val_scores, t), val))),
        }
        report[setting.name] = {
            "temperatures": temperatures,
            "val": evaluate(val_scores, val, temperatures),
            "test": evaluate(model.scores(setting.test.features), setting.test, temperatures),
        }
        test = report[setting.name]["test"]
        print(f"{args.backbone} {setting.name}: temperature {temperatures['per_photo']:.1f} per photo, "
              f"{temperatures['per_burst']:.1f} per burst   ECE per photo "
              f"{test['raw']['per_photo']['ece']:.3f} -> {test['calibrated']['per_photo']['ece']:.3f}, "
              f"per burst {test['raw']['per_burst']['ece']:.3f} -> {test['calibrated']['per_burst']['ece']:.3f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    name = f"calibration_{args.backbone}_step{args.step}"
    (RESULTS / f"{name}.json").write_text(json.dumps(report, indent=2))
    plots.reliability_and_coverage(report, f"Calibration, {args.backbone}, step {args.step}", RESULTS / f"{name}.png")


def probabilities(scores, temperature):
    """Softmax of scores / temperature. Species never seen in training get probability 0."""
    z = scores / temperature
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def negative_log_likelihood(p, true):
    return float(-np.log(np.clip(p[np.arange(len(true)), true], 1e-12, None)).mean())


def fit_temperature(loss) -> float:
    """The temperature that minimises loss(temperature), searched on a log scale."""
    result = minimize_scalar(lambda log_t: loss(np.exp(log_t)), bounds=(-5, 12), method="bounded")
    return float(np.exp(result.x))


def evaluate(scores, photos, temperatures) -> dict:
    raw = probabilities(scores, 1.0)
    photo_p = probabilities(scores, temperatures["per_photo"])
    burst_p, burst_true = per_burst(probabilities(scores, temperatures["per_burst"]), photos)
    return {
        "raw": {"per_photo": calibration(raw, photos.species), "per_burst": calibration(*per_burst(raw, photos))},
        "calibrated": {"per_photo": calibration(photo_p, photos.species), "per_burst": calibration(burst_p, burst_true)},
        "coverage_per_photo": coverage(photo_p, photos.species),
        "coverage_per_burst": coverage(burst_p, burst_true),
    }


def per_burst(p, photos):
    """One probability vector per burst: the photos' probabilities averaged, each weighted by
    MegaDetector's confidence (equally if it found nothing in the whole burst)."""
    burst_index, _ = pd.factorize(photos.bursts)
    weight = photos.detector_confidence
    total = np.bincount(burst_index, weights=weight)
    weight = np.where(total[burst_index] > 0, weight, 1.0)
    summed = np.zeros((burst_index.max() + 1, p.shape[1]))
    np.add.at(summed, burst_index, p * weight[:, None])
    burst_p = summed / summed.sum(1, keepdims=True)
    burst_true = np.zeros(len(burst_p), dtype=int)
    burst_true[burst_index] = photos.species
    return burst_p, burst_true


def calibration(p, true) -> dict:
    confidence, predicted = p.max(1), p.argmax(1)
    correct = predicted == true
    edges = np.linspace(0, 1, N_BINS + 1)
    which = np.clip(np.digitize(confidence, edges) - 1, 0, N_BINS - 1)
    bins = []
    for b in range(N_BINS):
        inside = which == b
        bins.append({"count": int(inside.sum()),
                     "confidence": float(confidence[inside].mean()) if inside.any() else None,
                     "accuracy": float(correct[inside].mean()) if inside.any() else None})
    ece = sum(b["count"] * abs(b["confidence"] - b["accuracy"]) for b in bins if b["count"]) / len(true)
    return {"ece": float(ece), "nll": negative_log_likelihood(p, true), "accuracy": float(correct.mean()),
            "mean_confidence": float(confidence.mean()), "bins": bins}


def coverage(p, true) -> list:
    """Keep only the most confident share of photos (or bursts); the rest go to a human."""
    confidence, predicted = p.max(1), p.argmax(1)
    rows = []
    for share in KEPT_SHARES:
        threshold = np.quantile(confidence, 1 - share) if share < 1 else 0.0
        kept = confidence >= threshold
        rows.append({"kept": float(kept.mean()), "threshold": float(threshold),
                     "macro_f1": macro_f1(true[kept], predicted[kept]),
                     "accuracy": float((predicted[kept] == true[kept]).mean())})
    return rows


if __name__ == "__main__":
    main()
