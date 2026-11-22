"""The project's score: macro F1 per image and after the burst vote, with camera-bootstrap CIs.

The detector-weighted burst vote is Paul's (linprobe.py, 2026-10-19); this is George's
implementation of it (exp3/metrics.py). On ResNet-50 it moves official-OOD test macro F1
from 0.405 (per image) to 0.495 (per burst); an unweighted vote only reaches 0.410.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from iwildcam.protocol import N_SPECIES


def burst_vote(predicted, bursts, detector_confidence) -> np.ndarray:
    """Let the photos of each burst vote on its species, and give every photo the winner.

    A photo's vote counts as much as MegaDetector's confidence that it shows an animal. In a
    burst where MegaDetector found nothing at all, every photo counts the same.
    """
    burst_index, _ = pd.factorize(np.asarray(bursts))
    vote = np.asarray(detector_confidence, dtype=np.float64)
    burst_total = np.bincount(burst_index, weights=vote)
    vote = np.where(burst_total[burst_index] > 0, vote, 1.0)
    tally = np.zeros((burst_index.max() + 1, N_SPECIES))
    np.add.at(tally, (burst_index, predicted), vote)
    return tally.argmax(1)[burst_index]


def macro_f1(true, predicted) -> float:
    """F1 of each species present in `true`, averaged: rare species count as much as common."""
    true, predicted = np.asarray(true), np.asarray(predicted)
    correct = np.bincount(true[true == predicted], minlength=N_SPECIES)
    n_true = np.bincount(true, minlength=N_SPECIES)
    n_predicted = np.bincount(predicted, minlength=N_SPECIES)
    present = n_true > 0
    return float((2 * correct[present] / np.maximum(n_true[present] + n_predicted[present], 1)).mean())


def score(true, predicted, bursts, detector_confidence) -> dict:
    """Macro F1 per image and after the burst vote. Report both, always."""
    voted = burst_vote(predicted, bursts, detector_confidence)
    return {"per_image": macro_f1(true, predicted), "per_burst": macro_f1(true, voted)}


def camera_bootstrap(true, predicted_a, cameras, predicted_b=None, n: int = 2000, seed: int = 0) -> dict:
    """95% interval of macro F1 (or of the difference a - b) resampling whole cameras.

    Cameras, not images, are the unit: images of one camera are not independent. With only
    32 validation cameras the intervals are wide; say so rather than over-read a gap.
    Pass burst-voted predictions to get the per-burst interval.
    """
    true, cameras = np.asarray(true), np.asarray(cameras)
    rows_of = [np.flatnonzero(cameras == c) for c in np.unique(cameras)]
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        rows = np.concatenate([rows_of[i] for i in rng.integers(len(rows_of), size=len(rows_of))])
        s = macro_f1(true[rows], predicted_a[rows])
        if predicted_b is not None:
            s -= macro_f1(true[rows], predicted_b[rows])
        stats.append(s)
    point = macro_f1(true, predicted_a) - (0 if predicted_b is None else macro_f1(true, predicted_b))
    low, high = np.percentile(stats, [2.5, 97.5])
    return {"point": point, "low": float(low), "high": float(high),
            "share_above_0": float(np.mean(np.array(stats) > 0))}
