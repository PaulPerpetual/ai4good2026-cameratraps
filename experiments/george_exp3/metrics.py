"""The burst vote and the project's score, macro F1."""
import numpy as np
import pandas as pd

from settings import N_SPECIES


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
    """F1 of each species that appears in `true`, averaged, so rare species count as much as
    common ones."""
    correct = np.bincount(true[true == predicted], minlength=N_SPECIES)
    n_true = np.bincount(true, minlength=N_SPECIES)
    n_predicted = np.bincount(predicted, minlength=N_SPECIES)
    present = n_true > 0
    f1 = 2 * correct[present] / np.maximum(n_true[present] + n_predicted[present], 1)
    return float(f1.mean())


def score(true, predicted, bursts, detector_confidence) -> dict:
    """Macro F1 per photo, and after the burst vote."""
    voted = burst_vote(predicted, bursts, detector_confidence)
    return {"per_photo": macro_f1(true, predicted), "per_burst": macro_f1(true, voted)}
