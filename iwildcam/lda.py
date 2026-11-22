"""Linear discriminant (George, exp3/lda.py) classifier on frozen features, optionally camera-aware.

Each species gets a centre (its average feature vector). A photo is given the species whose
centre is closest, measured against how much photos of one species normally vary. The
camera-aware version also counts how a species' centre moves from camera to camera as normal
variation, so the classifier relies less on what changes with the camera.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from iwildcam.protocol import N_SPECIES


@dataclass
class LinearDiscriminant:
    feature_mean: np.ndarray     # used to standardise the features
    feature_std: np.ndarray
    weights: np.ndarray          # (features, species)
    bias: np.ndarray             # (species,)
    trained_species: np.ndarray  # species that had training photos; the others are never predicted

    def scores(self, features: np.ndarray) -> np.ndarray:
        standardised = (features - self.feature_mean) / self.feature_std
        scores = standardised @ self.weights + self.bias
        scores[:, ~self.trained_species] = -np.inf
        return scores

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.scores(features).argmax(1)


@dataclass
class TrainingStatistics:
    """Everything the classifier needs from the training photos. Computing these is the slow
    part; turning them into a classifier for any shrinkage and camera weight is fast."""
    feature_mean: np.ndarray
    feature_std: np.ndarray
    centres: np.ndarray          # (species, features)
    within_species: np.ndarray   # covariance of photos around their species' centre
    camera_shift: np.ndarray     # covariance of species centres between cameras
    trained_species: np.ndarray
    bursts_per_species: np.ndarray


def burst_weights(bursts) -> np.ndarray:
    """1 / (photos in its burst) for every photo, so that each burst counts once."""
    burst_index, _ = pd.factorize(np.asarray(bursts))
    return 1.0 / np.bincount(burst_index)[burst_index]


def fit(features, species, cameras, bursts, shrinkage: float, camera_weight: float = 0.0):
    """Fit the classifier on the training photos.

    shrinkage      pulls the estimated variation towards the same amount in every direction,
                   which keeps it stable with 2048 features
    camera_weight  how strongly camera-to-camera shifts count as variation (0: plain classifier)
    """
    return classifier(statistics(features, species, cameras, bursts), shrinkage, camera_weight)


def statistics(features, species, cameras, bursts, species_balance: float = 0.0) -> TrainingStatistics:
    """species_balance  0: every burst counts the same in the within-species variation, so common
                     species dominate it; 1: every species counts the same."""
    mean, std = features.mean(0), features.std(0) + 1e-6
    standardised = (features - mean) / std
    weight = burst_weights(bursts)

    bursts_per_species = np.bincount(species, weights=weight, minlength=N_SPECIES)
    trained = bursts_per_species > 0
    centres = _species_centres(standardised, species, weight, trained)
    distance_to_centre = standardised - centres[species]

    variation_weight = weight
    if species_balance:
        variation_weight = weight / bursts_per_species[species] ** species_balance
    return TrainingStatistics(
        feature_mean=mean, feature_std=std, centres=centres,
        within_species=_covariance(distance_to_centre, variation_weight),
        camera_shift=_camera_shift_covariance(distance_to_centre, species, cameras, weight),
        trained_species=trained, bursts_per_species=bursts_per_species)


def classifier(stats: TrainingStatistics, shrinkage: float, camera_weight: float = 0.0) -> LinearDiscriminant:
    variation = stats.within_species
    n_features = len(variation)
    variation = ((1 - shrinkage) * variation
                 + shrinkage * np.trace(variation) / n_features * np.eye(n_features))
    if camera_weight:
        variation = variation + camera_weight * stats.camera_shift
    weights = np.linalg.solve(variation, stats.centres.T)
    bias = -0.5 * np.einsum("sf,fs->s", stats.centres, weights)
    return LinearDiscriminant(stats.feature_mean, stats.feature_std, weights, bias, stats.trained_species)


def _species_centres(features, species, weight, trained):
    centres = np.zeros((N_SPECIES, features.shape[1]))
    for s in np.flatnonzero(trained):
        of_species = species == s
        centres[s] = np.average(features[of_species], axis=0, weights=weight[of_species])
    return centres


def _covariance(deviations, weight):
    weighted = deviations * np.sqrt(weight)[:, None]
    return weighted.T @ weighted / weight.sum()


def _camera_shift_covariance(distance_to_centre, species, cameras, weight):
    """How far each species' photos at one camera sit, on average, from that species' centre,
    and the covariance of those shifts over all species-camera pairs."""
    pair = pd.Series(species.astype(str)) + "_" + pd.Series(np.asarray(cameras).astype(str))
    pair_index, _ = pd.factorize(pair)
    pair_weight = np.bincount(pair_index, weights=weight)
    shift = np.zeros((len(pair_weight), distance_to_centre.shape[1]))
    np.add.at(shift, pair_index, distance_to_centre * weight[:, None])
    shift /= pair_weight[:, None]
    return (shift * pair_weight[:, None]).T @ shift / pair_weight.sum()
