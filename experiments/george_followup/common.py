"""Shared by the follow-up scripts: the two test settings (OOD and ID) and where results go."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

import exp3_path  # noqa: F401
import lda
from data import detector_confidence, load_features, random_burst_split
from settings import CLASSIFIER, MY_FOLDER, VIEWS

RESULTS = MY_FOLDER / "followup_results"


@dataclass
class Photos:
    table: pd.DataFrame      # one row per photo: file_name, location, burst_id, y, ...
    features: np.ndarray

    @property
    def species(self):
        return self.table.y.to_numpy().astype(int)

    @property
    def bursts(self):
        return self.table.burst_id.to_numpy()

    @property
    def detector_confidence(self):
        return detector_confidence(self.table.file_name)


@dataclass
class Setting:
    """One way of testing: fit on `train`, choose things on `val`, report on `test`."""
    name: str          # "ood": unseen test cameras; "id": random_burst, same cameras in train and test
    train: Photos
    val: Photos
    test: Photos


def settings(backbone: str, step: int = 2) -> list[Setting]:
    """OOD and ID settings on the saved features of Experiment 3's step 1, 2 or 3."""
    views = tuple(VIEWS) if step == 3 else ("crop",)
    seen, seen_features = load_features(backbone, test=False, views=views)
    test, test_features = load_features(backbone, test=True, views=views)

    def part(table, features, mask):
        mask = np.asarray(mask)
        return Photos(table[mask].reset_index(drop=True), features[mask])

    ood = Setting("ood", part(seen, seen_features, seen.official_ood == "train"),
                  part(seen, seen_features, seen.official_ood == "val"),
                  Photos(test, test_features))

    everything = pd.concat([seen, test], ignore_index=True)
    every_feature = np.concatenate([seen_features, test_features])
    split = random_burst_split(everything).to_numpy()
    in_distribution = Setting("id", *(part(everything, every_feature, split == p) for p in ("train", "val", "test")))
    return [ood, in_distribution]


def training_statistics(photos: Photos, species_balance: float = 0.0) -> lda.TrainingStatistics:
    return lda.statistics(photos.features, photos.species, photos.table.location.to_numpy(), photos.bursts,
                          species_balance)


def classifier(stats, backbone: str, step: int, setting: str) -> lda.LinearDiscriminant:
    """The classifier with the shrinkage and camera weight Experiment 3 chose."""
    return lda.classifier(stats, *CLASSIFIER[(step, backbone, setting)])
