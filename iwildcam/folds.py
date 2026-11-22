"""Camera-wise cross-validation folds inside the training cameras.

Every setting (classifier shrinkage, augmentation weight, threshold...) is chosen by the
mean score over these folds; the 32 official validation cameras are then reported as a check.
The folds are fixed once, written to data/camera_folds.csv and committed, so everyone uses
the same ones.

Assignment (Shayan's pilot_a.py / gap_measure.py): cameras are sorted by region, then
randomly within region, and dealt round-robin to the folds, so each fold has cameras from
every region. Region from longitude: americas < -30 <= africa < 60 <= asia.

    uv run python -m iwildcam.folds        # (re)writes data/camera_folds.csv
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from iwildcam.data import DATA, load_task
from iwildcam.protocol import CAMERA_FOLDS, CV_SEED, N_CV_FOLDS, SPLIT


def region(longitude: float | None) -> str:
    if longitude is None:
        return "none"
    return "americas" if longitude < -30 else "africa" if longitude < 60 else "asia"


def make_folds(n_folds: int = N_CV_FOLDS, seed: int = CV_SEED) -> pd.DataFrame:
    task = load_task(require_files=False)
    cams = np.unique(task.df.location[task.df[SPLIT] == "train"])
    gps = json.loads((DATA / "gps_locations.json").read_text())
    reg = {c: region(gps.get(str(c), {}).get("longitude")) for c in cams}
    rng = np.random.default_rng(seed)
    order = sorted(cams, key=lambda c: (reg[c], rng.random()))
    return pd.DataFrame({"location": order, "region": [reg[c] for c in order],
                         "fold": [i % n_folds for i in range(len(order))]}).sort_values("location")


def camera_folds() -> dict[int, int]:
    """camera -> fold, for the training cameras."""
    table = pd.read_csv(CAMERA_FOLDS)
    return dict(zip(table.location, table.fold))


def fold_of(locations) -> np.ndarray:
    """Fold of each row's camera; -1 for cameras outside the training cameras (val, test)."""
    folds = camera_folds()
    return np.array([folds.get(c, -1) for c in np.asarray(locations)])


if __name__ == "__main__":
    table = make_folds()
    table.to_csv(CAMERA_FOLDS, index=False)
    print(table.groupby(["fold", "region"]).size().unstack(fill_value=0))
    print(f"wrote {CAMERA_FOLDS}")
