"""The feature store: frozen features of every image, one folder per backbone/size/view/cameras.

Layout of a folder (George's format from scripts/cache_features.py, kept as is):

    index.csv.gz   one row per image: file_name, location, category, is_empty, burst_id,
                   datetime, hour, official_ood, y (-1 for empty frames), sorted by file_name
    features.npy   float16, (rows, dim), row i belongs to index row i
    status.npy     int8 per row: 0 not done, 1 ok, 2 could not be decoded
    config.json    everything that defines the folder (backbone, weights, size, crop rule,
                   detections file and its hash, row counts)
    manifest.json  written once complete; a folder without it is unfinished

Three kinds of folder per backbone, and who may read them:

    <name>              train + val cameras, labelled and empty      anyone, any time
    <name>_testempty    empty frames of the test cameras             anyone (unsupervised adaptation)
    <name>_test         labelled test images                         only via iwildcam.ledger
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from iwildcam.backbones import REGISTRY
from iwildcam.data import load_task, official_test_cameras
from iwildcam.protocol import FEATURE_STORE, SPLIT

COLUMNS = ["file_name", "location", "category", "is_empty", "burst_id", "datetime", "hour", SPLIT]
NOT_DONE, OK, UNDECODABLE = 0, 1, 2


def folder(backbone: str, cameras: str = "trainval", view: str = "crop") -> Path:
    return FEATURE_STORE / REGISTRY[backbone].folder_name(view, cameras)


def load(backbone: str, view: str = "crop", empty: bool = False, cameras: str = "trainval",
         ) -> tuple[pd.DataFrame, np.ndarray]:
    """Rows and float32 features of a finished store folder.

    cameras  "trainval" or "testempty". Labelled test features are read with
             `iwildcam.ledger.load_test`, which records the access.
    empty    False: labelled rows only; True: empty frames only.
    """
    if cameras == "test":
        raise PermissionError("labelled test features are read through iwildcam.ledger.load_test")
    path = folder(backbone, cameras, view)
    return _read(path, empty=empty if cameras == "trainval" else True)


def _read(path: Path, empty: bool) -> tuple[pd.DataFrame, np.ndarray]:
    if not (path / "manifest.json").exists():
        raise FileNotFoundError(f"{path} is missing or unfinished (no manifest.json)")
    rows = pd.read_csv(path / "index.csv.gz")
    features = np.load(path / "features.npy", mmap_mode="r")
    is_empty = rows.is_empty.astype(str).str.lower().eq("true").to_numpy()
    keep = (is_empty if empty else ~is_empty) & (np.load(path / "status.npy") == OK)
    return rows[keep].reset_index(drop=True), np.asarray(features[keep], dtype=np.float32)


def select_rows(cameras: str) -> pd.DataFrame:
    """The images a store folder holds, sorted by file_name.

    trainval   labelled train/val images, then the empty frames of those same cameras
    test       labelled images of the official test cameras
    testempty  empty frames of the official test cameras
    """
    task = load_task(require_files=False)
    test_cams = set(official_test_cameras())
    lab = task.df[COLUMNS + ["y"]]
    emp = task.empty[COLUMNS].assign(y=-1)
    if cameras == "trainval":
        lab = lab[lab[SPLIT].isin(["train", "val"])]
        emp = emp[emp[SPLIT].isin(["train", "val"]) & emp.location.isin(set(lab.location))]
        rows = pd.concat([lab, emp])
        assert not set(rows.location) & test_cams, "a test camera was selected"
    elif cameras == "test":
        rows = lab[lab[SPLIT].eq("test")]
        assert set(rows.location) <= test_cams
    elif cameras == "testempty":
        rows = emp[emp[SPLIT].eq("test")]
        assert set(rows.location) <= test_cams
        assert (rows.y == -1).all()
    else:
        raise ValueError(cameras)
    assert rows.file_name.is_unique and len(rows)
    assert rows.groupby("location")[SPLIT].nunique().eq(1).all(), "a camera spans two folds"
    return rows.sort_values("file_name").reset_index(drop=True)


def describe(path: Path) -> dict:
    return json.loads((path / "config.json").read_text())
