"""The data protocol: every fixed choice that makes our numbers comparable, in one place.

PROTOCOL.md explains each choice and where it was decided. Code imports these names; it does
not restate them. Changing a value here is a protocol change: note it in PROTOCOL.md.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------------------
# Where things live (vinzant). Override GROUP with IWILDCAM_GROUP to run elsewhere.
# ---------------------------------------------------------------------------------------
GROUP = Path(os.environ.get("IWILDCAM_GROUP", "/zfs/ai4good/groups/camera-traps"))
ORIGINALS = GROUP / "raw" / "train"          # full-resolution JPEGs (train and test), read only
FEATURE_STORE = GROUP / "features"           # one folder per (backbone, size, view, cameras)
REPO = Path(__file__).resolve().parent.parent
TEST_LEDGER = REPO / "results" / "test_ledger.jsonl"
CAMERA_FOLDS = REPO / "data" / "camera_folds.csv"

# ---------------------------------------------------------------------------------------
# Split. The official out-of-distribution split: 216 train / 32 val / 46 test cameras.
# ---------------------------------------------------------------------------------------
SPLIT = "official_ood"
N_CV_FOLDS = 5                # camera-wise cross-validation inside the training cameras
CV_SEED = 0

# ---------------------------------------------------------------------------------------
# Animal crop. Most confident MegaDetector v5a animal box with confidence >= 0.2, made
# square in pixels, 15% padding per side, shifted to stay inside the frame, cut from the
# original. Below 0.2 (or no box): the whole frame.
# ---------------------------------------------------------------------------------------
DETECTOR = "md5a"
DETECTIONS = GROUP / "pdelap-background-20261018" / "data" / "md-v5a.json"
MIN_DETECTION_CONFIDENCE = 0.2
CROP_PADDING = 0.15
CROP_RULE = ("most confident MegaDetector v5a animal box if conf >= 0.2, squared in pixels, "
             "padded 15% per side, shifted inside the frame, cut from the original; else whole frame")

# ---------------------------------------------------------------------------------------
# Scoring. Macro F1 over the species present in the evaluated set, per image and after
# the detector-weighted burst vote.
# ---------------------------------------------------------------------------------------
N_SPECIES = 60
