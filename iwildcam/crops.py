"""MegaDetector crops: the boxes, the detector's confidence, and a dataset of cropped images.

The crop rule is fixed in `protocol.py`. Written by George for Experiment 3 (exp3/data.py);
the rule itself matches Paul's linprobe.py, so every cached feature uses the same crop.
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from iwildcam.data import load_task
from iwildcam.protocol import CROP_PADDING, DETECTIONS, MIN_DETECTION_CONFIDENCE, ORIGINALS


@lru_cache(maxsize=1)
def best_animal_boxes() -> dict:
    """file name -> (confidence, x, y, width, height) of its most confident animal box,
    in fractions of the frame."""
    best = {}
    for photo in json.loads(DETECTIONS.read_text())["images"]:
        animals = [d for d in photo.get("detections") or [] if d["category"] == "1"]
        if animals:
            top = max(animals, key=lambda d: d["conf"])
            best[photo["file"]] = (top["conf"], *top["bbox"])
    return best


def detector_confidence(file_names) -> np.ndarray:
    """How sure MegaDetector is that each photo shows an animal (0 if it found none).
    Used as the vote weight in the burst vote."""
    boxes = best_animal_boxes()
    return np.array([boxes.get(name, (0.0,))[0] for name in file_names], dtype=np.float64)


def crop_boxes(file_names, padding: float = CROP_PADDING) -> np.ndarray:
    """(left, top, right, bottom) of each photo's crop, as fractions of the frame."""
    task = load_task(require_files=False)
    sizes = pd.concat([task.df, task.empty])[["file_name", "width", "height"]]
    sizes = sizes.drop_duplicates("file_name").set_index("file_name").loc[list(file_names)]
    boxes = best_animal_boxes()
    return np.array([square_box(boxes.get(name), width, height, padding)
                     for name, width, height in zip(file_names, sizes.width, sizes.height)])


def square_box(detection, width, height, padding=CROP_PADDING):
    if detection is None or detection[0] < MIN_DETECTION_CONFIDENCE:
        return 0.0, 0.0, 1.0, 1.0
    _, x, y, w, h = detection
    centre_x, centre_y = (x + w / 2) * width, (y + h / 2) * height
    side = min(max(w * width, h * height) * (1 + 2 * padding), width, height)
    left = min(max(centre_x - side / 2, 0), width - side)
    top = min(max(centre_y - side / 2, 0), height - side)
    return left / width, top / height, (left + side) / width, (top + side) / height


class Crops(Dataset):
    """The animal crop of each photo cut from the full-resolution original (or the whole
    photo with view="whole"), passed through `transform`. Returns (image, index); index
    lets the caller see which rows failed to decode."""

    def __init__(self, file_names, transform, view: str = "crop"):
        self.file_names = list(file_names)
        self.boxes = (np.tile([0.0, 0.0, 1.0, 1.0], (len(self.file_names), 1)) if view == "whole"
                      else crop_boxes(self.file_names))
        self.transform = transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, i):
        photo = Image.open(ORIGINALS / self.file_names[i]).convert("RGB")
        width, height = photo.size
        left, top, right, bottom = self.boxes[i]
        crop = photo.crop((round(left * width), round(top * height),
                           round(right * width), round(bottom * height)))
        return self.transform(crop), i
