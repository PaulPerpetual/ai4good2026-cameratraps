"""Photos, MegaDetector crops, saved features and the two ways of splitting cameras."""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

from iwildcam.data import IMAGENET_MEAN, IMAGENET_STD, load_fold, load_task
from settings import DETECTIONS, IMAGE_SIZE, MIN_DETECTION_CONFIDENCE, ORIGINALS, VIEWS, feature_folder

# ---------------------------------------------------------------------------------------
# Which photos
# ---------------------------------------------------------------------------------------
COLUMNS = ["file_name", "location", "category", "is_empty", "burst_id", "datetime", "hour",
           "official_ood", "y"]


def labelled_photos(test: bool) -> pd.DataFrame:
    """Labelled photos of the training and validation cameras, or of the test cameras."""
    photos = load_task(require_files=False).df
    if test:
        photos = photos[photos.official_ood == "test"]
    else:
        photos = photos[photos.official_ood.isin(["train", "val"])]
    return photos[COLUMNS].sort_values("file_name").reset_index(drop=True)


def random_burst_split(photos: pd.DataFrame) -> pd.Series:
    """train / val / test of each photo in the in-distribution split, where whole bursts are
    assigned at random, so the same cameras appear in training and test."""
    task = load_task(require_files=False)
    split_of = {}
    for part, rows in load_fold(task, "random_burst").items():
        for name in task.df.file_name.to_numpy()[rows]:
            split_of[name] = part
    return photos.file_name.map(split_of)


# ---------------------------------------------------------------------------------------
# Saved features
# ---------------------------------------------------------------------------------------

def load_features(backbone: str, test: bool, views=("crop",), size: int = IMAGE_SIZE,
                  detector: str = "md5a") -> tuple[pd.DataFrame, np.ndarray]:
    """Labelled photos and their features. With several views, the views are averaged."""
    photos, features = _load_one_view(feature_folder(backbone, views[0], test, size, detector))
    for view in views[1:]:
        other_photos, other_features = _load_one_view(feature_folder(backbone, view, test, size, detector))
        assert (other_photos.file_name == photos.file_name).all(), f"{view} lists other photos"
        features = features + other_features
    return photos, features / len(views)


def _load_one_view(folder: Path) -> tuple[pd.DataFrame, np.ndarray]:
    photos = pd.read_csv(folder / "index.csv.gz")
    features = np.load(folder / "features.npy", mmap_mode="r")
    keep = ~photos.is_empty.astype(str).str.lower().eq("true").to_numpy()
    if (folder / "status.npy").exists():               # older folders also list empty frames
        keep &= np.load(folder / "status.npy") == 1    # and photos that failed to open
    return photos[keep].reset_index(drop=True), np.asarray(features[keep], dtype=np.float32)


# ---------------------------------------------------------------------------------------
# MegaDetector crops
# ---------------------------------------------------------------------------------------

@lru_cache(maxsize=2)
def _best_animal_boxes(detector: str = "md5a") -> dict:
    """file name -> (confidence, x, y, width, height) of its most confident animal box."""
    best = {}
    for photo in json.loads(DETECTIONS[detector].read_text())["images"]:
        animals = [d for d in photo.get("detections") or [] if d["category"] == "1"]
        if animals:
            top = max(animals, key=lambda d: d["conf"])
            best[photo["file"]] = (top["conf"], *top["bbox"])
    return best


def detector_confidence(file_names, detector: str = "md5a") -> np.ndarray:
    """How sure MegaDetector is that each photo shows an animal (0 if it found none)."""
    boxes = _best_animal_boxes(detector)
    return np.array([boxes.get(name, (0.0,))[0] for name in file_names], dtype=np.float64)


def animal_box_area(file_names, detector: str = "md5a") -> np.ndarray:
    """Area of the most confident animal box as a fraction of the frame (0 if none found)."""
    boxes = _best_animal_boxes(detector)
    return np.array([boxes[name][3] * boxes[name][4] if name in boxes else 0.0 for name in file_names])


def crop_boxes(file_names, padding: float, detector: str = "md5a") -> np.ndarray:
    """Crop of each photo as fractions of the frame (left, top, right, bottom).

    The most confident animal box, made square, padded on each side and shifted to stay
    inside the frame. The whole photo when MegaDetector is not confident enough.
    """
    task = load_task(require_files=False)
    sizes = pd.concat([task.df, task.empty])[["file_name", "width", "height"]]
    sizes = sizes.drop_duplicates("file_name").set_index("file_name").loc[list(file_names)]
    boxes = _best_animal_boxes(detector)
    return np.array([_square_box(boxes.get(name), width, height, padding)
                     for name, width, height in zip(file_names, sizes.width, sizes.height)])


def _square_box(detection, width, height, padding):
    if detection is None or detection[0] < MIN_DETECTION_CONFIDENCE:
        return 0.0, 0.0, 1.0, 1.0
    _, x, y, w, h = detection
    centre_x, centre_y = (x + w / 2) * width, (y + h / 2) * height
    side = min(max(w * width, h * height) * (1 + 2 * padding), width, height)
    left = min(max(centre_x - side / 2, 0), width - side)
    top = min(max(centre_y - side / 2, 0), height - side)
    return left / width, top / height, (left + side) / width, (top + side) / height


# ---------------------------------------------------------------------------------------
# Images for the networks
# ---------------------------------------------------------------------------------------
_TO_NETWORK_INPUT = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                     v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]


def _resize(size):
    return v2.Resize((size, size), interpolation=v2.InterpolationMode.BILINEAR, antialias=True)


def plain(size: int = IMAGE_SIZE):
    """Squash to size x size and normalise as the backbones expect."""
    return v2.Compose([_resize(size), *_TO_NETWORK_INPUT])


PLAIN = plain()
AUGMENTED = v2.Compose([_resize(IMAGE_SIZE), v2.RandomHorizontalFlip(), v2.RandomGrayscale(p=0.25),
                        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                        *_TO_NETWORK_INPUT])


class Crops(Dataset):
    """The animal crop of each photo, cut from the full-resolution original. With view
    "whole", the whole photo instead.

    Returns (image, species, weight); species and weight are -1 and 1 when not given.
    """

    def __init__(self, file_names, view="crop", transform=PLAIN, species=None, weights=None, detector="md5a"):
        self.file_names = list(file_names)
        if view == "whole":
            self.boxes = np.tile([0.0, 0.0, 1.0, 1.0], (len(self.file_names), 1))
            self.mirror = False
        else:
            self.boxes = crop_boxes(self.file_names, VIEWS[view]["padding"], detector)
            self.mirror = VIEWS[view]["mirror"]
        self.transform = transform
        self.species = species
        self.weights = weights

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, i):
        photo = Image.open(ORIGINALS / self.file_names[i]).convert("RGB")
        width, height = photo.size
        left, top, right, bottom = self.boxes[i]
        crop = photo.crop((round(left * width), round(top * height),
                           round(right * width), round(bottom * height)))
        if self.mirror:
            crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
        species = -1 if self.species is None else int(self.species[i])
        weight = 1.0 if self.weights is None else float(self.weights[i])
        return self.transform(crop), species, weight
