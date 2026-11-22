"""Run the compact MegaDetector (v6, YOLOv10-compact, 2.3M parameters) over every labelled photo.

    ../.venv-megadetector/bin/python detect.py
    ../.venv-megadetector/bin/python detect.py --limit 20 --device cpu --out /tmp/check.json   # quick check

Uses its own environment (PytorchWildlife). Writes the boxes in the same format as Pia's
MegaDetector v5a file, so exp3/data.py reads either: settings.DETECTIONS["md6c"].
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from PytorchWildlife.models.detection import MegaDetectorV6
from torch.utils.data import DataLoader, Dataset

GROUP = Path("/zfs/ai4good/groups/camera-traps")
ORIGINALS = GROUP / "raw" / "train"
PHOTO_LISTS = [GROUP / "features" / "resnet50_448_crop" / "index.csv.gz",                 # train + val cameras
               GROUP / "users" / "george" / "features_test" / "resnet50_448_crop_test" / "index.csv.gz"]
OUT = GROUP / "users" / "george" / "detections" / "md6c.json"
KEEP_ABOVE = 0.01          # low, so the burst vote still sees weak detections, as with v5a
CATEGORY = {0: "1", 1: "2", 2: "3"}   # MegaDetector's file format: 1 animal, 2 person, 3 vehicle


class Photos(Dataset):
    def __init__(self, names):
        self.names = names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        return self.names[i], np.array(Image.open(ORIGINALS / self.names[i]).convert("RGB"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, help="only the first N photos (to check the script)")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    names = labelled_photos()[:args.limit]
    detector = MegaDetectorV6(device=args.device, pretrained=True, version="MDV6-yolov10-c")
    loader = DataLoader(Photos(names), batch_size=None, num_workers=16, collate_fn=keep_as_array)
    images = []
    for i, (name, image) in enumerate(loader):
        result = detector.single_image_detection(image, img_path=name, det_conf_thres=KEEP_ABOVE)
        found = result["detections"]
        images.append({"file": name, "detections": [
            {"category": CATEGORY[int(c)], "conf": float(conf),
             "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]}
            for c, conf, (x1, y1, x2, y2) in zip(found.class_id, found.confidence, result["normalized_coords"])]})
        if i % 10000 == 0:
            print(f"{i:,} / {len(names):,}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"info": {"detector": "MegaDetector v6, MDV6-yolov10-c, 1280 px",
                                        "kept_above": KEEP_ABOVE},
                               "detection_categories": {"1": "animal", "2": "person", "3": "vehicle"},
                               "images": images}))
    print(f"wrote {args.out}")


def keep_as_array(sample):
    """The detector wants a NumPy image; the DataLoader would otherwise make it a tensor."""
    return sample


def labelled_photos() -> list[str]:
    tables = [pd.read_csv(path) for path in PHOTO_LISTS]
    photos = pd.concat(tables)
    labelled = ~photos.is_empty.astype(str).str.lower().eq("true")
    return sorted(photos.file_name[labelled].unique())


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
