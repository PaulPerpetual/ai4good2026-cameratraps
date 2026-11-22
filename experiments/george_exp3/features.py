"""Run a frozen backbone over every labelled photo's animal crop and save the features.

    python features.py --backbone dinov2l --view crop            # training + validation cameras
    python features.py --backbone dinov2l --view crop --test     # test cameras
    python features.py --backbone dinov2s --view whole --size 224  # other settings (followup/)

Runs once per backbone, view and set of cameras; every other script reads the saved files.
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import Crops, labelled_photos, plain
from models import Backbone
from settings import BACKBONES, DETECTIONS, IMAGE_SIZE, VIEWS, feature_folder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, choices=list(BACKBONES))
    parser.add_argument("--view", default="crop", choices=[*VIEWS, "whole"])
    parser.add_argument("--size", type=int, default=IMAGE_SIZE, help="input side in pixels")
    parser.add_argument("--detector", default="md5a", choices=list(DETECTIONS), help="whose boxes to crop to")
    parser.add_argument("--test", action="store_true", help="the test cameras instead of training + validation")
    parser.add_argument("--sample", type=int, help="only this many random photos (to check the script)")
    parser.add_argument("--out", type=Path, help="default: the folder the other scripts read")
    parser.add_argument("--share-gpu", action="store_true", help="run even if someone else is on the GPU")
    args = parser.parse_args()

    out = args.out or feature_folder(args.backbone, args.view, args.test, args.size, args.detector)
    if (out / "features.npy").exists():
        raise SystemExit(f"{out} already has features")
    if not args.share_gpu:
        stop_if_gpu_busy()

    photos = labelled_photos(test=args.test)
    if args.sample:
        photos = photos.sample(args.sample, random_state=0).sort_values("file_name").reset_index(drop=True)
    print(f"{len(photos):,} photos from {photos.location.nunique()} cameras -> {out}")

    crops = Crops(photos.file_name, args.view, transform=plain(args.size), detector=args.detector)
    features = extract(Backbone(args.backbone).cuda().eval(), crops)

    out.mkdir(parents=True)
    photos.to_csv(out / "index.csv.gz", index=False)
    np.save(out / "features.npy", features)
    (out / "config.json").write_text(json.dumps({
        "backbone": args.backbone, "view": args.view, **VIEWS.get(args.view, {}), "image_size": args.size,
        "detector": None if args.view == "whole" else args.detector,
        "cameras": "test" if args.test else "train and val", "photos": len(photos)}, indent=2))


@torch.inference_mode()
def extract(backbone, crops) -> np.ndarray:
    loader = DataLoader(crops, batch_size=64, num_workers=16, pin_memory=True)
    batches = []
    for images, _, _ in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            batches.append(backbone(images.cuda(non_blocking=True)).float().cpu().numpy())
    features = np.concatenate(batches)
    assert np.isfinite(features).all()
    return features.astype(np.float16)


def stop_if_gpu_busy():
    """The GPU is shared by the group: do not start on top of someone else's job."""
    running = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout.split()
    if running:
        raise SystemExit(f"the GPU is in use (processes {running}); try later or pass --share-gpu")


if __name__ == "__main__":
    main()
