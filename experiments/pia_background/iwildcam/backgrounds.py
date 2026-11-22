"""Background ablations using frozen detector/SAM artifacts and unlabelled empties."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

MODES = ("whole", "crop", "neutral", "masked_crop", "source", "target")


def fingerprint_artifacts(cfg):
    """Include preprocessing content in experiment identity, even if a path is reused."""
    cfg = dict(cfg)
    method = cfg.get("background", {})
    if method.get("mode", "whole") != "whole":
        path = Path(method["manifest"])
        cfg["background_manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return cfg


def background_pool(task, split, mode):
    """Select empties by their shipped split; never consume labelled target images."""
    if mode not in ("source", "target"):
        return task.empty.iloc[:0]
    if split != "official_ood":
        raise ValueError("Background replacement requires official_ood camera-disjoint splits")
    return task.empty[task.empty[split] == ("train" if mode == "source" else "test")]


def composite(foreground, background, position=None):
    """Fit an RGBA foreground inside a background without changing aspect ratio."""
    foreground = foreground.copy()
    foreground.thumbnail(background.size, Image.Resampling.LANCZOS)
    w, h = foreground.size
    if position is None:
        position = ((background.width - w) // 2, (background.height - h) // 2)
    result = background.convert("RGB").copy()
    result.paste(foreground, position, foreground.getchannel("A"))
    return result


class BackgroundProcessor:
    """PIL preprocessing before normal resize/augmentation. Randomness uses torch seeds."""

    def __init__(self, task, split, method, training):
        self.mode = method.get("mode", "whole")
        if self.mode not in MODES:
            raise ValueError(f"Unknown background mode: {self.mode}")
        self.training = training
        self.probability = float(method.get("probability", 0.5))
        if not 0 <= self.probability <= 1:
            raise ValueError("background.probability must be between 0 and 1")
        self.root = Path(method.get("manifest", ".")).resolve().parent
        self.records = {}
        if self.mode != "whole":
            manifest = json.loads(Path(method["manifest"]).read_text())
            if self.mode in ("neutral", "masked_crop", "source", "target") and not manifest.get("sam_model"):
                raise ValueError("This experiment requires a manifest prepared with SAM")
            self.records = manifest["images"]
            missing = set(task.df.file_name) - self.records.keys()
            if missing:
                raise ValueError(f"Manifest has no record for {len(missing)} images; rebuild it")
            for record in self.records.values():
                if record.get("foreground") and not (self.root / record["foreground"]).is_file():
                    raise FileNotFoundError(self.root / record["foreground"])
        self.image_dir = task.image_dir
        pool = background_pool(task, split, self.mode)
        # Uniform camera sampling prevents prolific cameras dominating the augmentation.
        self.groups = [g.file_name.tolist() for _, g in pool.groupby("location")]
        self.cameras = sorted(int(c) for c in pool.location.unique())
        if training and self.mode in ("source", "target") and not self.groups:
            raise ValueError(f"No available empty frames for {self.mode} backgrounds")

    def __call__(self, name, image):
        import torch

        if self.mode == "whole" or (not self.training and self.mode in ("source", "target")):
            return image
        record = self.records[name]
        box = record.get("box")
        if box is None:
            return image  # Shared fallback: never drop hard images from evaluation.
        x, y, w, h = box
        bounds = (int(x * image.width), int(y * image.height),
                  max(int(x * image.width) + 1, int((x + w) * image.width)),
                  max(int(y * image.height) + 1, int((y + h) * image.height)))
        if self.mode == "crop":
            return image.crop(bounds)
        if self.mode == "masked_crop":
            crop = image.crop(bounds)
            if not record.get("foreground"):
                return crop  # Preserve crop geometry if SAM failed.
            with Image.open(self.root / record["foreground"]) as asset:
                alpha = asset.getchannel("A")
            if alpha.size != crop.size:
                raise ValueError(f"SAM mask and crop sizes differ for {name}; use the preparation image resolution")
            # Keep the same box, scale, position, and foreground RGB as the crop baseline.
            return Image.composite(crop, Image.new("RGB", crop.size, (127, 127, 127)), alpha)
        if self.mode in ("source", "target") and torch.rand(()).item() >= self.probability:
            return image
        if not record.get("foreground"):
            return image  # Failed/empty SAM mask, recorded during preparation.
        with Image.open(self.root / record["foreground"]) as asset:
            foreground = asset.convert("RGBA")
        if self.mode == "neutral":
            return composite(foreground, Image.new("RGB", image.size, (127, 127, 127)))
        group = self.groups[torch.randint(len(self.groups), ()).item()]
        name = group[torch.randint(len(group), ()).item()]
        with Image.open(self.image_dir / name) as asset:
            background = asset.convert("RGB").resize(image.size)
        # Match original relative box size, with random placement on the new camera.
        foreground = foreground.resize((bounds[2] - bounds[0], bounds[3] - bounds[1]))
        position = (torch.randint(image.width - foreground.width + 1, ()).item(),
                    torch.randint(image.height - foreground.height + 1, ()).item())
        return composite(foreground, background, position)


def coverage_scores(task, fold, pred, true):
    """Separate test cameras with and without accessible empty reference images."""
    from .train_utils import per_camera_scores

    cameras = task.camera[fold["test"]]
    available = set(task.empty.location)
    reports = {}
    from sklearn.metrics import f1_score, accuracy_score
    for name, mask in (("with_empty", np.isin(cameras, list(available))),
                       ("without_empty", ~np.isin(cameras, list(available)))):
        reports[name] = {"n": int(mask.sum()), "n_cameras": int(len(np.unique(cameras[mask])))}
        if mask.any():
            reports[name].update(
                macro_f1_present=float(f1_score(true[mask], pred[mask], average="macro",
                                               labels=np.unique(true[mask]), zero_division=0)),
                accuracy=float(accuracy_score(true[mask], pred[mask])),
                per_camera=per_camera_scores(pred[mask], true[mask], cameras[mask]))
    return reports
