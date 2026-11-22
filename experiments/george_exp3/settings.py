"""Where everything lives, and every fixed setting behind the numbers on the slides."""
from dataclasses import dataclass
from pathlib import Path

N_SPECIES = 60

# ---------------------------------------------------------------------------------------
# Paths (vinzant)
# ---------------------------------------------------------------------------------------
GROUP = Path("/zfs/ai4good/groups/camera-traps")
ORIGINALS = GROUP / "raw" / "train"                                      # full-resolution photos
MY_FOLDER = GROUP / "users" / "george"
CHECKPOINTS = MY_FOLDER / "checkpoints"
RESULTS = MY_FOLDER / "exp3_results"

# MegaDetector boxes for every photo. v5a was run by Pia; the compact v6 by followup/detect.py.
DETECTIONS = {
    "md5a": GROUP / "pdelap-background-20261018" / "data" / "md-v5a.json",
    "md6c": MY_FOLDER / "detections" / "md6c.json",
}

# Our two backbones, and the name each has in the feature folders. The others are only used
# in the efficiency study (followup/); none of them was trained on iWildCam.
BACKBONES = {"resnet50": "resnet50", "dinov2l": "dinov2_vitl14_reg",
             "dinov2b": "dinov2_vitb14_reg", "dinov2s": "dinov2_vits14_reg",
             "mobilenetv3": "mobilenet_v3_large"}
DINOV2_REPO = "facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8"
IMAGE_SIZE = 448


def feature_folder(backbone: str, view: str = "crop", test: bool = False,
                   size: int = IMAGE_SIZE, detector: str = "md5a") -> Path:
    """Folder holding the saved features of one backbone, view, input size, detector and set
    of cameras. Experiment 3's own folders keep their original names."""
    if size != IMAGE_SIZE or detector != "md5a" or view == "whole" or backbone not in ("resnet50", "dinov2l"):
        name = f"{backbone}_{size}_{view}" + ("" if detector == "md5a" else f"_{detector}")
        return MY_FOLDER / "features_eff" / (name + ("_test" if test else ""))
    name = f"{BACKBONES[backbone]}_{IMAGE_SIZE}_crop"
    if view != "crop":
        name += f"_{view}"
    if not test and view == "crop":
        return GROUP / "features" / name
    if not test:
        return MY_FOLDER / "features_views" / name
    if view == "crop":
        return MY_FOLDER / "features_test" / f"{name}_test"
    return MY_FOLDER / "features_test_views" / f"{name}_test"


# ---------------------------------------------------------------------------------------
# Cropping the animal
# ---------------------------------------------------------------------------------------
MIN_DETECTION_CONFIDENCE = 0.2   # below this, the whole photo is used

# The four views of step 3: padding around the square box, and whether to mirror it.
VIEWS = {
    "crop":  {"padding": 0.15, "mirror": False},
    "flip":  {"padding": 0.15, "mirror": True},
    "pad05": {"padding": 0.05, "mirror": False},
    "pad40": {"padding": 0.40, "mirror": False},
}

# ---------------------------------------------------------------------------------------
# Classifier (linear discriminant) for steps 1-3
# ---------------------------------------------------------------------------------------
# (shrinkage, camera weight) per step, backbone and test setting.
#   Step 1 (plain): OOD shrinkage chosen on held-out training cameras by Shayan's
#     val_compare.py; ID shrinkage chosen on random_burst's own validation part.
#   Steps 2 and 3 (camera-aware): both values chosen on held-out training cameras.
CLASSIFIER = {
    (1, "resnet50", "ood"): (0.03, 0),
    (1, "resnet50", "id"):  (0.01, 0),
    (1, "dinov2l",  "ood"): (0.3, 0),
    (1, "dinov2l",  "id"):  (0.03, 0),
    (2, "resnet50", "ood"): (0.03, 4),
    (2, "resnet50", "id"):  (0.03, 4),
    (2, "dinov2l",  "ood"): (0.1, 8),
    (2, "dinov2l",  "id"):  (0.1, 8),
}
for _backbone in ("resnet50", "dinov2l"):          # step 3 = step 2 on the averaged views
    for _test in ("ood", "id"):
        CLASSIFIER[(3, _backbone, _test)] = CLASSIFIER[(2, _backbone, _test)]

# Validation-camera scores (per photo, per burst) each step must reproduce before test.
EXPECTED_VALIDATION = {
    (1, "resnet50"): (0.5732, 0.7464),
    (1, "dinov2l"):  (0.7811, 0.9431),
    (2, "resnet50"): (0.5791, 0.7835),
    (2, "dinov2l"):  (0.7932, 0.9462),
    (3, "resnet50"): (0.5884, 0.7904),
    (3, "dinov2l"):  (0.7991, 0.9431),
}

# ---------------------------------------------------------------------------------------
# Fine-tuning (step 4)
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FineTuning:
    method: str            # "all layers" or "lora"
    epochs: int
    batch_size: int
    learning_rate: float   # backbone (all layers) or adapters (lora)
    head_learning_rate: float
    weight_decay: float
    balanced_loss: bool    # add log(species frequency) to the scores in the loss
    seeds: tuple
    lora_rank: int = 0
    adapter_strength: float = 1.0   # how much of the adapters' change is used at test time


FINE_TUNING = {
    "resnet50": FineTuning(method="all layers", epochs=6, batch_size=64, learning_rate=1e-5,
                           head_learning_rate=1e-5, weight_decay=0.1, balanced_loss=False,
                           seeds=(0, 1, 2)),
    "dinov2l": FineTuning(method="lora", epochs=3, batch_size=32, learning_rate=1e-4,
                          head_learning_rate=1e-5, weight_decay=0.01, balanced_loss=True,
                          seeds=(0,), lora_rank=8, adapter_strength=0.25),   # 0.25: best on validation
}
WARMUP_STEPS = 300

# Validation scores (per photo, per burst) of the saved fine-tuned weights, per seed.
EXPECTED_VALIDATION_FINETUNED = {
    ("resnet50", 0): (0.6293, 0.8427),
    ("resnet50", 1): (0.6166, 0.8568),
    ("resnet50", 2): (0.6104, 0.8285),
    ("dinov2l", 0): (0.7985, 0.9554),
}


def checkpoint_path(backbone: str, seed: int) -> Path:
    """The fine-tuned weights used on the slides."""
    if backbone == "resnet50":
        return CHECKPOINTS / f"resnet50_lpft_s{seed}_best.pt"
    return CHECKPOINTS / f"dinov2l_lora_s{seed}_final.pt"
