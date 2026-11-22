
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from PIL import Image, ImageFilter
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ==================================================
# 1. SETTINGS
# ==================================================

ROOT = Path("/zfs/ai4good/groups/camera-traps")
WORK = ROOT / "users/junwu/copypaste"
DATA = ROOT / "camera-trap/data"

IMAGE_DIR = DATA / "images_h448"
METADATA = DATA / "metadata.csv.gz"
ANIMALS_JSON = WORK / "train_animal_candidates.json"

CHECKPOINT = WORK / "checkpoints/sam2.1_hiera_tiny.pt"
CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"

OUTPUT = WORK / "generated_v6_500"

NUM_IMAGES = 500
MAX_ATTEMPTS = 12000
SEED = 42

# CPU avoids using an unallocated cluster GPU.
DEVICE = "cuda"

# Only use official training images.
ALLOWED_SPLIT = "train"

MIN_DETECTION_CONF = 0.5
MIN_SAM_SCORE = 0.70

random.seed(SEED)

OUTPUT.mkdir(parents=True, exist_ok=True)

print("Starting Copy-Paste v6", flush=True)


# ==================================================
# 2. LOAD METADATA
# ==================================================

meta = pd.read_csv(METADATA)

meta["image_id"] = meta["image_id"].astype(str)

meta = (
    meta.drop_duplicates(subset="image_id")
    .set_index("image_id")
)

camera_candidates = [
    "location",
    "location_id",
    "camera_id",
    "camera",
    "camera_location",
]

camera_col = next(
    (col for col in camera_candidates if col in meta.columns),
    None
)

if camera_col is None:
    raise RuntimeError(
        "Camera column not found. Columns: "
        + str(meta.columns.tolist())
    )

print("Camera column:", camera_col, flush=True)

required_columns = ["is_empty", "official_ood"]

for col in required_columns:
    if col not in meta.columns:
        raise RuntimeError(f"Missing metadata column: {col}")


def camera_of(image_id):
    if image_id not in meta.index:
        return None

    value = meta.at[image_id, camera_col]

    if pd.isna(value):
        return None

    return str(value)


def image_path(image_id):
    path = IMAGE_DIR / f"{image_id}.jpg"

    if path.is_file():
        return path

    if image_id in meta.index and "file_name" in meta.columns:
        filename = str(meta.at[image_id, "file_name"])

        alternative = IMAGE_DIR / filename

        if alternative.is_file():
            return alternative

    return None


def in_training_split(image_id):
    if image_id not in meta.index:
        return False

    return (
        str(meta.at[image_id, "official_ood"]).lower()
        == ALLOWED_SPLIT
    )


# ==================================================
# 3. VERIFIED EMPTY BACKGROUNDS
# ==================================================

empty_mask = (
    meta["is_empty"]
    .astype(str)
    .str.lower()
    .eq("true")
)

training_mask = (
    meta["official_ood"]
    .astype(str)
    .str.lower()
    .eq(ALLOWED_SPLIT)
)

background_ids = meta.index[
    empty_mask & training_mask
].tolist()

backgrounds = []

for image_id in background_ids:
    camera = camera_of(image_id)
    path = image_path(image_id)

    if camera is None or path is None:
        continue

    backgrounds.append({
        "image_id": image_id,
        "camera": camera,
        "path": path
    })

print("Usable empty backgrounds:", len(backgrounds), flush=True)

if not backgrounds:
    raise RuntimeError("No valid empty backgrounds found")


# ==================================================
# 4. ANIMAL SOURCES
# ==================================================

with open(ANIMALS_JSON) as f:
    animals = json.load(f)

sources = []

for animal in animals:
    image_id = str(animal["image_id"])

    if not in_training_split(image_id):
        continue

    camera = camera_of(image_id)
    path = image_path(image_id)

    if camera is None or path is None:
        continue

    boxes = animal.get("boxes", [])

    # Prefer images with exactly one detected animal.
    if len(boxes) != 1:
        continue

    detection = boxes[0]

    if float(detection.get("conf", 0)) < MIN_DETECTION_CONF:
        continue

    bbox = detection.get("bbox", [])

    if len(bbox) != 4:
        continue

    sources.append({
        "image_id": image_id,
        "species": animal["species"],
        "camera": camera,
        "path": path,
        "bbox": bbox,
        "detection_conf": float(detection["conf"])
    })

print("Usable animal sources:", len(sources), flush=True)

if not sources:
    raise RuntimeError("No animal sources found")


# ==================================================
# 5. LOAD SAM 2
# ==================================================

print("Loading SAM 2...", flush=True)

model = build_sam2(
    CONFIG,
    str(CHECKPOINT),
    device=DEVICE
)

predictor = SAM2ImagePredictor(model)

print("SAM 2 ready", flush=True)


# ==================================================
# 6. HELPER FUNCTIONS
# ==================================================

def convert_box(bbox, width, height):
    x, y, w, h = map(float, bbox)

    x1 = max(0, x * width)
    y1 = max(0, y * height)

    x2 = min(width, (x + w) * width)
    y2 = min(height, (y + h) * height)

    if x2 <= x1 or y2 <= y1:
        return None

    return np.array(
        [x1, y1, x2, y2],
        dtype=np.float32
    )


def segment_animal(image, bbox):
    box = convert_box(
        bbox,
        image.width,
        image.height
    )

    if box is None:
        return None

    with torch.inference_mode():
        predictor.set_image(np.asarray(image))

        masks, scores, _ = predictor.predict(
            box=box,
            multimask_output=False
        )

    score = float(scores[0])

    if score < MIN_SAM_SCORE:
        return None

    mask = masks[0].astype(np.uint8)

    if not mask.any():
        return None

    ratio = float(mask.mean())

    if ratio < 0.001 or ratio > 0.60:
        return None

    # Reject masks spreading far beyond the detector box.
    box_area = (
        (box[2] - box[0])
        * (box[3] - box[1])
    )

    mask_area = float(mask.sum())

    if mask_area > 1.5 * box_area:
        return None

    alpha = Image.fromarray(
        mask * 255,
        mode="L"
    )

    bounds = alpha.getbbox()

    if bounds is None:
        return None

    crop = image.crop(bounds)
    alpha = alpha.crop(bounds)

    return crop, alpha, score, ratio


def resize_animal(crop, alpha, background):
    max_width = max(1, int(background.width * 0.35))
    max_height = max(1, int(background.height * 0.40))

    crop.thumbnail(
        (max_width, max_height),
        Image.Resampling.LANCZOS
    )

    alpha = alpha.resize(
        crop.size,
        Image.Resampling.BILINEAR
    )

    alpha = alpha.filter(
        ImageFilter.GaussianBlur(radius=1.0)
    )

    return crop, alpha


def choose_position(background, crop):
    if (
        crop.width >= background.width
        or crop.height >= background.height
    ):
        return None

    max_x = background.width - crop.width
    max_y = background.height - crop.height

    # Approximate ground-contact position.
    # This is a heuristic, not ground segmentation.
    min_ground = int(background.height * 0.70)
    max_ground = int(background.height * 0.95)

    for _ in range(50):
        px = random.randint(0, max_x)

        ground_y = random.randint(
            min_ground,
            max_ground
        )

        py = ground_y - crop.height

        if py < 0 or py > max_y:
            continue

        return px, py

    return None


def choose_background(source_camera):
    # Try only backgrounds from different cameras.
    for _ in range(100):
        target = random.choice(backgrounds)

        if target["camera"] != source_camera:
            return target

    return None


# ==================================================
# 7. GENERATE SYNTHETIC IMAGES
# ==================================================

results = []
attempts = 0

while (
    len(results) < NUM_IMAGES
    and attempts < MAX_ATTEMPTS
):
    attempts += 1

    source = random.choice(sources)

    target = choose_background(source["camera"])

    if target is None:
        continue

    try:
        with Image.open(source["path"]) as im:
            src = im.convert("RGB")

        with Image.open(target["path"]) as im:
            dst = im.convert("RGB")

        segmented = segment_animal(
            src,
            source["bbox"]
        )

        if segmented is None:
            continue

        crop, alpha, sam_score, mask_ratio = segmented

        crop, alpha = resize_animal(
            crop,
            alpha,
            dst
        )

        position = choose_position(dst, crop)

        if position is None:
            continue

        px, py = position

        # Copy only the segmented animal.
        dst.paste(crop, position, alpha)

        filename = f"v6_{len(results):04d}.jpg"

        dst.save(
            OUTPUT / filename,
            quality=92
        )

        results.append({
            "filename": filename,
            "species": source["species"],
            "source_image": source["image_id"],
            "background_image": target["image_id"],
            "source_camera": source["camera"],
            "target_camera": target["camera"],
            "source_detection_conf": source["detection_conf"],
            "sam2_score": sam_score,
            "mask_area_ratio": mask_ratio,
            "paste_box_xyxy": [
                px,
                py,
                px + crop.width,
                py + crop.height
            ],
            "official_split": ALLOWED_SPLIT
        })

        # Save progress after each successful image.
        with open(OUTPUT / "labels.json", "w") as f:
            json.dump(results, f, indent=2)

        print(
            f"Generated {len(results)}/{NUM_IMAGES}",
            flush=True
        )

    except Exception as e:
        print(
            f"Skipped attempt {attempts}: {e}",
            flush=True
        )


# ==================================================
# 8. FINAL REPORT
# ==================================================

print("--------------------------------")
print("Copy-Paste v6 completed")
print("Generated:", len(results))
print("Attempts:", attempts)
print("Output:", OUTPUT)
print("--------------------------------")
