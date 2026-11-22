
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ==========================================
# 1. Configuration
# ==========================================

ROOT = Path("/zfs/ai4good/groups/camera-traps")
WORK = ROOT / "users/junwu/copypaste"
DATA = ROOT / "camera-trap/data"
IMAGES = DATA / "images_h448"

OUT = WORK / "generated_v5"
OUT.mkdir(parents=True, exist_ok=True)

NUM_IMAGES = 20
MAX_ATTEMPTS = 500
DEVICE = "cpu"  # Switch to cuda only in an allocated GPU session

random.seed(42)

# ==========================================
# 2. Load animal candidates and metadata
# ==========================================

with open(WORK / "train_animal_candidates.json") as f:
    animals = json.load(f)

meta = pd.read_csv(DATA / "metadata.csv.gz")
meta["image_id"] = meta["image_id"].astype(str)
meta = meta.drop_duplicates("image_id").set_index("image_id")

# Find the camera/location column automatically.
camera_candidates = [
    "location", "location_id",
    "camera_id", "camera", "camera_location"
]

camera_col = next(
    (c for c in camera_candidates if c in meta.columns),
    None
)

if camera_col is None:
    raise RuntimeError(
        "Cannot identify camera column. Available columns: "
        + str(meta.columns.tolist())
    )

print("Camera column:", camera_col, flush=True)

def camera_of(image_id):
    if image_id not in meta.index:
        return None

    value = meta.at[image_id, camera_col]

    if pd.isna(value):
        return None

    return str(value)


def image_path(image_id):
    p = IMAGES / f"{image_id}.jpg"

    if p.is_file():
        return p

    if image_id in meta.index and "file_name" in meta.columns:
        filename = str(meta.at[image_id, "file_name"])
        q = IMAGES / filename

        if q.is_file():
            return q

    return None


# ==========================================
# 3. Genuine empty training backgrounds
# ==========================================

empty_mask = (
    meta["is_empty"]
    .astype(str)
    .str.lower()
    .eq("true")
)

if "official_ood" not in meta.columns:
    raise RuntimeError("Missing official_ood column")

train_mask = (
    meta["official_ood"]
    .astype(str)
    .str.lower()
    .eq("train")
)

background_ids = meta.index[
    empty_mask & train_mask
].tolist()

backgrounds = []

for image_id in background_ids:
    cam = camera_of(image_id)
    path = image_path(image_id)

    if cam is not None and path is not None:
        backgrounds.append((image_id, cam, path))

print("Verified empty backgrounds:", len(backgrounds), flush=True)

if not backgrounds:
    raise RuntimeError("No usable empty training backgrounds")


# ==========================================
# 4. Prepare animal sources
# ==========================================

sources = []

for animal in animals:
    image_id = str(animal["image_id"])
    cam = camera_of(image_id)
    path = image_path(image_id)
    boxes = animal.get("boxes", [])

    if cam is None or path is None:
        continue

    if len(boxes) != 1:
        continue

    if float(boxes[0].get("conf", 0)) < 0.5:
        continue

    sources.append((animal, cam, path))

print("Usable animal sources:", len(sources), flush=True)

if not sources:
    raise RuntimeError("No usable animal sources")


# ==========================================
# 5. Load SAM 2
# ==========================================

print("Loading SAM 2...", flush=True)

model = build_sam2(
    "configs/sam2.1/sam2.1_hiera_t.yaml",
    str(WORK / "checkpoints/sam2.1_hiera_tiny.pt"),
    device=DEVICE
)

predictor = SAM2ImagePredictor(model)

print("SAM 2 ready!", flush=True)


# ==========================================
# 6. Bounding box conversion
# ==========================================

def convert_box(bbox, width, height):
    x, y, w, h = map(float, bbox)

    return np.array([
        max(0, x * width),
        max(0, y * height),
        min(width, (x + w) * width),
        min(height, (y + h) * height)
    ], dtype=np.float32)


# ==========================================
# 7. Generate cross-camera Copy-Paste
# ==========================================

results = []
attempts = 0

while len(results) < NUM_IMAGES and attempts < MAX_ATTEMPTS:
    attempts += 1

    animal, source_camera, source_path = random.choice(sources)
    source_id = str(animal["image_id"])

    target_id, target_camera, target_path = random.choice(backgrounds)

    # Strictly different cameras.
    if source_camera == target_camera:
        continue

    try:
        with Image.open(source_path) as im:
            src = im.convert("RGB")

        with Image.open(target_path) as im:
            dst = im.convert("RGB")

        detection = animal["boxes"][0]
        bbox = convert_box(
            detection["bbox"],
            src.width,
            src.height
        )

        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue

        # SAM 2 segmentation
        with torch.inference_mode():
            predictor.set_image(np.asarray(src))

            masks, scores, _ = predictor.predict(
                box=bbox,
                multimask_output=False
            )

        score = float(scores[0])

        if score < 0.7:
            continue

        mask = masks[0].astype(np.uint8)

        if not mask.any():
            continue

        # Reject implausibly large masks.
        mask_ratio = float(mask.mean())

        if not (0.001 < mask_ratio < 0.65):
            continue

        alpha = Image.fromarray(mask * 255, mode="L")
        bounds = alpha.getbbox()

        if bounds is None:
            continue

        crop = src.crop(bounds)
        alpha = alpha.crop(bounds)

        # Reduce excessive animal size.
        max_width = max(1, int(dst.width * 0.35))
        max_height = max(1, int(dst.height * 0.40))

        crop.thumbnail(
            (max_width, max_height),
            Image.Resampling.LANCZOS
        )

        alpha = alpha.resize(
            crop.size,
            Image.Resampling.BILINEAR
        )

        # Slight edge feathering.
        alpha = alpha.filter(
            ImageFilter.GaussianBlur(radius=1.0)
        )

        if crop.width >= dst.width or crop.height >= dst.height:
            continue

        # Place animal in lower portion of scene.
        px = random.randint(0, dst.width - crop.width)

        ground_y = random.randint(
            int(dst.height * 0.70),
            int(dst.height * 0.95)
        )

        py = max(
            0,
            min(ground_y - crop.height, dst.height - crop.height)
        )

        # Composite with a transparent mask.
        dst.paste(crop, (px, py), alpha)

        filename = f"v5_{len(results):04d}.jpg"

        dst.save(
            OUT / filename,
            quality=92
        )

        results.append({
            "filename": filename,
            "species": animal["species"],
            "source_image": source_id,
            "background_image": target_id,
            "source_camera": source_camera,
            "target_camera": target_camera,
            "sam2_score": score,
            "mask_area_ratio": mask_ratio
        })

        print(
            f"Generated {len(results)}/{NUM_IMAGES}",
            flush=True
        )

    except Exception as e:
        print("Skipped:", str(e), flush=True)


# ==========================================
# 8. Save labels and provenance
# ==========================================

with open(OUT / "labels.json", "w") as f:
    json.dump(results, f, indent=2)

print("================================")
print("Generated:", len(results))
print("Attempts:", attempts)
print("Saved to:", OUT)
print("================================")
