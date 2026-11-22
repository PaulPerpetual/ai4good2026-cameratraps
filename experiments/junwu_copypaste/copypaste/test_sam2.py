
import os
import random
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# =====================================
# 1. Paths and settings
# =====================================

BASE = Path(__file__).resolve().parent
PROJECT = BASE.parents[2]

DATA_FILE = BASE / "train_animal_candidates.json"
RAW_DIR = PROJECT / "raw"
OUTPUT_DIR = BASE / "sam2_test"
CHECKPOINT = BASE / "checkpoints/sam2.1_hiera_tiny.pt"

CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"

NUM_IMAGES = 5
DEVICE = "cpu"  # Initial test without occupying GPU

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Project:", PROJECT)
print("Raw directory:", RAW_DIR)
print("Device:", DEVICE, flush=True)


# =====================================
# 2. Load animal candidates
# =====================================

with open(DATA_FILE, "r") as f:
    candidates = json.load(f)

print("Animal candidates:", len(candidates), flush=True)

random.seed(42)
sample_pool = random.sample(
    candidates,
    min(100, len(candidates))
)

selected = {}

for item in sample_pool:
    boxes = item.get("boxes", [])

    if not boxes:
        continue

    box = max(
        boxes,
        key=lambda b: float(b.get("conf", 0))
    )

    if len(box.get("bbox", [])) != 4:
        continue

    image_id = str(item["image_id"])

    selected[image_id] = {
        "species": item.get("species"),
        "box": box["bbox"]
    }

print("Searching for source images...", flush=True)


# =====================================
# 3. Find images automatically
# =====================================

found = {}

if not RAW_DIR.exists():
    raise FileNotFoundError(
        f"Raw data directory not found: {RAW_DIR}"
    )

for root, dirs, files in os.walk(RAW_DIR):
    for filename in files:
        path = Path(filename)

        if path.suffix.lower() not in (
            ".jpg", ".jpeg", ".png"
        ):
            continue

        image_id = path.stem

        if image_id in selected:
            found[image_id] = Path(root) / filename

    if len(found) >= NUM_IMAGES:
        break

print("Found images:", len(found), flush=True)

if not found:
    raise RuntimeError(
        "No matching images found in raw/. "
        "Check the source image directory."
    )


# =====================================
# 4. Load SAM 2
# =====================================

print("Loading SAM 2...", flush=True)

model = build_sam2(
    CONFIG,
    str(CHECKPOINT),
    device=DEVICE
)

predictor = SAM2ImagePredictor(model)

print("SAM 2 ready!", flush=True)


# =====================================
# 5. Convert MegaDetector box
# =====================================

def convert_box(bbox, width, height):
    x, y, w, h = map(float, bbox)

    # MegaDetector uses normalized XYWH boxes.
    x1 = max(0, x * width)
    y1 = max(0, y * height)
    x2 = min(width, (x + w) * width)
    y2 = min(height, (y + h) * height)

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid bounding box")

    return np.array(
        [x1, y1, x2, y2],
        dtype=np.float32
    )


# =====================================
# 6. Run segmentation
# =====================================

results = []

for image_id, image_path in list(found.items())[:NUM_IMAGES]:
    try:
        print("Processing:", image_id, flush=True)

        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)

        width, height = image.size

        bbox = convert_box(
            selected[image_id]["box"],
            width,
            height
        )

        with torch.inference_mode():
            predictor.set_image(image_array)

            masks, scores, _ = predictor.predict(
                box=bbox,
                multimask_output=False
            )

        mask = masks[0].astype(bool)

        if not mask.any():
            print("Empty mask, skipping.")
            continue

        name = f"animal_{len(results):03d}"

        # Save original with bounding box
        original = image.copy()
        draw = ImageDraw.Draw(original)
        draw.rectangle(
            bbox.tolist(),
            outline="red",
            width=4
        )
        original.save(
            OUTPUT_DIR / f"{name}_original.jpg"
        )

        # Save segmentation overlay
        overlay = image_array.copy()

        overlay[mask] = (
            overlay[mask].astype(np.float32) * 0.45
            + np.array([0, 255, 0]) * 0.55
        ).astype(np.uint8)

        Image.fromarray(overlay).save(
            OUTPUT_DIR / f"{name}_mask.jpg"
        )

        # Save animal with transparent background
        rgba = image.convert("RGBA")
        rgba.putalpha(
            Image.fromarray(
                (mask * 255).astype(np.uint8)
            )
        )

        rgba.save(
            OUTPUT_DIR / f"{name}_cutout.png"
        )

        results.append({
            "image_id": image_id,
            "species": selected[image_id]["species"],
            "score": float(scores[0]),
            "source": str(image_path)
        })

        print(
            "Saved:", name,
            "Score:", round(float(scores[0]), 3),
            flush=True
        )

    except Exception as e:
        print("Skipped:", image_id, str(e), flush=True)


# =====================================
# 7. Save results
# =====================================

with open(OUTPUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print("-----------------------------")
print("Completed:", len(results))
print("Saved to:", OUTPUT_DIR)
print("-----------------------------")
