
import os
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

BASE = Path(__file__).resolve().parent
PROJECT = BASE.parents[2]
RAW = PROJECT / "raw"

OUT = BASE / "generated_v4"
OUT.mkdir(exist_ok=True)

NUM_IMAGES = 20
MAX_ATTEMPTS = 300
random.seed(42)

with open(BASE / "train_animal_candidates.json") as f:
    animals = json.load(f)

print("Animal candidates:", len(animals), flush=True)

# Find image files by ID
image_paths = {}
print("Indexing source images...", flush=True)

for root, dirs, files in os.walk(RAW):
    for filename in files:
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_paths[Path(filename).stem] = Path(root) / filename

print("Indexed images:", len(image_paths), flush=True)

# Load SAM 2
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device, flush=True)

model = build_sam2(
    "configs/sam2.1/sam2.1_hiera_t.yaml",
    str(BASE / "checkpoints/sam2.1_hiera_tiny.pt"),
    device=device
)
predictor = SAM2ImagePredictor(model)

def get_box(bbox, w, h):
    x, y, bw, bh = map(float, bbox)
    return np.array(
        [x*w, y*h, (x+bw)*w, (y+bh)*h],
        dtype=np.float32
    )

# Use candidate images as a temporary background pool.
# For a valid experiment, replace with verified empty frames.
backgrounds = [
    a for a in animals
    if str(a["image_id"]) in image_paths
]

results = []
attempts = 0

while len(results) < NUM_IMAGES and attempts < MAX_ATTEMPTS:
    attempts += 1

    animal = random.choice(animals)
    source_id = str(animal["image_id"])
    source_path = image_paths.get(source_id)

    if source_path is None or not animal.get("boxes"):
        continue

    target = random.choice(backgrounds)
    target_id = str(target["image_id"])
    target_path = image_paths.get(target_id)

    if target_id == source_id or target_path is None:
        continue

    try:
        with Image.open(source_path) as im:
            src = im.convert("RGB")

        with Image.open(target_path) as im:
            dst = im.convert("RGB")

        box = max(
            animal["boxes"],
            key=lambda b: float(b.get("conf", 0))
        )

        bbox = get_box(
            box["bbox"], src.width, src.height
        )

        with torch.inference_mode():
            predictor.set_image(np.asarray(src))
            masks, scores, _ = predictor.predict(
                box=bbox,
                multimask_output=False
            )

        mask = masks[0].astype(np.uint8) * 255

        if mask.sum() == 0:
            continue

        alpha = Image.fromarray(mask, "L")
        bounds = alpha.getbbox()

        if bounds is None:
            continue

        crop = src.crop(bounds).convert("RGBA")
        alpha = alpha.crop(bounds)

        crop.thumbnail((
            int(dst.width * 0.35),
            int(dst.height * 0.40)
        ))

        alpha = alpha.resize(
            crop.size,
            Image.Resampling.BILINEAR
        )

        alpha = alpha.filter(
            ImageFilter.GaussianBlur(radius=1.5)
        )

        if crop.width >= dst.width or crop.height >= dst.height:
            continue

        px = random.randint(0, dst.width - crop.width)
        py = random.randint(
            max(0, int(dst.height * 0.65) - crop.height),
            dst.height - crop.height
        )

        dst.paste(crop, (px, py), alpha)

        filename = f"sam2_{len(results):04d}.jpg"
        dst.save(OUT / filename, quality=92)

        results.append({
            "filename": filename,
            "species": animal["species"],
            "source_image": source_id,
            "background_image": target_id,
            "sam2_score": float(scores[0])
        })

        print(
            f"Generated {len(results)}/{NUM_IMAGES}",
            flush=True
        )

    except Exception as e:
        print("Skipped:", str(e), flush=True)

with open(OUT / "labels.json", "w") as f:
    json.dump(results, f, indent=2)

print("Finished:", len(results), flush=True)
print("Saved to:", OUT, flush=True)
