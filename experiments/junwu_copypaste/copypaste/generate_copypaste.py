import json
import random
from pathlib import Path
from collections import defaultdict

import pandas as pd
from PIL import Image

ROOT = Path("/zfs/ai4good/groups/camera-traps")
WORK = ROOT / "users/junwu/copypaste"
RAW = ROOT / "raw"
DATA = ROOT / "camera-trap/data"
OUT = WORK / "generated_v2"

OUT.mkdir(parents=True, exist_ok=True)
random.seed(42)

# 1. Load labelled animal candidates
with open(WORK / "train_animal_candidates.json") as f:
    animals = json.load(f)

# 2. Load camera metadata
metadata = pd.read_csv(DATA / "metadata.csv.gz")
metadata["image_id"] = metadata["image_id"].astype(str)
meta = metadata.set_index("image_id")

# 3. Load MegaDetector predictions
with open(RAW / "iwildcam2020_megadetector_results.json") as f:
    detections = json.load(f)["images"]

# 4. Identify likely empty backgrounds
empty_ids = []

for record in detections:
    boxes = record.get("detections") or []
    has_animal = any(
        str(d.get("category")) == "1"
        and d.get("conf", 0) >= 0.2
        for d in boxes
    )
    if not has_animal:
        empty_ids.append(str(record["id"]))

print("Potential empty backgrounds:", len(empty_ids))

# 5. Group background images by camera
backgrounds = defaultdict(list)

for image_id in empty_ids:
    if image_id not in meta.index:
        continue

    row = meta.loc[image_id]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    if str(row.get("official_ood")) != "train":
        continue

    camera = row.get("location")
    if pd.notna(camera):
        backgrounds[str(camera)].append(image_id)

print("Background cameras:", len(backgrounds))

# 6. Find image file
IMAGE_ROOTS = [
    DATA / "images_h448",
    RAW / "train",
]

def find_image(image_id):
    for folder in IMAGE_ROOTS:
        path = folder / f"{image_id}.jpg"
        if path.exists():
            return path

    for folder in IMAGE_ROOTS:
        matches = list(folder.rglob(f"{image_id}.jpg"))
        if matches:
            return matches[0]

    return None

# 7. Generate cross-camera images
results = []
cameras = list(backgrounds.keys())

for i in range(20):
    animal = random.choice(animals)
    source_id = str(animal["image_id"])

    if source_id not in meta.index:
        continue

    source_camera = str(meta.loc[source_id, "location"])

    possible_cameras = [
        c for c in cameras
        if c != source_camera and backgrounds[c]
    ]

    if not possible_cameras:
        continue

    target_camera = random.choice(possible_cameras)
    target_id = random.choice(backgrounds[target_camera])

    source_path = find_image(source_id)
    target_path = find_image(target_id)

    if source_path is None or target_path is None:
        continue

    with Image.open(source_path) as im:
        source = im.convert("RGB")

    with Image.open(target_path) as im:
        target = im.convert("RGB")

    box = random.choice(animal["boxes"])["bbox"]
    x, y, w, h = box

    W, H = source.size

    crop = source.crop((
        max(0, int(x * W)),
        max(0, int(y * H)),
        min(W, int((x + w) * W)),
        min(H, int((y + h) * H)),
    ))

    if crop.width == 0 or crop.height == 0:
        continue

    crop.thumbnail((
        max(1, target.width // 3),
        max(1, target.height // 3),
    ))

    px = random.randint(0, target.width - crop.width)
    py = random.randint(0, target.height - crop.height)

    target.paste(crop, (px, py))

    filename = f"cross_camera_{i:03d}.jpg"
    target.save(OUT / filename)

    results.append({
        "filename": filename,
        "species": animal["species"],
        "source_camera": source_camera,
        "target_camera": target_camera,
        "source_image": source_id,
        "background_image": target_id,
    })

with open(OUT / "labels.json", "w") as f:
    json.dump(results, f, indent=2)

print("Generated:", len(results))
print("Saved to:", OUT)
