import json
import random
from pathlib import Path

ROOT = Path("/zfs/ai4good/groups/camera-traps")
RAW = ROOT / "raw"
OUT = ROOT / "users/junwu/copypaste"

ANNOTATIONS = RAW / "iwildcam2020_train_annotations.json"
DETECTIONS = RAW / "iwildcam2020_megadetector_results.json"

OUT.mkdir(parents=True, exist_ok=True)

print("Loading training annotations...")
with open(ANNOTATIONS) as f:
    ann = json.load(f)

print("Loading MegaDetector results...")
with open(DETECTIONS) as f:
    det = json.load(f)

# Build an image ID -> species label mapping
# using the training annotations only.
labels = {}

if isinstance(ann, dict):
    annotations = ann.get("annotations", [])
else:
    annotations = ann

for item in annotations:
    image_id = str(item.get("image_id", ""))
    species = item.get("category_id")

    if image_id and species is not None:
        labels[image_id] = species

print("Labelled image IDs:", len(labels))
print("MegaDetector records:", len(det["images"]))

# Match animal detections with training labels.
matched = []

for image in det["images"]:
    image_id = str(image.get("id", ""))

    if image_id not in labels:
        continue

    animal_boxes = [
        detection
        for detection in image.get("detections", [])
        if str(detection.get("category")) == "1"
        and detection.get("conf", 0) >= 0.8
        and len(detection.get("bbox", [])) == 4
    ]

    if not animal_boxes:
        continue

    matched.append({
        "image_id": image_id,
        "species": labels[image_id],
        "boxes": animal_boxes
    })

print("Matched training images:", len(matched))

random.seed(42)
random.shuffle(matched)

output_file = OUT / "animal_candidates.json"

with open(output_file, "w") as f:
    json.dump(matched, f, indent=2)

print("Saved to:", output_file)
print("First candidate:", matched[0] if matched else "None")


import pandas as pd

metadata_path = (
    ROOT / "camera-trap/data/metadata.csv.gz"
)

metadata = pd.read_csv(metadata_path)

# Keep only the official training split
train = metadata[
    (metadata["official_ood"] == "train")
    & (metadata["is_empty"] == False)
].copy()

# Match image IDs with MegaDetector candidates
train_ids = set(train["image_id"].astype(str))

train_candidates = [
    item for item in matched
    if item["image_id"] in train_ids
]

print("Official training images:", len(train))
print("Usable animal candidates:", len(train_candidates))

output = OUT / "train_animal_candidates.json"

with open(output, "w") as f:
    json.dump(train_candidates, f)

print("Saved:", output)
