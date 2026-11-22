import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

ROOT = Path("/zfs/ai4good/groups/camera-traps")
WORK = ROOT / "users/junwu/domain_adaptation"
ORIGINAL = ROOT / "features/dinov2_vitl14_reg_448"
SYNTHETIC = WORK / "synthetic_source_v2"

ap = argparse.ArgumentParser()
ap.add_argument(
    "--method",
    choices=["random_source", "same_y_source"],
    required=True
)
args = ap.parse_args()

OUTPUT = WORK / "features_source_v2" / args.method
OUTPUT.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 8
DEVICE = "cuda"

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

with open(ORIGINAL / "config.json") as f:
    config = json.load(f)

model_name = config["model"]
model_repo = config["repo"]
size = int(config["size"])

assert model_name == "dinov2_vitl14_reg"
assert size == 448

index = pd.read_csv(ORIGINAL / "index.csv.gz")
labels = pd.read_csv(SYNTHETIC / "labels.csv")

required = {"method", "image_file", "source_image", "y"}
if not required.issubset(labels.columns):
    raise ValueError(f"Missing columns: {required - set(labels.columns)}")

original = index.copy()
original["image_id"] = (
    original["file_name"].astype(str).str.removesuffix(".jpg")
)
original = original[
    (original["official_ood"] == "train") &
    (original["y"] >= 0)
].drop_duplicates("image_id")

lookup = original.set_index("image_id")["y"].to_dict()

selected = []

for row in labels[labels["method"] == args.method].to_dict("records"):
    source_id = str(row["source_image"]).removesuffix(".jpg")
    path = Path(str(row["image_file"]))

    if not path.is_absolute():
        path = SYNTHETIC / path

    if source_id not in lookup:
        continue

    if int(row["y"]) != int(lookup[source_id]):
        raise ValueError(f"Label mismatch: {source_id}")

    if not path.is_file():
        continue

    selected.append({
        **row,
        "image_file": str(path),
        "source_image": source_id,
        "y": int(lookup[source_id])
    })

print("Method:", args.method, flush=True)
print("Matched images:", len(selected), flush=True)

if not selected:
    raise RuntimeError("No matching synthetic images found")


class SyntheticDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows
        self.transform = v2.Compose([
            v2.Resize(
                (size, size),
                interpolation=v2.InterpolationMode.BILINEAR,
                antialias=True
            ),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        with Image.open(self.rows[i]["image_file"]) as im:
            return self.transform(im.convert("RGB"))


dataset = SyntheticDataset(selected)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

print("Loading DINOv2...", flush=True)

model = torch.hub.load(
    model_repo,
    model_name,
    trust_repo=True,
    skip_validation=True
).to(DEVICE).eval()

print("DINOv2 ready", flush=True)

all_features = []

with torch.inference_mode():
    for batch_idx, images in enumerate(loader):
        images = images.to(DEVICE, non_blocking=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):
            output = model.forward_features(images)
            cls = output["x_norm_clstoken"]
            patch = output["x_norm_patchtokens"].mean(1)
            features = torch.cat([cls, patch], dim=1)

        features = features.float().cpu().numpy().astype(np.float16)

        if not np.isfinite(features).all():
            raise RuntimeError("Non-finite features")

        all_features.append(features)

        print(
            f"Processed {min((batch_idx+1)*BATCH_SIZE, len(dataset))}/{len(dataset)}",
            flush=True
        )

X = np.concatenate(all_features, axis=0)
assert X.shape == (len(selected), 2048)

np.save(OUTPUT / "features.npy", X)
pd.DataFrame(selected).to_csv(OUTPUT / "index.csv", index=False)

with open(OUTPUT / "config.json", "w") as f:
    json.dump({
        "model": model_name,
        "repo": model_repo,
        "size": size,
        "dimension": 2048,
        "method": args.method,
        "number_of_images": len(selected)
    }, f, indent=2)

print("DONE:", X.shape, "saved to", OUTPUT, flush=True)
