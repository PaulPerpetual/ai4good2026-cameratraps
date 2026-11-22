
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2


# =====================================
# 1. Paths
# =====================================

ROOT = Path("/zfs/ai4good/groups/camera-traps")
WORK = ROOT / "users/junwu/copypaste"

ORIGINAL = ROOT / "features/dinov2_vitl14_reg_448"
SYNTHETIC = WORK / "generated_v6_500"
OUTPUT = WORK / "synthetic_features_v6"

OUTPUT.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 8
DEVICE = "cuda"

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

# =====================================
# 2. Read original feature configuration
# =====================================

with open(ORIGINAL / "config.json") as f:
    config = json.load(f)

model_name = config["model"]
model_repo = config["repo"]
size = int(config["size"])

assert model_name == "dinov2_vitl14_reg"
assert size == 448

print("Model:", model_name, flush=True)
print("Image size:", size, flush=True)


# =====================================
# 3. Match synthetic labels to original y
# =====================================

index = pd.read_csv(ORIGINAL / "index.csv.gz")

# Each original image has an established class label.
original_labels = dict(
    zip(index["file_name"].astype(str), index["y"])
)

original_categories = dict(
    zip(index["file_name"].astype(str), index["category"])
)

with open(SYNTHETIC / "labels.json") as f:
    synthetic = json.load(f)

selected = []

for item in synthetic:
    filename = item["filename"]
    source_name = str(item["source_image"]) + ".jpg"

    if source_name not in original_labels:
        continue

    y = int(original_labels[source_name])

    if y < 0:
        continue

    if not (SYNTHETIC / filename).is_file():
        continue

    # Source image must belong to training data.
    source_rows = index.loc[index["file_name"] == source_name]

    if len(source_rows) != 1:
        continue

    if source_rows.iloc[0]["official_ood"] != "train":
        continue

    selected.append({
        **item,
        "y": y,
        "category": original_categories[source_name]
    })

print("Total synthetic:", len(synthetic), flush=True)
print("Matched:", len(selected), flush=True)

if not selected:
    raise RuntimeError("No synthetic images matched original labels")


# =====================================
# 4. Dataset and preprocessing
# =====================================

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
        path = SYNTHETIC / self.rows[i]["filename"]

        with Image.open(path) as img:
            image = img.convert("RGB")
            tensor = self.transform(image)

        return tensor


dataset = SyntheticDataset(selected)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)


# =====================================
# 5. Load DINOv2
# =====================================

print("Loading DINOv2...", flush=True)

model = torch.hub.load(
    model_repo,
    model_name,
    trust_repo=True,
    skip_validation=True
)

model = model.to(DEVICE).eval()

print("DINOv2 ready", flush=True)


# =====================================
# 6. Extract embeddings
# =====================================

all_features = []

with torch.inference_mode():
    for batch_idx, images in enumerate(loader):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):
            output = model.forward_features(images)

            cls = output["x_norm_clstoken"]
            patch = output["x_norm_patchtokens"].mean(1)

            features = torch.cat(
                [cls, patch],
                dim=1
            )

        features = (
            features.float()
            .cpu()
            .numpy()
            .astype(np.float16)
        )

        if not np.isfinite(features).all():
            raise RuntimeError("Non-finite features detected")

        all_features.append(features)

        print(
            f"Processed {min((batch_idx + 1) * BATCH_SIZE, len(dataset))}/{len(dataset)}",
            flush=True
        )


# =====================================
# 7. Save features and labels
# =====================================

X = np.concatenate(all_features, axis=0)

assert X.shape == (len(selected), 2048)

np.save(OUTPUT / "features.npy", X)

pd.DataFrame(selected).to_csv(
    OUTPUT / "index.csv",
    index=False
)

with open(OUTPUT / "config.json", "w") as f:
    json.dump({
        "model": model_name,
        "repo": model_repo,
        "size": size,
        "dimension": 2048,
        "feature_method": "CLS + mean normalized patch tokens",
        "normalization": "ImageNet",
        "source": str(SYNTHETIC),
        "number_of_images": len(selected)
    }, f, indent=2)

print("============================", flush=True)
print("Extraction completed!", flush=True)
print("Features shape:", X.shape, flush=True)
print("Saved to:", OUTPUT, flush=True)
print("============================", flush=True)
