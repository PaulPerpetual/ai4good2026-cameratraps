from pathlib import Path
import numpy as np
import pandas as pd

root = Path("/zfs/ai4good/groups/camera-traps/features/dinov2_vitl14_reg_448")

X = np.load(root / "features.npy", mmap_mode="r")
df = pd.read_csv(root / "index.csv.gz")

assert len(df) == len(X), "Feature/index mismatch"

empty = df["is_empty"].astype(str).str.lower().eq("true")
train = df["official_ood"].eq("train")

# Estimate background direction using training empty images
idx = np.flatnonzero((train & empty).to_numpy())

print("Training empty images:", len(idx))

if len(idx) == 0:
    raise ValueError("No training empty images found")

B = np.asarray(X[idx], dtype=np.float32)
B = B[np.isfinite(B).all(axis=1)]

if len(B) == 0:
    raise ValueError("No valid background features")

b = B.mean(axis=0)
b = b / (np.linalg.norm(b) + 1e-8)

np.save("background_direction.npy", b)

print("Background direction saved:", b.shape)
print("Ready for debiasing experiment!")
