
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(
    "/zfs/ai4good/groups/camera-traps/features/"
    "dinov2_vitl14_reg_448"
)

# Load existing DINOv2 features
X = np.load(DATA / "features.npy", mmap_mode="r")
df = pd.read_csv(DATA / "index.csv.gz")

print("Features:", X.shape)
print("Metadata:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head(3).to_string(index=False))

# Save metadata information
with open("data_info.txt", "w") as f:
    f.write(f"Features: {X.shape}\n")
    f.write(f"Metadata: {df.shape}\n")
    f.write(f"Columns: {df.columns.tolist()}\n")

print("Experiment initialized successfully!")




