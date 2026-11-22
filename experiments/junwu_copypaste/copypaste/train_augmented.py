
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import normalize
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

ROOT = Path("/zfs/ai4good/groups/camera-traps")
WORK = ROOT / "users/junwu/copypaste"

ORIGINAL = ROOT / "features/dinov2_vitl14_reg_448"
SYNTHETIC = WORK / "synthetic_features_v6"
META = ROOT / "camera-trap/data/metadata.csv.gz"

OUT = WORK / "augmented_results"
OUT.mkdir(parents=True, exist_ok=True)

print("Loading original features...", flush=True)

index = pd.read_csv(ORIGINAL / "index.csv.gz")
features = np.load(ORIGINAL / "features.npy", mmap_mode="r")
status = np.load(ORIGINAL / "status.npy")

valid = (index["y"] >= 0) & (status == 1)

train_rows = np.flatnonzero(
    (valid & (index["official_ood"] == "train")).to_numpy()
)
val_rows = np.flatnonzero(
    (valid & (index["official_ood"] == "val")).to_numpy()
)

X_train = np.array(features[train_rows], dtype=np.float32)
X_val = np.array(features[val_rows], dtype=np.float32)

y_train = index.iloc[train_rows]["y"].to_numpy(dtype=np.int64)
y_val = index.iloc[val_rows]["y"].to_numpy(dtype=np.int64)

print("Loading synthetic features...", flush=True)

syn_index = pd.read_csv(SYNTHETIC / "index.csv")
syn_features = np.load(SYNTHETIC / "features.npy")

assert len(syn_index) == len(syn_features)
assert syn_features.shape[1] == X_train.shape[1]
assert syn_index["filename"].is_unique

# Check synthetic source labels against original metadata.
meta = pd.read_csv(META, usecols=[
    "image_id", "category_id", "category",
    "official_ood"
])

meta["image_id"] = meta["image_id"].astype(str)
meta = meta.drop_duplicates("image_id").set_index("image_id")

original_lookup = (
    index.drop_duplicates("file_name")
    .set_index("file_name")
)

keep = []
reasons = {}

for i, row in syn_index.iterrows():
    source_id = str(row["source_image"])
    source_file = source_id + ".jpg"

    reason = None

    if source_id not in meta.index:
        reason = "source missing from metadata"
    elif source_file not in original_lookup.index:
        reason = "source missing from DINOv2 index"
    else:
        m = meta.loc[source_id]
        o = original_lookup.loc[source_file]

        if str(m["official_ood"]) != "train":
            reason = "source not in training split"
        elif str(o["official_ood"]) != "train":
            reason = "source feature not in training split"
        elif int(m["category_id"]) != int(row["species"]):
            reason = "species ID mismatch"
        elif str(m["category"]) != str(o["category"]):
            reason = "species name mismatch"
        elif int(o["y"]) != int(row["y"]):
            reason = "classification label mismatch"
        elif int(row["y"]) < 0:
            reason = "invalid classification label"

    if reason is None:
        keep.append(i)
    else:
        reasons[reason] = reasons.get(reason, 0) + 1

print("Synthetic features:", len(syn_index), flush=True)
print("Verified synthetic images:", len(keep), flush=True)
print("Excluded:", reasons, flush=True)

if not keep:
    raise RuntimeError("No verified synthetic images")

X_syn = np.asarray(syn_features[keep], dtype=np.float32)
y_syn = syn_index.iloc[keep]["y"].to_numpy(dtype=np.int64)

assert np.isfinite(X_train).all()
assert np.isfinite(X_val).all()
assert np.isfinite(X_syn).all()

assert set(y_syn).issubset(set(y_train))

# Identical feature normalization as Baseline.
normalize(X_train, norm="l2", copy=False)
normalize(X_val, norm="l2", copy=False)
normalize(X_syn, norm="l2", copy=False)

X_aug = np.concatenate([X_train, X_syn], axis=0)
y_aug = np.concatenate([y_train, y_syn], axis=0)

print("Original training:", len(y_train), flush=True)
print("Synthetic training:", len(y_syn), flush=True)
print("Augmented training:", len(y_aug), flush=True)
print("Validation:", len(y_val), flush=True)

# Same hyperparameters as train_baseline.py
model = SGDClassifier(
    loss="log_loss",
    alpha=1e-4,
    max_iter=100,
    tol=1e-3,
    random_state=42,
    n_jobs=-1
)

print("Training augmented classifier...", flush=True)

model.fit(X_aug, y_aug)

pred = model.predict(X_val)

metrics = {
    "experiment": "copy_paste_v6",
    "original_train": len(y_train),
    "synthetic_train": len(y_syn),
    "validation": len(y_val),
    "accuracy": float(accuracy_score(y_val, pred)),
    "macro_f1": float(f1_score(
        y_val, pred, average="macro", zero_division=0
    )),
    "weighted_f1": float(f1_score(
        y_val, pred, average="weighted", zero_division=0
    )),
}

print("\n==============================")
print("COPY-PASTE AUGMENTED RESULTS")
print("==============================")
print(f"Accuracy:    {metrics['accuracy']:.4f}")
print(f"Macro-F1:    {metrics['macro_f1']:.4f}")
print(f"Weighted-F1: {metrics['weighted_f1']:.4f}")
print("==============================")

baseline = {
    "accuracy": 0.8470,
    "macro_f1": 0.6281,
    "weighted_f1": 0.8468
}

print("\nCHANGE VS BASELINE")
for key in baseline:
    diff = metrics[key] - baseline[key]
    print(f"{key}: {diff:+.4f}")

joblib.dump(model, OUT / "classifier.joblib")

with open(OUT / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

report = classification_report(
    y_val,
    pred,
    labels=model.classes_,
    output_dict=True,
    zero_division=0
)

pd.DataFrame(report).transpose().to_csv(
    OUT / "classification_report.csv"
)

print("\nSaved to:", OUT, flush=True)
