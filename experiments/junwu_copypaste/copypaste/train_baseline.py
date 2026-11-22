
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)
from sklearn.preprocessing import normalize


# =====================================
# 1. Paths and settings
# =====================================

ROOT = Path("/zfs/ai4good/groups/camera-traps")

FEATURE_DIR = (
    ROOT / "features/dinov2_vitl14_reg_448"
)

OUTPUT = (
    ROOT / "users/junwu/copypaste/baseline_results"
)

OUTPUT.mkdir(parents=True, exist_ok=True)

SEED = 42

print("Starting DINOv2 Baseline", flush=True)


# =====================================
# 2. Load shared features
# =====================================

index = pd.read_csv(
    FEATURE_DIR / "index.csv.gz"
)

features = np.load(
    FEATURE_DIR / "features.npy",
    mmap_mode="r"
)

status = np.load(
    FEATURE_DIR / "status.npy"
)

assert len(index) == len(features) == len(status)

print("Feature shape:", features.shape, flush=True)


# =====================================
# 3. Select labeled animal images
# =====================================

valid = (
    (index["y"] >= 0)
    & (status == 1)
)

train_mask = (
    valid
    & (index["official_ood"] == "train")
)

val_mask = (
    valid
    & (index["official_ood"] == "val")
)

train_rows = np.flatnonzero(train_mask.to_numpy())
val_rows = np.flatnonzero(val_mask.to_numpy())

X_train = np.asarray(
    features[train_rows],
    dtype=np.float32
)

X_val = np.asarray(
    features[val_rows],
    dtype=np.float32
)

y_train = index.iloc[train_rows]["y"].to_numpy(
    dtype=np.int64
)

y_val = index.iloc[val_rows]["y"].to_numpy(
    dtype=np.int64
)

print("Train:", len(y_train), flush=True)
print("Validation:", len(y_val), flush=True)

if not np.isfinite(X_train).all():
    raise RuntimeError("Non-finite training features")

if not np.isfinite(X_val).all():
    raise RuntimeError("Non-finite validation features")


# =====================================
# 4. Normalize embeddings
# =====================================

normalize(X_train, norm="l2", copy=False)
normalize(X_val, norm="l2", copy=False)

print("Features normalized", flush=True)


# =====================================
# 5. Train classifier
# =====================================

model = SGDClassifier(
    loss="log_loss",
    alpha=1e-4,
    max_iter=100,
    tol=1e-3,
    random_state=SEED,
    n_jobs=-1
)

print("Training classifier...", flush=True)

model.fit(X_train, y_train)

print("Training completed!", flush=True)


# =====================================
# 6. Evaluation
# =====================================

predictions = model.predict(X_val)

accuracy = accuracy_score(
    y_val, predictions
)

macro_f1 = f1_score(
    y_val,
    predictions,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_val,
    predictions,
    average="weighted",
    zero_division=0
)

print("\n==========================")
print("DINOv2 BASELINE RESULTS")
print("==========================")

print(f"Accuracy:    {accuracy:.4f}")
print(f"Macro-F1:    {macro_f1:.4f}")
print(f"Weighted-F1: {weighted_f1:.4f}")
print("==========================", flush=True)


# =====================================
# 7. Save model and results
# =====================================

joblib.dump(
    model,
    OUTPUT / "classifier.joblib"
)

metrics = {
    "experiment": "baseline",
    "model": "DINOv2 + SGD logistic classifier",
    "seed": SEED,
    "train_images": len(y_train),
    "validation_images": len(y_val),
    "accuracy": float(accuracy),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1)
}

with open(OUTPUT / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

report = classification_report(
    y_val,
    predictions,
    labels=np.unique(
        np.concatenate([y_train, y_val])
    ),
    output_dict=True,
    zero_division=0
)

pd.DataFrame(report).transpose().to_csv(
    OUTPUT / "classification_report.csv"
)

print("\nSaved to:", OUTPUT, flush=True)
