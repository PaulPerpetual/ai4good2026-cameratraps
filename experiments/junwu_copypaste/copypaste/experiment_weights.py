
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import normalize


# ==========================================
# 1. Settings
# ==========================================

ROOT = Path("/zfs/ai4good/groups/camera-traps")
WORK = ROOT / "users/junwu/copypaste"

ORIGINAL = ROOT / "features/dinov2_vitl14_reg_448"
SYNTHETIC = WORK / "synthetic_features_v6"
METADATA = ROOT / "camera-trap/data/metadata.csv.gz"

OUT = WORK / "weight_experiment_results"
OUT.mkdir(parents=True, exist_ok=True)

WEIGHTS = [0, 1, 2, 5, 10]
SEED = 42

print("Copy-Paste Weight Experiment", flush=True)


# ==========================================
# 2. Load original features
# ==========================================

index = pd.read_csv(ORIGINAL / "index.csv.gz")

features = np.load(
    ORIGINAL / "features.npy",
    mmap_mode="r"
)

status = np.load(ORIGINAL / "status.npy")

assert len(index) == len(features) == len(status)

valid = (index["y"] >= 0) & (status == 1)

train_rows = np.flatnonzero(
    (
        valid &
        (index["official_ood"] == "train")
    ).to_numpy()
)

val_rows = np.flatnonzero(
    (
        valid &
        (index["official_ood"] == "val")
    ).to_numpy()
)

X_train = np.array(
    features[train_rows],
    dtype=np.float32
)

y_train = index.iloc[train_rows]["y"].to_numpy(
    dtype=np.int64
)

X_val = np.array(
    features[val_rows],
    dtype=np.float32
)

y_val = index.iloc[val_rows]["y"].to_numpy(
    dtype=np.int64
)

print("Original training:", len(y_train), flush=True)
print("Validation:", len(y_val), flush=True)


# ==========================================
# 3. Load and verify synthetic features
# ==========================================

syn_index = pd.read_csv(
    SYNTHETIC / "index.csv"
)

syn_features = np.load(
    SYNTHETIC / "features.npy",
    mmap_mode="r"
)

meta = pd.read_csv(
    METADATA,
    usecols=[
        "image_id",
        "category_id",
        "category",
        "official_ood",
        "is_empty"
    ]
)

meta["image_id"] = meta["image_id"].astype(str)

meta = (
    meta.drop_duplicates("image_id")
    .set_index("image_id")
)

original_lookup = (
    index.drop_duplicates("file_name")
    .set_index("file_name")
)

assert len(syn_index) == len(syn_features)
assert syn_index["filename"].is_unique

keep = []
excluded = {}

for i, row in syn_index.iterrows():
    src_id = str(row["source_image"])
    bg_id = str(row["background_image"])
    src_file = src_id + ".jpg"

    reason = None

    if src_id not in meta.index:
        reason = "source missing"

    elif bg_id not in meta.index:
        reason = "background missing"

    elif src_file not in original_lookup.index:
        reason = "source not in feature index"

    else:
        src = meta.loc[src_id]
        bg = meta.loc[bg_id]
        original = original_lookup.loc[src_file]

        if str(src["official_ood"]) != "train":
            reason = "source not train"

        elif str(bg["official_ood"]) != "train":
            reason = "background not train"

        elif str(bg["is_empty"]).lower() != "true":
            reason = "background not empty"

        elif str(original["official_ood"]) != "train":
            reason = "original index not train"

        elif int(src["category_id"]) != int(row["species"]):
            reason = "species ID mismatch"

        elif str(src["category"]) != str(original["category"]):
            reason = "species name mismatch"

        elif int(original["y"]) != int(row["y"]):
            reason = "y label mismatch"

        elif int(row["y"]) not in set(y_train):
            reason = "unknown class"

    if reason is None:
        keep.append(i)
    else:
        excluded[reason] = excluded.get(reason, 0) + 1

print("Synthetic available:", len(syn_index), flush=True)
print("Synthetic verified:", len(keep), flush=True)
print("Excluded:", excluded, flush=True)

if not keep:
    raise RuntimeError("No valid synthetic features")

X_syn = np.array(
    syn_features[keep],
    dtype=np.float32
)

y_syn = syn_index.iloc[keep]["y"].to_numpy(
    dtype=np.int64
)

assert X_train.shape[1] == X_syn.shape[1]
assert np.isfinite(X_train).all()
assert np.isfinite(X_val).all()
assert np.isfinite(X_syn).all()


# ==========================================
# 4. Normalize features
# ==========================================

normalize(X_train, norm="l2", copy=False)
normalize(X_val, norm="l2", copy=False)
normalize(X_syn, norm="l2", copy=False)

X_aug = np.concatenate(
    [X_train, X_syn],
    axis=0
)

y_aug = np.concatenate(
    [y_train, y_syn],
    axis=0
)

print("Features ready", flush=True)


# ==========================================
# 5. Train with different weights
# ==========================================

results = []

for weight in WEIGHTS:

    print(
        f"\nTraining synthetic weight = {weight}",
        flush=True
    )

    model = SGDClassifier(
        loss="log_loss",
        alpha=1e-4,
        max_iter=100,
        tol=1e-3,
        random_state=SEED,
        n_jobs=-1
    )

    if weight == 0:

        # True original-data-only baseline.
        model.fit(X_train, y_train)

    else:

        sample_weights = np.concatenate([
            np.ones(len(y_train), dtype=np.float64),
            np.full(
                len(y_syn),
                weight,
                dtype=np.float64
            )
        ])

        # Normalize mean weight to one,
        # holding the total loss scale comparable.
        sample_weights /= sample_weights.mean()

        model.fit(
            X_aug,
            y_aug,
            sample_weight=sample_weights
        )

    predictions = model.predict(X_val)

    accuracy = accuracy_score(
        y_val,
        predictions
    )

    macro_f1 = f1_score(
        y_val,
        predictions,
        average="macro",
        labels=np.unique(y_train),
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_val,
        predictions,
        average="weighted",
        zero_division=0
    )

    result = {
        "weight": weight,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1)
    }

    results.append(result)

    print(
        f"Accuracy: {accuracy:.4f} | "
        f"Macro-F1: {macro_f1:.4f} | "
        f"Weighted-F1: {weighted_f1:.4f}",
        flush=True
    )


# ==========================================
# 6. Compare with experiment baseline
# ==========================================

df = pd.DataFrame(results)

baseline = df.loc[df["weight"] == 0].iloc[0]

df["delta_macro_f1"] = (
    df["macro_f1"] - baseline["macro_f1"]
)

df["delta_accuracy"] = (
    df["accuracy"] - baseline["accuracy"]
)

print("\n====================================")
print("WEIGHT EXPERIMENT RESULTS")
print("====================================")

print(
    df.round(4).to_string(index=False),
    flush=True
)

df.to_csv(
    OUT / "weight_results.csv",
    index=False
)

with open(OUT / "experiment_config.json", "w") as f:
    json.dump({
        "weights": WEIGHTS,
        "seed": SEED,
        "original_train": len(y_train),
        "synthetic_train": len(y_syn),
        "validation": len(y_val),
        "classifier": "SGDClassifier",
        "loss": "log_loss",
        "alpha": 1e-4,
        "max_iter": 100,
        "weight_normalization": "mean equals one"
    }, f, indent=2)

print("\nSaved to:", OUT, flush=True)
