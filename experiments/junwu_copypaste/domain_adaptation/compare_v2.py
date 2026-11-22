import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path("/zfs/ai4good/groups/camera-traps")
WORK = ROOT / "users/junwu/domain_adaptation"
ORIGINAL = ROOT / "features/dinov2_vitl14_reg_448"
OUT = WORK / "results_v2"
OUT.mkdir(parents=True, exist_ok=True)

print("Loading original features...", flush=True)

index = pd.read_csv(ORIGINAL / "index.csv.gz")
features = np.load(ORIGINAL / "features.npy", mmap_mode="r")
status = np.load(ORIGINAL / "status.npy")

assert len(index) == len(features) == len(status)

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

normalize(X_train, norm="l2", copy=False)
normalize(X_val, norm="l2", copy=False)

train_lookup = (
    index.iloc[train_rows]
    .assign(source_image=lambda d:
            d["file_name"].astype(str).str.removesuffix(".jpg"))
    .drop_duplicates("source_image")
    .set_index("source_image")["y"]
    .to_dict()
)

def load_synthetic(method):
    folder = WORK / "features_v2" / method

    syn_index = pd.read_csv(folder / "index.csv")
    X = np.load(folder / "features.npy").astype(np.float32)

    assert len(syn_index) == len(X)
    assert X.shape[1] == X_train.shape[1]

    keep = []
    for i, row in syn_index.iterrows():
        source = str(row["source_image"]).removesuffix(".jpg")
        if source not in train_lookup:
            continue
        if int(row["y"]) != int(train_lookup[source]):
            raise ValueError(f"Label mismatch: {source}")
        keep.append(i)

    if len(keep) != len(syn_index):
        raise RuntimeError(
            f"{method}: only {len(keep)}/{len(syn_index)} "
            "synthetic images have verified training sources"
        )

    X = X[keep]
    y = syn_index.iloc[keep]["y"].to_numpy(dtype=np.int64)

    if not np.isfinite(X).all():
        raise RuntimeError(f"Non-finite features: {method}")

    normalize(X, norm="l2", copy=False)
    return X, y

results = []

def evaluate(name, X, y):
    print(f"\nTraining {name}: {len(y)} images", flush=True)

    model = SGDClassifier(
        loss="log_loss",
        alpha=1e-4,
        max_iter=100,
        tol=1e-3,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)
    pred = model.predict(X_val)

    result = {
        "experiment": name,
        "train_size": len(y),
        "accuracy": accuracy_score(y_val, pred),
        "macro_f1": f1_score(
            y_val, pred, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            y_val, pred, average="weighted", zero_division=0
        )
    }
    results.append(result)
    print(result, flush=True)

evaluate("baseline", X_train, y_train)

for method in ["random_target", "matched_target"]:
    X_syn, y_syn = load_synthetic(method)

    X_aug = np.concatenate([X_train, X_syn])
    y_aug = np.concatenate([y_train, y_syn])

    evaluate(method, X_aug, y_aug)

df = pd.DataFrame(results)

base = df.iloc[0]
df["delta_accuracy"] = df["accuracy"] - base["accuracy"]
df["delta_macro_f1"] = df["macro_f1"] - base["macro_f1"]

df.to_csv(OUT / "comparison.csv", index=False)

print("\n========== RESULTS ==========", flush=True)
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nSaved:", OUT / "comparison.csv", flush=True)
