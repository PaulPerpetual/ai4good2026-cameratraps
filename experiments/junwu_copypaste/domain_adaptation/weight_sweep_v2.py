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
OUT.mkdir(exist_ok=True)

print("Loading original DINOv2 features...", flush=True)

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

normalize(X_train, copy=False)
normalize(X_val, copy=False)

results = []

def train_and_evaluate(name, X, y, weights=None, weight=0):
    model = SGDClassifier(
        loss="log_loss",
        alpha=1e-4,
        max_iter=100,
        tol=1e-3,
        random_state=42,
        n_jobs=-1
    )

    print(f"Training {name}, weight={weight}...", flush=True)
    model.fit(X, y, sample_weight=weights)

    predictions = model.predict(X_val)

    result = {
        "experiment": name,
        "weight": weight,
        "accuracy": accuracy_score(y_val, predictions),
        "macro_f1": f1_score(
            y_val, predictions, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            y_val, predictions, average="weighted", zero_division=0
        )
    }

    results.append(result)
    print(result, flush=True)


# Baseline: original data only
train_and_evaluate(
    "baseline", X_train, y_train, weight=0
)

# Two target-background methods
for method in ["random_target", "matched_target"]:

    folder = WORK / "features_v2" / method

    syn_index = pd.read_csv(folder / "index.csv")
    X_syn = np.load(folder / "features.npy").astype(np.float32)
    y_syn = syn_index["y"].to_numpy(dtype=np.int64)

    assert len(X_syn) == len(y_syn) == 477
    assert X_syn.shape[1] == X_train.shape[1]
    assert set(y_syn).issubset(set(y_train))

    normalize(X_syn, copy=False)

    X_aug = np.concatenate([X_train, X_syn])
    y_aug = np.concatenate([y_train, y_syn])

    for weight in [1, 2, 5, 10]:
        sample_weights = np.concatenate([
            np.ones(len(y_train)),
            np.full(len(y_syn), weight)
        ])

        train_and_evaluate(
            method, X_aug, y_aug,
            weights=sample_weights,
            weight=weight
        )

df = pd.DataFrame(results)

baseline = df.iloc[0]
df["delta_accuracy"] = df["accuracy"] - baseline["accuracy"]
df["delta_macro_f1"] = df["macro_f1"] - baseline["macro_f1"]

df.to_csv(OUT / "weight_sweep.csv", index=False)

print("\n========== WEIGHT SWEEP RESULTS ==========")
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nSaved:", OUT / "weight_sweep.csv")
