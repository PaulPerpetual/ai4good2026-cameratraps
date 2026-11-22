from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path("/zfs/ai4good/groups/camera-traps")
WORK = ROOT / "users/junwu/domain_adaptation"
ORIG = ROOT / "features/dinov2_vitl14_reg_448"
OUT = WORK / "results_v2"
OUT.mkdir(parents=True, exist_ok=True)

METHODS = [
    "random_source",
    "same_y_source",
    "random_target",
    "matched_target"
]
WEIGHT = 5

# Load synthetic features and verify pair alignment
synthetic = {}
for method in METHODS:
    parent = ("features_source_v2"
              if method.endswith("_source") else "features_v2")
    folder = WORK / parent / method

    idx = pd.read_csv(folder / "index.csv")
    X = np.load(folder / "features.npy").astype(np.float32)

    assert len(idx) == len(X), method
    assert idx["pair_id"].is_unique, method
    assert idx["y"].notna().all(), method

    synthetic[method] = (idx, X)

common = set.intersection(*[
    set(synthetic[m][0]["pair_id"]) for m in METHODS
])
common = sorted(common)
if not common:
    raise RuntimeError("No common pair IDs")

print("Common pairs:", len(common), flush=True)

reference = None
aligned = {}

for method in METHODS:
    idx, X = synthetic[method]
    positions = idx.set_index("pair_id").index.get_indexer(common)
    assert (positions >= 0).all()

    idx = idx.iloc[positions].reset_index(drop=True)
    X = X[positions]

    identity = list(zip(
        idx["source_image"].astype(str).str.removesuffix(".jpg"),
        idx["y"].astype(int)
    ))
    if reference is None:
        reference = identity
    elif identity != reference:
        raise RuntimeError(f"Source/label mismatch: {method}")

    normalize(X, norm="l2", copy=False)
    aligned[method] = (idx, X)

# Load original training and validation data
index = pd.read_csv(ORIG / "index.csv.gz")
features = np.load(ORIG / "features.npy", mmap_mode="r")
status = np.load(ORIG / "status.npy")

valid = (index["y"] >= 0) & (status == 1)
train_ids = np.flatnonzero(
    (valid & (index["official_ood"] == "train")).to_numpy()
)
val_ids = np.flatnonzero(
    (valid & (index["official_ood"] == "val")).to_numpy()
)

X_train = np.array(features[train_ids], dtype=np.float32)
X_val = np.array(features[val_ids], dtype=np.float32)
y_train = index.iloc[train_ids]["y"].to_numpy(dtype=np.int64)
y_val = index.iloc[val_ids]["y"].to_numpy(dtype=np.int64)

normalize(X_train, norm="l2", copy=False)
normalize(X_val, norm="l2", copy=False)

train_labels = (
    index.iloc[train_ids]
    .assign(source_image=lambda d:
            d["file_name"].astype(str).str.removesuffix(".jpg"))
    .drop_duplicates("source_image")
    .set_index("source_image")["y"]
    .to_dict()
)

for source, y in reference:
    if source not in train_labels or int(train_labels[source]) != y:
        raise RuntimeError(f"Invalid training source: {source}")

print("Original training:", len(y_train), flush=True)
print("Validation:", len(y_val), flush=True)

results = []

for method in ["baseline"] + METHODS:
    if method == "baseline":
        X_fit, y_fit = X_train, y_train
        weights = None
    else:
        idx, X_syn = aligned[method]
        y_syn = idx["y"].to_numpy(dtype=np.int64)
        X_fit = np.concatenate([X_train, X_syn])
        y_fit = np.concatenate([y_train, y_syn])
        weights = np.concatenate([
            np.ones(len(y_train)),
            np.full(len(y_syn), WEIGHT)
        ])

    print("Training:", method, flush=True)

    model = SGDClassifier(
        loss="log_loss",
        alpha=1e-4,
        max_iter=100,
        tol=1e-3,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_fit, y_fit, sample_weight=weights)
    pred = model.predict(X_val)

    results.append({
        "experiment": method,
        "synthetic_count": 0 if method == "baseline" else len(common),
        "accuracy": accuracy_score(y_val, pred),
        "macro_f1": f1_score(
            y_val, pred, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            y_val, pred, average="weighted", zero_division=0
        )
    })

df = pd.DataFrame(results)
df["delta_macro_f1"] = df["macro_f1"] - df.iloc[0]["macro_f1"]
df["delta_accuracy"] = df["accuracy"] - df.iloc[0]["accuracy"]

dest = OUT / "five_way_comparison.csv"
df.to_csv(dest, index=False)

print("\n========== FIVE-WAY RESULTS ==========")
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nSaved:", dest, flush=True)
