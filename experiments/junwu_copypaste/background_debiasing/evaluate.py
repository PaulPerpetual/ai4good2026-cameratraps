from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler

DATA = Path(
    "/zfs/ai4good/groups/camera-traps/features/"
    "dinov2_vitl14_reg_448"
)

df = pd.read_csv(DATA / "index.csv.gz")
X = np.load(DATA / "features.npy", mmap_mode="r")
b = np.load("background_direction.npy").astype(np.float32)

assert len(df) == len(X)
assert X.shape[1] == len(b)

split = df["official_ood"].astype(str).str.lower()
empty = df["is_empty"].astype(str).str.lower().eq("true")
labels = df["category"]

train_mask = (split == "train") & ~empty & labels.notna()
val_mask = (split == "val") & ~empty & labels.notna()

train_idx = np.flatnonzero(train_mask.to_numpy())
val_idx = np.flatnonzero(val_mask.to_numpy())

print("Training animals:", len(train_idx))
print("Validation animals:", len(val_idx), flush=True)

if len(train_idx) == 0 or len(val_idx) == 0:
    raise ValueError(
        "No train/val animal samples. Check official_ood values."
    )

X_train = np.asarray(X[train_idx], dtype=np.float32)
X_val = np.asarray(X[val_idx], dtype=np.float32)

y_train = labels.iloc[train_idx].astype(str).to_numpy()
y_val = labels.iloc[val_idx].astype(str).to_numpy()

valid_train = np.isfinite(X_train).all(axis=1)
valid_val = np.isfinite(X_val).all(axis=1)

X_train, y_train = X_train[valid_train], y_train[valid_train]
X_val, y_val = X_val[valid_val], y_val[valid_val]

# Keep evaluation restricted to classes seen during training.
known = np.isin(y_val, np.unique(y_train))
if not known.all():
    raise ValueError("Validation contains unseen classes.")

results = []

for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
    print(f"Running lambda={lam}", flush=True)

    A = X_train - lam * np.outer(X_train @ b, b)
    V = X_val - lam * np.outer(X_val @ b, b)

    scaler = StandardScaler()
    A = scaler.fit_transform(A)
    V = scaler.transform(V)

    model = LogisticRegression(
        max_iter=300,
        C=1.0,
        solver="lbfgs"
    )
    model.fit(A, y_train)
    pred = model.predict(V)

    f1 = f1_score(y_val, pred, average="macro", zero_division=0)
    acc = accuracy_score(y_val, pred)

    print(f"Macro F1={f1:.4f}, Accuracy={acc:.4f}", flush=True)

    results.append({
        "lambda": lam,
        "macro_f1": f1,
        "accuracy": acc
    })

pd.DataFrame(results).to_csv("debias_results.csv", index=False)
print("\nResults saved to debias_results.csv")
