"""Synthetic cache with the real index: tests code paths only, the numbers mean nothing."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, sys.argv[2])                      # pilot_a folder, for cache_features
from cache_features import select_rows
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
rows = select_rows()
rng = np.random.default_rng(1)
D = 32
means = rng.normal(size=(60, D)) * 0.55
cams = {c: rng.normal(size=D) * 0.6 for c in rows["location"].unique()}
shift = np.stack([cams[c] for c in rows["location"]])
y = rows["y"].to_numpy()
X = shift + rng.normal(size=(len(rows), D))
X[y >= 0] += means[y[y >= 0]]
np.save(out / "features.npy", X.astype(np.float16))
np.save(out / "status.npy", np.ones(len(rows), dtype=np.uint8))
rows.to_csv(out / "index.csv.gz", index=False)
print(len(rows), "rows", rows["official_ood"].value_counts().to_dict(), "labelled", int((y >= 0).sum()))
