import numpy as np, sys
import val_compare as vc
rng = np.random.default_rng(0)
C, D, n = 5, 7, 400
y = rng.integers(0, 4, n)                      # class 4 absent
X = rng.normal(size=(n, D)) + rng.normal(size=(C, D))[y]
w = rng.uniform(0.1, 1, n)
present = np.bincount(y, minlength=C) > 0
M, S = vc.lda_parts(X, y, w, present, C)
# weighted means and covariance by hand
for c in range(4):
    assert np.allclose(M[c], (X[y == c] * w[y == c, None]).sum(0) / w[y == c].sum())
Sh = sum(w[i] * np.outer(X[i] - M[y[i]], X[i] - M[y[i]]) for i in range(n)) / w.sum()
assert np.allclose(S, Sh)
Xq = rng.normal(size=(50, D))
for s in (1.0, 0.1, 0.0001):
    sc = vc.lda_scores(Xq, M, S, s, present)
    Ss = (1 - s) * S + s * np.trace(S) / D * np.eye(D)
    P = np.linalg.inv(Ss)
    maha = np.array([[-0.5 * (x - M[c]) @ P @ (x - M[c]) for c in range(4)] for x in Xq])
    assert (sc[:, :4].argmax(1) == maha.argmax(1)).all() and np.isneginf(sc[:, 4]).all()
    # scores differ from the Mahalanobis form only by a per-row constant
    diff = sc[:, :4] - maha; assert np.allclose(diff, diff[:, :1])
print("linear discriminant matches the Mahalanobis form; absent class masked: ok")
# pooling of adjusted scores equals adjusted pooled scores
from pilot_a import pool
z = rng.normal(size=(30, C)); b = rng.integers(0, 6, 30); b[:6] = np.arange(6); logf = rng.normal(size=C)
assert np.allclose(pool(z - 0.7 * logf, b, 6), pool(z, b, 6) - 0.7 * logf)
print("mean of adjusted scores equals adjusted mean: ok")
assert vc.first_max(["a", "b", "c"], {"a": 1.0, "b": 2.0, "c": 2.0}) == "b"
print("tie goes to the earlier entry: ok")
