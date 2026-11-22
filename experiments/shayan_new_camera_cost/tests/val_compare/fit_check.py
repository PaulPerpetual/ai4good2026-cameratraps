import numpy as np, warnings
warnings.simplefilter("ignore")
import torch
import val_compare as vc, pilot_a
torch.set_num_threads(8)
rng = np.random.default_rng(3)
n, D, C = 6000, 40, 12
y = rng.integers(0, C, n); X = (rng.normal(size=(n, D)) + rng.normal(size=(C, D))[y] * 0.8).astype(np.float32)
w = rng.uniform(0.05, 1, n).astype(np.float32)
for l2, it in ((1e-2, 2000), (1e-4, 2000), (1e-4, 40), (1e-2, 5)):
    W1, b1, d1 = pilot_a.fit_head(X, y, w, None, l2, C, it)
    W2, b2, d2 = vc.fit_linear(X, y, w, l2, C, it)
    assert np.array_equal(W1, W2) and np.array_equal(b1, b2), "weights differ from fit_head"
    assert d1["iterations"] == d2["iterations"] and abs(d1["grad_max"] - d2["grad_max"]) < 1e-12
    print(f"l2={l2:g} max_iter={it}: identical weights; iterations {d2['iterations']}, evaluations {d2['evaluations']}, "
          f"fit_head hit_cap={d1['hit_cap']}, fit_linear hit_cap={d2['hit_cap']}, grad_max {d2['grad_max']:.1e}")
