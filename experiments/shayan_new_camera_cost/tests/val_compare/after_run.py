"""Checks on the real Run 5 outputs, run on the laptop after the files are copied back.

    python after_run.py results/val_compare_v1 results/rare_v1 results/pilot_a_v2_gpu

1. The grid points that Run 4 also computed (l2 0.01 with tau 0, 0.5 and 1; shrinkage 0.1)
   must give Run 4's numbers: same folds, same data, same code path.
2. Fit diagnostics by l2: iterations, loss evaluations against the cap, gradients.
3. The pilot's ordinary-head fits (l2 0.01, 0.001, 0.0001, five folds and the final fit):
   same iteration counts here, and how many evaluations they took.
4. Diagnostics of the settings the nested cross-validation chose in each outer fold.
"""

import json
import sys
from pathlib import Path

import pandas as pd

out, rare, pilot = (Path(p) for p in sys.argv[1:4])
cv = pd.read_csv(out / "cv_scores.csv")
fits = pd.DataFrame(json.loads((out / "fit_diagnostics.json").read_text()))
sel = json.loads((out / "selection.json").read_text())
res = json.loads((out / "results.json").read_text()) if (out / "results.json").exists() else None

print("1. Agreement with Run 4 at the shared grid points (pooled, per image, accuracy)")
r4 = json.loads((rare / "results.json").read_text())["B_heads"]
lin, lda = cv[cv["classifier"] == "linear head"], cv[cv["classifier"] == "linear discriminant"]
shared = [("ordinary", lin[(lin.l2 == 0.01) & (lin.tau == 0.0)]),
          ("ordinary - 0.5 log freq", lin[(lin.l2 == 0.01) & (lin.tau == 0.5)]),
          ("ordinary - log freq", lin[(lin.l2 == 0.01) & (lin.tau == 1.0)]),
          ("linear discriminant", lda[lda.shrinkage == 0.1])]
worst = 0.0
for name, row in shared:
    assert len(row) == 1
    here = row.iloc[0][["cv_macro_f1_pooled", "cv_macro_f1_per_image", "cv_accuracy_pooled"]].tolist()
    then = [r4[name]["macro_f1_pooled"], r4[name]["macro_f1_per_image"], r4[name]["accuracy_pooled"]]
    worst = max(worst, max(abs(a - b) for a, b in zip(here, then)))
    print(f"   {name:26s} here {here}  Run 4 {then}")
print(f"   largest difference {worst:.4f}")

print("\n2. Fits by l2 (cap: 2000 iterations or 2500 evaluations)")
fits["kind"] = fits["fit"].str.replace(r"\d", "#", regex=True)
g = fits.groupby("l2").agg(fits=("l2", "size"), it_median=("iterations", "median"), it_max=("iterations", "max"),
                           ev_max=("evaluations", "max"), capped=("hit_cap", "sum"), grad_max=("grad_max", "max"),
                           grad_norm_max=("grad_norm", "max"), seconds=("seconds", "sum")).sort_index(ascending=False)
print(g.to_string())
print("   by training set:", fits.groupby("kind").size().to_dict(), " total seconds in fits:", round(fits["seconds"].sum()))

print("\n3. The pilot's ordinary-head fits, refitted here")
pf = pd.DataFrame(json.loads((pilot / "results.json").read_text())["optimisation"]["detail"])
pf = pf[pf["head"] == "ord"]
tag = {f"fold{i}": f"fold {i}" for i in range(5)} | {"final": "final"}
for _, r in pf.iterrows():
    m = fits[(fits["fit"] == tag[r["fit"]]) & (fits["l2"] == r["l2"])]
    if len(m) == 1:
        m = m.iloc[0]
        print(f"   {r['fit']:6s} l2={r['l2']:<7g} pilot iterations {r['iterations']:5d}   here {m['iterations']:5d} iterations, "
              f"{m['evaluations']:5d} evaluations, cap {bool(m['hit_cap'])}, |grad| {m['grad_max']:.1e}")

if res:
    print("\n4. Settings chosen inside each outer fold (nested), and their fits")
    for h, choices in res["held_out_training_cameras"]["settings_chosen_per_fold"].items():
        print(f"   {h}: {choices}")
        if "l2" in choices[0]:
            for f, c in enumerate(choices):
                # behind outer fold f's nested prediction: the outer model, and the inner models that leave f out
                rel = fits[(fits["l2"] == c["l2"]) & ((fits["fit"] == f"fold {f}") | fits["fit"].str.contains(f"without.*{f}"))]
                print(f"      fold {f}: l2={c['l2']:g}  {len(rel)} fits, capped {int(rel['hit_cap'].sum())}, "
                      f"most evaluations {rel['evaluations'].max()}, max |grad| {rel['grad_max'].max():.1e}")
    print("\n   chosen on all folds:", json.dumps(sel["chosen"]))
