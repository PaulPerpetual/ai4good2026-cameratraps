import json, sys
from pathlib import Path
import numpy as np, pandas as pd
out = Path(sys.argv[1])
res = json.loads((out / "results.json").read_text())
vp = pd.read_csv(out / "val_predictions.csv.gz")
tp = pd.read_csv(out / "train_camera_predictions.csv.gz")
cvt = pd.read_csv(out / "cv_scores.csv")

def mf1(y, p):   # independent of the script: per-class F1 from a crosstab, averaged over classes in y
    f = []
    for c in np.unique(y):
        tp_, fp, fn = ((p == c) & (y == c)).sum(), ((p == c) & (y != c)).sum(), ((p != c) & (y == c)).sum()
        f.append(2 * tp_ / max(2 * tp_ + fp + fn, 1))
    return float(np.mean(f))

slug = {"ordinary": "ordinary", "frequency-adjusted": "freq_adjusted", "linear discriminant": "lda"}
for r in res["validation"]["rows"]:
    col = "pooled_" + slug.get(r["classifier"], "pilot_calibrated")
    assert abs(mf1(vp["y"].to_numpy(), vp[col].to_numpy()) - r["macro_f1_pooled"]) < 6e-5, r["classifier"]
    assert abs((vp[col] == vp["y"]).mean() - r["accuracy_pooled"]) < 6e-5
    if r["macro_f1_per_image"] is not None:
        assert abs(mf1(vp["y"].to_numpy(), vp["per_image_" + slug[r["classifier"]]].to_numpy()) - r["macro_f1_per_image"]) < 6e-5
print("validation rows recomputed from val_predictions: ok")
# every image of a burst shares the pooled prediction; rows and cameras as expected
for c in [c for c in vp.columns if c.startswith("pooled_")]:
    assert vp.groupby("burst_id")[c].nunique().eq(1).all(), c
assert len(vp) == res["data"]["val_images"] and vp["location"].nunique() == res["data"]["val_cameras"]
assert len(tp) == res["data"]["train_images"] and tp["location"].nunique() == res["data"]["train_cameras"]
assert not set(vp["location"]) & set(tp["location"])
print("bursts share one prediction; train and validation cameras disjoint: ok")
ts = res["held_out_training_cameras"]
for h, s in slug.items():
    assert abs(mf1(tp["y"].to_numpy(), tp["nested_" + s].to_numpy()) - ts["macro_f1_pooled"][h]) < 6e-5
    assert abs(mf1(tp["y"].to_numpy(), tp["allfolds_" + s].to_numpy()) - res["selection"]["chosen"][h]["cv_macro_f1_pooled"]) < 6e-5
print("training-camera numbers recomputed from train_camera_predictions: ok")
# the chosen settings are the argmax of the table, with the stated tie-break
lin = cvt[cvt["classifier"] == "linear head"]; lda = cvt[cvt["classifier"] == "linear discriminant"]
o = lin[lin["tau"] == 0]; ch = res["selection"]["chosen"]
assert ch["ordinary"]["cv_macro_f1_pooled"] == o["cv_macro_f1_pooled"].max()
assert ch["frequency-adjusted"]["cv_macro_f1_pooled"] == lin["cv_macro_f1_pooled"].max()
assert ch["linear discriminant"]["cv_macro_f1_pooled"] == lda["cv_macro_f1_pooled"].max()
assert len(lin) == 56 and len(lda) == 9
print("chosen settings are the table's maxima: ok")
# fold structure: 5 folds, every camera in one fold
assert tp.groupby("location")["fold"].nunique().eq(1).all() and sorted(tp["fold"].unique()) == [0, 1, 2, 3, 4]
# contrasts are consistent with rows
rows = {r["classifier"]: r for r in res["validation"]["rows"]}
for k, v in res["validation"]["contrasts"].items():
    assert abs(v["diff"] - (rows[v["a"]]["macro_f1_pooled"] - rows[v["b"]]["macro_f1_pooled"])) < 2.1e-4, k
    exp = "supported" if v["diff"] >= v["margin"] and v["ci95"][0] > 0 else "not supported" if v["ci95"][1] < v["margin"] else "inconclusive"
    assert v["outcome"] == exp
print("contrasts consistent with the table and the outcome rule: ok")
pc = pd.read_csv(out / "per_class_val.csv")
assert len(pc) == res["data"]["val_species_present"]
assert abs(pc["f1_lda"].mean() - rows["linear discriminant"]["macro_f1_pooled"]) < 1e-3
print("per-class table: ok;", len(pc), "species")
