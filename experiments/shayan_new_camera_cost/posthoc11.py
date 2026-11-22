"""Post-hoc diagnostics of Run 11 (labelled post hoc; saved outputs only, no fits)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "/zfs/ai4good/groups/camera-traps/users/shayan/pilot_a")
from pilot_a import Data, macro_f1
R = Path("/zfs/ai4good/groups/camera-traps/users/shayan/pilot_a/results")
G = "/zfs/ai4good/groups/camera-traps"
C = 60
d = Data(Path(f"{G}/features/dinov2_vitl14_reg_448_crop"), Path("."), C)
tr_bursts = np.bincount(d.b_y[d.b_fold == "train"], minlength=C)
bmeta = pd.DataFrame({"burst_id": d.img.groupby(d.burst)["burst_id"].first().to_numpy(), "y": d.b_y, "frames": d.b_n,
                      "pure": pd.Series(d.y == d.b_y[d.burst]).groupby(d.burst).all().to_numpy()}).set_index("burst_id")
plans = pd.read_csv(R / "select_v1/plans.csv.gz")
out = {"label": "post hoc diagnostics, saved outputs only"}
sel = {}
for k in (1, 8):
    for s in ("random", "uncertain", "spread"):
        p = plans[(plans["set"] == "vitl_crop") & (plans["selector"] == s) & (plans["k"] == k)]
        m = bmeta.loc[p["burst_id"]]
        sel[f"{s} k={k}"] = {"bursts_per_repeat": round(len(p) / p["repeat"].nunique(), 1),
                             "rare_share(<50 training bursts)": round(float((tr_bursts[m["y"]] < 50).mean()), 3),
                             "species": int(m["y"].nunique()), "mixed_share": round(float((~m["pure"]).mean()), 3),
                             "mean_frames": round(float(m["frames"].mean()), 2)}
out["selected_labels"] = sel
q = pd.read_csv(R / "select_v1/query_images.csv.gz")
b = pd.read_csv(R / "budget_v1/query_images.csv.gz", usecols=["file_name", "vitl_crop|0|REF|burst"]).set_index("file_name")
q["REF"] = b.loc[q["file_name"], "vitl_crop|0|REF|burst"].to_numpy()
y = q["y"].to_numpy(); cam = q["camera"].to_numpy()
unc = q["vitl_crop|0|uncertain_1|burst"].to_numpy()
rnd = [q[f"vitl_crop|{r}|random_1|burst"].to_numpy() for r in range(3)]
ref = q["REF"].to_numpy()
def f1c(yy, pp, c):
    return 2 * ((yy == c) & (pp == c)).sum() / max((yy == c).sum() + (pp == c).sum(), 1)
present = np.unique(y)
dif = {}
for c in present:
    du = f1c(y, unc, c) - np.mean([f1c(y, r, c) for r in rnd])
    prec = lambda pp: ((y == c) & (pp == c)).sum() / max((pp == c).sum(), 1)
    rec = lambda pp: ((y == c) & (pp == c)).sum() / max((y == c).sum(), 1)
    dif[int(c)] = (du, prec(unc) - np.mean([prec(r) for r in rnd]), rec(unc) - np.mean([rec(r) for r in rnd]), int((y == c).sum()), int(tr_bursts[c]))
tot = sum(v[0] for v in dif.values()) / len(present)
top = sorted(dif.items(), key=lambda kv: -kv[1][0])[:6]
out["per_species"] = {"macro_f1_difference": round(float(tot), 4),
                      "top_contributors (class: F1 diff, precision diff, recall diff, query images, training bursts)":
                          {c: [round(float(v[0]), 3), round(float(v[1]), 3), round(float(v[2]), 3), v[3], v[4]] for c, v in top},
                      "share_of_difference_from_top_2": round(float(sum(v[0] for _, v in top[:2]) / len(present) / tot), 3) if tot else None,
                      "species_better/worse/same": [int(sum(v[0] > 1e-9 for v in dif.values())), int(sum(v[0] < -1e-9 for v in dif.values())),
                                                    int(sum(abs(v[0]) <= 1e-9 for v in dif.values()))]}
diffall = macro_f1(y, unc, C) - np.mean([macro_f1(y, r, C) for r in rnd])
loo = {}
for c0 in np.unique(cam):
    m = cam != c0
    loo[int(c0)] = macro_f1(y[m], unc[m], C) - np.mean([macro_f1(y[m], r[m], C) for r in rnd])
vals = np.array(list(loo.values()))
out["leave_one_camera_out"] = {"full": round(float(diffall), 4), "min": round(float(vals.min()), 4), "max": round(float(vals.max()), 4),
                               "most_influential_camera": int(min(loo, key=loo.get))}
rng = np.random.default_rng(5)
cams = np.unique(cam); by = {c: np.flatnonzero(cam == c) for c in cams}
pt = macro_f1(y, unc, C) - macro_f1(y, ref, C)
bs = []
for _ in range(1000):
    ix = np.concatenate([by[c] for c in rng.choice(cams, len(cams))])
    bs.append(macro_f1(y[ix], unc[ix], C) - macro_f1(y[ix], ref[ix], C))
out["uncertain_1_minus_REF"] = {"value": round(float(pt), 4), "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]}
print(json.dumps(out, indent=1))
