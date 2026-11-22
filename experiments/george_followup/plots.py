"""Figures for the follow-up studies, in one consistent style."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Categorical colours in a fixed order: blue, orange, aqua, yellow, magenta.
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "lines.linewidth": 2,
    "legend.frameon": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
})


def reliability_and_coverage(report: dict, title: str, path):
    """Left: confidence vs actual accuracy before and after calibration (OOD, per photo).
    Middle: the same per burst. Right: macro F1 of the photos kept vs share kept."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, unit in zip(axes[:2], ("per_photo", "per_burst")):
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", label="perfect")
        for colour, version in zip(COLOURS, ("raw", "calibrated")):
            bins = report["ood"]["test"][version][unit]["bins"]
            used = [b for b in bins if b["count"] > 0]
            ax.plot([b["confidence"] for b in used], [b["accuracy"] for b in used], marker="o",
                    markersize=5, color=colour,
                    label=f"{version}  (ECE {report['ood']['test'][version][unit]['ece']:.3f})")
        ax.set(title=f"OOD, {unit.replace('_', ' ')}", xlabel="confidence", ylabel="accuracy", xlim=(0, 1), ylim=(0, 1))
        ax.legend(loc="upper left")
    ax = axes[2]
    for colour, setting in zip(COLOURS, ("ood", "id")):
        curve = report[setting]["test"]["coverage_per_photo"]
        ax.plot([c["kept"] for c in curve], [c["macro_f1"] for c in curve], marker="o", markersize=5,
                color=colour, label=setting.upper())
    ax.invert_xaxis()
    ax.set(title="Least confident photos to a human", xlabel="share of photos kept",
           ylabel="macro F1 of photos kept")
    ax.legend(loc="lower right")
    fig.suptitle(title, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def per_species(table, title: str, path):
    """F1 of every species against its number of training bursts, plain vs adjusted classifier."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for colour, (column, label) in zip(COLOURS, (("f1_plain", "plain"), ("f1_chosen", "rare-species changes"))):
        ax.scatter(table.training_bursts, table[column], s=28, color=colour, label=label,
                   edgecolor="#fcfcfb", linewidth=1, zorder=3)
    for limit in table.attrs.get("group_limits", []):
        ax.axvline(limit, color=MUTED, linewidth=1, linestyle=":")
    ax.set_xscale("log")
    ax.set(title=title, xlabel="training bursts of the species (log scale)", ylabel="F1 on test cameras", ylim=(-0.03, 1.03))
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def accuracy_vs_cost(rows, cost_key: str, title: str, path):
    """One point per combination: macro F1 on unseen cameras against compute cost."""
    fig, ax = plt.subplots(figsize=(8, 5))
    backbones = sorted({r["backbone"] for r in rows})
    for colour, backbone in zip(COLOURS, backbones):
        mine = sorted((r for r in rows if r["backbone"] == backbone), key=lambda r: r[cost_key])
        for marker, view, detector, label in (("o", "crop", "md5a", "crop (MD v5a)"), ("^", "crop", "md6c", "crop (MD v6-compact)"),
                                              ("s", "whole", "md5a", "whole photo")):
            points = [r for r in mine if r["view"] == view and (view == "whole" or r["detector"] == detector)]
            if points:
                ax.scatter([r[cost_key] for r in points], [r["ood_per_photo"] for r in points], s=48,
                           marker=marker, color=colour, edgecolor="#fcfcfb", linewidth=1.5, zorder=3,
                           label=f"{backbone}, {label}")
                for r in points:
                    ax.annotate(str(r["size"]), (r[cost_key], r["ood_per_photo"]), textcoords="offset points",
                                xytext=(5, 4), fontsize=8, color=MUTED)
    ax.set_xscale("log")
    ax.set(title=title, xlabel=cost_key.replace("_", " "), ylabel="macro F1 per photo, unseen cameras (OOD)")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def by_animal_size(rows, title: str, path):
    """Per-photo accuracy for small, medium and large animals, for each input size of a backbone."""
    groups = ["small", "medium", "large"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [f"{r['backbone']} {r['view']} {r['size']}" for r in rows]
    x = np.arange(len(rows))
    width = 0.26
    for i, (colour, group) in enumerate(zip(COLOURS, groups)):
        ax.bar(x + (i - 1) * width, [r["accuracy_by_size"][group] for r in rows], width - 0.03,
               color=colour, label=f"{group} animals")
    ax.set_xticks(x, labels, rotation=30, ha="right", fontsize=8)
    ax.set(title=title, ylabel="accuracy per photo, unseen cameras", ylim=(0, 1))
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
