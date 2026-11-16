"""Render the README figures from bench/results/scores.jsonl and metrics.json."""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "..", "docs")
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8983"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "axes.spines.top": False, "axes.spines.right": False})


def main():
    rows = [json.loads(line) for line in open(os.path.join(HERE, "results", "scores.jsonl"))]
    metrics = json.load(open(os.path.join(HERE, "results", "metrics.json")))
    cls = np.array([r["cls"] for r in rows])
    m = (cls == 0) | (cls == 1)

    fig, ax = plt.subplots(figsize=(6.2, 4.6), dpi=200)
    for mode, colour, name in (("ref", BLUE, "with reference rows"),
                               ("free", ORANGE, "data-free")):
        s = np.array([r[f"score_{mode}"] for r in rows])
        fpr, tpr, _ = roc_curve(cls[m], s[m])
        auc = metrics[mode]["out_of_sample_all"]["auc"]
        ax.plot(fpr, tpr, color=colour, lw=2, label=f"deadfall, {name} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], color=GREY, lw=1, ls=(0, (4, 3)), label="chance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("clean models wrongly flagged")
    ax.set_ylabel("backdoored models caught")
    ax.grid(color=GRID, lw=0.6)
    ax.set_title(f"Out-of-sample detection, {int((cls == 1).sum())} backdoored vs "
                 f"{int((cls == 0).sum())} clean models", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, loc="lower right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "roc.png"))

    per = {mode: metrics[mode]["C_per_dataset"] for mode in ("free", "ref")}
    names = sorted(per["ref"], key=lambda d: per["ref"][d].get("auc", 0))
    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=200)
    y = np.arange(len(names))
    for mode, colour, name in (("ref", BLUE, "with reference rows"), ("free", ORANGE, "data-free")):
        v = [per[mode][d].get("auc", np.nan) for d in names]
        ax.scatter(v, y, s=46, color=colour, label=name, zorder=3, edgecolor=SURFACE, linewidth=1.5)
    for i, d in enumerate(names):
        ax.plot([per["free"][d].get("auc", np.nan), per["ref"][d].get("auc", np.nan)], [i, i],
                color=GRID, lw=2, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([d.replace("_", " ") for d in names], color=INK)
    ax.set_xlim(0.5, 1.005)
    ax.set_xlabel("AUC on a dataset the detector never saw (leave-one-dataset-out)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, loc="lower left", labelcolor=INK)
    ax.set_title("Generalisation to unseen real datasets", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "per_dataset.png"))


if __name__ == "__main__":
    main()
