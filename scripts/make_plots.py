"""
Plotting utilities for protocol results.

Generates publication-quality figures from the JSON outputs of
scripts/run_p*.py.

Usage:
  python scripts/make_plots.py p1 experiments/results/p1_seed42_sev5.json
  python scripts/make_plots.py p2 experiments/results/p2_emergent_seed42_sev5.json
  python scripts/make_plots.py p3 experiments/results/p3_doseresponse_seed42_sev5.json
  python scripts/make_plots.py p4 experiments/results/p4_whereness_seed42_sev5.json
  python scripts/make_plots.py p5 experiments/results/p5_ablation_seed42_sev5.json

Output: PNG + PDF in the same folder as the JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def _setup_mpl():
    """Lazy import + style setup."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
    })
    return plt


def _save(fig, out_path: Path):
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"[saved] {out_path}.png  and  .pdf")


# ----------------------------------------------------------------------------
# P1: per-corruption accuracy bar chart + mean table
# ----------------------------------------------------------------------------

def plot_p1(json_path: Path):
    plt = _setup_mpl()
    data = json.loads(json_path.read_text())
    results = data["results"]                     # method -> corruption -> acc
    methods = list(results.keys())
    corruptions = list(next(iter(results.values())).keys())

    fig, ax = plt.subplots(figsize=(11, 4.5))
    n_meth = len(methods)
    x = list(range(len(corruptions)))
    width = 0.8 / n_meth

    for i, m in enumerate(methods):
        accs = [results[m][c] for c in corruptions]
        ax.bar([xi + i * width for xi in x], accs, width=width, label=m)

    ax.set_xticks([xi + width * (n_meth - 1) / 2 for xi in x])
    ax.set_xticklabels(corruptions, rotation=45, ha="right")
    ax.set_ylabel("accuracy")
    ax.set_title(f"P1: Online TTA per corruption  (sev={data['args']['severity']})")
    ax.legend(loc="lower right", ncols=n_meth)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, json_path.with_suffix(""))


# ----------------------------------------------------------------------------
# P2: pretrained vs random init — adaptation gain
# ----------------------------------------------------------------------------

def plot_p2(json_path: Path):
    plt = _setup_mpl()
    data = json.loads(json_path.read_text())
    results = data["results"]                     # init -> method -> corruption -> acc

    # Compute adaptation gain per (init, method): mean over corruptions of
    #   results[init][method][c] - results[init]["source"][c]
    inits = list(results.keys())
    methods = [m for m in next(iter(results.values())).keys() if m != "source"]
    corruptions = list(next(iter(next(iter(results.values())).values())).keys())

    gain = defaultdict(dict)
    for init in inits:
        for m in methods:
            per_c = [results[init][m][c] - results[init]["source"][c]
                     for c in corruptions]
            gain[init][m] = sum(per_c) / len(per_c)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    n_init = len(inits)
    x = list(range(len(methods)))
    width = 0.8 / n_init

    for i, init in enumerate(inits):
        gains = [gain[init][m] for m in methods]
        ax.bar([xi + i * width for xi in x], gains, width=width,
               label=init, alpha=0.85)

    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks([xi + width * (n_init - 1) / 2 for xi in x])
    ax.set_xticklabels(methods)
    ax.set_ylabel("mean adaptation gain  (acc - source)")
    ax.set_title(f"P2: Emergent capacity — pretrained vs random init")
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, json_path.with_suffix(""))


# ----------------------------------------------------------------------------
# P3: dose-response across pretraining checkpoints
# ----------------------------------------------------------------------------

def plot_p3(json_path: Path):
    plt = _setup_mpl()
    data = json.loads(json_path.read_text())
    results = data["results"]                     # epoch_label -> method -> corruption -> acc
    epoch_labels = list(results.keys())
    methods = [m for m in next(iter(results.values())).keys() if m != "source"]
    corruptions = list(next(iter(next(iter(results.values())).values())).keys())

    # Convert epoch labels to numeric for x-axis
    def _epoch_num(s):
        try:
            return int(s)
        except ValueError:
            return float("inf")  # 'final' goes last
    epoch_labels_sorted = sorted(epoch_labels, key=_epoch_num)
    epoch_nums = [_epoch_num(e) for e in epoch_labels_sorted]
    # 'final' as last position visually
    if epoch_nums and epoch_nums[-1] == float("inf"):
        epoch_nums[-1] = (max(epoch_nums[:-1]) if len(epoch_nums) > 1 else 1) * 1.2

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in methods:
        gains = []
        for e in epoch_labels_sorted:
            per_c = [results[e][m][c] - results[e]["source"][c]
                     for c in corruptions]
            gains.append(sum(per_c) / len(per_c))
        ax.plot(epoch_nums, gains, marker="o", label=m, linewidth=2)

    ax.set_xlabel("pretraining epoch")
    ax.set_ylabel("mean adaptation gain")
    ax.set_title("P3: Dose-response — adaptive capacity emerges with pretraining")
    ax.axhline(0, color="black", lw=0.6, alpha=0.5)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    # Annotate x-tick labels with original strings
    ax.set_xticks(epoch_nums)
    ax.set_xticklabels(epoch_labels_sorted)
    _save(fig, json_path.with_suffix(""))


# ----------------------------------------------------------------------------
# P4: where-emergence — gradient profile across stages
# ----------------------------------------------------------------------------

def plot_p4(json_path: Path):
    plt = _setup_mpl()
    data = json.loads(json_path.read_text())
    results = data["results"]                     # corruption -> method -> dict
    corruptions = list(results.keys())
    methods = list(next(iter(results.values())).keys())
    stages_order = ["stem", "stage1", "stage2", "stage3", "stage4", "head"]

    fig, axes = plt.subplots(1, len(corruptions),
                             figsize=(3.2 * len(corruptions), 4),
                             sharey=False)
    if len(corruptions) == 1:
        axes = [axes]

    for ax, c in zip(axes, corruptions):
        for m in methods:
            per_stage_list = results[c][m]["per_stage"]   # list[dict[stage->norm]]
            # Average across batches
            sums = defaultdict(list)
            for d in per_stage_list:
                for k, v in d.items():
                    sums[k].append(v)
            means = [sum(sums.get(s, [0])) / max(len(sums.get(s, [1])), 1)
                     for s in stages_order]
            # Normalize for shape comparison (gradient magnitudes differ across methods)
            total = sum(means) or 1.0
            normalized = [v / total for v in means]
            ax.plot(stages_order, normalized, marker="o", label=m, linewidth=2)
        ax.set_title(c, fontsize=10)
        ax.set_ylabel("relative ||grad||")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("P4: Where-emergence — gradient profile across stages "
                 "(normalized; shape comparison)", y=1.04)
    _save(fig, json_path.with_suffix(""))


# ----------------------------------------------------------------------------
# P5: hierarchy ablation (two-axis: hierarchy × update set)
# ----------------------------------------------------------------------------

def plot_p5(json_path: Path):
    plt = _setup_mpl()
    data = json.loads(json_path.read_text())
    results = data["results"]
    hierarchy = list(data["hierarchy_variants"].keys())     # 4 items
    update_sets = list(data["update_set_variants"].keys())  # 2 items

    # Figure 1: heatmap of mean accuracy (hierarchy x update_set)
    src_mean = sum(results["source"].values()) / len(results["source"])
    matrix = []
    for h in hierarchy:
        row = []
        for u in update_sets:
            key = f"{h}__{u}"
            vals = list(results[key].values())
            row.append(sum(vals) / len(vals))
        matrix.append(row)

    import numpy as np
    matrix_arr = np.array(matrix)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(matrix_arr, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(update_sets)))
    ax.set_xticklabels(update_sets)
    ax.set_yticks(range(len(hierarchy)))
    ax.set_yticklabels(hierarchy)
    for i in range(len(hierarchy)):
        for j in range(len(update_sets)):
            ax.text(j, i, f"{matrix_arr[i, j]:.3f}",
                    ha="center", va="center",
                    color="white" if matrix_arr[i, j] < matrix_arr.mean() else "black",
                    fontsize=10)
    ax.set_xlabel("update set")
    ax.set_ylabel("hierarchy depth")
    ax.set_title(f"P5: Two-axis ablation (mean acc)  source={src_mean:.3f}")
    fig.colorbar(im, ax=ax, label="accuracy")
    _save(fig, json_path.with_suffix(""))


# ----------------------------------------------------------------------------

PLOTTERS = {"p1": plot_p1, "p2": plot_p2, "p3": plot_p3, "p4": plot_p4, "p5": plot_p5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("protocol", choices=list(PLOTTERS.keys()))
    ap.add_argument("json_path", type=str)
    args = ap.parse_args()
    p = Path(args.json_path)
    if not p.exists():
        print(f"[error] not found: {p}", file=sys.stderr)
        sys.exit(1)
    PLOTTERS[args.protocol](p)


if __name__ == "__main__":
    main()
