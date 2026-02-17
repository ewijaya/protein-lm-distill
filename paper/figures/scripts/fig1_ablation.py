#!/usr/bin/env python3
"""Figure 1: Ablation study showing V-shape pattern.

3-panel grouped bar chart: PPL Ratio, KL Divergence, ECE.
Highlights that individual methods perform worse but combined performs better.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    COLORS,
    DOUBLE_COL,
    MODEL_NAMES,
    SINGLE_COL,
    load_ablation_results,
    np,
    plt,
    savefig,
)

# Load data
results = load_ablation_results()
methods = ["baseline", "uncertainty", "calibration", "both"]
labels = [MODEL_NAMES[m] for m in methods]

# Extract metrics
ppl_ratios = [results[m]["perplexity_ratio"] for m in methods]
kl_divs = [results[m]["kl_divergence"] for m in methods]
eces = [results[m]["student_ece"]["ece"] for m in methods]

# Colors for each method
bar_colors = [COLORS[m] for m in methods]

# Short x-axis labels (full names go in caption)
short_labels = ["Baseline", "+Uncert.", "+Calib.", "+Both"]

# Create figure — double-column width for 3 panels
fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, SINGLE_COL * 0.65))

metrics = [
    ("Perplexity Ratio", ppl_ratios, "(a)"),
    ("KL Divergence", kl_divs, "(b)"),
    ("ECE", eces, "(c)"),
]

for ax, (title, values, panel) in zip(axes, metrics):
    x = np.arange(len(methods))
    bars = ax.bar(x, values, width=0.55, color=bar_colors, edgecolor="white",
                  linewidth=0.5)

    # Baseline reference line
    ax.axhline(y=values[0], color=COLORS["baseline"], linestyle="--",
               linewidth=0.8, alpha=0.5, zorder=0)

    ax.set_ylabel(title)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=7, rotation=30, ha="right")
    ax.set_xlim(-0.5, len(methods) - 0.5)

    # Panel label
    ax.text(-0.15, 1.05, panel, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom")

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.tight_layout()
savefig(fig, "fig1_ablation")
