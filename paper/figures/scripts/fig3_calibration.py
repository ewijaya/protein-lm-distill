#!/usr/bin/env python3
"""Figure 3: Reliability diagrams for Teacher, Baseline-medium, Synergy-medium.

1x3 panel of calibration plots with ECE values and perfect-calibration diagonal.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import COLORS, DOUBLE_COL, load_result, np, plt, savefig

# Load data
teacher_data = load_result("eval_synergy_medium_v2.json")["teacher_ece"]
baseline_data = load_result("eval_baseline_medium.json")["student_ece"]
synergy_data = load_result("eval_synergy_medium_v2.json")["student_ece"]

panels = [
    ("(a) Teacher", teacher_data, COLORS["teacher"]),
    ("(b) Baseline-medium", baseline_data, COLORS["baseline"]),
    ("(c) Synergy-medium", synergy_data, COLORS["synergy"]),
]

fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, DOUBLE_COL * 0.3))

for ax, (title, data, color) in zip(axes, panels):
    bin_stats = data["bin_stats"]
    ece = data["ece"]
    total_count = sum(b["count"] for b in bin_stats)

    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Perfect")

    # Plot bars for each bin
    for b in bin_stats:
        if b["count"] == 0 or b["avg_confidence"] is None:
            continue
        bin_center = (b["bin_lower"] + b["bin_upper"]) / 2
        bin_width = b["bin_upper"] - b["bin_lower"]
        # Width proportional to bin count
        scaled_width = bin_width * (b["count"] / total_count) * len(bin_stats)
        scaled_width = min(scaled_width, bin_width)  # cap at full bin width

        ax.bar(bin_center, b["avg_accuracy"], width=scaled_width,
               color=color, alpha=0.7, edgecolor="white", linewidth=0.5)

    # Plot accuracy line
    confs = []
    accs = []
    for b in bin_stats:
        if b["count"] > 0 and b["avg_confidence"] is not None:
            confs.append(b["avg_confidence"])
            accs.append(b["avg_accuracy"])
    if confs:
        ax.plot(confs, accs, "o-", color=color, markersize=3, linewidth=1,
                label="Model")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Confidence")
    if ax == axes[0]:
        ax.set_ylabel("Accuracy")
    ax.set_title(title, fontsize=9)

    # ECE annotation
    ax.text(0.05, 0.92, f"ECE = {ece:.3f}", transform=ax.transAxes,
            fontsize=7, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", alpha=0.8))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.tight_layout()
savefig(fig, "fig3_calibration")
