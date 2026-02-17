#!/usr/bin/env python3
"""Figure 3: Calibration comparison (ECE) across model scales.

Horizontal bar chart of ECE values for Teacher, Synergy, Baseline
at all three scales (medium, small, tiny).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import COLORS, SINGLE_COL, load_scaling_results, np, plt, savefig

# Load scaling results for ECE across all scales
scaling = load_scaling_results()
scales = ["medium", "small", "tiny"]
scale_labels = ["Medium", "Small", "Tiny"]

# Extract ECE values
teacher_eces = []
synergy_eces = []
baseline_eces = []

for s in scales:
    teacher_eces.append(scaling[s]["synergy"]["teacher_ece"]["ece"])
    synergy_eces.append(scaling[s]["synergy"]["student_ece"]["ece"])
    baseline_eces.append(scaling[s]["baseline"]["student_ece"]["ece"])

fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.7))

y = np.arange(len(scales))
bar_h = 0.22

bars_t = ax.barh(y + bar_h, teacher_eces, height=bar_h,
                 color=COLORS["teacher"], label="Teacher", edgecolor="white", linewidth=0.5)
bars_s = ax.barh(y, synergy_eces, height=bar_h,
                 color=COLORS["synergy"], label="Synergy (Ours)", edgecolor="white", linewidth=0.5)
bars_b = ax.barh(y - bar_h, baseline_eces, height=bar_h,
                 color=COLORS["baseline"], label="Baseline", edgecolor="white", linewidth=0.5)

# Value labels
for bars in [bars_t, bars_s, bars_b]:
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{width:.3f}", va="center", fontsize=7)

ax.set_yticks(y)
ax.set_yticklabels(scale_labels, fontsize=8)
ax.set_xlabel("ECE (lower is better)")
ax.set_xlim(0, max(max(teacher_eces), max(synergy_eces), max(baseline_eces)) * 1.25)
ax.legend(fontsize=7, loc="upper right", frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.invert_yaxis()

fig.tight_layout()
savefig(fig, "fig3_calibration")
