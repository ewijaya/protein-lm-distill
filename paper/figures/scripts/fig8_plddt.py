#!/usr/bin/env python3
"""Figure 8: pLDDT structural quality comparison.

Grouped box plot of ESMFold pLDDT scores: teacher, then synergy vs baseline
at medium scale side-by-side, plus small and tiny synergy models.
Data from results/plddt_benchmark.json.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import COLORS, SINGLE_COL, plt, savefig, load_result
from matplotlib.patches import Patch

data = load_result("plddt_benchmark.json")

# Groups: Teacher | Medium (synergy vs baseline) | Small | Tiny
group_labels = ["Teacher", "Medium", "Small", "Tiny"]
positions = [0, 1.8, 2.4, 3.6, 4.8]
box_data = [
    data["teacher"]["plddt_scores"],
    data["synergy-medium"]["plddt_scores"],
    data["baseline-medium"]["plddt_scores"],
    data["synergy-small"]["plddt_scores"],
    data["synergy-tiny"]["plddt_scores"],
]
box_colors = [
    COLORS["teacher"],
    COLORS["synergy"],
    COLORS["baseline"],
    COLORS["synergy"],
    COLORS["synergy"],
]
box_labels = ["Teacher", "Synergy", "Baseline", "Synergy", "Synergy"]
means = [d["mean_plddt"] for d in [
    data["teacher"], data["synergy-medium"], data["baseline-medium"],
    data["synergy-small"], data["synergy-tiny"],
]]

fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.7))

bp = ax.boxplot(box_data, positions=positions, widths=0.45,
                patch_artist=True, showmeans=True,
                meanprops=dict(marker="D", markerfacecolor="white",
                               markeredgecolor="black", markersize=3.5),
                medianprops=dict(color="black", linewidth=1),
                flierprops=dict(marker="o", markersize=2.5, alpha=0.5))

for patch, color in zip(bp["boxes"], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Annotate means
for pos, m in zip(positions, means):
    ax.text(pos, m + 2, f"{m:.1f}", ha="center", va="bottom", fontsize=6,
            fontweight="bold")

# Reference line at pLDDT=70
ax.axhline(y=70, color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
ax.text(positions[-1] + 0.3, 71, "pLDDT=70", ha="right", va="bottom",
        fontsize=6, color="gray")

# Group labels
ax.set_xticks([0, 2.1, 3.6, 4.8])
ax.set_xticklabels(group_labels, fontsize=8)

# Bracket for Medium pair
bracket_y = -2
ax.annotate("", xy=(1.8, bracket_y), xytext=(2.4, bracket_y),
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
            annotation_clip=False)

ax.set_ylabel("pLDDT Score")
ax.set_ylim(15, 95)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend
legend_elements = [
    Patch(facecolor=COLORS["teacher"], alpha=0.7, label="Teacher"),
    Patch(facecolor=COLORS["synergy"], alpha=0.7, label="Synergy"),
    Patch(facecolor=COLORS["baseline"], alpha=0.7, label="Baseline"),
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=6,
          framealpha=0.8)

fig.tight_layout()
savefig(fig, "fig8_plddt")
