#!/usr/bin/env python3
"""Figure 6: Inference speed comparison.

Horizontal bar chart with inference time and speedup annotations.
Data from results/throughput_benchmark.json.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import COLORS, SINGLE_COL, np, plt, savefig, load_result

# Load actual benchmark data
data = load_result("throughput_benchmark.json")

models = ["Teacher", "Medium", "Small", "Tiny"]
keys = ["teacher", "synergy-medium", "synergy-small", "synergy-tiny"]
times = [data[k]["avg_time_per_seq_s"] for k in keys]
speedups = [data[k]["speedup_vs_teacher"] for k in keys]
colors = [
    COLORS["teacher"],
    COLORS["synergy"],
    COLORS["synergy"],
    COLORS["synergy"],
]

fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.55))

y = np.arange(len(models))
bars = ax.barh(y, times, height=0.55, color=colors, edgecolor="white",
               linewidth=0.5)

# Annotate speedup values
for i, (bar, spd) in enumerate(zip(bars, speedups)):
    if spd > 1.0:
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{spd:.1f}x faster", ha="left", va="center", fontsize=7,
                fontweight="bold", color=COLORS["synergy"])
    else:
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                "(reference)", ha="left", va="center", fontsize=7,
                color="gray")

# Annotate time values inside bars
for bar, t in zip(bars, times):
    ax.text(bar.get_width() - 0.05, bar.get_y() + bar.get_height() / 2,
            f"{t:.2f}s", ha="right", va="center", fontsize=7,
            color="white", fontweight="bold")

ax.set_yticks(y)
ax.set_yticklabels(models, fontsize=8)
ax.set_xlabel("Inference Time (seconds per sequence)")
ax.set_xlim(0, max(times) * 1.5)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.invert_yaxis()  # Teacher at top

fig.tight_layout()
savefig(fig, "fig6_speed")
