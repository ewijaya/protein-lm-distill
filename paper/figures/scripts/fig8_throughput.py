#!/usr/bin/env python3
"""Figure 9: Throughput and GPU memory benchmark.

Dual-axis bar chart: sequences/min (left) and GPU memory (right).
Data from results/throughput_benchmark.json.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import SINGLE_COL, np, plt, savefig, load_result
from matplotlib.patches import Patch

data = load_result("throughput_benchmark.json")

labels = ["Teacher", "Medium", "Small", "Tiny"]
keys = ["teacher", "synergy-medium", "synergy-small", "synergy-tiny"]
throughput = [data[k]["sequences_per_min"] for k in keys]
memory = [data[k]["peak_gpu_memory_mb"] / 1024 for k in keys]  # Convert to GB

COLOR_THROUGHPUT = "#2166ac"  # blue
COLOR_MEMORY = "#b2182b"     # dark red

fig, ax1 = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.75))

x = np.arange(len(labels))
width = 0.35

# Throughput bars (left axis)
bars1 = ax1.bar(x - width / 2, throughput, width, color=COLOR_THROUGHPUT,
                alpha=0.85, edgecolor="white", linewidth=0.5)
ax1.set_ylabel("Sequences / min", color=COLOR_THROUGHPUT)
ax1.tick_params(axis="y", labelcolor=COLOR_THROUGHPUT)

# Annotate throughput
for bar, t in zip(bars1, throughput):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
             f"{t:.0f}", ha="center", va="bottom", fontsize=6,
             fontweight="bold", color=COLOR_THROUGHPUT)

# GPU memory bars (right axis)
ax2 = ax1.twinx()
bars2 = ax2.bar(x + width / 2, memory, width, color=COLOR_MEMORY,
                alpha=0.6, edgecolor="white", linewidth=0.5, hatch="//")
ax2.set_ylabel("GPU Memory (GB)", color=COLOR_MEMORY)
ax2.tick_params(axis="y", labelcolor=COLOR_MEMORY)

# Annotate memory
for bar, m in zip(bars2, memory):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
             f"{m:.1f}", ha="center", va="bottom", fontsize=6,
             fontweight="bold", color=COLOR_MEMORY)

ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=8)

ax1.spines["top"].set_visible(False)
ax2.spines["top"].set_visible(False)

# Legend below x-axis labels
legend_elements = [
    Patch(facecolor=COLOR_THROUGHPUT, alpha=0.85, label="Throughput (seq/min)"),
    Patch(facecolor=COLOR_MEMORY, alpha=0.6, hatch="//", label="GPU Memory (GB)"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=6,
           frameon=False, bbox_to_anchor=(0.5, -0.02))

fig.tight_layout(rect=[0, 0.06, 1, 1])  # leave room for legend at bottom
savefig(fig, "fig8_throughput")
