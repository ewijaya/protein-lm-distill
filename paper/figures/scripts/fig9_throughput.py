#!/usr/bin/env python3
"""Figure 9: Throughput and GPU memory benchmark.

Dual-axis bar chart: sequences/min (left) and GPU memory (right).
Data from results/throughput_benchmark.json.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import COLORS, SINGLE_COL, np, plt, savefig, load_result

data = load_result("throughput_benchmark.json")

labels = ["Teacher", "Medium", "Small", "Tiny"]
keys = ["teacher", "synergy-medium", "synergy-small", "synergy-tiny"]
throughput = [data[k]["sequences_per_min"] for k in keys]
memory = [data[k]["peak_gpu_memory_mb"] / 1024 for k in keys]  # Convert to GB
colors = [COLORS["teacher"], COLORS["synergy"], COLORS["synergy"], COLORS["synergy"]]

fig, ax1 = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.7))

x = np.arange(len(labels))
width = 0.35

# Throughput bars (left axis)
bars1 = ax1.bar(x - width / 2, throughput, width, color=colors, alpha=0.85,
                edgecolor="white", linewidth=0.5, label="Throughput")
ax1.set_ylabel("Sequences / min", color=COLORS["synergy"])
ax1.tick_params(axis="y", labelcolor=COLORS["synergy"])

# Annotate throughput
for bar, t in zip(bars1, throughput):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
             f"{t:.0f}", ha="center", va="bottom", fontsize=6,
             fontweight="bold", color=COLORS["synergy"])

# GPU memory bars (right axis)
ax2 = ax1.twinx()
bars2 = ax2.bar(x + width / 2, memory, width, color=colors, alpha=0.35,
                edgecolor="gray", linewidth=0.5, hatch="//", label="GPU Memory")
ax2.set_ylabel("GPU Memory (GB)", color="gray")
ax2.tick_params(axis="y", labelcolor="gray")

# Annotate memory
for bar, m in zip(bars2, memory):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
             f"{m:.1f}", ha="center", va="bottom", fontsize=6, color="gray")

ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=8)

ax1.spines["top"].set_visible(False)
ax2.spines["top"].set_visible(False)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=COLORS["synergy"], alpha=0.85, label="Throughput (seq/min)"),
    Patch(facecolor="gray", alpha=0.35, hatch="//", label="GPU Memory (GB)"),
]
ax1.legend(handles=legend_elements, loc="upper left", fontsize=6,
           framealpha=0.8)

fig.tight_layout()
savefig(fig, "fig9_throughput")
