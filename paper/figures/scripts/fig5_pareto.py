#!/usr/bin/env python3
"""Figure 5: Pareto frontier - compression ratio vs perplexity ratio.

Baseline and synergy models plotted with different markers.
Pareto frontier connected for synergy models.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import COLORS, SINGLE_COL, load_scaling_results, np, plt, savefig

# Load data
results = load_scaling_results()
scales = ["tiny", "small", "medium"]
scale_labels = ["Tiny", "Small", "Medium"]

baseline_cr = [results[s]["baseline"]["compression_ratio"] for s in scales]
baseline_ppl = [results[s]["baseline"]["perplexity_ratio"] for s in scales]
synergy_cr = [results[s]["synergy"]["compression_ratio"] for s in scales]
synergy_ppl = [results[s]["synergy"]["perplexity_ratio"] for s in scales]

fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.85))

# Plot baseline points
ax.scatter(baseline_cr, baseline_ppl, color=COLORS["baseline"], marker="s",
           s=60, zorder=5, label="Baseline (Standard KD)", edgecolors="white",
           linewidth=0.5)

# Plot synergy points
ax.scatter(synergy_cr, synergy_ppl, color=COLORS["synergy"], marker="o",
           s=60, zorder=5, label="Synergy (Ours)", edgecolors="white",
           linewidth=0.5)

# Connect Pareto frontier for synergy (sort by compression ratio)
sorted_idx = np.argsort(synergy_cr)
sorted_cr = [synergy_cr[i] for i in sorted_idx]
sorted_ppl = [synergy_ppl[i] for i in sorted_idx]
ax.plot(sorted_cr, sorted_ppl, color=COLORS["synergy"], linewidth=1,
        linestyle="-", alpha=0.7, zorder=4)

# Label each point
offset = {"Tiny": (8, 8), "Small": (8, 8), "Medium": (8, -12)}
for i, label in enumerate(scale_labels):
    ox, oy = offset[label]
    ax.annotate(label, (baseline_cr[i], baseline_ppl[i]),
                xytext=(ox, oy), textcoords="offset points",
                fontsize=7, color=COLORS["baseline"],
                arrowprops=dict(arrowstyle="-", color=COLORS["baseline"],
                                linewidth=0.5))
    ax.annotate(label, (synergy_cr[i], synergy_ppl[i]),
                xytext=(ox, oy), textcoords="offset points",
                fontsize=7, color=COLORS["synergy"],
                arrowprops=dict(arrowstyle="-", color=COLORS["synergy"],
                                linewidth=0.5))

ax.set_xlabel("Compression Ratio")
ax.set_ylabel("Perplexity Ratio (student / teacher)")
ax.legend(fontsize=7, frameon=False, loc="upper left")


ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
savefig(fig, "fig5_pareto")
