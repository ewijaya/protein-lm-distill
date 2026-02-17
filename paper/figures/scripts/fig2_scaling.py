#!/usr/bin/env python3
"""Figure 2: Scaling behavior - baseline vs synergy PPL ratio at 3 scales.

Grouped bar chart with percentage improvement annotations and compression ratios.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import COLORS, SINGLE_COL, load_scaling_results, np, plt, savefig

# Load data
results = load_scaling_results()
scales = ["tiny", "small", "medium"]
scale_labels = ["Tiny", "Small", "Medium"]

baseline_ppl = [results[s]["baseline"]["perplexity_ratio"] for s in scales]
synergy_ppl = [results[s]["synergy"]["perplexity_ratio"] for s in scales]
compression = [results[s]["baseline"]["compression_ratio"] for s in scales]

# Percentage improvements
improvements = [
    (b - s) / b * 100 for b, s in zip(baseline_ppl, synergy_ppl)
]

# Create figure
fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.75))

x = np.arange(len(scales))
width = 0.35

bars_b = ax.bar(x - width / 2, baseline_ppl, width, label="Baseline (Standard KD)",
                color=COLORS["baseline"], edgecolor="white", linewidth=0.5)
bars_s = ax.bar(x + width / 2, synergy_ppl, width, label="Synergy (Ours)",
                color=COLORS["synergy"], edgecolor="white", linewidth=0.5)

# Annotate percentage improvement above synergy bars
for i, (bar, imp) in enumerate(zip(bars_s, improvements)):
    ax.annotate(f"{imp:.0f}%",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=7, fontweight="bold",
                color=COLORS["synergy"])

# Add compression ratios as secondary x-axis labels
ax.set_xticks(x)
ax.set_xticklabels(
    [f"{lab}\n({cr:.1f}x)" for lab, cr in zip(scale_labels, compression)],
    fontsize=7,
)

ax.set_ylabel("Perplexity Ratio (student / teacher)", fontsize=8)
ax.set_xlabel("Model Scale (compression ratio)")
ax.legend(fontsize=7, frameon=False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout(pad=1.2)
savefig(fig, "fig2_scaling")
