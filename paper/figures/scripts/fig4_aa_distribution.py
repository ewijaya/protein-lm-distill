#!/usr/bin/env python3
"""Figure 4: Amino acid distribution heatmap.

Rows: Natural, Teacher, Synergy-medium, Baseline-medium.
Columns: amino acids in AA_ORDER.
Diverging colormap centered on natural distribution.
"""

import sys
from pathlib import Path

import matplotlib.colors as mcolors

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    AA_ORDER,
    NATURAL_AA_DIST,
    SINGLE_COL,
    load_result,
    np,
    plt,
    savefig,
)

# Load data
synergy_med = load_result("eval_synergy_medium_v2.json")
baseline_med = load_result("eval_baseline_medium.json")

teacher_aa = synergy_med["teacher_generation"]["aa_distribution"]
synergy_aa = synergy_med["student_generation"]["aa_distribution"]
baseline_aa = baseline_med["student_generation"]["aa_distribution"]

# Build matrix: rows are distributions, columns are amino acids
row_labels = ["Natural", "Teacher", "Synergy-medium", "Baseline-medium"]
distributions = [NATURAL_AA_DIST, teacher_aa, synergy_aa, baseline_aa]

matrix = np.zeros((len(distributions), len(AA_ORDER)))
for i, dist in enumerate(distributions):
    for j, aa in enumerate(AA_ORDER):
        matrix[i, j] = dist.get(aa, 0.0)

# Compute divergence from natural
natural_row = matrix[0]
divergence = matrix - natural_row[np.newaxis, :]

# Symmetric limits for diverging colormap
vmax = np.max(np.abs(divergence))

fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.5))

cmap = plt.cm.RdBu_r
norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

im = ax.imshow(divergence, cmap=cmap, norm=norm, aspect="auto")

ax.set_xticks(np.arange(len(AA_ORDER)))
ax.set_xticklabels(AA_ORDER, fontsize=6)
ax.set_yticks(np.arange(len(row_labels)))
ax.set_yticklabels(row_labels, fontsize=7)

# Add text annotations
for i in range(divergence.shape[0]):
    for j in range(divergence.shape[1]):
        val = divergence[i, j]
        text_color = "white" if abs(val) > vmax * 0.6 else "black"
        ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                fontsize=4, color=text_color)

cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label("Deviation from natural", fontsize=7)
cbar.ax.tick_params(labelsize=6)

ax.set_xlabel("Amino Acid")
ax.set_title("Amino Acid Distribution Deviation from Natural", fontsize=9)

fig.tight_layout()
savefig(fig, "fig4_aa_distribution")
