#!/usr/bin/env python3
"""Figure 4: Amino acid distribution deviation from natural.

Heatmap showing deviation of generated AA frequencies from UniProt natural
distribution. Rows: Teacher, Synergy-medium, Baseline-medium.
"""

import sys
from pathlib import Path

import matplotlib.colors as mcolors

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    AA_ORDER,
    DOUBLE_COL,
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

# Build matrix (exclude Natural row — it's all zeros by definition)
row_labels = ["Teacher", "Synergy-Medium", "Baseline-Medium"]
distributions = [teacher_aa, synergy_aa, baseline_aa]
natural_freqs = np.array([NATURAL_AA_DIST.get(aa, 0.0) for aa in AA_ORDER])

matrix = np.zeros((len(distributions), len(AA_ORDER)))
for i, dist in enumerate(distributions):
    for j, aa in enumerate(AA_ORDER):
        matrix[i, j] = dist.get(aa, 0.0)

# Divergence from natural
divergence = matrix - natural_freqs[np.newaxis, :]

# Symmetric colormap limits
vmax = np.max(np.abs(divergence))

fig, ax = plt.subplots(figsize=(DOUBLE_COL, SINGLE_COL * 0.55))

cmap = plt.cm.RdBu_r
norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

im = ax.imshow(divergence, cmap=cmap, norm=norm, aspect="auto")

# X-axis: amino acids
ax.set_xticks(np.arange(len(AA_ORDER)))
ax.set_xticklabels(AA_ORDER, fontsize=8, fontfamily="monospace")
ax.set_xlabel("Amino Acid")

# Y-axis: model labels
ax.set_yticks(np.arange(len(row_labels)))
ax.set_yticklabels(row_labels, fontsize=8)

# Colorbar
cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, aspect=15)
cbar.set_label("$\\Delta$ freq.", fontsize=8)
cbar.ax.tick_params(labelsize=7)

# Remove title (caption handles it)
ax.tick_params(length=2, width=0.5)

fig.tight_layout()
savefig(fig, "fig4_aa_distribution")
