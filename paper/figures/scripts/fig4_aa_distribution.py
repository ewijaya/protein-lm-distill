#!/usr/bin/env python3
"""Figure 4: Amino acid distribution deviation from natural.

Heatmap showing deviation of generated AA frequencies from UniProt natural
distribution, with MAD summary bar on the right.
Rows: Teacher, Synergy-medium, Baseline-medium.
"""

import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec

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

# Build matrix
row_labels = ["Teacher", "Synergy-Medium", "Baseline-Medium"]
distributions = [teacher_aa, synergy_aa, baseline_aa]
natural_freqs = np.array([NATURAL_AA_DIST.get(aa, 0.0) for aa in AA_ORDER])

matrix = np.zeros((len(distributions), len(AA_ORDER)))
for i, dist in enumerate(distributions):
    for j, aa in enumerate(AA_ORDER):
        matrix[i, j] = dist.get(aa, 0.0)

# Divergence from natural
divergence = matrix - natural_freqs[np.newaxis, :]

# Mean absolute deviation per model
mad = np.mean(np.abs(divergence), axis=1)

# Symmetric colormap limits
vmax = np.max(np.abs(divergence))

# Layout: heatmap (wide) + MAD bar (narrow)
fig = plt.figure(figsize=(DOUBLE_COL, SINGLE_COL * 0.55))
gs = gridspec.GridSpec(1, 2, width_ratios=[6, 1], wspace=0.15)

ax_heat = fig.add_subplot(gs[0])
ax_bar = fig.add_subplot(gs[1])

# --- Heatmap ---
cmap = plt.cm.RdBu_r
norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
im = ax_heat.imshow(divergence, cmap=cmap, norm=norm, aspect="auto")

ax_heat.set_xticks(np.arange(len(AA_ORDER)))
ax_heat.set_xticklabels(AA_ORDER, fontsize=8, fontfamily="monospace")
ax_heat.set_xlabel("Amino Acid")

ax_heat.set_yticks(np.arange(len(row_labels)))
ax_heat.set_yticklabels(row_labels, fontsize=8)

cbar = fig.colorbar(im, ax=ax_heat, shrink=0.8, pad=0.02, aspect=15)
cbar.set_label("$\\Delta$ freq.", fontsize=8)
cbar.ax.tick_params(labelsize=7)

ax_heat.tick_params(length=2, width=0.5)

# --- MAD bar chart ---
bar_colors = ["#555555", "#555555", "#555555"]
bars = ax_bar.barh(np.arange(len(row_labels)), mad, color=bar_colors,
                   edgecolor="white", linewidth=0.5, height=0.6)

# Value labels on each bar
for i, (bar, val) in enumerate(zip(bars, mad)):
    ax_bar.text(val + 0.0003, i, f"{val:.4f}", va="center", fontsize=7)

ax_bar.set_yticks([])
ax_bar.set_xlabel("MAD", fontsize=8)
ax_bar.set_xlim(0, max(mad) * 1.45)
ax_bar.spines["top"].set_visible(False)
ax_bar.spines["right"].set_visible(False)
ax_bar.tick_params(labelsize=7, length=2, width=0.5)
ax_bar.invert_yaxis()  # match heatmap row order (top to bottom)

fig.subplots_adjust(left=0.12, right=0.95, bottom=0.2, top=0.95)
savefig(fig, "fig4_aa_distribution")
