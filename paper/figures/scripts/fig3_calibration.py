#!/usr/bin/env python3
"""Figure 3: Calibration comparison (ECE) across model scales.

Two-panel figure:
  (a) Horizontal bar chart of ECE values for Teacher, Synergy, Baseline
      at all three scales (medium, small, tiny).
  (b) Confidence distribution histograms for medium-scale models showing
      how each model distributes its prediction confidence.
"""

import sys
from pathlib import Path

import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent))
from common import COLORS, DOUBLE_COL, SINGLE_COL, load_result, load_scaling_results, np, plt, savefig

# Load scaling results for ECE across all scales
scaling = load_scaling_results()
scales = ["medium", "small", "tiny"]
scale_labels = ["Medium", "Small", "Tiny"]

# Extract ECE values
teacher_eces = []
synergy_eces = []
baseline_eces = []

for s in scales:
    # Teacher ECE comes from synergy eval files (same teacher)
    teacher_eces.append(scaling[s]["synergy"]["teacher_ece"]["ece"])
    synergy_eces.append(scaling[s]["synergy"]["student_ece"]["ece"])
    baseline_eces.append(scaling[s]["baseline"]["student_ece"]["ece"])

# Load medium-scale confidence distributions for panel (b)
teacher_data = load_result("eval_synergy_medium_v2.json")["teacher_ece"]
baseline_data = load_result("eval_baseline_medium.json")["student_ece"]
synergy_data = load_result("eval_synergy_medium_v2.json")["student_ece"]

fig = plt.figure(figsize=(DOUBLE_COL, SINGLE_COL * 0.7))
gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1], wspace=0.35)

ax_ece = fig.add_subplot(gs[0])
ax_conf = fig.add_subplot(gs[1])

# --- Panel (a): ECE bar chart ---
y = np.arange(len(scales))
bar_h = 0.22

bars_t = ax_ece.barh(y + bar_h, teacher_eces, height=bar_h,
                      color=COLORS["teacher"], label="Teacher", edgecolor="white", linewidth=0.5)
bars_s = ax_ece.barh(y, synergy_eces, height=bar_h,
                      color=COLORS["synergy"], label="Synergy (Ours)", edgecolor="white", linewidth=0.5)
bars_b = ax_ece.barh(y - bar_h, baseline_eces, height=bar_h,
                      color=COLORS["baseline"], label="Baseline", edgecolor="white", linewidth=0.5)

# Value labels
for bars in [bars_t, bars_s, bars_b]:
    for bar in bars:
        width = bar.get_width()
        ax_ece.text(width + 0.003, bar.get_y() + bar.get_height() / 2,
                    f"{width:.3f}", va="center", fontsize=6.5)

ax_ece.set_yticks(y)
ax_ece.set_yticklabels(scale_labels, fontsize=8)
ax_ece.set_xlabel("ECE (lower is better)")
ax_ece.set_xlim(0, max(max(teacher_eces), max(synergy_eces), max(baseline_eces)) * 1.25)
ax_ece.legend(fontsize=6.5, loc="upper right", frameon=False)
ax_ece.spines["top"].set_visible(False)
ax_ece.spines["right"].set_visible(False)
ax_ece.invert_yaxis()
ax_ece.set_title("(a) Expected Calibration Error", fontsize=9)

# --- Panel (b): Confidence distribution for medium ---
def get_conf_histogram(data):
    """Extract confidence values weighted by bin counts."""
    confs = []
    counts = []
    for b in data["bin_stats"]:
        if b["count"] > 0 and b["avg_confidence"] is not None:
            confs.append(b["avg_confidence"])
            counts.append(b["count"])
    return confs, counts

for label, data, color in [
    ("Teacher", teacher_data, COLORS["teacher"]),
    ("Synergy", synergy_data, COLORS["synergy"]),
    ("Baseline", baseline_data, COLORS["baseline"]),
]:
    confs, counts = get_conf_histogram(data)
    total = sum(counts)
    fracs = [c / total for c in counts]
    ax_conf.scatter(confs, fracs, color=color, s=20, zorder=5, edgecolors="white", linewidth=0.3)
    ax_conf.plot(confs, fracs, color=color, linewidth=1, alpha=0.7, label=label)

ax_conf.set_xlabel("Confidence")
ax_conf.set_ylabel("Fraction of predictions")
ax_conf.set_xlim(0, 1.05)
ax_conf.legend(fontsize=6.5, frameon=False)
ax_conf.spines["top"].set_visible(False)
ax_conf.spines["right"].set_visible(False)
ax_conf.set_title("(b) Confidence distribution (Medium)", fontsize=9)

fig.subplots_adjust(left=0.08, right=0.97, bottom=0.18, top=0.88, wspace=0.35)
savefig(fig, "fig3_calibration")
