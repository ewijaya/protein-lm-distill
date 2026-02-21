#!/usr/bin/env python3
"""Figure 10: Fine-tuning sample efficiency curves.

Two panels (Conotoxin, Lysozyme) showing test perplexity vs training set
size for all five models.  Panel (a) conotoxin: students dominate teacher
at all N.  Panel (b) lysozyme: students converge toward teacher, setting
up the hit-rate story in Figure 11.  AMP omitted (supplementary).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import DOUBLE_COL, PROJECT_ROOT, plt, savefig

# ── Data ────────────────────────────────────────────────────────────
RESULTS_DIR = PROJECT_ROOT / "results" / "finetune"
FAMILIES = ["conotoxin", "lysozyme"]
FAMILY_LABELS = {
    "conotoxin": "Conotoxins",
    "lysozyme": "Lysozymes",
}
MODELS = ["teacher", "medium", "small", "tiny", "baseline-tiny"]
SUBSETS = [50, 100, 200, 500, 1000]

# ── Model styling ──────────────────────────────────────────────────
MODEL_STYLE = {
    "teacher": {
        "label": "Teacher (738M)",
        "color": "#555555",
        "marker": "s",
        "ls": "-",
        "lw": 2.0,
        "ms": 6,
        "zorder": 5,
    },
    "medium": {
        "label": "Synergy-Medium (194M)",
        "color": "#1b9e77",
        "marker": "^",
        "ls": "-",
        "lw": 1.5,
        "ms": 6,
        "zorder": 4,
    },
    "small": {
        "label": "Synergy-Small (78M)",
        "color": "#d95f02",
        "marker": "D",
        "ls": "-",
        "lw": 1.5,
        "ms": 5,
        "zorder": 4,
    },
    "tiny": {
        "label": "Synergy-Tiny (37M)",
        "color": "#7570b3",
        "marker": "o",
        "ls": "-",
        "lw": 1.5,
        "ms": 6,
        "zorder": 4,
    },
    "baseline-tiny": {
        "label": "Baseline-Tiny (37M)",
        "color": "#e7298a",
        "marker": "v",
        "ls": "--",
        "lw": 1.3,
        "ms": 6,
        "zorder": 3,
    },
}


def load_finetune_results():
    """Load result JSONs into a nested dict[family][model][n] = ppl."""
    data = {}
    for fam in FAMILIES:
        data[fam] = {}
        for model in MODELS:
            data[fam][model] = {}
            for n in SUBSETS:
                fp = RESULTS_DIR / f"{fam}-{model}-{n}.json"
                if fp.exists() and fp.stat().st_size > 0:
                    with open(fp) as f:
                        d = json.load(f)
                    data[fam][model][n] = d["test_perplexity"]
    return data


# ── Figure ──────────────────────────────────────────────────────────
data = load_finetune_results()

fig, axes = plt.subplots(
    1, 2,
    figsize=(DOUBLE_COL, DOUBLE_COL * 0.40),
    sharey=False,
)

for ax, fam in zip(axes, FAMILIES):
    for model in MODELS:
        sty = MODEL_STYLE[model]
        xs = sorted(data[fam][model].keys())
        ys = [data[fam][model][n] for n in xs]

        ax.plot(
            xs, ys,
            color=sty["color"],
            marker=sty["marker"],
            markersize=sty["ms"],
            markeredgecolor="white",
            markeredgewidth=0.5,
            linestyle=sty["ls"],
            linewidth=sty["lw"],
            label=sty["label"],
            zorder=sty["zorder"],
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(FAMILY_LABELS[fam], fontsize=9, fontweight="bold", pad=6)
    ax.set_xlabel("Fine-tuning sequences", fontsize=8)
    ax.set_xticks(SUBSETS)
    ax.set_xticklabels(["50", "100", "200", "500", "1k"], fontsize=7)
    ax.tick_params(axis="x", which="minor", bottom=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", axis="y", linewidth=0.3, alpha=0.4)

axes[0].set_ylabel("Test Perplexity", fontsize=9)

# ── Annotations ─────────────────────────────────────────────────────
# Panel (a): highlight teacher vs Medium gap at N=50
ax_cono = axes[0]
teacher_50 = data["conotoxin"]["teacher"][50]
medium_50 = data["conotoxin"]["medium"][50]
ax_cono.annotate(
    f"{teacher_50:.0f}",
    xy=(50, teacher_50),
    xytext=(12, 4),
    textcoords="offset points",
    fontsize=6.5,
    fontweight="bold",
    color=MODEL_STYLE["teacher"]["color"],
)
ax_cono.annotate(
    f"{medium_50:.0f}",
    xy=(50, medium_50),
    xytext=(12, -8),
    textcoords="offset points",
    fontsize=6.5,
    fontweight="bold",
    color=MODEL_STYLE["medium"]["color"],
)
# Ratio callout between the two points
ratio = teacher_50 / medium_50
geo_mean = (teacher_50 * medium_50) ** 0.5  # midpoint on log scale
ax_cono.annotate(
    f"{ratio:.1f}x gap",
    xy=(55, geo_mean),
    fontsize=7,
    fontweight="bold",
    fontstyle="italic",
    color="#333333",
    ha="left",
    va="center",
)

# Panel (b): annotate convergence at N=1000
ax_lyso = axes[1]
teacher_1k = data["lysozyme"]["teacher"][1000]
medium_1k = data["lysozyme"]["medium"][1000]
ax_lyso.annotate(
    f"{teacher_1k:.0f}",
    xy=(1000, teacher_1k),
    xytext=(-30, -10),
    textcoords="offset points",
    fontsize=6.5,
    fontweight="bold",
    color=MODEL_STYLE["teacher"]["color"],
)
ax_lyso.annotate(
    f"{medium_1k:.0f}",
    xy=(1000, medium_1k),
    xytext=(-30, 6),
    textcoords="offset points",
    fontsize=6.5,
    fontweight="bold",
    color=MODEL_STYLE["medium"]["color"],
)

# Panel labels
for i, ax in enumerate(axes):
    ax.text(
        -0.02, 1.08,
        chr(ord("a") + i),
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="right",
    )

# Shared legend below the panels
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="lower center",
    ncol=5,
    fontsize=6.5,
    frameon=False,
    bbox_to_anchor=(0.5, -0.06),
    columnspacing=1.0,
    handlelength=2.2,
)

fig.tight_layout(w_pad=3.0)
fig.subplots_adjust(bottom=0.20)
savefig(fig, "fig10_finetune_efficiency")
