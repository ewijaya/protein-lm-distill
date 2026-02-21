#!/usr/bin/env python3
"""Figure 11: HMMER hit-rate advantage of distilled students on lysozyme.

Two panels: (a) Lysozyme  (b) Conotoxin — showing that students generate
more family-specific sequences than the teacher despite having higher
perplexity.  Small@500 (89.5%) exceeds Teacher@1000 (69.0%) on lysozyme.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import DOUBLE_COL, PROJECT_ROOT, np, plt, savefig

# ── Data ────────────────────────────────────────────────────────────
RESULTS_DIR = PROJECT_ROOT / "results" / "finetune"
FAMILIES = ["lysozyme", "conotoxin"]
FAMILY_LABELS = {"lysozyme": "Lysozyme (PF00959)", "conotoxin": "Conotoxin (PF02950)"}
MODELS = ["teacher", "medium", "small", "tiny", "baseline-tiny"]
SUBSETS = [50, 100, 200, 500, 1000]

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


def load_hitrate_data():
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
                    hr = d["hmmer"]["hit_rate"]
                    if hr is not None:
                        data[fam][model][n] = hr * 100  # convert to %
    return data


# ── Figure ──────────────────────────────────────────────────────────
data = load_hitrate_data()

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
    ax.set_title(FAMILY_LABELS[fam], fontsize=9, fontweight="bold", pad=6)
    ax.set_xlabel("Fine-tuning sequences", fontsize=8)
    ax.set_xticks(SUBSETS)
    ax.set_xticklabels(["50", "100", "200", "500", "1k"], fontsize=6.5)
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.set_ylim(-3, 103)
    ax.set_ylabel("HMMER Hit Rate (%)", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", axis="y", linewidth=0.3, alpha=0.4)

# ── Lysozyme annotations ───────────────────────────────────────────
ax_lyso = axes[0]

# Annotate Small@1000 = 94%
ax_lyso.annotate(
    "94.0%",
    xy=(1000, 94.0),
    xytext=(-32, 8),
    textcoords="offset points",
    fontsize=6.5,
    fontweight="bold",
    color=MODEL_STYLE["small"]["color"],
    arrowprops=dict(
        arrowstyle="-",
        color=MODEL_STYLE["small"]["color"],
        linewidth=0.6,
    ),
)

# Annotate Teacher@1000 = 69%
ax_lyso.annotate(
    "69.0%",
    xy=(1000, 69.0),
    xytext=(6, -14),
    textcoords="offset points",
    fontsize=6.5,
    fontweight="bold",
    color=MODEL_STYLE["teacher"]["color"],
    arrowprops=dict(
        arrowstyle="-",
        color=MODEL_STYLE["teacher"]["color"],
        linewidth=0.6,
    ),
)

# Bracket / delta annotation between teacher and small at N=1000
mid_y = (69.0 + 94.0) / 2
ax_lyso.annotate(
    "",
    xy=(1150, 69.0),
    xytext=(1150, 94.0),
    arrowprops=dict(
        arrowstyle="<->",
        color="#333333",
        linewidth=0.8,
        shrinkA=2,
        shrinkB=2,
    ),
    annotation_clip=False,
)
ax_lyso.text(
    1250, mid_y, "+25 pp",
    fontsize=6,
    fontweight="bold",
    color="#333333",
    ha="left",
    va="center",
    clip_on=False,
)

# ── Conotoxin annotation ──────────────────────────────────────────
ax_cono = axes[1]

# Annotate Medium@1000 = 42.5%
ax_cono.annotate(
    "42.5%",
    xy=(1000, 42.5),
    xytext=(-32, 8),
    textcoords="offset points",
    fontsize=6.5,
    fontweight="bold",
    color=MODEL_STYLE["medium"]["color"],
    arrowprops=dict(
        arrowstyle="-",
        color=MODEL_STYLE["medium"]["color"],
        linewidth=0.6,
    ),
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

# Shared legend
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
savefig(fig, "fig11_finetune_hitrate")
