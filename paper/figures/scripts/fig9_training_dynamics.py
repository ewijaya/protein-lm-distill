#!/usr/bin/env python3
"""Figure 7: Training dynamics showing early divergence.

Line plot of training loss at early steps for synergy-tiny-v1, v2, and baseline.
Highlights the critical window in the first ~200 steps with warmup shading.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import COLORS, SINGLE_COL, np, plt, savefig

# Hardcoded training loss data from mechanistic explanation
data = {
    "synergy-tiny-v1": {
        "steps": [0, 50, 200, 1000],
        "losses": [6.62, 5.37, 4.99, 4.41],
    },
    "synergy-tiny-v2": {
        "steps": [0, 50, 200, 1000],
        "losses": [7.93, 5.39, 4.99, 4.40],
    },
    "baseline-tiny": {
        "steps": [0, 50, 200, 1000],
        "losses": [7.94, 6.18, 5.61, 4.79],
    },
}

line_styles = {
    "synergy-tiny-v1": {"color": COLORS["synergy"], "linestyle": "--",
                        "marker": "^", "label": "Synergy-tiny-v1"},
    "synergy-tiny-v2": {"color": COLORS["synergy"], "linestyle": "-",
                        "marker": "o", "label": "Synergy-tiny-v2"},
    "baseline-tiny": {"color": COLORS["baseline"], "linestyle": "-",
                      "marker": "s", "label": "Baseline-tiny"},
}

fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.75))

# Shade warmup region (first 200 steps)
ax.axvspan(0, 200, alpha=0.1, color="orange", label="Warmup period")

for name, d in data.items():
    style = line_styles[name]
    ax.plot(d["steps"], d["losses"], linewidth=1.5, markersize=5,
            color=style["color"], linestyle=style["linestyle"],
            marker=style["marker"], label=style["label"], zorder=5)

    # Label final values
    ax.text(d["steps"][-1] + 20, d["losses"][-1],
            f"{d['losses'][-1]:.2f}", fontsize=7, va="center",
            color=style["color"])

# Annotate critical window
ax.annotate("Critical window", xy=(100, 5.8), fontsize=7, ha="center",
            style="italic", color="darkorange")

ax.set_xlabel("Training Steps")
ax.set_ylabel("Training Loss")
ax.set_xlim(-30, 1100)
ax.legend(fontsize=7, frameon=False, loc="upper right")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
savefig(fig, "fig9_training_dynamics")
