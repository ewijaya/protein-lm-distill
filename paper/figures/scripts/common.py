"""Shared styling, color palette, and data loading utilities for paper figures."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np  # noqa: F401 - used by figure scripts

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = Path(__file__).parent.parent / "pdf"

# Nature Communications style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,  # TrueType fonts in PDF
    "ps.fonttype": 42,
})

# Color palette
COLORS = {
    "baseline": "#4575b4",
    "uncertainty": "#fc8d59",
    "calibration": "#91cf60",
    "both": "#d73027",
    "synergy": "#d73027",
    "teacher": "#666666",
    "natural": "#984ea3",
}

# Model display names
MODEL_NAMES = {
    "baseline": "Baseline\n(Standard KD)",
    "uncertainty": "+Uncertainty",
    "calibration": "+Calibration",
    "both": "+Both\n(Combined)",
}

# Nature Communications single-column width: 88mm, double-column: 180mm
SINGLE_COL = 88 / 25.4  # inches
DOUBLE_COL = 180 / 25.4  # inches


def load_result(filename):
    """Load a results JSON file."""
    path = RESULTS_DIR / filename
    with open(path) as f:
        return json.load(f)


def load_ablation_results():
    """Load all 4 ablation results."""
    return {
        "baseline": load_result("ablation_baseline.json"),
        "uncertainty": load_result("ablation_uncertainty.json"),
        "calibration": load_result("ablation_calibration.json"),
        "both": load_result("ablation_both.json"),
    }


def load_scaling_results():
    """Load baseline vs synergy results for all 3 scales."""
    return {
        "tiny": {
            "baseline": load_result("eval_baseline_tiny.json"),
            "synergy": load_result("eval_synergy_tiny_v2.json"),
        },
        "small": {
            "baseline": load_result("eval_baseline_small.json"),
            "synergy": load_result("eval_synergy_small.json"),  # v1 kept
        },
        "medium": {
            "baseline": load_result("eval_baseline_medium.json"),
            "synergy": load_result("eval_synergy_medium_v2.json"),
        },
    }


# Natural amino acid distribution (from UniProt, used in evaluate.py)
NATURAL_AA_DIST = {
    "A": 0.0825, "R": 0.0553, "N": 0.0406, "D": 0.0545, "C": 0.0137,
    "Q": 0.0393, "E": 0.0675, "G": 0.0707, "H": 0.0227, "I": 0.0596,
    "L": 0.0966, "K": 0.0584, "M": 0.0242, "F": 0.0386, "P": 0.0470,
    "S": 0.0656, "T": 0.0534, "W": 0.0108, "Y": 0.0292, "V": 0.0687,
}

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")


def savefig(fig, name):
    """Save figure to PDF output directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.pdf"
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)
