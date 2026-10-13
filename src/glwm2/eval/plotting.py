"""One plotting style for every figure: Okabe-Ito colour-blind-safe palette, vector PDF output."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from .. import PAPER_DIR, RESULTS_DIR

OKABE_ITO = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

SPLIT_COLORS = {
    "id_val": "#999999",
    "id_test": "#000000",
    "ood_val": "#F0E442",
    "ood_depth": "#D55E00",
    "ood_length": "#0072B2",
    "ood_both": "#CC79A7",
}

FAMILY_COLORS = {
    "reference": "#000000",
    "posthoc": "#56B4E9",
    "joint": "#E69F00",
    "ours": "#009E73",
}


def set_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": mpl.cycler(color=OKABE_ITO[1:] + OKABE_ITO[:1]),
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def savefig(fig, name: str, paper: bool = True) -> Path:
    """Save as vector PDF into paper/figures (or results/figures) plus a PNG preview."""
    out_dir = PAPER_DIR / "figures" if paper else RESULTS_DIR / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{name}.pdf"
    fig.tight_layout()
    fig.savefig(pdf, bbox_inches="tight")
    prev = RESULTS_DIR / "figures"
    prev.mkdir(parents=True, exist_ok=True)
    fig.savefig(prev / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    return pdf
