"""Shared visual style for the manuscript figures (Phase 5).

One palette and one set of sizes for every figure. Dataset colours are fixed
(the same dataset has the same colour everywhere); the three slots are the
first three of the validated reference palette, which stay distinguishable for
colour-blind readers when shown together. NF-v2 and NF-v3 are told apart by
fill (v3 solid, v2 hatched or lighter), never by a new hue.
"""
from __future__ import annotations

import matplotlib as mpl

DATASET_COLOR = {"ToN": "#2a78d6", "BoT": "#eb6834", "CIC": "#1baf7a"}
INK = "#0b0b0b"          # primary text
INK_2 = "#52514e"        # secondary text, axes
GRID = "#d9d8d4"         # recessive grid and box edges
SURFACE = "#ffffff"      # print surface
NEUTRAL = "#f0efec"      # neutral fill (boxes, diverging midpoint)
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ("#2a78d6", "#f0efec", "#e34948")

FULL_WIDTH_IN = 7.0      # two-column journal width
HALF_WIDTH_IN = 3.4      # single-column width


def apply() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
