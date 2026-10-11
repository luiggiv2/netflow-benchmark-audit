"""Figure 1 (P5.4): how the NF datasets were produced and what this study does
with them. Schematic only; no data values. The production steps follow
Sarhan et al. (2022) for NF-v2 and Luay et al. (2026) for NF-v3.

Run: `.venv/bin/python scripts/51_figure_pipeline.py` -> manuscript/figures/F1_pipeline.{pdf,png}
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import figstyle as S
from nfv3_config import WORK

OUT = WORK / "manuscript/figures"


PAD = 0.008   # FancyBboxPatch padding: the drawn edge lies PAD outside (x, y, w, h)
GAP = 0.006   # clearance between an arrow end and a box edge


def box(ax, x, y, w, h, title, body, face=S.NEUTRAL, edge=S.GRID, bold_edge=False):
    """Draw a box and return its drawn edges (left, right, bottom, top)."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={PAD},rounding_size=0.012",
                                fc=face, ec=S.INK_2 if bold_edge else edge, lw=1.0 if bold_edge else 0.8))
    ax.text(x + 0.012, y + h - 0.02, title, ha="left", va="top", fontsize=7.5, weight="bold", color=S.INK)
    ax.text(x + 0.012, y + h - 0.075, body, ha="left", va="top", fontsize=6.6, color=S.INK, linespacing=1.35)
    return x - PAD, x + w + PAD, y - PAD, y + h + PAD


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=1.0,
                                 color=S.INK_2, shrinkA=0, shrinkB=0))


def harrow(ax, src, dst, y):
    """Horizontal arrow from the right edge of src to the left edge of dst, at height y."""
    arrow(ax, src[1] + GAP, y, dst[0] - GAP, y)


def main() -> None:
    S.apply()
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(S.FULL_WIDTH_IN, 3.7))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.01, 0.995, "Dataset production (by the dataset authors)", fontsize=8, weight="bold", va="top")
    ax.text(0.01, 0.465, "This study", fontsize=8, weight="bold", va="top")
    ax.plot([0.01, 0.99], [0.48, 0.48], color=S.GRID, lw=0.8, ls=(0, (3, 2)))

    # Parent datasets: one coloured marker per dataset.
    par = box(ax, 0.01, 0.52, 0.20, 0.42, "Parent datasets",
        "Packet captures (pcap) and\nground-truth attack records")
    for i, (k, name) in enumerate([("ToN", "ToN-IoT"), ("BoT", "Bot-IoT"), ("CIC", "CSE-CIC-IDS2018")]):
        y = 0.74 - i * 0.055
        ax.add_patch(plt.Rectangle((0.025, y - 0.014), 0.018, 0.028, color=S.DATASET_COLOR[k]))
        ax.text(0.05, y, name, va="center", fontsize=6.6)

    v2 = box(ax, 0.27, 0.745, 0.30, 0.195, "NF-v2 export",
        "nProbe, 43 NetFlow fields\nLabels: 5-tuple match with the records\nExport settings not documented")
    v3 = box(ax, 0.27, 0.52, 0.30, 0.205, "NF-v3 export",
        "nProbe, 53 fields (+2 timestamps, +8 IAT)\nLabels: 5-tuple and time overlap\nActive timeout 2 min; original timestamps")
    rel = box(ax, 0.625, 0.52, 0.16, 0.42, "Released files",
        "CSV per dataset and\nrelease, public on\nUQ eSpace\n\nSix files audited\n(Table 2)")
    use = box(ax, 0.845, 0.52, 0.145, 0.42, "Common use",
        "Random split\nwithin a dataset;\ntrain on one,\ntest on another")

    aud = box(ax, 0.10, 0.03, 0.38, 0.38, "Audit of the files (RQ1, RQ2)",
        "Exact duplicates and label conflicts\nValue anomalies and documentation errors\n"
        "Heuristics of Flood et al. per class\nShortcut probes: one feature, depth-3 trees,\n"
        "timestamps alone\nOrigin of each finding: parent, one release,\nrelease-dependent, NetFlow export,\ndocumentation or undetermined",
        face=S.SURFACE, bold_edge=True)
    eva = box(ax, 0.55, 0.03, 0.40, 0.38, "Consequences for evaluation (RQ3, RQ4)",
        "In-domain: random vs grouped split, with and\nwithout conflicts, shortcuts, timestamps\n"
        "and duplicate records (five seeds)\nCross-dataset: Benign, DoS, DDoS core task\n"
        "Replication of a published result\nSHAP: do explanations reveal the shortcuts?",
        face=S.SURFACE, bold_edge=True)
    # Horizontal flows in the upper half, anchored to the box edges.
    y_v2, y_v3 = (v2[2] + v2[3]) / 2, (v3[2] + v3[3]) / 2
    harrow(ax, par, v2, y_v2)
    harrow(ax, par, v3, y_v3)
    harrow(ax, v2, rel, y_v2)
    harrow(ax, v3, rel, y_v3)
    harrow(ax, rel, use, (rel[2] + rel[3]) / 2)
    # Right-angle connector from the released files down to the audit.
    xr, xa, y_mid = (rel[0] + rel[1]) / 2, (aud[0] + aud[1]) / 2, 0.445
    ax.plot([xr, xr, xa], [rel[2] - GAP, y_mid, y_mid], color=S.INK_2, lw=1.0, solid_capstyle="butt")
    arrow(ax, xa, y_mid, xa, aud[3] + GAP)
    harrow(ax, aud, eva, (aud[2] + aud[3]) / 2)

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"F1_pipeline.{ext}", bbox_inches="tight", pad_inches=0.02)
    print("wrote", OUT / "F1_pipeline.pdf")


if __name__ == "__main__":
    main()
