"""Figure 5 of the manuscript (P5.6, RQ2): how simple the classification task is.

Per NF release, the macro-F1 of shallow decision trees (attack class vs
benign, balanced samples, grouped test split) for every native attack class:
depth 1 and depth 3, with the 40 common features without the source port (set
A) and without ports, TTL, TCP window and DNS fields (set C); for NF-v3 also a
depth-3 tree on the flow start time alone. All probes use the common grouped
split keyed on set A (reviewer point M2, author decision 11, 2026-10-06).

Inputs: data/audit/shortcut_probes_common/<dataset>.json (scripts/34_shortcut_probes.py
--common) and data/exploration/timestamp_probe_common.txt
(scripts/exploration/timestamp_probe.py --common).
Plotted values: data/paper/rq2_figure.json and .txt.
Run: `.venv/bin/python scripts/53_figure_rq2.py`
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import figstyle as S
from nfv3_config import WORK

KEY = {"ToN": "NF-ToN-IoT", "BoT": "NF-BoT-IoT", "CIC": "NF-CSE-CIC-IDS2018"}
COLS = [("A", "f1_d1", "1 rule\nall features"), ("C", "f1_d1", "1 rule\nno identity"),
        ("A", "f1_d3", "3 levels\nall features"), ("C", "f1_d3", "3 levels\nno identity")]
THR = 0.95


def provenance() -> dict:
    def git(*a):
        return subprocess.run(["git", *a], cwd=WORK, capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)data/paper",
                         ":(exclude)manuscript/figures", ":(exclude)manuscript/tables")),  # own outputs ignored
            "script_sha1": hashlib.sha1((WORK / "scripts/53_figure_rq2.py").read_bytes()).hexdigest(),
            "generated_utc": datetime.now(timezone.utc).isoformat()}


def timestamp_scores() -> dict:
    out, ds = {}, None
    for line in open(WORK / "data/exploration/timestamp_probe_common.txt"):
        m = re.match(r"== (NF-\S+)", line)
        if m:
            ds = m.group(1)
            continue
        m = re.match(r"\s+(\S+)\s+[\d,]+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", line)
        if m and ds:
            out[(ds, m.group(1))] = float(m.group(3))
    return out


def main() -> None:
    S.apply()
    ts = timestamp_scores()
    rec: dict = {"provenance": provenance(), "threshold": THR, "values": {}}
    fig, axes = plt.subplots(1, 2, figsize=(S.FULL_WIDTH_IN, 2.9), gridspec_kw={"width_ratios": [4, 5]}, sharey=True)
    rng = np.random.default_rng(7)
    lines = [f"RQ2 figure (P5.6). Commit {rec['provenance']['commit']}"
             f"{' (dirty)' if rec['provenance']['dirty'] else ''}", f"Classes with macro-F1 >= {THR}; median"]
    for ax, v in zip(axes, ("v2", "v3")):
        cols = COLS + ([("TS", None, "3 levels\nstart time")] if v == "v3" else [])
        for ci, (fs, metric, label) in enumerate(cols):
            pts = []
            for k, name in KEY.items():
                ds = f"{name}-{v}"
                d = json.load(open(WORK / f"data/audit/shortcut_probes_common/{ds}.json"))
                for cls, c in d["classes"].items():
                    if fs == "TS":
                        val = ts.get((ds, cls))
                    else:
                        val = c.get("sets", {}).get(fs, {}).get(metric)
                    if val is None:
                        continue
                    pts.append((k, cls, float(val)))
            vals = np.array([p[2] for p in pts])
            x = ci + rng.uniform(-0.22, 0.22, len(pts))
            for xx, (k, _, val) in zip(x, pts):
                ax.scatter(xx, val, s=11, color=S.DATASET_COLOR[k], alpha=0.85, lw=0, zorder=3)
            med = float(np.median(vals))
            ax.plot([ci - 0.3, ci + 0.3], [med, med], color=S.INK, lw=1.4, zorder=4)
            n_ok = int((vals >= THR).sum())
            ax.text(ci, 0.27, f"{n_ok}/{len(vals)}", ha="center", fontsize=5.5, color=S.INK_2)
            rec["values"][f"{v} {label.replace(chr(10), ' ')}"] = pts
            lines.append(f"  {v} {label.replace(chr(10), ' '):32} {n_ok}/{len(vals)}  median {med:.3f}")
        ax.set_xticks(range(len(cols)), [c[2] for c in cols], fontsize=6.5)
        ax.axhline(THR, color=S.INK_2, lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.set_xlim(-0.6, len(cols) - 0.4)
        ax.set_title(f"({'a' if v == 'v2' else 'b'}) NF-{v}", loc="left", fontsize=8)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
    axes[0].set_ylim(0.2, 1.03)
    axes[0].set_ylabel("Macro-F1, class vs benign")
    axes[0].set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    axes[0].text(-0.55, 0.325, "classes with macro-F1 \u2265 0.95:", fontsize=6.3, color=S.INK_2, ha="left")
    handles = [Line2D([], [], marker="o", ls="", color=S.DATASET_COLOR[k], label=k) for k in KEY]
    handles.append(Line2D([], [], color=S.INK, lw=1.4, label="median"))
    handles.append(Line2D([], [], color=S.INK_2, lw=0.7, ls=(0, (3, 2)), label="macro-F1 = 0.95 (threshold)"))
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.subplots_adjust(wspace=0.05)
    for ext in ("pdf", "png"):
        fig.savefig(WORK / f"manuscript/figures/F5_shortcut_probes.{ext}", bbox_inches="tight", pad_inches=0.03)
    (WORK / "data/paper/rq2_figure.json").write_text(json.dumps(rec, indent=1))
    (WORK / "data/paper/rq2_figure.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
