"""Figures 9 and 10 of the manuscript (P5.6, RQ4).

F9   composition of the five most important features of XGBoost (mean |SHAP|):
     ports and service (destination port, L7_PROTO, ICMP fields; regrouped
     2026-10-10, P6.1b block 1), testbed-identity features (TTL, TCP window,
     DNS), timestamps, other traffic features; G0 for BIN and CAN, and G-TS for NF-v3 (mean over seeds)
F10  the same G0 model explained on its own test rows and on another dataset:
     Spearman correlation of the two rankings against the drop in macro-F1
     (SHAP: data/eval/shap/*.jsonl; scores: data/eval/cross_inspection.json)

Plotted values: data/paper/rq4_figures.json and .txt.
Run: `.venv/bin/python scripts/55_figures_rq4.py`
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import spearmanr

import figstyle as S
from nfv3_config import IDENTITY_COLS, PORT_COLS, SERVICE_COLS, TIMESTAMP_COLS, WORK

KEY = {"ToN": "NF-ToN-IoT", "BoT": "NF-BoT-IoT", "CIC": "NF-CSE-CIC-IDS2018"}
SHORT = {v: k for k, v in KEY.items()}
CATS = [("ports and service", "#0d366b"), ("identity", "#3987e5"), ("timestamps", "#eb6834"), ("other", "#e2e1dd")]


def provenance() -> dict:
    def git(*a):
        return subprocess.run(["git", *a], cwd=WORK, capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)data/paper",
                         ":(exclude)manuscript/figures", ":(exclude)manuscript/tables")),  # own outputs ignored
            "script_sha1": hashlib.sha1((WORK / "scripts/55_figures_rq4.py").read_bytes()).hexdigest(),
            "generated_utc": datetime.now(timezone.utc).isoformat()}


def cat(f: str) -> str:
    if f in PORT_COLS or f in SERVICE_COLS:
        return "ports and service"
    if f in IDENTITY_COLS:
        return "identity"
    if f in TIMESTAMP_COLS:
        return "timestamps"
    return "other"


def records():
    out = []
    for v in ("v2", "v3"):
        for name in KEY.values():
            for line in open(WORK / f"data/eval/shap/{name}-{v}.jsonl"):
                out.append(json.loads(line))
    return out


def short(ds: str) -> str:
    base, v = ds.rsplit("-", 1)
    return f"{SHORT[base]}-{v}"


def fig9(recs, rec, lines):
    comp = defaultdict(list)
    for r in recs:
        if r["model"] != "xgb" or (r["condition"], r["task"]) not in {("G0", "BIN"), ("G0", "CAN"), ("G-TS", "BIN"), ("G-TS", "CAN")}:
            continue
        imp = r["importance_test"]
        top = sorted(imp, key=imp.get, reverse=True)[:5]
        comp[(r["dataset"], r["condition"], r["task"])].append([sum(cat(f) == c for f in top) / 5 for c, _ in CATS])
    order = [f"{KEY[k]}-{v}" for v in ("v2", "v3") for k in KEY]
    fig, axes = plt.subplots(1, 2, figsize=(S.FULL_WIDTH_IN, 2.7), gridspec_kw={"width_ratios": [6, 3]}, sharey=True)
    rec["F9"] = {}
    for ax, cond, dss, title in ((axes[0], "G0", order, "(a) G0: all features except timestamps"),
                                 (axes[1], "G-TS", order[3:], "(b) G-TS: timestamps added")):
        x = 0
        ticks, labels = [], []
        for ds in dss:
            for task in ("BIN", "CAN"):
                m = np.mean(comp[(ds, cond, task)], axis=0)
                bottom = 0
                for (c, col), val in zip(CATS, m):
                    ax.bar(x, val, 0.8, bottom=bottom, color=col, edgecolor="white", lw=0.8)
                    bottom += val
                rec["F9"][f"{ds} {cond} {task}"] = dict(zip([c for c, _ in CATS], m.round(3).tolist()))
                lines.append(f"  F9 {ds:22} {cond:4} {task}: " + ", ".join(f"{c} {v:.2f}" for (c, _), v in zip(CATS, m)))
                ticks.append(x)
                labels.append(task)
                x += 1
            ax.text(x - 1.5, -0.24, short(ds), ha="center", fontsize=7, transform=ax.get_xaxis_transform())
            x += 0.6
        ax.set_xticks(ticks, labels, fontsize=6)
        ax.set_title(title, loc="left", fontsize=8)
        ax.set_ylim(0, 1)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("Share of the top-5 features")
    names = {"ports and service": "ports and service (L7_PROTO, ICMP)", "identity": "identity (TTL, TCP window, DNS)"}
    handles = [Patch(color=col, label=names.get(c, c)) for c, col in CATS]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.subplots_adjust(wspace=0.05, bottom=0.22)
    for ext in ("pdf", "png"):
        fig.savefig(WORK / f"manuscript/figures/F9_shap_top5.{ext}", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def fig10(recs, rec, lines):
    tabs = json.load(open(WORK / "data/eval/cross_inspection.json"))["part1"]["tables"]
    drop = {(t["task"], t["source"], t["target"]): 100 * (t["in_domain_G0"] - t["target_median"]["G0"])
            for t in tabs if t["model"] == "xgb"}
    rho = defaultdict(list)
    for r in recs:
        if r["model"] != "xgb" or r["condition"] != "G0" or r["task"] not in ("BIN", "CORE"):
            continue
        src = r["importance_test"]
        feats = list(src)
        for tgt, t in (r.get("targets") or {}).items():
            imp = t["importance"]
            rho[(r["task"], r["dataset"], tgt)].append(spearmanr([src[f] for f in feats], [imp[f] for f in feats])[0])
    fig, ax = plt.subplots(figsize=(S.HALF_WIDTH_IN + 0.6, 2.8))
    pts = []
    for (task, s, t), vals in sorted(rho.items()):
        d = drop[(task, s, t)]
        y = float(np.median(vals))
        k = short(s).split("-")[0]
        ax.scatter(d, y, s=26, marker="o" if task == "BIN" else "^", color=S.DATASET_COLOR[k],
                   edgecolor="white", lw=0.6, zorder=3)
        pts.append((task, s, t, round(d, 1), round(y, 3)))
    ys = [p[4] for p in pts]
    lines.append(f"  F10 {len(pts)} pairs; Spearman median {np.median(ys):.2f} (min {min(ys):.2f}); "
                 f"drop {min(p[3] for p in pts):.1f} to {max(p[3] for p in pts):.1f} points")
    for task in ("BIN", "CORE"):
        sub = [p[4] for p in pts if p[0] == task]
        lines.append(f"  F10 {task}: median Spearman {np.median(sub):.3f}")
    rec["F10"] = pts
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Drop in macro-F1 on the target (points)")
    ax.set_ylabel("Spearman, source vs target ranking")
    ax.grid(True)
    ax.set_axisbelow(True)
    handles = [Line2D([], [], marker="o", ls="", color=S.DATASET_COLOR[k], label=k) for k in KEY]
    handles += [Line2D([], [], marker="o", ls="", color=S.INK_2, label="BIN"),
                Line2D([], [], marker="^", ls="", color=S.INK_2, label="CORE")]
    ax.legend(handles=handles, frameon=False, fontsize=6.3, loc="lower left", ncol=2)
    for ext in ("pdf", "png"):
        fig.savefig(WORK / f"manuscript/figures/F10_shap_source_target.{ext}", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    S.apply()
    recs = records()
    rec: dict = {"provenance": provenance()}
    lines = [f"RQ4 figures (P5.6). Commit {rec['provenance']['commit']}"
             f"{' (dirty)' if rec['provenance']['dirty'] else ''}"]
    fig9(recs, rec, lines)
    fig10(recs, rec, lines)
    (WORK / "data/paper/rq4_figures.json").write_text(json.dumps(rec, indent=1))
    (WORK / "data/paper/rq4_figures.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
