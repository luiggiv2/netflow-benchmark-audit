"""Figures 6-8 and Table 6 of the manuscript (P5.6, RQ3).

F6  in-domain macro-F1 under the random (R0) and grouped (G0) split, five seeds,
    with the conflict ceiling of G0 (data/eval/in_domain/*.jsonl)
F7  cross-dataset matrices: median macro-F1 of G0 models on every other dataset
    of the same release, in-domain score on the diagonal
    (data/eval/cross_inspection.json, part 1)
F8  removing shortcut features and transfer: G0 vs G-NP and G-NI repeated on the
    split of G0 (G-NP0, G-NI0: same training and test rows, only the columns
    change; data/eval/cross_g0split/), BIN task, every direction and model;
    the counts of G-NP and G-NI on their own split are kept in the trace
    (third reviewer report, major issue 6)
T6  replication of Layeghy and Portmann (2023) (data/eval/replication/*.jsonl)

Plotted values: data/paper/rq3_figures.json and .txt.
Run: `.venv/bin/python scripts/54_figures_rq3.py`
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import subprocess
from collections import defaultdict
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import figstyle as S
from nfv3_config import WORK

KEY = {"ToN": "NF-ToN-IoT", "BoT": "NF-BoT-IoT", "CIC": "NF-CSE-CIC-IDS2018"}
SHORT = {v: k for k, v in KEY.items()}
OUT_FIG = WORK / "manuscript/figures"
OUT_TAB = WORK / "manuscript/tables"


def provenance() -> dict:
    def git(*a):
        return subprocess.run(["git", *a], cwd=WORK, capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)data/paper",
                         ":(exclude)manuscript/figures", ":(exclude)manuscript/tables")),  # own outputs ignored
            "script_sha1": hashlib.sha1((WORK / "scripts/54_figures_rq3.py").read_bytes()).hexdigest(),
            "generated_utc": datetime.now(timezone.utc).isoformat()}


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT_FIG / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def short(ds: str) -> str:
    base, v = ds.rsplit("-", 1)
    return f"{SHORT[base]}-{v}"


def fig6(rec, lines):
    runs = defaultdict(list)
    ceil = defaultdict(list)
    for v in ("v2", "v3"):
        for name in KEY.values():
            ds = f"{name}-{v}"
            for line in open(WORK / f"data/eval/in_domain/{ds}.jsonl"):
                r = json.loads(line)
                if r["condition"] in ("G0", "R0"):
                    runs[(ds, r["task"], r["model"], r["condition"])].append(r["macro_f1"])
                    if r["condition"] == "G0":
                        ceil[(ds, r["task"])].append(r["conflict_ceiling_macro_f1"])
    fig, axes = plt.subplots(2, 2, figsize=(S.FULL_WIDTH_IN, 5.0), sharey=True)
    rec["F6"] = {}
    for r, task in enumerate(("BIN", "CAN")):
        for c, v in enumerate(("v2", "v3")):
            ax = axes[r, c]
            order = [f"{KEY[k]}-{v}" for k in KEY]
            ticks, labels = [], []
            for i, ds in enumerate(order):
                col = S.DATASET_COLOR[short(ds).split("-")[0]]
                for j, model in enumerate(("xgb", "rf")):
                    x0 = i + (j - 0.5) * 0.4
                    r0 = runs[(ds, task, model, "R0")]
                    g0 = runs[(ds, task, model, "G0")]
                    ax.scatter(np.full(len(r0), x0 - 0.06), r0, s=12, marker="o", facecolor="white",
                               edgecolor=S.INK_2, lw=0.7, zorder=3)
                    ax.scatter(np.full(len(g0), x0 + 0.06), g0, s=14, marker="o", color=col, lw=0,
                               alpha=0.9, zorder=4)
                    rec["F6"][f"{ds} {task} {model}"] = {"R0": r0, "G0": g0}
                    lines.append(f"  F6 {ds:22} {task} {model:3} R0 {min(r0):.3f}-{max(r0):.3f}  "
                                 f"G0 {min(g0):.3f}-{max(g0):.3f}")
                    ticks.append(x0)
                    labels.append("XGB" if model == "xgb" else "RF")
                cval = float(np.median(ceil[(ds, task)]))
                ax.plot([i - 0.42, i + 0.42], [cval, cval], color=S.INK, lw=1.0, ls=(0, (2, 1.5)), zorder=2)
                rec["F6"][f"{ds} {task} ceiling"] = cval
                ax.text(i, -0.10, short(ds).split("-")[0], ha="center", va="top", fontsize=7.5,
                        transform=ax.get_xaxis_transform())
            ax.set_xticks(ticks, labels, fontsize=6)
            ax.tick_params(axis="x", length=0)
            ax.set_xlim(-0.6, 2.6)
            ax.set_title(f"({'abcd'[2 * r + c]}) {task} task, NF-{v}", loc="left", fontsize=8)
            ax.yaxis.grid(True)
            ax.set_axisbelow(True)
        axes[r, 0].set_ylabel("Macro-F1 (test split)")
    axes[0, 0].set_ylim(0, 1.03)
    handles = [Line2D([], [], marker="o", ls="", mfc="white", mec=S.INK_2, label="R0 random split"),
               Line2D([], [], marker="o", ls="", mfc=S.INK_2, mec=S.INK_2, label="G0 grouped split"),
               Line2D([], [], color=S.INK, lw=1.0, ls=(0, (2, 1.5)), label="conflict ceiling of G0")]
    handles += [Patch(color=S.DATASET_COLOR[k], label=k) for k in KEY]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.99),
               fontsize=6.5, columnspacing=1.2)
    fig.subplots_adjust(wspace=0.06, hspace=0.45)
    save(fig, "F6_in_domain_splits")


def fig7(rec, lines):
    tabs = json.load(open(WORK / "data/eval/cross_inspection.json"))["part1"]["tables"]
    idx = {(t["task"], t["model"], t["source"], t["target"]): t for t in tabs}
    fig, axes = plt.subplots(2, 2, figsize=(5.0, 5.0))
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("seq", S.SEQ_BLUE)
    rec["F7"] = {}
    for r, task in enumerate(("BIN", "CORE")):
        for c, v in enumerate(("v2", "v3")):
            ax = axes[r, c]
            names = [f"{KEY[k]}-{v}" for k in KEY]
            m = np.full((3, 3), np.nan)
            for i, src in enumerate(names):
                for j, tgt in enumerate(names):
                    if i == j:
                        cand = [t for t in tabs if t["task"] == task and t["model"] == "xgb" and t["source"] == src]
                        m[i, j] = cand[0]["in_domain_G0"]
                    else:
                        m[i, j] = idx[(task, "xgb", src, tgt)]["target_median"]["G0"]
            ax.imshow(m, cmap=cmap, vmin=0, vmax=1)
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                            color="white" if m[i, j] > 0.55 else S.INK)
            ax.add_patch(plt.Rectangle((-0.5, -0.5), 3, 3, fill=False, ec=S.GRID, lw=0.6))
            ax.set_xticks(range(3), list(KEY), fontsize=7)
            ax.set_yticks(range(3), list(KEY), fontsize=7)
            ax.set_xlabel("Tested on", fontsize=7)
            ax.set_ylabel("Trained on", fontsize=7)
            ax.set_title(f"({'abcd'[2 * r + c]}) {task}, NF-{v}", loc="left", fontsize=8)
            ax.spines[:].set_visible(False)
            ax.tick_params(length=0)
            rec["F7"][f"{task} {v}"] = m.tolist()
            lines.append(f"  F7 {task} {v} XGBoost G0: " + "; ".join(
                f"{list(KEY)[i]}->{list(KEY)[j]} {m[i, j]:.2f}" for i in range(3) for j in range(3)))
    fig.subplots_adjust(hspace=0.45, wspace=0.25)
    save(fig, "F7_cross_dataset")


def g0split_medians() -> dict:
    """(source, target, model, condition) -> median target macro-F1 over seeds, BIN, split of G0."""
    vals = defaultdict(list)
    for name in KEY.values():
        for v in ("v2", "v3"):
            src = f"{name}-{v}"
            for line in open(WORK / f"data/eval/cross_g0split/{src}.jsonl"):
                r = json.loads(line)
                if r["task"] == "BIN":
                    for tgt, t in r["targets"].items():
                        vals[(src, tgt, r["model"], r["condition"])].append(t["macro_f1"])
    assert all(len(x) == 5 for x in vals.values())
    return {k: float(np.median(x)) for k, x in vals.items()}


def fig8(rec, lines):
    tabs = json.load(open(WORK / "data/eval/cross_inspection.json"))["part1"]["tables"]
    for cond in ("G-NP", "G-NI"):  # split of each condition (also changes the training rows): trace only
        d = [t["target_median"][cond] - t["target_median"]["G0"] for t in tabs if t["task"] == "BIN"]
        lines.append(f"  F8 trace BIN {cond} (split of the condition): worse than G0 in {sum(x < 0 for x in d)} "
                     f"of {len(d)} cases (strict, medians over seeds)")
    g0s = g0split_medians()
    fig, axes = plt.subplots(1, 2, figsize=(S.FULL_WIDTH_IN, 3.0), sharey=True)
    rec["F8"] = {}
    for ax, cond, title in zip(axes, ("G-NP0", "G-NI0"), ("(a) without the destination port",
                                                         "(b) without ports and identity features")):
        pts = []
        for t in tabs:
            if t["task"] != "BIN":
                continue
            src = short(t["source"]).split("-")[0]
            v = t["source"].rsplit("-", 1)[1]
            x, y = t["target_median"]["G0"], g0s[(t["source"], t["target"], t["model"], cond)]
            pts.append((t["source"], t["target"], t["model"], x, y))
            ax.scatter(x, y, s=22, marker="o" if v == "v3" else "s",
                       facecolor=S.DATASET_COLOR[src] if t["model"] == "xgb" else "white",
                       edgecolor=S.DATASET_COLOR[src], lw=1.1, zorder=3)
        ax.plot([0, 1], [0, 1], color=S.INK_2, lw=0.8, ls=(0, (3, 2)), zorder=1)
        worse = sum(p[4] < p[3] for p in pts)
        ax.text(0.97, 0.05, f"below the line (worse): {worse} of {len(pts)}", ha="right", fontsize=6.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_aspect("equal")
        ax.set_xlabel("Target macro-F1, G0 (all features)")
        ax.set_title(title, loc="left", fontsize=8)
        ax.grid(True)
        ax.set_axisbelow(True)
        rec["F8"][cond] = pts
        lines.append(f"  F8 BIN {cond} (split of G0): worse than G0 in {worse} of {len(pts)} cases "
                     "(strict, medians over seeds)")
    axes[0].set_ylabel("Target macro-F1, features removed")
    handles = [Patch(color=S.DATASET_COLOR[k], label=k) for k in KEY]
    handles += [Line2D([], [], marker="o", ls="", mfc=S.INK_2, mec=S.INK_2, label="XGBoost"),
                Line2D([], [], marker="o", ls="", mfc="white", mec=S.INK_2, label="random forest"),
                Line2D([], [], marker="s", ls="", mfc="white", mec=S.INK_2, label="square NF-v2, circle NF-v3"),
                Line2D([], [], color=S.INK_2, lw=0.8, ls=(0, (3, 2)), label="no change (below: worse)")]
    h = {x.get_label(): x for x in handles}
    order = ["ToN", "XGBoost", "BoT", "random forest", "CIC", "square NF-v2, circle NF-v3", "no change (below: worse)"]
    fig.legend(handles=[h[o] for o in order], loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.10),
               fontsize=6.3)
    fig.subplots_adjust(wspace=0.12)
    save(fig, "F8_shortcuts_transfer")


def table6(rec, lines):
    steps = ["L0", "L0k", "L0t", "L1", "L2", "L3", "L4"]
    rows, data, cross_rows = [], {}, {}
    for name in ("NF-ToN-IoT", "NF-BoT-IoT", "NF-CSE-CIC-IDS2018"):
        ds = f"{name}-v2"
        per = defaultdict(list)
        leak = defaultdict(list)
        pub = None
        for line in open(WORK / f"data/eval/replication/{ds}.jsonl"):
            r = json.loads(line)
            per[r["step"]].append(r["in_domain"]["attack_f1"])
            leak[r["step"]].append(r["test_rows_with_vector_in_train"])
            pub = r["published_in_domain_attack_f1"]
            for tgt, c in r["cross"].items():  # panel (b), reviewer point M3
                xr = cross_rows.setdefault((name, tgt.replace("-v2", "")), {"pub": c["published_attack_f1"]})
                xr.setdefault(r["step"], []).append(c["attack_f1"])
        med = {s: float(np.median(per[s])) for s in steps}
        data[ds] = {"published": pub, "median_attack_f1": med, "leak_L0": float(np.median(leak["L0"])),
                    "L1_range": [float(min(per["L1"])), float(max(per["L1"]))]}
        l1 = f"{med['L1']:.2f} ({min(per['L1']):.2f}–{max(per['L1']):.2f})"
        rows.append([name, f"{pub:.2f}"] + [l1 if s == "L1" else f"{med[s]:.2f}" for s in steps]
                    + [f"{100 * data[ds]['leak_L0']:.1f}"])
        lines.append(f"  T6 {ds}: published {pub}; " + ", ".join(f"{s} {med[s]:.2f}" for s in steps)
                     + f"; L1 range {min(per['L1']):.2f}-{max(per['L1']):.2f}")
    xrows = []
    for (src, tgt), c in cross_rows.items():
        cells = [src, tgt, f"{c['pub']:.2f}"] + [f"{np.median(c[s]):.2f} ({min(c[s]):.2f}–{max(c[s]):.2f})"
                                                for s in ("L0", "L1")]
        xrows.append(cells)
        lines.append(f"  T6b {src} -> {tgt}: published {c['pub']}; L0 {np.median(c['L0']):.2f}; "
                     f"L1 {np.median(c['L1']):.2f}")
    data["cross"] = {f"{a}->{b}": {"published": c["pub"], "L0": [float(x) for x in c["L0"]],
                                   "L1": [float(x) for x in c["L1"]]} for (a, b), c in cross_rows.items()}
    rec["T6"] = data
    hdr = ["Dataset (v2)", "Published", "L0", "L0k", "L0t", "L1", "L2", "L3", "L4", "Leak L0 (%)"]
    # numbered citation in the caption and note (fourth reviewer report, R27); \citep in the .tex version
    title = "Replication of the results of Layeghy and Portmann [@layeghy2023explainable]"
    note = ("How to read: attack-class F1 (%) of a random forest on the binary task, median over five seeds. "
            "Published: value reported by Layeghy and Portmann [@layeghy2023explainable]. L0: their setting as read here (stratified "
            "sample of one "
            "million flows, 39 features, random 70/30 split). L0k and L0t: ordered alternatives for their "
            "undocumented split (five folds without shuffling; first 70 % for training). L1: grouped split. "
            "L2: L1 without conflicting rows. L3: L1 with the flow duration repaired (F8). L4: L1 without the "
            "flow duration. Leak L0: share of test rows whose feature vector also occurs in the training data "
            "under L0. A value within 2 points of the published one in L0 counts as reproduced; the drop from L0 "
            "to L1 combines the leak of the random split and the composition of the grouped test split, which "
            "depends on where the few large vectors fall (Section 5.3.1). Parentheses: minimum and maximum over "
            "seeds. "
            "Panel (b): the same models tested on all rows of each other NF-v2 dataset.")
    hdr_x = ["Trained on", "Tested on", "Published", "L0", "L1"]
    with open(OUT_TAB / "T6_replication.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(hdr)
        w.writerows(rows)
        w.writerow([])
        w.writerow(hdr_x)
        w.writerows(xrows)
    esc = lambda x: str(x).replace("%", "\\%").replace("_", "\\_")
    tex = [r"\begin{table*}[t]", r"\centering", r"\caption{" + title.replace("[@layeghy2023explainable]", r"\citep{layeghy2023explainable}") + ".}", r"\label{tab:replication}",
           r"\footnotesize", r"\begin{tabular}{l" + "r" * 9 + "}", r"\toprule", " & ".join(map(esc, hdr)) + r" \\",
           r"\midrule"] + [" & ".join(map(esc, r)) + r" \\" for r in rows] + [
           r"\midrule", r"\multicolumn{10}{l}{\textit{(b) Cross-dataset: " + " / ".join(hdr_x) + r"}} \\"] + [
           " & ".join(map(esc, r[:2])) + r" & & " + " & ".join(map(esc, r[2:])) + r" \\" for r in xrows] + [
           r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
           r"\parbox{\textwidth}{\footnotesize \textit{Note.} " + esc(note).replace("[@layeghy2023explainable]", r"\citep{layeghy2023explainable}") + "}", r"\end{table*}"]
    (OUT_TAB / "T6_replication.tex").write_text("\n".join(tex) + "\n")
    md = [f"**Table 6.** {title}.", "", "*(a) In-domain.*", "", "| " + " | ".join(hdr) + " |",
          "|" + "---|" * len(hdr)]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    md += ["", "*(b) Cross-dataset.*", "", "| " + " | ".join(hdr_x) + " |", "|" + "---|" * len(hdr_x)]
    md += ["| " + " | ".join(r) + " |" for r in xrows] + ["", f"*Note.* {note}"]
    (OUT_TAB / "T6_replication.md").write_text("\n".join(md) + "\n")


def table_s4(rec, lines):
    """Supplementary Table S4: Figure 7 for the random forest (median macro-F1, G0)."""
    tabs = json.load(open(WORK / "data/eval/cross_inspection.json"))["part1"]["tables"]
    idx = {(t["task"], t["model"], t["source"], t["target"]): t for t in tabs}
    md = ["**Table S4.** Cross-dataset scores of random-forest models trained under G0.", "",
          "| Task | Release | Trained on | Tested on ToN | Tested on BoT | Tested on CIC |", "|---|---|---|---|---|---|"]
    rec["S4"] = {}
    for task in ("BIN", "CORE"):
        for v in ("v2", "v3"):
            names = [f"{KEY[k]}-{v}" for k in KEY]
            for i, src in enumerate(names):
                row = []
                for j, tgt in enumerate(names):
                    if i == j:
                        val = [t for t in tabs if t["task"] == task and t["model"] == "rf" and t["source"] == src][0]["in_domain_G0"]
                        row.append(f"**{val:.2f}**")
                    else:
                        val = idx[(task, "rf", src, tgt)]["target_median"]["G0"]
                        row.append(f"{val:.2f}")
                rec["S4"][f"{task} {src}"] = row
                md.append(f"| {task} | {v} | {list(KEY)[i]} | " + " | ".join(row) + " |")
    md += ["", "*Note.* Median macro-F1 over five seeds, as in Figure 7 but for the random forest. Bold: in-domain score on "
           "the source test split; other cells: score on all rows of the target dataset (for CORE, all its Benign, DoS "
           "and DDoS rows)."]
    (OUT_TAB / "S4_cross_rf.md").write_text("\n".join(md) + "\n")
    lines.append("  S4 random-forest cross-dataset table written")


def main() -> None:
    if "--tables-only" in sys.argv:  # tables without redrawing the figures or rewriting data/paper
        rec, lines = {}, []
        table6(rec, lines)
        table_s4(rec, lines)
        print("\n".join(lines))
        return
    S.apply()
    rec: dict = {"provenance": provenance()}
    lines = [f"RQ3 figures and Table 6 (P5.6). Commit {rec['provenance']['commit']}"
             f"{' (dirty)' if rec['provenance']['dirty'] else ''}"]
    fig6(rec, lines)
    fig7(rec, lines)
    fig8(rec, lines)
    table6(rec, lines)
    table_s4(rec, lines)
    (WORK / "data/paper/rq3_figures.json").write_text(json.dumps(rec, indent=1))
    (WORK / "data/paper/rq3_figures.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(l for l in lines if " F6 " not in l))


if __name__ == "__main__":
    main()
