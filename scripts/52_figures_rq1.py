"""Figures 2-4 of the manuscript (P5.6, RQ1).

F2  repeated rows and minority-label rows per dataset, NF-v2 vs NF-v3, on the
    40 features the models see (key 40; key 41 with the source port shown as a
    marker), and records duplicated in every column per class (data/audit/dupconf/*.json,
    data/audit/flood_comparison.csv)
F4  heuristics of Flood et al. per class: parent, NF-v2, NF-v3
    (data/audit/flood_comparison.csv)
F3  distribution of FLOW_DURATION_MILLISECONDS in NF-v2 and NF-v3
    (data/processed/<dataset>/data.parquet, one column)

Every plotted number is written to data/paper/rq1_figures.json and .txt.
Run: `.venv/bin/python scripts/52_figures_rq1.py`
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import figstyle as S
from nfv3_config import WORK

OUT_FIG = WORK / "manuscript/figures"
OUT_DATA = WORK / "data/paper"
KEY = {"ToN": "NF-ToN-IoT", "BoT": "NF-BoT-IoT", "CIC": "NF-CSE-CIC-IDS2018"}
FLOOD_DS = {"ToN_IoT": "ToN", "Bot_IoT": "BoT", "CSE-CIC-IDS2018": "CIC"}
WRAP_MS = 2 ** 32 / 1000


def provenance() -> dict:
    def git(*a):
        return subprocess.run(["git", *a], cwd=WORK, capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)data/paper",
                         ":(exclude)manuscript/figures", ":(exclude)manuscript/tables")),  # own outputs ignored
            "script_sha1": hashlib.sha1((WORK / "scripts/52_figures_rq1.py").read_bytes()).hexdigest(),
            "generated_utc": datetime.now(timezone.utc).isoformat()}


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT_FIG / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def version_bars(ax, values, ylabel, fmt):
    """values[k] = (v2, v3) for k in ToN, BoT, CIC. v2 hatched and light, v3 solid; legend explains the fill."""
    w = 0.36
    for i, k in enumerate(KEY):
        c = S.DATASET_COLOR[k]
        v2, v3 = values[k]
        x2, x3 = i - w / 2 - 0.01, i + w / 2 + 0.01
        ax.bar(x2, v2, w, color="white", edgecolor=c, hatch="////", lw=1.0)
        ax.bar(x3, v3, w, color=c, edgecolor=c, lw=1.0)
        for xx, v in ((x2, v2), (x3, v3)):
            ax.text(xx, v, fmt(v), ha="center", va="bottom", fontsize=5.3, color=S.INK_2)
    ax.set_xticks(range(3), list(KEY), fontsize=7.5)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)


def fig2(rec):
    dup, conf, dup41 = {}, {}, {}
    for k, name in KEY.items():
        d = {}
        for v in ("v2", "v3"):
            keys = json.load(open(WORK / f"data/audit/dupconf/{name}-{v}.json"))["keys"]
            j, j41 = keys["40"], keys["41"]
            d[v] = (100 * j["duplicate_rows"] / j["rows"], 100 * j["native"]["minority_rows"] / j["rows"],
                    100 * j41["duplicate_rows"] / j41["rows"])
        dup[k] = (d["v2"][0], d["v3"][0])
        conf[k] = (d["v2"][1], d["v3"][1])
        dup41[k] = (d["v2"][2], d["v3"][2])
    full = []
    for r in csv.DictReader(open(WORK / "data/audit/flood_comparison.csv")):
        s = []
        for v in ("v2", "v3"):
            raw, kept = float(r[f"rows_raw_{v}"]), float(r[f"rows_{v}"])
            s.append(100 * (1 - kept / raw) if raw else 0.0)
        if max(s) >= 1.0:
            full.append((FLOOD_DS[r["dataset"]], r["class"], s[0], s[1]))
    full.sort(key=lambda t: (list(KEY).index(t[0]), -t[3]))
    rec["F2"] = {"duplicate_rows_pct_key40": dup, "minority_rows_pct_native_key40": conf,
                 "duplicate_rows_pct_key41": dup41, "full_record_duplicates_pct_by_class": full}

    fig = plt.figure(figsize=(S.FULL_WIDTH_IN, 2.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 0.8, 1.0], wspace=1.0)
    a, b, c = (fig.add_subplot(gs[0, i]) for i in range(3))
    version_bars(a, dup, "Rows (%)", lambda v: f"{v:.1f}")
    for i, k in enumerate(KEY):  # key 41: what the source port hides
        a.scatter([i - 0.19, i + 0.19], dup41[k], marker="_", s=60, color=S.INK, lw=1.4, zorder=5)
    a.set_ylim(0, 105)
    a.set_title("(a) Repeated rows", loc="left", fontsize=8)
    version_bars(b, conf, "Rows (%)", lambda v: f"{v:.2f}")
    b.set_title("(b) Minority-label rows", loc="left", fontsize=8)
    y = np.arange(len(full))[::-1]
    for yy, (k, cls, v2, v3) in zip(y, full):
        col = S.DATASET_COLOR[k]
        c.barh(yy, v3, 0.6, color=col, edgecolor=col, lw=1.0)
        c.text(v3 + 1, yy, f"{v3:.0f}", va="center", fontsize=5.3, color=S.INK_2)
    c.set_yticks(y, [f"{k} {cls}" for k, cls, _, _ in full], fontsize=6.3)
    c.set_xlim(0, 60)
    c.set_xlabel("Records of the class (%)")
    c.set_title("(c) Fully repeated records, NF-v3", loc="left", fontsize=8)
    c.text(0, -0.24, "BoT: no class affected; NF-v2: at most 0.7 % in any class", fontsize=6.3, color=S.INK_2,
           transform=c.transAxes)
    c.xaxis.grid(True)
    c.set_axisbelow(True)
    handles = [Patch(facecolor="white", edgecolor=S.INK_2, hatch="////", label="NF-v2"),
               Patch(facecolor=S.INK_2, edgecolor=S.INK_2, label="NF-v3"),
               Line2D([], [], marker="_", ls="", color=S.INK, markersize=8, mew=1.4,
                      label="(a) with the source port")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.30, 1.08))
    save(fig, "F2_duplicates_conflicts")


def fig3(rec):
    rows = list(csv.DictReader(open(WORK / "data/audit/flood_comparison.csv")))
    panels = [("PDD", "PDD_orig", "PDD_v2", "PDD_v3"), ("TC", "TC_orig", "TC_v2", "TC_v3"),
              ("WL (balanced)", None, "WLbal_v2", "WLbal_v3"), ("HDF (no ports)", "HDF_orig", "HDF_np_v2", "HDF_np_v3"),
              ("UGT", "UGT_orig", "UGT_v2", "UGT_v3")]
    labels = [f"{FLOOD_DS[r['dataset']]} {r['class']}" for r in rows]
    y = np.arange(len(rows))[::-1]
    fig, axes = plt.subplots(1, 5, figsize=(S.FULL_WIDTH_IN, 4.6), sharey=True)
    rec["F3"] = {}
    for ax, (title, po, p2, p3) in zip(axes, panels):
        vals = []
        for yy, r in zip(y, rows):
            col = S.DATASET_COLOR[FLOOD_DS[r["dataset"]]]
            f = lambda c: float(r[c]) if c and r[c] not in ("", "nan") else np.nan
            vo, v2, v3 = f(po), f(p2), f(p3)
            vals.append((labels[len(rows) - 1 - yy], vo, v2, v3))
            ax.plot([np.nanmin([v2, v3]), np.nanmax([v2, v3])], [yy, yy], color=S.GRID, lw=1.2, zorder=1)
            if not np.isnan(vo):
                ax.scatter(vo, yy, s=14, marker="o", facecolor="white", edgecolor=S.INK_2, lw=0.8, zorder=2)
            ax.scatter(v2, yy, s=16, marker="s", facecolor="white", edgecolor=col, lw=1.1, zorder=3)
            ax.scatter(v3, yy, s=16, marker="o", facecolor=col, edgecolor=col, zorder=4)
        rec["F3"][title] = vals
        ax.set_xlim(-0.05, 1.05)
        ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
        ax.set_title(title, fontsize=8)
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)
        n_ton = sum(FLOOD_DS[r["dataset"]] == "ToN" for r in rows)
        n_bot = sum(FLOOD_DS[r["dataset"]] == "BoT" for r in rows)
        for cut in (len(rows) - n_ton - 0.5, len(rows) - n_ton - n_bot - 0.5):
            ax.axhline(cut, color=S.GRID, lw=0.6, ls=(0, (3, 2)))
    axes[0].set_yticks(y, labels, fontsize=6.0)
    handles = [Line2D([], [], marker="o", ls="", mfc="white", mec=S.INK_2, label="Parent (Flood et al.)"),
               Line2D([], [], marker="s", ls="", mfc="white", mec=S.INK_2, label="NF-v2"),
               Line2D([], [], marker="o", ls="", mfc=S.INK_2, mec=S.INK_2, label="NF-v3")]
    handles += [Patch(color=S.DATASET_COLOR[k], label=KEY[k]) for k in KEY]
    handles += [Line2D([], [], color=S.GRID, lw=1.6, label="change from NF-v2 to NF-v3"),
                Line2D([], [], color=S.GRID, lw=0.8, ls=(0, (3, 2)), label="boundary between datasets")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.55, 1.03),
               fontsize=6.5, columnspacing=1.4)
    fig.supxlabel("Heuristic value (0 to 1)", fontsize=8, y=0.02)
    fig.subplots_adjust(wspace=0.12)
    save(fig, "F4_flood_heuristics")


def fig4(rec):
    rec["F4"] = {}
    dur = {}
    for v in ("v2", "v3"):
        for k, name in KEY.items():
            d = pq.read_table(WORK / f"data/processed/{name}-{v}/data.parquet",
                              columns=["FLOW_DURATION_MILLISECONDS"]).column(0).to_numpy().astype(np.int64)
            dur[(k, v)] = d
            rec["F4"][f"{name}-{v}"] = {"rows": int(len(d)), "zero_pct": 100 * float((d == 0).mean()),
                                        "ge_4e6_ms_pct": 100 * float((d >= 4_000_000).mean()),
                                        "other_pct": 100 * float(((d > 0) & (d < 4_000_000)).mean()),
                                        "max_ms": int(d.max())}
    fig = plt.figure(figsize=(S.FULL_WIDTH_IN, 2.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.9, 1.1], wspace=0.35)
    a, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    # (a) NF-v2: every value is 0 or close to 2^32/1000 ms
    yk = list(KEY)[::-1]
    for i, k in enumerate(yk):
        r = rec["F4"][f"{KEY[k]}-v2"]
        col = S.DATASET_COLOR[k]
        z, w = r["zero_pct"], r["ge_4e6_ms_pct"]
        light = plt.matplotlib.colors.to_rgba(col, 0.22)
        a.barh(i, z, 0.6, facecolor=light, edgecolor=col, lw=1.0)
        a.barh(i, w, 0.6, left=z, color=col, edgecolor=col, lw=1.0)
        lab_z, lab_w = f"0 ms: {z:.1f} %", f"\u2248 71.6 min: {w:.1f} %"
        if z >= 25:
            a.text(z / 2, i, lab_z, ha="center", va="center", fontsize=5.6, color=S.INK)
        else:
            a.text(0, i + 0.34, lab_z, ha="left", va="bottom", fontsize=5.6, color=S.INK)
        if w >= 25:
            a.text(z + w / 2, i, lab_w, ha="center", va="center", fontsize=5.6, color="white")
        else:
            a.text(100, i + 0.34, lab_w, ha="right", va="bottom", fontsize=5.6, color=S.INK)
    a.set_yticks(range(3), yk)
    a.set_xlim(0, 100)
    a.set_ylim(-0.5, 2.75)
    a.set_xlabel("Rows of the dataset (%)")
    a.set_title("(a) NF-v2: only two kinds of duration value", loc="left", fontsize=8)
    # (b) NF-v3: realistic durations, capped by the active timeout
    bins = np.linspace(0, 6, 61)
    for k in KEY:
        d = dur[(k, "v3")]
        h, _ = np.histogram(np.log10(d[d > 0]), bins=bins)
        b.step(bins[:-1], 100 * h / len(d), where="post", color=S.DATASET_COLOR[k], lw=0.8, alpha=0.9, label=k)
    b.axvline(np.log10(120_000), color=S.INK_2, lw=0.8, ls=(0, (3, 2)))
    b.text(np.log10(120_000) + 0.08, 55, "active timeout\n(2 min)", fontsize=6.3, va="top")
    b.set_xticks(range(0, 7), ["1 ms", "10 ms", "100 ms", "1 s", "10 s", "100 s", "1000 s"], fontsize=6.3)
    b.set_xlabel("Non-zero flow duration (log scale)")
    b.set_ylabel("Rows of the dataset (%)")
    b.set_title("(b) NF-v3: distribution of durations", loc="left", fontsize=8)
    b.yaxis.grid(True)
    b.set_axisbelow(True)
    b.legend(frameon=False, loc="upper center", fontsize=6.5, ncol=3)
    save(fig, "F3_duration_artefact")


def main() -> None:
    S.apply()
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    rec: dict = {"provenance": provenance()}
    fig2(rec)
    fig3(rec)
    fig4(rec)
    (OUT_DATA / "rq1_figures.json").write_text(json.dumps(rec, indent=1, default=float))
    lines = [f"RQ1 figures (P5.6). Commit {rec['provenance']['commit']}"
             f"{' (dirty)' if rec['provenance']['dirty'] else ''}", "", "F2 (percent of rows; v2, v3)"]
    for k in KEY:
        lines.append(f"  {KEY[k]:20} repeated (key 40) {rec['F2']['duplicate_rows_pct_key40'][k][0]:6.2f} "
                     f"{rec['F2']['duplicate_rows_pct_key40'][k][1]:6.2f}   minority (key 40) "
                     f"{rec['F2']['minority_rows_pct_native_key40'][k][0]:5.2f} "
                     f"{rec['F2']['minority_rows_pct_native_key40'][k][1]:5.2f}   repeated (key 41) "
                     f"{rec['F2']['duplicate_rows_pct_key41'][k][0]:6.2f} {rec['F2']['duplicate_rows_pct_key41'][k][1]:6.2f}")
    lines.append("  records repeated in every column (% of class; v2, v3):")
    lines += [f"    {k} {c:16} {a:6.2f} {b:6.2f}" for k, c, a, b in rec["F2"]["full_record_duplicates_pct_by_class"]]
    lines.append("F4 FLOW_DURATION_MILLISECONDS")
    lines += [f"  {n:24} zero {d['zero_pct']:6.2f} %  >=4,000,000 ms {d['ge_4e6_ms_pct']:6.2f} %  other {d['other_pct']:6.2f} %  max {d['max_ms']:,}"
              for n, d in rec["F4"].items()]
    (OUT_DATA / "rq1_figures.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
