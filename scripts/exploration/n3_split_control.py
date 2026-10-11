"""n3_split_control.py — second reviewer report, point N3, 2026-10-08: does
removing the destination port (G-NP) or the identity features (G-NI) still
worsen cross-dataset transfer when the training rows do not change?

G-NP and G-NI (data/eval/cross/) key the grouped split on the reduced feature
set, so their training rows differ from G0. G-NP0 and G-NI0
(data/eval/cross_g0split/, 44_cross_dataset.py --conditions G-NP0 G-NI0) use
the split of G0, so training and test rows are those of G0 (checked below)
and only the columns change.

Part 1: sign test of each condition against G0 on the target, as in
45_inspect_cross.py part 4 (a case is a direction x model, median over seeds;
thresholds 0 = strict, 0.01, 0.02, 0.05; all 24 cases and the cases whose G0
target median is at least 0.05), for G-NP, G-NI, G-NP0 and G-NI0.
Part 2: for G-NP0 and G-NI0, paired per-seed differences against G0 on the
target (median over all runs, and per case the number of seeds below G0), and
the in-domain change on the same test rows.
Part 3: the BIN sign test of G-NP0 and G-NI0 without the source models whose
in-domain median falls by more than 0.05 on the same test rows.

Usage: python scripts/exploration/n3_split_control.py
Output: data/exploration/n3_split_control.txt
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ["NF-ToN-IoT-v2", "NF-BoT-IoT-v2", "NF-CSE-CIC-IDS2018-v2", "NF-ToN-IoT-v3", "NF-BoT-IoT-v3",
            "NF-CSE-CIC-IDS2018-v3"]
THRESHOLDS = [0.0, 0.01, 0.02, 0.05]
FLOOR = 0.05
SEEDS = [1, 2, 3, 4, 5]


def load() -> dict:
    """runs[(source, task, model, condition, seed)] = record"""
    runs = {}
    for d in ("cross", "cross_g0split"):
        for src in DATASETS:
            for line in open(ROOT / f"data/eval/{d}/{src}.jsonl"):
                r = json.loads(line)
                if r["task"] in ("BIN", "CORE"):
                    runs[(src, r["task"], r["model"], r["condition"], r["seed"])] = r
    return runs


def main() -> None:
    R = load()
    L = ["N3 check: feature removal with and without a change of training rows (cross-dataset, BIN and CORE)",
         f"Generated: {datetime.now(timezone.utc).isoformat()}", ""]
    # check: G-NP0 / G-NI0 runs have the training and test rows of G0
    same, total = 0, 0
    for (src, task, m, c, s), r in R.items():
        if c in ("G-NP0", "G-NI0"):
            g0 = R[(src, task, m, "G0", s)]
            total += 1
            same += r["train_rows"] == g0["train_rows"] and r["test_rows"] == g0["in_domain"]["rows"]
    L += [f"Check: {same} of {total} G-NP0/G-NI0 runs have the training and test row counts of the G0 run "
          "(same source, task, model and seed)", ""]

    med = lambda vals: float(np.median(vals))  # noqa: E731
    target = lambda src, task, m, c, t: [R[(src, task, m, c, s)]["targets"][t]["macro_f1"] for s in SEEDS]  # noqa: E731
    cases = defaultdict(list)  # (task, cond) -> list of (src, tgt, model, g0 median, cond median, per-seed diffs)
    for src in DATASETS:
        for task in ("BIN", "CORE"):
            for m in ("xgb", "rf"):
                for t in R[(src, task, m, "G0", 1)]["targets"]:
                    g0 = target(src, task, m, "G0", t)
                    for c in ("G-NP", "G-NI", "G-NP0", "G-NI0"):
                        cv = target(src, task, m, c, t)
                        diffs = [a - b for a, b in zip(cv, g0)] if c.endswith("0") else None
                        cases[(task, c)].append((src, t, m, med(g0), med(cv), diffs))

    L += ["Part 1. Sign test against G0 on the target (case = direction x model, median over seeds)",
          f"  {'task':4s} {'cond':6s} {'scope':9s} {'thr':>5s} {'n':>3s} {'better':>6s} {'similar':>7s} {'worse':>5s} {'p':>9s}"]
    for task in ("BIN", "CORE"):
        for c in ("G-NP", "G-NP0", "G-NI", "G-NI0"):
            for scope in ("all", "off floor"):
                diffs = [cm - g0 for _, _, _, g0, cm, _ in cases[(task, c)] if scope == "all" or g0 >= FLOOR]
                for thr in THRESHOLDS:
                    if thr == 0:
                        b, w = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
                    else:
                        b, w = sum(d >= thr for d in diffs), sum(d <= -thr for d in diffs)
                    pv = binomtest(b, b + w, 0.5).pvalue if b + w else float("nan")
                    L.append(f"  {task:4s} {c:6s} {scope:9s} {thr:5.2f} {len(diffs):3d} {b:6d} {len(diffs) - b - w:7d} "
                             f"{w:5d} {pv:9.2g}")
        L.append("")

    L.append("Part 2. Paired per-seed differences against G0 (G-NP0, G-NI0: same training and test rows)")
    for task in ("BIN", "CORE"):
        for c in ("G-NP0", "G-NI0"):
            allp = [d for *_, ds in cases[(task, c)] for d in ds]
            L.append(f"  {task} {c}: target macro-F1 minus G0, median over {len(allp)} runs {np.median(allp):+.3f} "
                     f"[{min(allp):+.3f}, {max(allp):+.3f}]; runs below G0 {sum(d < 0 for d in allp)}, above "
                     f"{sum(d > 0 for d in allp)}")
            ind = [R[(src, task, m, c, s)]["in_domain"]["macro_f1"] - R[(src, task, m, "G0", s)]["in_domain"]["macro_f1"]
                   for src in DATASETS for m in ("xgb", "rf") for s in SEEDS]
            L.append(f"      in-domain (same test rows) minus G0: median {np.median(ind):+.3f} [{min(ind):+.3f}, "
                     f"{max(ind):+.3f}]")
            for src, t, m, g0, cm, ds in cases[(task, c)]:
                L.append(f"      {src:24s} -> {t:24s} {m:3s} G0 {g0:.3f}  {c} {cm:.3f}  per seed "
                         + " ".join(f"{d:+.3f}" for d in ds) + f"  (below G0 in {sum(d < 0 for d in ds)} of 5)")
        L.append("")
    L.append("Part 3. BIN sign test of G-NP0 and G-NI0 without the cases whose source model loses more than 0.05 in-domain")
    L.append("  (median over seeds, same test rows as G0), so a worse transfer cannot come from a weaker source model")
    for c in ("G-NP0", "G-NI0"):
        diffs, dropped = [], set()
        for src, t, m, g0, cm, _ in cases[("BIN", c)]:
            ind = med([R[(src, "BIN", m, c, s)]["in_domain"]["macro_f1"] - R[(src, "BIN", m, "G0", s)]["in_domain"]["macro_f1"]
                       for s in SEEDS])
            if ind < -0.05:
                dropped.add(f"{src} {m} ({ind:+.3f})")
                continue
            diffs.append(cm - g0)
        for thr in (0.0, 0.02):
            b = sum(d > thr if thr == 0 else d >= thr for d in diffs)
            w = sum(d < -thr if thr == 0 else d <= -thr for d in diffs)
            pv = binomtest(b, b + w, 0.5).pvalue if b + w else float("nan")
            L.append(f"  {c} thr {thr:.2f}: {len(diffs)} cases, better {b}, similar {len(diffs) - b - w}, worse {w}, p {pv:.2g}")
        L.append(f"    sources left out: {'; '.join(sorted(dropped)) or 'none'}")
    out = ROOT / "data/exploration/n3_split_control.txt"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
