"""r3m1_nosvc_transfer.py — third reviewer report, major issue 1, 2026-10-10
(plan task P6.1b, block 1, option (b) authorised by the author): does
cross-dataset transfer improve when the service fields are removed together
with the ports and the identity features?

G-NIS0 (data/eval/cross_g0split_nosvc/, 44_cross_dataset.py --conditions
G-NIS0) trains without ports, TTL, TCP window, DNS fields, L7_PROTO,
ICMP_TYPE and ICMP_IPV4_TYPE, on the grouped split of G0, so its training and
test rows are those of G0 (checked below), as for G-NP0 and G-NI0
(data/eval/cross_g0split/, analysed in n3_split_control.py).

Part 1: sign test against G0 on the target, as in n3_split_control.py part 1
(a case is a direction x model, median over seeds; thresholds 0 = strict,
0.01, 0.02, 0.05; all cases and the cases whose G0 target median is at least
0.05), for G-NP0, G-NI0 and G-NIS0.
Part 2: G-NIS0 against G-NI0 (the only change is the three service fields):
the same sign test, and the paired per-seed differences.
Part 3: paired per-seed differences of G-NIS0 against G0 on the target, the
in-domain change on the same test rows, and the per-case values.
Part 4: the BIN sign test of G-NIS0 against G0 without the cases whose source
model loses more than 0.05 in-domain (as n3_split_control.py part 3).
Part 5: in-domain macro-F1 of G0, G-NI0 and G-NIS0 per source, task and model
(median over seeds, same test rows), and the change from G-NI0 to G-NIS0.

Usage: python scripts/exploration/r3m1_nosvc_transfer.py
Output: data/exploration/r3m1_nosvc_transfer.txt
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
CONDS = ["G-NP0", "G-NI0", "G-NIS0"]
OUT = ROOT / "data/exploration/r3m1_nosvc_transfer.txt"


def load() -> dict:
    """runs[(source, task, model, condition, seed)] = record"""
    runs = {}
    for d in ("cross", "cross_g0split", "cross_g0split_nosvc"):
        for src in DATASETS:
            for line in open(ROOT / f"data/eval/{d}/{src}.jsonl"):
                r = json.loads(line)
                if r["task"] in ("BIN", "CORE"):
                    runs[(src, r["task"], r["model"], r["condition"], r["seed"])] = r
    return runs


def sign(diffs: list[float], thr: float) -> tuple[int, int, int, float]:
    if thr == 0:
        b, w = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    else:
        b, w = sum(d >= thr for d in diffs), sum(d <= -thr for d in diffs)
    pv = binomtest(b, b + w, 0.5).pvalue if b + w else float("nan")
    return b, len(diffs) - b - w, w, pv


def main() -> None:
    R = load()
    L = ["Major issue 1 (third reviewer report): transfer without ports, identity and service fields (G-NIS0)",
         f"Generated: {datetime.now(timezone.utc).isoformat()}",
         "G-NIS0 = G-NI0 without L7_PROTO, ICMP_TYPE and ICMP_IPV4_TYPE; all on the grouped split of G0.", ""]
    n_runs = sum(1 for k in R if k[3] == "G-NIS0")
    same = sum(R[k]["train_rows"] == R[(*k[:3], "G0", k[4])]["train_rows"]
               and R[k]["test_rows"] == R[(*k[:3], "G0", k[4])]["in_domain"]["rows"] for k in R if k[3] == "G-NIS0")
    feats = sorted({(k[0], R[k]["features"]) for k in R if k[3] == "G-NIS0"})
    L += [f"Check: {same} of {n_runs} G-NIS0 runs have the training and test row counts of the G0 run "
          "(same source, task, model and seed)",
          "Features per source: " + ", ".join(f"{s} {f}" for s, f in feats), ""]

    med = lambda vals: float(np.median(vals))  # noqa: E731
    target = lambda src, task, m, c, t: [R[(src, task, m, c, s)]["targets"][t]["macro_f1"] for s in SEEDS]  # noqa: E731
    cases = defaultdict(list)  # (task, cond) -> (src, tgt, model, g0 median, cond median, per-seed diffs vs G0)
    vs_ni = defaultdict(list)  # task -> (src, tgt, model, G-NI0 median, G-NIS0 median, per-seed diffs)
    for src in DATASETS:
        for task in ("BIN", "CORE"):
            for m in ("xgb", "rf"):
                for t in R[(src, task, m, "G0", 1)]["targets"]:
                    g0 = target(src, task, m, "G0", t)
                    for c in CONDS:
                        cv = target(src, task, m, c, t)
                        cases[(task, c)].append((src, t, m, med(g0), med(cv), [a - b for a, b in zip(cv, g0)]))
                    ni, nis = target(src, task, m, "G-NI0", t), target(src, task, m, "G-NIS0", t)
                    vs_ni[task].append((src, t, m, med(ni), med(nis), [a - b for a, b in zip(nis, ni)]))

    L += ["Part 1. Sign test against G0 on the target (case = direction x model, median over seeds)",
          f"  {'task':4s} {'cond':6s} {'scope':9s} {'thr':>5s} {'n':>3s} {'better':>6s} {'similar':>7s} {'worse':>5s} {'p':>9s}"]
    for task in ("BIN", "CORE"):
        for c in CONDS:
            for scope in ("all", "off floor"):
                diffs = [cm - g0 for _, _, _, g0, cm, _ in cases[(task, c)] if scope == "all" or g0 >= FLOOR]
                for thr in THRESHOLDS:
                    b, s, w, pv = sign(diffs, thr)
                    L.append(f"  {task:4s} {c:6s} {scope:9s} {thr:5.2f} {len(diffs):3d} {b:6d} {s:7d} {w:5d} {pv:9.2g}")
        L.append("")

    L.append("Part 2. G-NIS0 against G-NI0 (only the three service fields differ; case medians over seeds)")
    L.append(f"  {'task':4s} {'thr':>5s} {'n':>3s} {'better':>6s} {'similar':>7s} {'worse':>5s} {'p':>9s}")
    for task in ("BIN", "CORE"):
        diffs = [nis - ni for _, _, _, ni, nis, _ in vs_ni[task]]
        for thr in THRESHOLDS:
            b, s, w, pv = sign(diffs, thr)
            L.append(f"  {task:4s} {thr:5.2f} {len(diffs):3d} {b:6d} {s:7d} {w:5d} {pv:9.2g}")
        allp = [d for *_, ds in vs_ni[task] for d in ds]
        L.append(f"  {task}: per-seed G-NIS0 minus G-NI0, median over {len(allp)} runs {np.median(allp):+.3f} "
                 f"[{min(allp):+.3f}, {max(allp):+.3f}]; case medians {np.median(diffs):+.3f} "
                 f"[{min(diffs):+.3f}, {max(diffs):+.3f}]")
    L.append("")

    L.append("Part 3. Paired per-seed differences of G-NIS0 against G0 (same training and test rows)")
    for task in ("BIN", "CORE"):
        allp = [d for *_, ds in cases[(task, "G-NIS0")] for d in ds]
        L.append(f"  {task} G-NIS0: target macro-F1 minus G0, median over {len(allp)} runs {np.median(allp):+.3f} "
                 f"[{min(allp):+.3f}, {max(allp):+.3f}]; runs below G0 {sum(d < 0 for d in allp)}, above "
                 f"{sum(d > 0 for d in allp)}")
        ind = [R[(src, task, m, "G-NIS0", s)]["in_domain"]["macro_f1"] - R[(src, task, m, "G0", s)]["in_domain"]["macro_f1"]
               for src in DATASETS for m in ("xgb", "rf") for s in SEEDS]
        L.append(f"      in-domain (same test rows) minus G0: median {np.median(ind):+.3f} [{min(ind):+.3f}, "
                 f"{max(ind):+.3f}]")
        ni = {(s, t, m): v for s, t, m, _, v, _ in cases[(task, "G-NI0")]}
        for src, t, m, g0, cm, ds in cases[(task, "G-NIS0")]:
            L.append(f"      {src:24s} -> {t:24s} {m:3s} G0 {g0:.3f}  G-NI0 {ni[(src, t, m)]:.3f}  G-NIS0 {cm:.3f}  "
                     f"per seed vs G0 " + " ".join(f"{d:+.3f}" for d in ds) + f"  (below G0 in {sum(d < 0 for d in ds)} of 5)")
        L.append("")

    L.append("Part 4. BIN sign test of G-NIS0 against G0 without the cases whose source model loses more than 0.05")
    L.append("  in-domain (median over seeds, same test rows as G0)")
    diffs, dropped = [], set()
    for src, t, m, g0, cm, _ in cases[("BIN", "G-NIS0")]:
        ind = med([R[(src, "BIN", m, "G-NIS0", s)]["in_domain"]["macro_f1"] - R[(src, "BIN", m, "G0", s)]["in_domain"]["macro_f1"]
                   for s in SEEDS])
        if ind < -0.05:
            dropped.add(f"{src} {m} ({ind:+.3f})")
            continue
        diffs.append(cm - g0)
    for thr in (0.0, 0.02):
        b, s, w, pv = sign(diffs, thr)
        L.append(f"  thr {thr:.2f}: {len(diffs)} cases, better {b}, similar {s}, worse {w}, p {pv:.2g}")
    L.append(f"    sources left out: {'; '.join(sorted(dropped)) or 'none'}")
    L.append("")
    L.append("Part 5. In-domain macro-F1 (median over seeds, same test rows): G0, G-NI0, G-NIS0, and G-NIS0 - G-NI0")
    for task in ("BIN", "CORE"):
        ch = []
        for src in DATASETS:
            for m in ("xgb", "rf"):
                v = {c: med([R[(src, task, m, c, s)]["in_domain"]["macro_f1"] for s in SEEDS]) for c in ("G0", "G-NI0", "G-NIS0")}
                ch.append((v["G-NIS0"] - v["G-NI0"], src, m))
                L.append(f"  {task:4s} {src:24s} {m:3s} G0 {v['G0']:.3f}  G-NI0 {v['G-NI0']:.3f}  G-NIS0 {v['G-NIS0']:.3f}"
                         f"  change {v['G-NIS0'] - v['G-NI0']:+.3f}")
        big = [f"{s} {m} {d:+.3f}" for d, s, m in sorted(ch) if abs(d) >= 0.05]
        L.append(f"  {task}: G-NIS0 - G-NI0 median {np.median([d for d, *_ in ch]):+.3f}; |change| >= 0.05 in "
                 f"{len(big)} of {len(ch)}: {'; '.join(big) or 'none'}")
    OUT.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
