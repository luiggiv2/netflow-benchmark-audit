"""rq4_trace_checks.py — reviewer point (e), 2026-10-06: two RQ4 statements of
the manuscript that had no report line in data/.

  1. Which feature ranks first in every seed: for every XGBoost run of
     condition G0 (and G-TS for reference), the top-1 feature of the mean
     |SHAP| ranking on the source test rows, per seed.
  2. Share of random-forest permutation importances that are exactly zero
     (and <= 0), over all features of the 90 RF runs (G0, BIN/CAN/CORE).
  3. Stability of the XGBoost rankings (Q2 of 47_shap.py): median over cells of
     the per-cell median Jaccard@10 and Spearman, across seeds under G0 and
     against G0 for the other conditions (read from data/eval/shap_report.txt).

Reads data/eval/shap/*.jsonl and data/eval/shap_report.txt.
Usage: python scripts/exploration/rq4_trace_checks.py
Output: data/exploration/rq4_trace_checks.txt
"""

from __future__ import annotations

import ast
import json
import re
import statistics as st
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nfv3_config import DATASETS, WORK  # noqa: E402


def as_dict(x):
    return x if isinstance(x, dict) else ast.literal_eval(x)


def main() -> None:
    top1, perm = {}, []
    for ds in DATASETS:
        for line in open(WORK / f"data/eval/shap/{ds}.jsonl"):
            r = json.loads(line)
            imp = as_dict(r["importance_test"])
            if r["model"] == "xgb" and r["condition"] in ("G0", "G-TS"):
                top1.setdefault((ds, r["condition"], r["task"]), {})[r["seed"]] = max(imp, key=imp.get)
            elif r["model"] == "rf":
                perm += list(imp.values())
    L = ["RQ4 trace checks (reviewer point e)", f"Generated: {datetime.now(timezone.utc).isoformat()}", "",
         "1. XGBoost top-1 feature (mean |SHAP|, source test rows) per seed; '=' marks one feature in all seeds"]
    for (ds, cond, task), seeds in sorted(top1.items()):
        c = Counter(seeds.values())
        feat, n = c.most_common(1)[0]
        mark = "=" if n == len(seeds) else " "
        L.append(f"  {mark} {ds:24s} {cond:4s} {task:4s} {feat:28s} {n}/{len(seeds)} seeds"
                 + ("" if n == len(seeds) else f"  (others: {dict(c)})"))
    # second reviewer report, minor 5 (2026-10-08): count of G0 cells with the destination port first in every seed
    g0c = {k: v for k, v in top1.items() if k[1] == "G0"}
    dport = sorted(f"{ds} {task}" for (ds, _, task), v in g0c.items() if set(v.values()) == {"L4_DST_PORT"})
    L.append(f"  G0 cells with L4_DST_PORT first in every seed: {len(dport)} of {len(g0c)} ({'; '.join(dport)})")
    zero = sum(v == 0 for v in perm)
    nonpos = sum(v <= 0 for v in perm)
    L += ["", f"2. RF permutation importances (macro-F1 decrease): {len(perm):,} values over the RF runs; "
          f"exactly 0: {zero:,} ({100 * zero / len(perm):.1f} %); <= 0: {nonpos:,} ({100 * nonpos / len(perm):.1f} %)"]
    q2 = (WORK / "data/eval/shap_report.txt").read_text().split("Q2.")[1].split("Q2b.")[0]
    num = r"([\d.]+) \[[^\]]+\]"
    g0 = [(float(a), float(b)) for a, b in re.findall(r" G0 +\w+ +xgb seeds: J@10 " + num + r"  rho " + num, q2)]
    vs = [(float(a), float(b)) for a, b in re.findall(r"vs G0: J@10 " + num + r"  rho " + num, q2)]
    L += ["", f"3. XGBoost ranking stability (median over cells of the per-cell medians): across seeds under G0 "
          f"({len(g0)} cells) Jaccard@10 {st.median(x[0] for x in g0):.2f}, Spearman {st.median(x[1] for x in g0):.2f}; "
          f"other conditions against G0 ({len(vs)} cells) Jaccard@10 {st.median(x[0] for x in vs):.2f}, "
          f"Spearman {st.median(x[1] for x in vs):.2f}"]
    # per condition (second reviewer report, minor 3, 2026-10-08): the pooled value above includes G-TS
    vc = re.findall(r"(G-NP|G-NI|G-TS) +\w+ +xgb seeds: .*?vs G0: J@10 " + num + r"  rho " + num, q2)
    groups = {"G-NP": ["G-NP"], "G-NI": ["G-NI"], "G-TS": ["G-TS"], "G-NP and G-NI": ["G-NP", "G-NI"]}
    for name, conds in groups.items():
        xs = [(float(a), float(b)) for c, a, b in vc if c in conds]
        L.append(f"   {name} against G0 ({len(xs)} cells): Jaccard@10 {st.median(x[0] for x in xs):.2f}, "
                 f"Spearman {st.median(x[1] for x in xs):.2f}")
    # second reviewer report, minor 4 (2026-10-08): source vs target rankings of the same G0 model (Q3)
    q3 = (WORK / "data/eval/shap_report.txt").read_text().split("Q3.")[1].split("Q4.")[0]
    L += ["", "4. Same G0 XGBoost model, source test rows vs target rows (Q3; per-pair medians over seeds), "
          "median over pairs"]
    for task in ("BIN", "CORE"):
        xs = [(float(a), float(b)) for a, b in re.findall(r" " + task + r" +xgb -> \S+ +J@10 " + num + r"  rho " + num, q3)]
        L.append(f"   {task} ({len(xs)} pairs): Jaccard@10 {st.median(x[0] for x in xs):.2f} "
                 f"(min {min(x[0] for x in xs):.2f}), Spearman {st.median(x[1] for x in xs):.2f} "
                 f"(min {min(x[1] for x in xs):.2f})")
    out = WORK / "data/exploration/rq4_trace_checks.txt"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
