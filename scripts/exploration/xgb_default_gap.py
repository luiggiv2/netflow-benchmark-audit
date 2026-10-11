"""xgb_default_gap.py — second reviewer report, point N8, 2026-10-08: trace of
"with the library default, 14 runs of the canonical task on
NF-CSE-CIC-IDS2018-v3 scored more than 10 points below the random forest"
(Section 7, internal validity).

Per dataset and task, the in-domain XGBoost runs (all conditions and seeds of
P4.2) whose macro-F1 is more than GAP below the random forest run of the same
condition, task and seed; with the library default
(data/eval/in_domain_xgbdefault/) and with max_delta_step=1
(data/eval/in_domain/, the setting used in the paper). The random forest runs
are the same in both directories.

Usage: python scripts/exploration/xgb_default_gap.py
Output: data/exploration/xgb_default_gap.txt
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ["NF-ToN-IoT-v2", "NF-BoT-IoT-v2", "NF-CSE-CIC-IDS2018-v2", "NF-ToN-IoT-v3", "NF-BoT-IoT-v3",
            "NF-CSE-CIC-IDS2018-v3"]
GAP = 0.10


def load(d: str, ds: str) -> dict:
    out = {}
    for line in open(ROOT / f"data/eval/{d}/{ds}.jsonl"):
        r = json.loads(line)
        out[(r["condition"], r["task"], r["model"], r["seed"])] = r["macro_f1"]
    return out


def main() -> None:
    L = ["In-domain XGBoost runs more than 10 macro-F1 points below the random forest (same condition, task, seed)",
         f"Generated: {datetime.now(timezone.utc).isoformat()}", ""]
    for d, tag in (("in_domain_xgbdefault", "library default"), ("in_domain", "max_delta_step=1 (paper)")):
        L.append(f"== XGBoost {tag} ({d})")
        total = n_runs = 0
        for ds in DATASETS:
            r = load(d, ds)
            for task in ("BIN", "CAN"):
                xs = [(c, s, v, r[(c, task, "rf", s)]) for (c, t, m, s), v in r.items()
                      if t == task and m == "xgb" and (c, task, "rf", s) in r]
                low = sorted((c, s, v, f) for c, s, v, f in xs if v < f - GAP)
                n_runs += len(xs)
                total += len(low)
                if low:
                    L.append(f"  {ds:24s} {task}: {len(low)} of {len(xs)} runs; " + "; ".join(
                        f"{c} s{s} {v:.3f} vs RF {f:.3f}" for c, s, v, f in low))
        L.append(f"  total: {total} of {n_runs} runs")
        L.append("")
    out = ROOT / "data/exploration/xgb_default_gap.txt"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
