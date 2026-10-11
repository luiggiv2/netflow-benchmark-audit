"""r3n5_gnc_removed.py — third reviewer report, minor 5, 2026-10-10 (plan task
P6.1b, block 6): is the in-domain gain of G-NC "largely mechanical"?

G-NC drops every row of a feature vector that carries more than one label
(base key), not only the minority rows. For each dataset, task and model, from
the stored in-domain runs (data/eval/in_domain/*.jsonl, five seeds): the G-NC
gain over G0 (median and range of the paired per-seed differences, in points),
the gap between the G0 conflict ceiling and 1 (median over seeds, in points),
and the share of the G0 test rows that G-NC removes (1 - G-NC test rows / G0
test rows, same seed). No model is refitted.

Usage: python scripts/exploration/r3n5_gnc_removed.py
Output: data/exploration/r3n5_gnc_removed.txt
"""

from __future__ import annotations

import json
import statistics as st
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nfv3_config import WORK  # noqa: E402

DATASETS = ["NF-ToN-IoT-v3", "NF-BoT-IoT-v3", "NF-CSE-CIC-IDS2018-v3", "NF-ToN-IoT-v2", "NF-BoT-IoT-v2",
            "NF-CSE-CIC-IDS2018-v2"]
OUT = WORK / "data/exploration/r3n5_gnc_removed.txt"


def main() -> None:
    L = ["G-NC against G0 within each dataset: gain, ceiling gap and share of test rows removed",
         f"Generated: {datetime.now(timezone.utc).isoformat()}",
         "gain: G-NC minus G0 macro-F1 per seed (points), median [min, max]; ceiling gap: 100 x (1 - G0 conflict "
         "ceiling), median over seeds; removed: 1 - G-NC test rows / G0 test rows per seed, median [min, max]", ""]
    for ds in DATASETS:
        R = [json.loads(x) for x in open(WORK / f"data/eval/in_domain/{ds}.jsonl")]
        for task in ("BIN", "CAN"):
            for m in ("rf", "xgb"):
                g0 = {r["seed"]: r for r in R if r["condition"] == "G0" and r["task"] == task and r["model"] == m}
                nc = {r["seed"]: r for r in R if r["condition"] == "G-NC" and r["task"] == task and r["model"] == m}
                seeds = sorted(set(g0) & set(nc))
                gain = [100 * (nc[s]["macro_f1"] - g0[s]["macro_f1"]) for s in seeds]
                gap = st.median(100 * (1 - g0[s]["conflict_ceiling_macro_f1"]) for s in seeds)
                rem = [1 - nc[s]["test_rows"] / g0[s]["test_rows"] for s in seeds]
                L.append(f"  {ds:22s} {task} {m:3s} gain {st.median(gain):+6.2f} [{min(gain):+6.2f}, {max(gain):+6.2f}]"
                         f"  ceiling gap {gap:5.2f}  removed {100 * st.median(rem):5.1f} % "
                         f"[{100 * min(rem):.1f}, {100 * max(rem):.1f}]  (n={len(seeds)})")
        L.append("")
    OUT.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
