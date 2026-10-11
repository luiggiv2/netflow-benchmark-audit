"""rq2_threshold_sensitivity.py — internal audit, Part C: how much does the
RQ2 summary ("classes with macro F1 >= 0.95") depend on the 0.95 threshold?

Reads the stored outputs of scripts/34_shortcut_probes.py (no retraining) and
counts, per NF version, feature set (A = 41 features, C = without ports, TTL,
TCP window and DNS fields) and depth (1, 3), the classes reaching several
thresholds, plus the median macro F1 and the median loss from A to C.

Usage: python scripts/exploration/rq2_threshold_sensitivity.py
Output: data/exploration/rq2_threshold_sensitivity.txt
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nfv3_config import WORK  # noqa: E402

THRESHOLDS = [0.80, 0.90, 0.95, 0.99]
CELLS = [("A", 1), ("A", 3), ("C", 1), ("C", 3)]


def main() -> None:
    L = ["RQ2 summary under several macro-F1 thresholds (from data/audit/shortcut_probes/*.json)",
         f"Generated: {datetime.now(timezone.utc).isoformat()}", ""]
    for v in ("v2", "v3"):
        vals = {c: [] for c in CELLS}
        for f in sorted((WORK / "data/audit/shortcut_probes").glob(f"*-{v}.json")):
            for x in json.loads(f.read_text())["classes"].values():
                if "note" in x:
                    continue
                for s, d in CELLS:
                    vals[(s, d)].append(x["sets"][s][f"f1_d{d}"])
        n = len(vals[CELLS[0]])
        L.append(f"NF-{v}: {n} classes probed")
        for t in THRESHOLDS:
            L.append(f"  F1 >= {t:.2f}: " + "; ".join(
                f"{s} depth {d} {sum(y >= t for y in vals[(s, d)])}/{n}" for s, d in CELLS))
        L.append("  median F1: " + "; ".join(f"{s} depth {d} {np.median(vals[(s, d)]):.3f}" for s, d in CELLS))
        for d in (1, 3):
            loss = np.array(vals[("A", d)]) - np.array(vals[("C", d)])
            L.append(f"  F1 loss from A to C, depth {d}: median {np.median(loss):.3f}, max {loss.max():.3f}")
        L.append("")
    out = WORK / "data/exploration/rq2_threshold_sensitivity.txt"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
