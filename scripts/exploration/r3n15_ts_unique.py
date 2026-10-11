"""r3n15_ts_unique.py — third reviewer report, minor 15, 2026-10-10 (plan task
P6.1b, block 6): "with the absolute timestamps in the key, almost every feature
vector is unique" (05:124 and the Table 4 note) had no file in data/.

For each NF-v3 dataset, from the cached row hashes of the protocol feature sets
(results/eval/cache/<dataset>/hash_<set>.npy, the keys of the grouped splits):
rows, distinct vectors, and the share of rows whose vector occurs only once, for
the model base set (key of G0) and the base set plus both absolute timestamps
(key of G-TS). No data file is read and nothing stored is modified.

Usage: python scripts/exploration/r3n15_ts_unique.py
Output: data/exploration/r3n15_ts_unique.txt
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nfv3_config import WORK  # noqa: E402

DATASETS = ["NF-ToN-IoT-v3", "NF-BoT-IoT-v3", "NF-CSE-CIC-IDS2018-v3"]
SETS = {"base": "G0 key (48 features)", "ts": "G-TS key (48 features + 2 timestamps)"}
OUT = WORK / "data/exploration/r3n15_ts_unique.txt"


def main() -> None:
    L = ["Uniqueness of the feature vectors with and without the absolute timestamps (NF-v3)",
         f"Generated: {datetime.now(timezone.utc).isoformat()}",
         "Source: cached row hashes of the grouped-split keys (results/eval/cache/<dataset>/hash_<set>.npy)", ""]
    for ds in DATASETS:
        L.append(f"== {ds}")
        for s, name in SETS.items():
            h = np.load(WORK / f"results/eval/cache/{ds}/hash_{s}.npy")
            _, counts = np.unique(h, return_counts=True)
            single = int(np.sum(counts == 1))
            L.append(f"  {name}: rows {len(h):,}; distinct vectors {len(counts):,} ({100 * len(counts) / len(h):.2f} % of rows); "
                     f"rows on a vector that occurs once {single:,} ({100 * single / len(h):.2f} %); "
                     f"largest vector {int(counts.max()):,} rows")
            del h, counts
        L.append("")
    OUT.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
