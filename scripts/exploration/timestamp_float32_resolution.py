"""timestamp_float32_resolution.py — second reviewer report, minor 19,
2026-10-08: resolution of the absolute NF-v3 timestamps as the models of G-TS
see them. The model matrix (cache of 44_cross_dataset.py, loader of
42_in_domain.py) is float32, and both XGBoost (hist) and scikit-learn trees
work in float32. The loader shifts the timestamps by the dataset minimum
(42_in_domain.py), so the values are milliseconds since the first flow and the
float32 spacing is that of these offsets, not of epoch milliseconds.

Usage: python scripts/exploration/timestamp_float32_resolution.py
Output: data/exploration/timestamp_float32_resolution.txt
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "results/eval/cache"
DATASETS = ["NF-ToN-IoT-v3", "NF-BoT-IoT-v3", "NF-CSE-CIC-IDS2018-v3"]


def main() -> None:
    L = ["Float32 resolution of the NF-v3 absolute timestamps in the model matrix (offsets from the dataset minimum)",
         f"Generated: {datetime.now(timezone.utc).isoformat()}", ""]
    for ds in DATASETS:
        meta = json.loads((CACHE / ds / "meta.json").read_text())
        X = np.load(CACHE / ds / "X.npy", mmap_mode="r")
        for col in ("FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS"):
            v = np.asarray(X[:, meta["cols"].index(col)])
            lo, hi = float(v.min()), float(v.max())
            sp_hi = float(np.spacing(np.float32(hi)))  # coarsest spacing, at the largest offset
            L.append(f"  {ds:24s} {col:24s} dtype {X.dtype}; offsets {lo / 8.64e7:.2f} to {hi / 8.64e7:.2f} days;"
                     f" coarsest float32 spacing {sp_hi:g} ms; distinct values {np.unique(v).size:,} of {v.size:,} rows")
    out = ROOT / "data/exploration/timestamp_float32_resolution.txt"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
