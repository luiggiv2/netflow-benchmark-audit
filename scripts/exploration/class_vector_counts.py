"""Distinct feature vectors per native class under the Phase 4 keys
(nfv3_config.model_feature_sets), and the split of each class with seed 42
and seed 1. Read-only exploration for Phase 4 block A: explains classes that
the grouped split places entirely in one partition.

Usage: python scripts/exploration/class_vector_counts.py
Output: data/exploration/class_vector_counts.txt
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nfv3_config import (DATASETS, FEATURE_COLS_V2, PROCESSED_ROOT, WORK, feature_cols,  # noqa: E402
                         grouped_split_codes, model_feature_sets, tabular_row_hash)

MAX_VECTORS = 50  # report classes with at most this many distinct vectors in some key
KEYS = {"A40": [c for c in FEATURE_COLS_V2 if c != "L4_SRC_PORT"]}

L = ["Distinct feature vectors per native class (classes with <= "
     f"{MAX_VECTORS} distinct vectors under some key are listed)",
     f"Generated: {datetime.now(timezone.utc).isoformat()}",
     "Keys: base (all features but L4_SRC_PORT), noport, noid; A40 = 41 common features without",
     "L4_SRC_PORT (set A of the re-keyed probes). Rows per split for seeds 42 and 1 (train/val/test).", ""]
for ds in DATASETS:
    sets = {k: v for k, v in model_feature_sets(ds).items() if k != "ts"} | KEYS
    cols = sorted({c for v in sets.values() for c in v}, key=feature_cols(ds).index)
    df = pq.read_table(PROCESSED_ROOT / ds / "tabular.parquet", columns=cols + ["native_label"]).to_pandas()
    lab = df["native_label"].astype(str).to_numpy()
    L.append(f"== {ds}")
    for k, kc in sets.items():
        h = tabular_row_hash(df, kc)
        s42, s1 = grouped_split_codes(h, 42), grouped_split_codes(h, 1)
        for c in sorted(np.unique(lab)):
            m = lab == c
            nv = len(np.unique(h[m]))
            if nv <= MAX_VECTORS:
                b42, b1 = np.bincount(s42[m], minlength=3), np.bincount(s1[m], minlength=3)
                L.append(f"  {k:6s} {c:28s} rows {m.sum():>9,d}  distinct vectors {nv:>4d}  "
                         f"seed 42 {'/'.join(map(str, b42))}  seed 1 {'/'.join(map(str, b1))}")
    L.append("")
    del df
out = WORK / "data/exploration/class_vector_counts.txt"
out.write_text("\n".join(L) + "\n")
print("\n".join(L))
