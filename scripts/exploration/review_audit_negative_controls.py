"""Negative controls for audit checks I5 and N5: inject the defects found in
the Phase 1 review and confirm the checks report them."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("aud", SCRIPTS / "17_reproducibility_audit.py")
aud = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aud)

col = ["DNS_TTL_ANSWER"]
good = pq.read_table(SCRIPTS.parent / "data/processed/NF-BoT-IoT-v3/data.parquet",
                     columns=col).to_pandas()
wrapped = good.copy()
wrapped["DNS_TTL_ANSWER"] = good["DNS_TTL_ANSWER"].astype("int32")
a, b = aud.new_stats(col), aud.new_stats(col)
aud.update_stats(a, good, col)
aud.update_stats(b, wrapped, col)
print("I5 correct column ->",
      aud.compare_stats(a, a, col, merge_nonfinite=False, float_rtol=1e-6) or "pass")
print("I5 int32-wrapped column ->",
      aud.compare_stats(a, b, col, merge_nonfinite=False, float_rtol=1e-6))


def cross_split(h, s):
    o = np.lexsort((s, h))
    hs, ss = h[o], s[o]
    x = (hs[1:] == hs[:-1]) & (ss[1:] != ss[:-1])
    return int(np.unique(hs[1:][x]).size)


h = np.array([1, 1, 2, 3, 3, 4], dtype=np.uint64)
print("N5a clean ->", cross_split(h, np.array([0, 0, 1, 2, 2, 0], dtype=np.int8)))
print("N5a one vector in val and test ->",
      cross_split(h, np.array([0, 0, 1, 2, 1, 0], dtype=np.int8)))

t = pd.DataFrame({"x": [1.0, np.nan], "y": [2, 3]})
tr = pd.DataFrame({"x": [np.nan, 9.0], "y": [3, 3]})
print("N5b NaN-keyed test row matched in train ->",
      len(t.merge(tr, on=["x", "y"])), "(expected 1)")
