"""split_key_sensitivity.py — internal audit, Part C (finding B6): how much
near-duplicate leakage does the adopted grouped split leave?

The stored split is keyed on the full feature vector of each version (49
features in NF-v3, 41 in NF-v2), which includes both ports. Two flows that
differ only in the ephemeral source port, or only in the NF-v3 IAT columns,
get different keys and can fall in train and test. For coarser keys, this
script measures on the stored split the share of test rows whose reduced
vector also occurs in train:

  full         the split key itself (must give 0)
  no_src_port  full minus L4_SRC_PORT
  no_ports     full minus both ports
  common41     the 41 features common to NF-v2 and NF-v3 (NF-v3 only)
  common39     common41 minus both ports (NF-v3 only)

Reported for all test rows and per canonical class (rows with a canonical
class). Read-only; uses the stored `split` column of tabular.parquet.

Usage: python scripts/exploration/split_key_sensitivity.py
Output: data/exploration/split_key_sensitivity.txt
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nfv3_config import (DATASETS, FEATURE_COLS_V2, PROCESSED_ROOT, WORK,  # noqa: E402
                         feature_cols, is_v2)

BATCH = 1_000_000
PORTS = ["L4_SRC_PORT", "L4_DST_PORT"]


def keys_for(dataset: str) -> dict[str, list[str]]:
    full = feature_cols(dataset)
    k = {"full": full,
         "no_src_port": [c for c in full if c != "L4_SRC_PORT"],
         "no_ports": [c for c in full if c not in PORTS]}
    if not is_v2(dataset):
        k["common41"] = FEATURE_COLS_V2
        k["common39"] = [c for c in FEATURE_COLS_V2 if c not in PORTS]
    return k


def main() -> None:
    L = ["Split-key sensitivity: share of grouped-test rows whose reduced vector also occurs in grouped train",
         f"Generated: {datetime.now(timezone.utc).isoformat()}",
         "Stored split keyed on the full vector (49 features NF-v3, 41 NF-v2). Read-only.", ""]
    for ds in DATASETS:
        keys = keys_for(ds)
        cols = feature_cols(ds) + ["canonical_label", "split"]
        hs = {k: [] for k in keys}
        sp, can = [], []
        for b in pq.ParquetFile(PROCESSED_ROOT / ds / "tabular.parquet").iter_batches(batch_size=BATCH, columns=cols):
            df = b.to_pandas()
            for k, c in keys.items():
                hs[k].append(pd.util.hash_pandas_object(df[c], index=False).to_numpy())
            sp.append(df["split"].astype(str).to_numpy())
            can.append(df["canonical_label"].astype(str).fillna("").to_numpy())
        sp, can = np.concatenate(sp), np.concatenate(can)
        tr, te = sp == "train", sp == "test"
        L.append(f"== {ds}: {len(sp):,} rows; test {te.sum():,}")
        classes = sorted(c for c in np.unique(can[te]) if c not in ("", "nan"))
        L.append(f"  {'key':12s} {'n_feat':>6s} {'all test':>9s}  " + "  ".join(f"{c[:14]:>14s}" for c in classes))
        for k, c in keys.items():
            h = np.concatenate(hs[k])
            hit = np.isin(h[te], np.unique(h[tr]))
            ct = can[te]
            per = [f"{hit[ct == cl].mean():14.4f}" for cl in classes]
            L.append(f"  {k:12s} {len(c):6d} {hit.mean():9.4f}  " + "  ".join(per))
            del h
        L.append("")
        print("\n".join(L[-(len(keys) + 3):]), flush=True)
    (WORK / "data/exploration/split_key_sensitivity.txt").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
