"""five_tuple_split_check.py — reviewer point (e), 2026-10-06: the numbers of
supplementary S3 on the withdrawn five-tuple split ("groups of 1.6 to 2.9
rows", "45.7 % of the NF-ToN-IoT-v3 test rows had an identical vector in
training"). The original output (data/exploration/review_duplicates_5tuple_split.txt)
was removed on 2026-10-01 and the group sizes were never written to a file.

Recomputed for the three NF-v3 datasets with the key of the withdrawn
protocol (research/split_protocol.md §3): canonical label, source and
destination address, source and destination port, protocol. Rows outside
the canonical taxonomy are left out, as in that protocol. Per dataset: rows
per distinct key (mean and median), and, under a grouped 70/15/15 split of
the keys (grouped_split_codes, seed 42; the bucket function of the original
run is not recoverable, so the share can differ slightly), the share of test
rows whose 49-feature vector also occurs in train.

Usage: python scripts/exploration/five_tuple_split_check.py
Output: data/exploration/five_tuple_split_check.txt
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nfv3_config import (DATASETS, FEATURE_COLS, PROCESSED_ROOT, WORK, grouped_split_codes,  # noqa: E402
                         is_v2, tabular_row_hash)

KEY = ["canonical_label", "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL"]
BATCH = 1_000_000


def main() -> None:
    L = ["Withdrawn five-tuple split: group size and exact-vector leakage (NF-v3, canonical rows)",
         f"Generated: {datetime.now(timezone.utc).isoformat()}",
         "Key: canonical label, IPs, ports, protocol. Leakage: test rows whose 49-feature vector occurs in train",
         "(grouped split of the keys, seed 42).", ""]
    for ds in [d for d in DATASETS if not is_v2(d)]:
        kh, mask, vh = [], [], []
        for rb in pq.ParquetFile(PROCESSED_ROOT / ds / "data.parquet").iter_batches(batch_size=BATCH, columns=KEY):
            r = rb.to_pandas()
            mask.append((r["canonical_label"].fillna("").astype(str) != "").to_numpy())
            kh.append(pd.util.hash_pandas_object(r[KEY].astype(str), index=False).to_numpy())
        for tb in pq.ParquetFile(PROCESSED_ROOT / ds / "tabular.parquet").iter_batches(batch_size=BATCH,
                                                                                       columns=FEATURE_COLS):
            vh.append(tabular_row_hash(tb.to_pandas(), FEATURE_COLS))
        mask = np.concatenate(mask)
        kh, vh = np.concatenate(kh)[mask], np.concatenate(vh)[mask]
        _, size = np.unique(kh, return_counts=True)
        split = grouped_split_codes(kh, 42)
        test = split == 2
        leak = np.isin(vh[test], np.unique(vh[split == 0]))
        L.append(f"{ds:24s} canonical rows {len(kh):,}; keys {len(size):,}; rows per key mean {size.mean():.2f}, "
                 f"median {np.median(size):.0f}, max {size.max():,}; test rows with their vector in train "
                 f"{leak.sum():,} / {test.sum():,} ({100 * leak.mean():.1f} %)")
    out = WORK / "data/exploration/five_tuple_split_check.txt"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
