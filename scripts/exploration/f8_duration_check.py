"""f8_duration_check.py — internal audit of finding F8 (flow duration near
2^32 / 1000 ms in NF-v2).

Script 35 counts FLOW_DURATION_MILLISECONDS >= 4,294,000 ms. This check asks
whether that threshold captures the whole artefact:
  1. distribution of FLOW_DURATION_MILLISECONDS above 120 s, per dataset;
  2. lower edge of the cluster below 2^32 / 1000, and the share of rows and of
     each native class at or above the old threshold and at or above the edge;
  3. relation between FLOW_DURATION_MILLISECONDS and the directional
     durations: share of non-zero rows where
     4,294,967 - FLOW_DURATION_MILLISECONDS == max(DURATION_IN, DURATION_OUT)
     (exactly and within 1 ms), and share of zero-duration rows with a
     non-zero directional duration;
  4. for the rows that do not match within 1 ms: whether the implied duration
     exceeds the directional ones, and how long it is.

Usage: python scripts/exploration/f8_duration_check.py
Output: data/exploration/f8_duration_check.txt
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nfv3_config import DATASETS, PROCESSED_ROOT, WORK, is_v2  # noqa: E402

OLD_THRESHOLD = 4_294_000
WRAP = 2**32 // 1000            # 4,294,967 ms
BINS = [120_000, 600_000, 1_800_000, 4_000_000, OLD_THRESHOLD, 2**33]
COLS = ["FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT", "native_label"]


def main() -> None:
    L = ["F8 check: FLOW_DURATION_MILLISECONDS near 2^32 / 1000 ms",
         f"Generated: {datetime.now(timezone.utc).isoformat()}",
         f"Old threshold (script 35): >= {OLD_THRESHOLD:,} ms; 2^32 // 1000 = {WRAP:,} ms", ""]
    for ds in DATASETS:
        t = pq.read_table(PROCESSED_ROOT / ds / "data.parquet", columns=COLS)
        f, i, o = (t[c].to_numpy().astype(np.int64) for c in COLS[:3])
        lab = t["native_label"].to_pandas().astype(str).to_numpy()
        h = np.histogram(f, bins=BINS)[0]
        L.append(f"== {ds}: {len(f):,} rows; max {f.max():,} ms; zero {np.mean(f == 0):.4f}")
        L.append("  rows by duration bin (ms): " + "; ".join(
            f"[{BINS[k]:,}, {BINS[k + 1]:,}) {h[k]:,}" for k in range(len(h))))
        for c, x in (("DURATION_IN", i), ("DURATION_OUT", o)):
            L.append(f"  {c}: max {x.max():,} ms")
        if not is_v2(ds):
            L.append("")
            continue
        nz = f[f > 0]
        edge = int(nz.min())
        L.append(f"  non-zero FLOW_DURATION_MILLISECONDS: min {edge:,} ms "
                 f"(= {WRAP:,} - {WRAP - edge:,}); no value in [1, {edge:,})")
        old, new = f >= OLD_THRESHOLD, f >= edge
        L.append(f"  share of rows: >= old threshold {old.mean():.4f} ({old.sum():,}); "
                 f">= edge {new.mean():.4f} ({new.sum():,})")
        per = pd.DataFrame({"c": lab, "old": old, "new": new}).groupby("c").agg(
            rows=("new", "size"), old=("old", "mean"), new=("new", "mean")).sort_values("new", ascending=False)
        L.append("  per native class (rows; share >= edge; share >= old threshold):")
        for c, r in per.iterrows():
            L.append(f"    {c:28s} {int(r['rows']):>11,}  {r['new']:.4f}  {r['old']:.4f}")
        m = f > 0
        diff = (WRAP - f[m]) - np.maximum(i[m], o[m])
        z = ~m
        L.append(f"  non-zero rows where {WRAP:,} - FLOW_DURATION == max(DURATION_IN, DURATION_OUT): "
                 f"exact {np.mean(diff == 0):.4f}, within 1 ms {np.mean(np.abs(diff) <= 1):.4f}")
        L.append(f"  zero-duration rows with max(DURATION_IN, DURATION_OUT) > 0: "
                 f"{np.mean(np.maximum(i[z], o[z]) > 0):.4f}")
        L.append(f"  non-zero rows where {WRAP:,} - FLOW_DURATION >= max(DURATION_IN, DURATION_OUT): "
                 f"{np.mean(diff >= -1):.4f}")
        bad = np.abs(diff) > 1
        if bad.any():
            implied = WRAP - f[m]
            q = np.percentile(implied[bad], [50, 95, 100])
            L.append(f"  rows off by more than 1 ms: {bad.sum():,}; implied duration above the longest "
                     f"directional one {np.mean(diff[bad] > 0):.4f}; both directional durations 0 "
                     f"{np.mean(np.maximum(i[m], o[m])[bad] == 0):.4f}; implied duration median "
                     f"{q[0]:,.0f} ms, p95 {q[1]:,.0f} ms, max {q[2]:,.0f} ms")
            top = pd.Series(lab[m][bad]).value_counts().head(4)
            L.append("    top classes: " + "; ".join(f"{c} {n:,}" for c, n in top.items()))
        L.append("")
    out = WORK / "data/exploration/f8_duration_check.txt"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
