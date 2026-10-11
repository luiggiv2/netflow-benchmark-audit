"""src_port_signal.py — internal audit, Part C (C1): is L4_SRC_PORT attack
signal or testbed identity?

Per dataset and native class:
  - rows, distinct source ports, share of rows on the most frequent source
    port (and which port), share of rows with a source port below 1024;
  - single-feature separability, attack class vs benign: macro F1 of the
    20-tree random forest trained on L4_SRC_PORT alone, and on L4_DST_PORT
    alone, read from the stored Flood-heuristic outputs
    (data/audit/flood_heuristics/<dataset>.json, field hdf.per_feature).

Read-only. Usage: python scripts/exploration/src_port_signal.py
Output: data/exploration/src_port_signal.txt
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nfv3_config import DATASETS, PROCESSED_ROOT, WORK  # noqa: E402


def main() -> None:
    L = ["L4_SRC_PORT: concentration per class and single-feature separability vs benign",
         f"Generated: {datetime.now(timezone.utc).isoformat()}",
         "HDF src / dst = macro F1 of a 20-tree random forest on that port alone, class vs benign",
         "(from data/audit/flood_heuristics/<dataset>.json; rows deduplicated on the full record).", ""]
    summary = []
    for ds in DATASETS:
        t = pq.read_table(PROCESSED_ROOT / ds / "tabular.parquet", columns=["L4_SRC_PORT", "native_label"])
        df = pd.DataFrame({"p": t["L4_SRC_PORT"].to_numpy(),
                           "c": t["native_label"].to_pandas().astype(str).to_numpy()})
        fh = json.loads((WORK / "data/audit/flood_heuristics" / f"{ds}.json").read_text())["classes"]
        L.append(f"== {ds}")
        L.append(f"  {'class':28s} {'rows':>11s} {'distinct':>9s} {'top port':>9s} {'top %':>6s} "
                 f"{'<1024 %':>8s} {'HDF src':>8s} {'HDF dst':>8s}")
        for c, g in df.groupby("c", sort=True):
            vc = g["p"].value_counts()
            top, share = int(vc.index[0]), vc.iloc[0] / len(g)
            low = float((g["p"] < 1024).mean())
            pf = fh.get(c, {}).get("hdf", {}).get("per_feature", {})
            hs, hd = pf.get("L4_SRC_PORT"), pf.get("L4_DST_PORT")
            fmt = lambda x: f"{x:8.3f}" if x is not None else f"{'-':>8s}"  # noqa: E731
            L.append(f"  {c:28s} {len(g):>11,} {g['p'].nunique():>9,} {top:>9d} {100 * share:6.1f} "
                     f"{100 * low:8.1f} {fmt(hs)} {fmt(hd)}")
            if hs is not None:
                summary.append((ds, c, hs))
        L.append("")
    hs = np.array([x[2] for x in summary])
    L.append(f"Attack classes with HDF on L4_SRC_PORT alone: {len(hs)}; "
             f">= 0.90: {(hs >= 0.90).sum()}; >= 0.95: {(hs >= 0.95).sum()}; median {np.median(hs):.3f}")
    L.append("Classes with HDF src >= 0.95: " + "; ".join(
        f"{d.replace('NF-', '')} {c} {h:.3f}" for d, c, h in summary if h >= 0.95))
    out = WORK / "data/exploration/src_port_signal.txt"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
