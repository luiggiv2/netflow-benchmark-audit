"""f2b_window_check.py — reviewer point M9, 2026-10-06: direct check of the
mechanism of F2b (decision 10 of research/findings_classification.md): are
the Benign rows that share a feature vector with an attack class traffic of
the attacker hosts outside the attack windows?

For each group (dataset, attack label), on the 40-feature key without
L4_SRC_PORT (key 40 of 30_audit_duplicates_conflicts.py):
  - conflicting Benign rows: Benign rows whose vector also carries the label;
  - attacker hosts: source addresses of the rows of the label;
  - attack windows: the intervals [start, end] of the label's flows, merged
    when the gap between consecutive flows is at most GAP minutes (1, 5, 30);
  - per GAP, the share of conflicting Benign rows whose start time lies
    outside every window, overall and for the rows from attacker hosts.
All Benign rows from attacker hosts are given for reference. Group F2e (CIC
Infiltration) is included for comparison.

Usage: python scripts/exploration/f2b_window_check.py
Output: data/exploration/f2b_window_check.txt
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nfv3_config import FEATURE_COLS_V2, PROCESSED_ROOT, WORK, tabular_row_hash  # noqa: E402

KEY = [c for c in FEATURE_COLS_V2 if c != "L4_SRC_PORT"]
GROUPS = [("NF-ToN-IoT-v3", "scanning"), ("NF-ToN-IoT-v3", "ddos"), ("NF-ToN-IoT-v3", "password"),
          ("NF-BoT-IoT-v3", "Reconnaissance"), ("NF-CSE-CIC-IDS2018-v3", "Infiltration")]
GAPS_MIN = [1, 5, 30]
BATCH = 1_000_000


def load(ds: str) -> pd.DataFrame:
    raw = pq.read_table(PROCESSED_ROOT / ds / "data.parquet",
                        columns=["native_label", "IPV4_SRC_ADDR", "FLOW_START_MILLISECONDS",
                                 "FLOW_END_MILLISECONDS"]).to_pandas()
    h = np.concatenate([tabular_row_hash(b.to_pandas(), KEY) for b in pq.ParquetFile(
        PROCESSED_ROOT / ds / "tabular.parquet").iter_batches(batch_size=BATCH, columns=KEY)])
    raw["h"] = h
    raw["native_label"] = raw["native_label"].astype(str).astype("category")
    raw["IPV4_SRC_ADDR"] = raw["IPV4_SRC_ADDR"].astype("category")
    return raw


def windows(start: np.ndarray, end: np.ndarray, gap_ms: int) -> tuple[np.ndarray, np.ndarray]:
    o = np.argsort(start)
    s, e = start[o], np.maximum.accumulate(end[o])
    new = np.r_[True, s[1:] > e[:-1] + gap_ms]
    idx = np.flatnonzero(new)
    return s[idx], np.r_[e[idx[1:] - 1], e[-1]]


def inside(t: np.ndarray, ws: np.ndarray, we: np.ndarray) -> np.ndarray:
    i = np.searchsorted(ws, t, side="right") - 1
    return (i >= 0) & (t <= we[np.clip(i, 0, None)])


def ts(ms) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M")


def main() -> None:
    L = ["F2b mechanism check: Benign rows sharing a 40-feature vector with an attack label",
         f"Generated: {datetime.now(timezone.utc).isoformat()}",
         "Windows: flows of the label merged when the gap is <= GAP minutes; 'outside' = start time of the",
         "Benign row in no window. Attacker hosts = source addresses of the label's rows.", ""]
    cache = {}
    for ds, lab in GROUPS:
        if ds not in cache:
            cache.clear()
            cache[ds] = load(ds)
        d = cache[ds]
        is_lab = (d["native_label"] == lab).to_numpy()
        is_ben = (d["native_label"] == "Benign").to_numpy()
        h = d["h"].to_numpy()
        conf = is_ben & np.isin(h, np.intersect1d(np.unique(h[is_lab]), np.unique(h[is_ben])))
        hosts = set(d.loc[is_lab, "IPV4_SRC_ADDR"].astype(str).unique())
        from_att = d["IPV4_SRC_ADDR"].astype(str).isin(hosts).to_numpy()
        st = d["FLOW_START_MILLISECONDS"].to_numpy()
        en = d["FLOW_END_MILLISECONDS"].to_numpy()
        L.append(f"== {ds} Benign / {lab}: label rows {is_lab.sum():,} from {len(hosts)} source hosts "
                 f"({ts(st[is_lab].min())} to {ts(en[is_lab].max())} UTC)")
        L.append(f"   conflicting Benign rows {conf.sum():,}; from attacker hosts {(conf & from_att).sum():,} "
                 f"({100 * (conf & from_att).sum() / max(conf.sum(), 1):.1f} %); all Benign rows from attacker "
                 f"hosts {(is_ben & from_att).sum():,}")
        for g in GAPS_MIN:
            ws, we = windows(st[is_lab], en[is_lab], g * 60_000)
            out_c = conf & ~inside(st, ws, we)
            out_ca = out_c & from_att
            out_ba = is_ben & from_att & ~inside(st, ws, we)
            L.append(f"   GAP {g:>2} min: {len(ws):>6,} windows, {(we - ws).sum() / 3.6e6:7.1f} h; outside: "
                     f"conflicting Benign {100 * out_c.sum() / max(conf.sum(), 1):5.1f} %, of those from attacker "
                     f"hosts {100 * out_ca.sum() / max((conf & from_att).sum(), 1):5.1f} %; all Benign from "
                     f"attacker hosts {100 * out_ba.sum() / max((is_ben & from_att).sum(), 1):5.1f} %")
        L.append("")
    out = WORK / "data/exploration/f2b_window_check.txt"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
