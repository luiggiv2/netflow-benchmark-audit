"""P1.4 exploration: unique-IP counts and class concentration per dataset.

Loads IPV4_SRC_ADDR, IPV4_DST_ADDR, Attack and FLOW_START_MILLISECONDS for
each of the three NF-v3 datasets, applies the canonical class mapping and
reports:

- number of unique source and destination IPs (total)
- unique source IPs per attack class (native and canonical)
- for each canonical class, the number of source IPs holding 90% of its flows
- temporal coverage (min and max FLOW_START_MILLISECONDS)
- simulation of the 70/15/15 IP-hash split with seed 42, reporting per-class
  flow counts in each bucket and any at-risk classes (<1000 flows in test)

Runs the three datasets sequentially (memory-safe on 16 GB) and prints a
consolidated report at the end.
"""

from __future__ import annotations

import hashlib
import time
from collections import Counter
from pathlib import Path

import pandas as pd

WORK = Path(__file__).resolve().parents[2]  # repository root (was an absolute path)
MAPPING_PATH = WORK / "data/schemas/class_mapping.csv"

DATASETS = {
    "NF-ToN-IoT-v3": WORK / "02934b58528a226b_NFV3DATA-A11964_A11964/data/NF-ToN-IoT-v3.csv",
    "NF-BoT-IoT-v3": WORK / "d509c9db7490cf92_NFV3DATA-A11964_A11964/data/NF-BoT-IoT-v3.csv",
    "NF-CSE-CIC-IDS2018-v3": WORK / "f78acbaa2afe1595_NFV3DATA-A11964_A11964/data/NF-CICIDS2018-v3.csv",
}

SEED = 42
COLS = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Attack", "FLOW_START_MILLISECONDS"]


def hash_bucket(value: str, seed: int) -> int:
    """Deterministic hash-based bucket assignment for a source IP."""
    h = hashlib.md5(f"{seed}|{value}".encode()).hexdigest()
    return int(h[:8], 16) % 100


def assign_split(bucket: int) -> str:
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "val"
    return "test"


def load_canonical_map(mapping_path: Path) -> dict:
    m = pd.read_csv(mapping_path)
    m["canonical_label"] = m["canonical_label"].fillna("")
    return {
        (row.dataset, row.native_label): row.canonical_label
        for row in m.itertuples()
    }


def summarise_temporal(ts: pd.Series) -> str:
    lo = pd.to_datetime(ts.min(), unit="ms", utc=True)
    hi = pd.to_datetime(ts.max(), unit="ms", utc=True)
    span_days = (ts.max() - ts.min()) / 1000 / 86400
    return f"from {lo} to {hi} (~{span_days:.2f} days)"


def analyse(dataset_name: str, path: Path, mapping: dict) -> dict:
    t0 = time.time()
    print(f"\n{'='*76}")
    print(f"DATASET: {dataset_name}")
    print(f"{'='*76}")
    print(f"Loading {path.name} (usecols={COLS})...", flush=True)
    df = pd.read_csv(path, usecols=COLS, low_memory=False)
    print(f"  rows={len(df):,}  time={time.time()-t0:.1f}s")

    n_src = df["IPV4_SRC_ADDR"].nunique()
    n_dst = df["IPV4_DST_ADDR"].nunique()
    print(f"  unique IPV4_SRC_ADDR: {n_src:,}")
    print(f"  unique IPV4_DST_ADDR: {n_dst:,}")
    print(f"  temporal coverage:    {summarise_temporal(df['FLOW_START_MILLISECONDS'])}")

    df["canonical"] = [
        mapping.get((dataset_name, str(a)), "") for a in df["Attack"].astype(str)
    ]

    print(f"\nUnique source IPs per NATIVE attack class:")
    native = df.groupby("Attack")["IPV4_SRC_ADDR"].agg(["nunique", "count"])
    native.columns = ["unique_src_ips", "flow_count"]
    native = native.sort_values("flow_count", ascending=False)
    for cls, row in native.iterrows():
        marker = " ⚠" if row["unique_src_ips"] < 3 else ""
        print(
            f"  {str(cls):32s}  IPs={int(row['unique_src_ips']):>8,d}"
            f"  flows={int(row['flow_count']):>15,d}{marker}"
        )

    print(f"\nUnique source IPs per CANONICAL class (empty string = excluded):")
    canonical = df.groupby("canonical")["IPV4_SRC_ADDR"].agg(["nunique", "count"])
    canonical.columns = ["unique_src_ips", "flow_count"]
    canonical = canonical.sort_values("flow_count", ascending=False)
    for cls, row in canonical.iterrows():
        marker = " ⚠" if row["unique_src_ips"] < 3 and cls != "" else ""
        print(
            f"  {str(cls) if cls else '(excluded)':32s}  IPs={int(row['unique_src_ips']):>8,d}"
            f"  flows={int(row['flow_count']):>15,d}{marker}"
        )

    print(f"\nConcentration: source IPs holding 90% of flows per canonical class")
    for cls in canonical.index:
        if cls == "":
            continue
        sub = df[df["canonical"] == cls]
        ip_counts = sub["IPV4_SRC_ADDR"].value_counts()
        total = ip_counts.sum()
        cumul = ip_counts.cumsum()
        n90 = int((cumul <= 0.9 * total).sum()) + 1
        print(
            f"  {cls:32s}  total_flows={total:>15,d}  "
            f"unique_IPs={len(ip_counts):>8,d}  "
            f"IPs_holding_90pct={n90:>6,d}"
        )

    print(f"\nSplit simulation (hash of src IP, seed={SEED}, 70/15/15):")
    unique_src = df["IPV4_SRC_ADDR"].drop_duplicates().reset_index(drop=True)
    ip_to_bucket = {ip: hash_bucket(str(ip), SEED) for ip in unique_src}
    ip_to_split = {ip: assign_split(b) for ip, b in ip_to_bucket.items()}
    df["split"] = df["IPV4_SRC_ADDR"].map(ip_to_split)

    print(f"\n  IP-level distribution:")
    ip_split_count = Counter(ip_to_split.values())
    total_ips = sum(ip_split_count.values())
    for bucket in ("train", "val", "test"):
        n = ip_split_count.get(bucket, 0)
        print(f"    {bucket:5s}  IPs={n:>8,d}  ({n/total_ips*100:5.2f}%)")

    print(f"\n  Flow-level distribution per canonical class:")
    ct = pd.crosstab(df["canonical"], df["split"])
    for col in ("train", "val", "test"):
        if col not in ct.columns:
            ct[col] = 0
    ct = ct[["train", "val", "test"]]
    ct["total"] = ct.sum(axis=1)
    ct = ct.sort_values("total", ascending=False)
    print(f"    {'class':32s}  {'train':>15s}  {'val':>15s}  {'test':>15s}")
    for cls, row in ct.iterrows():
        label = str(cls) if cls else "(excluded)"
        marker = ""
        if cls != "" and int(row["test"]) < 1000:
            marker = " ⚠ <1000 in test"
        print(
            f"    {label:32s}  {int(row['train']):>15,d}"
            f"  {int(row['val']):>15,d}  {int(row['test']):>15,d}{marker}"
        )

    at_risk = [
        cls
        for cls in ct.index
        if cls != "" and int(ct.loc[cls, "test"]) < 1000
    ]
    if at_risk:
        print(f"\n  AT-RISK canonical classes (test < 1000 flows): {at_risk}")
    else:
        print(f"\n  All canonical classes have >= 1000 flows in test. OK.")

    print(f"\n  Total time for {dataset_name}: {time.time()-t0:.1f}s")
    return {
        "dataset": dataset_name,
        "rows": len(df),
        "unique_src_ips": n_src,
        "unique_dst_ips": n_dst,
        "at_risk_canonical_classes": at_risk,
    }


def main() -> None:
    mapping = load_canonical_map(MAPPING_PATH)
    results = []
    for name, path in DATASETS.items():
        results.append(analyse(name, path, mapping))

    print(f"\n\n{'='*76}")
    print("CONSOLIDATED SUMMARY")
    print(f"{'='*76}")
    for r in results:
        print(f"  {r['dataset']:32s}  src_IPs={r['unique_src_ips']:>8,d}  "
              f"dst_IPs={r['unique_dst_ips']:>8,d}  "
              f"rows={r['rows']:>12,d}  "
              f"at_risk={r['at_risk_canonical_classes']}")


if __name__ == "__main__":
    main()
