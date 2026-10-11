"""P1.4 temporal exploration.

For each of the three NF-v3 datasets, checks whether a temporal 70/15/15 split
preserves every canonical class in every bucket.

Loads only Attack and FLOW_START_MILLISECONDS; applies the canonical class
mapping; then for each canonical class computes:

  - min and max timestamp
  - temporal span in hours
  - concentration ratio (class span / dataset span)
  - simulation of a chronological 70/15/15 split: flows per class per bucket
  - flag when a canonical class has fewer than 1000 flows in any bucket

Runs sequentially and prints a consolidated report.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

WORK = Path(__file__).resolve().parents[2]  # repository root (was an absolute path)
MAPPING_PATH = WORK / "data/schemas/class_mapping.csv"

DATASETS = {
    "NF-ToN-IoT-v3": WORK / "02934b58528a226b_NFV3DATA-A11964_A11964/data/NF-ToN-IoT-v3.csv",
    "NF-BoT-IoT-v3": WORK / "d509c9db7490cf92_NFV3DATA-A11964_A11964/data/NF-BoT-IoT-v3.csv",
    "NF-CSE-CIC-IDS2018-v3": WORK / "f78acbaa2afe1595_NFV3DATA-A11964_A11964/data/NF-CICIDS2018-v3.csv",
}

COLS = ["Attack", "FLOW_START_MILLISECONDS"]
MIN_TEST = 1000


def load_canonical_map(mapping_path: Path) -> dict:
    m = pd.read_csv(mapping_path)
    m["canonical_label"] = m["canonical_label"].fillna("")
    return {
        (row.dataset, row.native_label): row.canonical_label
        for row in m.itertuples()
    }


def analyse(dataset_name: str, path: Path, mapping: dict) -> dict:
    t0 = time.time()
    print(f"\n{'='*80}")
    print(f"DATASET: {dataset_name}")
    print(f"{'='*80}", flush=True)
    df = pd.read_csv(path, usecols=COLS, low_memory=False)
    print(f"  rows={len(df):,}  load_time={time.time()-t0:.1f}s")

    df["canonical"] = [
        mapping.get((dataset_name, str(a)), "") for a in df["Attack"].astype(str)
    ]

    ts_min = df["FLOW_START_MILLISECONDS"].min()
    ts_max = df["FLOW_START_MILLISECONDS"].max()
    span_h = (ts_max - ts_min) / 1000 / 3600
    print(f"  full range: {pd.to_datetime(ts_min, unit='ms', utc=True)} to "
          f"{pd.to_datetime(ts_max, unit='ms', utc=True)}  span={span_h:.2f}h")

    print(f"\nTemporal footprint per CANONICAL class:")
    print(f"  {'class':22s}  {'flows':>15s}  {'first':>26s}  {'last':>26s}  {'span_h':>10s}  {'span%':>7s}")
    for cls in sorted(df["canonical"].unique()):
        sub = df[df["canonical"] == cls]["FLOW_START_MILLISECONDS"]
        lo, hi = sub.min(), sub.max()
        s_h = (hi - lo) / 1000 / 3600
        pct = (s_h / span_h * 100) if span_h > 0 else 0.0
        label = cls if cls else "(excluded)"
        print(
            f"  {label:22s}  {len(sub):>15,d}  "
            f"{str(pd.to_datetime(lo, unit='ms', utc=True))[:26]:>26s}  "
            f"{str(pd.to_datetime(hi, unit='ms', utc=True))[:26]:>26s}  "
            f"{s_h:>10.2f}  {pct:>6.1f}%"
        )

    # Temporal split simulation (row-count based, chronological)
    print(f"\nChronological 70/15/15 split simulation (by row count on sorted timestamp):")
    n = len(df)
    sorted_idx = df["FLOW_START_MILLISECONDS"].sort_values(kind="stable").index
    train_end = int(0.70 * n)
    val_end = int(0.85 * n)
    split = pd.Series("", index=df.index)
    split.iloc[sorted_idx[:train_end]] = "train"
    split.iloc[sorted_idx[train_end:val_end]] = "val"
    split.iloc[sorted_idx[val_end:]] = "test"
    df["split"] = split.values

    # Cutoff timestamps
    train_cutoff_ts = df.loc[sorted_idx[train_end - 1], "FLOW_START_MILLISECONDS"]
    val_cutoff_ts = df.loc[sorted_idx[val_end - 1], "FLOW_START_MILLISECONDS"]
    print(f"  train cutoff: {pd.to_datetime(train_cutoff_ts, unit='ms', utc=True)}")
    print(f"  val   cutoff: {pd.to_datetime(val_cutoff_ts, unit='ms', utc=True)}")

    ct = pd.crosstab(df["canonical"], df["split"])
    for col in ("train", "val", "test"):
        if col not in ct.columns:
            ct[col] = 0
    ct = ct[["train", "val", "test"]]
    ct["total"] = ct.sum(axis=1)
    ct = ct.sort_values("total", ascending=False)

    print(f"\n  {'class':22s}  {'train':>15s}  {'val':>15s}  {'test':>15s}  {'total':>15s}  status")
    at_risk = []
    for cls, row in ct.iterrows():
        label = cls if cls else "(excluded)"
        tr, va, te, tot = int(row["train"]), int(row["val"]), int(row["test"]), int(row["total"])
        status_parts = []
        if cls != "":
            if tr < MIN_TEST:
                status_parts.append(f"train<{MIN_TEST}")
            if va < MIN_TEST:
                status_parts.append(f"val<{MIN_TEST}")
            if te < MIN_TEST:
                status_parts.append(f"test<{MIN_TEST}")
        status = " ".join(status_parts) if status_parts else "OK"
        if status_parts:
            at_risk.append((cls, tr, va, te))
        print(f"  {label:22s}  {tr:>15,d}  {va:>15,d}  {te:>15,d}  {tot:>15,d}  {status}")

    if at_risk:
        print(f"\n  AT-RISK canonical classes under chronological split:")
        for cls, tr, va, te in at_risk:
            print(f"    {cls}: train={tr:,} val={va:,} test={te:,}")
    else:
        print(f"\n  All canonical classes have >= {MIN_TEST} flows in each bucket. OK.")

    # Also report DIVERSITY: how many distinct hours each class covers
    print(f"\n  Diversity check (distinct hours per canonical class):")
    df["hour_bucket"] = (df["FLOW_START_MILLISECONDS"] // (3600 * 1000)).astype("int64")
    for cls in sorted(df["canonical"].unique()):
        if cls == "":
            continue
        sub = df[df["canonical"] == cls]["hour_bucket"]
        n_hours = sub.nunique()
        print(f"    {cls:22s}  distinct_hours={n_hours:>6,d}")

    print(f"\n  Total time for {dataset_name}: {time.time()-t0:.1f}s")
    return {"dataset": dataset_name, "at_risk": at_risk}


def main() -> None:
    mapping = load_canonical_map(MAPPING_PATH)
    results = []
    for name, path in DATASETS.items():
        results.append(analyse(name, path, mapping))

    print(f"\n\n{'='*80}")
    print("CONSOLIDATED SUMMARY — temporal 70/15/15")
    print(f"{'='*80}")
    for r in results:
        if r["at_risk"]:
            classes = [c[0] for c in r["at_risk"]]
            print(f"  {r['dataset']:32s}  AT-RISK: {classes}")
        else:
            print(f"  {r['dataset']:32s}  OK")


if __name__ == "__main__":
    main()
