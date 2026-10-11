"""48_split_hashes.py — P4.6: SHA-1 of every data partition used in the study,
so that anyone with the official CSV files can check that the shared
functions of nfv3_config.py reproduce the same partitions (the CSV files are
not redistributed; research/license_verification.md).

Each partition is an int8 array with one code per row of the dataset, rows in
the order of the official CSV (the loader preserves it): 0 train, 1 val,
2 test. Hashed: SHA-1 of the array bytes. Partitions:

  phase1        stored grouped split (all features, seed 42), column `split`
  grouped/<key> grouped_split_codes on the Phase 4 keys base, noport, noid
                (and ts in NF-v3), seeds 1-5 (P4.2-P4.5)
  random        random_split_codes, seeds 1-5 (condition R0)
  replication   NF-v2 only, seeds 1-5 (P4.4, 46_replication.py): the row ids of
                the stratified 1,000,000-flow sample (int64), the L0 random
                70/30 codes and the L1 grouped 70/30 codes on that sample

Row hashes are read from the cache of 44_cross_dataset.py, which holds the
values computed by the shared functions (checked identical to P4.2).

Usage: python scripts/48_split_hashes.py
Outputs: data/schemas/split_hashes.csv and data/schemas/split_hashes.txt
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
from datetime import datetime, timezone

import numpy as np
import pyarrow.parquet as pq

from nfv3_config import (DATASETS, PROCESSED_ROOT, WORK, grouped_split_codes, is_v2,
                         random_split_codes)

SEEDS = [1, 2, 3, 4, 5]
OUT = WORK / "data/schemas"
CODES = {"train": 0, "val": 1, "test": 2}

_spec = importlib.util.spec_from_file_location("rep", WORK / "scripts/46_replication.py")
REP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(REP)
C = REP.C


def sha1(arr):
    return hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()


def counts(codes):
    c = np.bincount(codes.astype(np.int64), minlength=3)
    return int(c[0]), int(c[1]), int(c[2])


def main():
    rows = []

    def add(ds, part, key, seed, arr, dtype):
        tr, va, te = counts(arr) if dtype == "int8 codes" else ("", "", "")
        rows.append({"dataset": ds, "partition": part, "key": key, "seed": seed, "dtype": dtype,
                     "length": int(len(arr)), "train": tr, "val": va, "test": te, "sha1": sha1(arr)})

    for ds in DATASETS:
        stored = pq.read_table(PROCESSED_ROOT / ds / "tabular.parquet", columns=["split"]).column(0)
        stored = stored.to_pandas().astype(str).map(CODES).to_numpy().astype(np.int8)
        add(ds, "phase1", "all features", 42, stored, "int8 codes")
        d = C.open_cache(ds)
        n = len(d["label"])
        for key in d["sets"]:
            for seed in SEEDS:
                add(ds, "grouped", key, seed, grouped_split_codes(d["hashes"][key], seed), "int8 codes")
        for seed in SEEDS:
            add(ds, "random", "-", seed, random_split_codes(n, seed), "int8 codes")
        if is_v2(ds):
            for seed in SEEDS:
                sample = REP.stratified_sample(d, seed)
                rng = np.random.default_rng(seed)
                l0 = np.where(rng.random(len(sample)) < 0.70, 0, 2).astype(np.int8)
                l1 = np.where(grouped_split_codes(d["hashes"]["noport"][sample], seed) == 0, 0, 2).astype(np.int8)
                add(ds, "replication sample row ids", "-", seed, sample.astype(np.int64), "int64 row ids")
                add(ds, "replication L0 random 70/30", "-", seed, l0, "int8 codes")
                add(ds, "replication L1 grouped 70/30", "noport", seed, l1, "int8 codes")
        print(f"{ds}: {sum(r['dataset'] == ds for r in rows)} partitions", flush=True)

    with (OUT / "split_hashes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    L = ["Split hashes (P4.6): SHA-1 of every partition used in the study",
         f"Generated: {datetime.now(timezone.utc).isoformat()}  Producer: scripts/48_split_hashes.py",
         "Codes 0 train, 1 val, 2 test, one per row in official CSV order (replication: 0 train, 2 test, one",
         "per sampled row). Check: recompute with nfv3_config.py and compare with data/schemas/split_hashes.csv.", ""]
    for ds in DATASETS:
        r = [x for x in rows if x["dataset"] == ds]
        L.append(f"{ds}: {len(r)} partitions; phase1 {r[0]['sha1']} (train/val/test {r[0]['train']:,}/"
                 f"{r[0]['val']:,}/{r[0]['test']:,})")
    (OUT / "split_hashes.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
