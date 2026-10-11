"""40_split_verification.py — Phase 4, block A: verify the grouped splits
recomputed on the features each model sees (research/phase4_protocol.md §2).

For every dataset and every Phase 4 feature set (nfv3_config.model_feature_sets:
base, noport, noid, and ts in NF-v3), the grouped split is recomputed with the
shared functions for the replicate seeds 1, 2 and 3 and checked:

  R1  regression: the full Phase 1 key (49 / 41 features) with seed 42
      reproduces the stored `split` column (0 mismatches)
  V1  no 64-bit row hash in more than one split (every seed)
  V2  exact, hash-free comparison (seed 1): a sample of distinct test vectors
      is matched against every train row on the feature columns (pandas merge,
      NaN equal to NaN), as audit check N5
  V3  realised train/val/test shares per canonical class (seed 1); classes
      with fewer than MIN_TEST test rows are flagged

The NF-v3 timestamps are read from data.parquet, whose row order equals that
of tabular.parquet (checked on the `Label` column).

Usage:
  python scripts/40_split_verification.py --dataset NF-ToN-IoT-v3
  python scripts/40_split_verification.py --report
Outputs: data/eval/split_verification/<dataset>.json and
data/eval/split_verification.txt
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from nfv3_config import (DATASETS, PROCESSED_ROOT, TIMESTAMP_COLS, WORK, feature_cols,
                         grouped_split_codes, is_v2, model_feature_sets, tabular_row_hash)

SEEDS = [1, 2, 3]
EXACT_SEED = 1
EXACT_SAMPLE = 20_000
SAMPLE_SEED = 2026
MIN_TEST = 100
BATCH = 1_000_000
CODES = {"train": 0, "val": 1, "test": 2}
OUT = WORK / "data/eval"


def batches(dataset, cols, ts):
    """Yield (offset, DataFrame) over tabular.parquet, with the timestamps
    appended for NF-v3."""
    pf = pq.ParquetFile(PROCESSED_ROOT / dataset / "tabular.parquet")
    off = 0
    for b in pf.iter_batches(batch_size=BATCH, columns=cols):
        df = b.to_pandas()
        if ts is not None:
            for c in TIMESTAMP_COLS:
                df[c] = ts[c][off:off + len(df)]
        yield off, df
        off += len(df)


def load_timestamps(dataset):
    data = pq.read_table(PROCESSED_ROOT / dataset / "data.parquet",
                         columns=TIMESTAMP_COLS + ["Label"]).to_pandas()
    lab = pq.read_table(PROCESSED_ROOT / dataset / "tabular.parquet",
                        columns=["Label"]).to_pandas()["Label"]
    if len(data) != len(lab) or not (data["Label"].to_numpy() == lab.to_numpy()).all():
        raise RuntimeError("data.parquet and tabular.parquet rows are not aligned")
    return {c: data[c].to_numpy() for c in TIMESTAMP_COLS}


def run(dataset):
    t0 = time.time()
    sets = model_feature_sets(dataset)
    full = feature_cols(dataset)
    ts = None if is_v2(dataset) else load_timestamps(dataset)
    tab_cols = sorted(set(full) | {c for s in sets.values() for c in s} - set(TIMESTAMP_COLS),
                      key=full.index)

    # Pass 1: hashes for every key, labels, regression against the stored split.
    hashes = {k: [] for k in sets}
    labels, mismatch = [], 0
    for _, df in batches(dataset, tab_cols + ["canonical_label", "split"], ts):
        stored = df["split"].astype(str).map(CODES).to_numpy(dtype=np.int8)
        mismatch += int((grouped_split_codes(tabular_row_hash(df, full), 42) != stored).sum())
        for k, cols in sets.items():
            hashes[k].append(tabular_row_hash(df, cols))
        labels.append(df["canonical_label"].astype(object).fillna("").astype(str).replace("", "(excluded)").to_numpy())
    hashes = {k: np.concatenate(v) for k, v in hashes.items()}
    labels = np.concatenate(labels)
    n = len(labels)
    print(f"{dataset}: {n:,} rows; R1 mismatches {mismatch}", flush=True)

    res = {"dataset": dataset, "rows": n, "generated_utc": datetime.now(timezone.utc).isoformat(),
           "R1_stored_split_mismatches": mismatch, "keys": {}}
    exact_codes = {}
    for k, cols in sets.items():
        h = hashes[k]
        r = {"n_features": len(cols), "features": cols,
             "distinct_vectors": int(len(np.unique(h))), "V1": {}}
        for seed in SEEDS:
            s = grouped_split_codes(h, seed)
            u = [np.unique(h[s == c]) for c in range(3)]
            cross = sum(len(np.intersect1d(u[a], u[b], assume_unique=True))
                        for a, b in ((0, 1), (0, 2), (1, 2)))
            r["V1"][str(seed)] = {"hashes_in_two_splits": int(cross),
                                  "shares": [round(float((s == c).mean()), 4) for c in range(3)]}
            if seed == EXACT_SEED:
                exact_codes[k] = s
                per = {}
                for lab in np.unique(labels):
                    m = labels == lab
                    cnt = np.bincount(s[m], minlength=3)
                    per[lab] = {"train": int(cnt[0]), "val": int(cnt[1]), "test": int(cnt[2]),
                                "test_share": round(float(cnt[2] / m.sum()), 4),
                                "flag": bool(cnt[2] < MIN_TEST)}
                r["V3_per_class_seed1"] = per
        res["keys"][k] = r
        print(f"  {k:7s} F{len(cols)}  distinct {r['distinct_vectors']:,}  V1 "
              + " ".join(f"s{s}:{v['hashes_in_two_splits']}" for s, v in r["V1"].items()), flush=True)
    del hashes

    # Pass 2 (V2): exact comparison of sampled distinct test vectors with every train row.
    rng = np.random.default_rng(SAMPLE_SEED)
    samples = {}
    for k, s in exact_codes.items():
        n_test = int((s == 2).sum())
        samples[k] = (rng.random(n) < min(1.0, EXACT_SAMPLE / max(n_test, 1))) & (s == 2)
    parts = {k: [] for k in sets}
    for off, df in batches(dataset, tab_cols, ts):
        for k, cols in sets.items():
            m = samples[k][off:off + len(df)]
            parts[k].append(df.loc[m, cols])
    sample_df = {}
    for k in sets:
        d = pd.concat(parts[k], ignore_index=True).drop_duplicates()
        d["_row"] = np.arange(len(d))
        sample_df[k] = d
    matched = {k: set() for k in sets}
    for off, df in batches(dataset, tab_cols, ts):
        for k, cols in sets.items():
            tr = df.loc[exact_codes[k][off:off + len(df)] == 0, cols].drop_duplicates()
            matched[k].update(sample_df[k].merge(tr, on=cols, how="inner")["_row"].tolist())
    for k in sets:
        res["keys"][k]["V2_exact_seed1"] = {"sampled_distinct_test_vectors": int(len(sample_df[k])),
                                            "found_in_train": len(matched[k])}
        print(f"  {k:7s} V2 {len(matched[k])} of {len(sample_df[k]):,} sampled test vectors in train",
              flush=True)

    res["elapsed_s"] = round(time.time() - t0, 1)
    d = OUT / "split_verification"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{dataset}.json").write_text(json.dumps(res, indent=2))
    print(f"{dataset}: done in {res['elapsed_s']}s", flush=True)


def report():
    L = ["Phase 4 block A: grouped splits recomputed on the model feature sets",
         f"Generated: {datetime.now(timezone.utc).isoformat()}",
         "Keys: base = all features but L4_SRC_PORT; noport = base without L4_DST_PORT;",
         "noid = base without ports, TTL, TCP window, DNS; ts = base + both timestamps (NF-v3).",
         f"R1: Phase 1 key, seed 42, against the stored split. V1: hashes in two splits, seeds {SEEDS}.",
         f"V2: exact match of sampled distinct test vectors against every train row (seed {EXACT_SEED}).",
         f"Classes with fewer than {MIN_TEST} test rows (seed {EXACT_SEED}) are listed.", ""]
    ok = True
    for f in sorted((OUT / "split_verification").glob("*.json")):
        r = json.loads(f.read_text())
        ok &= r["R1_stored_split_mismatches"] == 0
        L.append(f"== {r['dataset']}: {r['rows']:,} rows; R1 mismatches "
                 f"{r['R1_stored_split_mismatches']}; {r['elapsed_s']} s")
        L.append(f"  {'key':7s} {'feat':>4s} {'distinct vectors':>17s} {'V1 max':>6s} "
                 f"{'V2 found/sampled':>17s}  shares seed 1 (train/val/test)")
        for k, x in r["keys"].items():
            v1 = max(v["hashes_in_two_splits"] for v in x["V1"].values())
            v2 = x["V2_exact_seed1"]
            ok &= v1 == 0 and v2["found_in_train"] == 0
            sh = x["V1"][str(EXACT_SEED)]["shares"]
            L.append(f"  {k:7s} {x['n_features']:4d} {x['distinct_vectors']:17,d} {v1:6d} "
                     f"{v2['found_in_train']:>7d}/{v2['sampled_distinct_test_vectors']:<9,d}  "
                     + "/".join(f"{s:.3f}" for s in sh))
            flags = [f"{c} ({p['test']} test rows)" for c, p in x["V3_per_class_seed1"].items() if p["flag"]]
            dev = [f"{c} {p['test_share']:.3f}" for c, p in x["V3_per_class_seed1"].items()
                   if abs(p["test_share"] - 0.15) > 0.05]
            if flags:
                L.append(f"          few test rows: {', '.join(flags)}")
            if dev:
                L.append(f"          test share off 0.15 by > 0.05: {', '.join(dev)}")
        L.append("")
    L.append(f"ALL CHECKS PASS: {ok}")
    text = "\n".join(L) + "\n"
    (OUT / "split_verification.txt").write_text(text)
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.dataset:
        run(a.dataset)
    if a.report:
        report()


if __name__ == "__main__":
    main()
