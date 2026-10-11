"""17_reproducibility_audit.py — independent audit of the loader and
tabular-builder artefacts (NF-v3 and NF-v2).

The audit re-reads the raw CSV and every produced parquet and recomputes each
property from scratch; it shares only constants (paths, dtypes, feature list)
with the pipeline. 10 checks per dataset:

Integrity
  I1  raw CSV SHA-1 matches the loader manifest
  I2  data.parquet SHA-1 matches the loader manifest
  I3  tabular.parquet SHA-1 matches the tabular manifest
  I5  value fidelity raw CSV -> data.parquet: per numeric column, NaN / +Inf /
      -Inf counts equal, and finite min, max and sum equal (exact for integer
      columns; relative tolerance 1e-6 for float32 columns)

Consistency
  C1  row counts equal in data.parquet and tabular.parquet
  C2  metadata columns byte-identical in both files (batch-boundary invariant
      SHA-1)
  C3  canonical x split crosstab identical in both files

Content
  N1  tabular.parquet holds no Inf
  N2  tabular values equal the normalised data.parquet values: per feature,
      NaN count = NaN + Inf in data, finite min / max / sum equal
  N5  no feature vector appears in more than one split: (a) full pass over
      64-bit row hashes of tabular.parquet; (b) exact, hash-free comparison of
      a sample of test rows against every train row

Check IDs are stable; I4, N3, N4 and N6 covered the textual representation,
retired on 2026-10-01.

Outputs: data/processed/<dataset>/audit_report.txt and audit_manifest.json.
Exit code 1 if any check fails on any audited dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from nfv3_config import (
    DATASETS,
    META_COLS,
    NFV3_DTYPES,
    PROCESSED_ROOT,
    RAW_CSV,
    WORK,
    feature_cols,
    raw_dtypes,
)

BATCH = 500_000
RAW_CHUNK = 2_000_000
FLOAT_RTOL = 1e-6
N5_SAMPLE = 20_000
AUDIT_SEED = 2026

NUMERIC_COLS = [c for c, t in NFV3_DTYPES.items() if t.startswith(("int", "float"))]
INT_COLS = {c for c in NUMERIC_COLS if NFV3_DTYPES[c].startswith("int")}


def make_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"17_audit::{log_path.parent.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s")
    for h in (logging.FileHandler(log_path, mode="w"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def sha1_file(path: Path, block: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Column statistics
# --------------------------------------------------------------------------- #


def new_stats(cols) -> dict:
    return {c: {"nan": 0, "posinf": 0, "neginf": 0, "min": None, "max": None,
                "sum": 0} for c in cols}


def update_stats(acc: dict, df: pd.DataFrame, cols) -> None:
    for c in cols:
        arr = df[c].to_numpy()
        s = acc[c]
        if np.issubdtype(arr.dtype, np.integer):
            if arr.size:
                lo, hi = int(arr.min()), int(arr.max())
                s["min"] = lo if s["min"] is None else min(s["min"], lo)
                s["max"] = hi if s["max"] is None else max(s["max"], hi)
                s["sum"] += int(arr.sum(dtype=np.int64))
        else:
            a = arr.astype(np.float64)
            s["nan"] += int(np.isnan(a).sum())
            s["posinf"] += int(np.isposinf(a).sum())
            s["neginf"] += int(np.isneginf(a).sum())
            fin = a[np.isfinite(a)]
            if fin.size:
                lo, hi = float(fin.min()), float(fin.max())
                s["min"] = lo if s["min"] is None else min(s["min"], lo)
                s["max"] = hi if s["max"] is None else max(s["max"], hi)
                s["sum"] += float(fin.sum())


def compare_stats(a: dict, b: dict, cols, *, merge_nonfinite: bool,
                  float_rtol: float) -> list[str]:
    """Return a list of mismatch descriptions (empty when equal).

    merge_nonfinite: compare a's NaN + Inf against b's NaN (b has no Inf).
    """
    bad = []
    for c in cols:
        x, y = a[c], b[c]
        if merge_nonfinite:
            if x["nan"] + x["posinf"] + x["neginf"] != y["nan"]:
                bad.append(f"{c}: non-finite {x['nan']+x['posinf']+x['neginf']} vs NaN {y['nan']}")
        else:
            for k in ("nan", "posinf", "neginf"):
                if x[k] != y[k]:
                    bad.append(f"{c}: {k} {x[k]} vs {y[k]}")
        for k in ("min", "max", "sum"):
            u, v = x[k], y[k]
            if u is None or v is None:
                if u != v:
                    bad.append(f"{c}: {k} {u} vs {v}")
                continue
            if c in INT_COLS or (isinstance(u, int) and isinstance(v, int)):
                if int(u) != int(v):
                    bad.append(f"{c}: {k} {u} vs {v}")
            elif not np.isclose(float(u), float(v), rtol=float_rtol, atol=0.0):
                bad.append(f"{c}: {k} {u!r} vs {v!r}")
    return bad


# --------------------------------------------------------------------------- #
# Passes over each artefact
# --------------------------------------------------------------------------- #


def raw_pass(path: Path, num_cols: list[str]) -> tuple[dict, int]:
    dtypes = {c: ("int64" if c in INT_COLS else "float64") for c in num_cols}
    acc = new_stats(num_cols)
    n = 0
    for chunk in pd.read_csv(path, usecols=num_cols, dtype=dtypes, chunksize=RAW_CHUNK):
        update_stats(acc, chunk, num_cols)
        n += len(chunk)
    return acc, n


def meta_update(per_col: dict, crosstab: dict, df: pd.DataFrame) -> None:
    for c in META_COLS:
        per_col[c].update(("|".join(df[c].astype(str).tolist()) + "|").encode())
    ct = pd.crosstab(df["canonical_label"].astype(str), df["split"].astype(str))
    for cls in ct.index:
        row = crosstab.setdefault(cls, {"train": 0, "val": 0, "test": 0})
        for sp in ct.columns:
            row[sp] += int(ct.at[cls, sp])


def meta_digest(per_col: dict) -> str:
    final = hashlib.sha1()
    for c in META_COLS:
        final.update(f"{c}:{per_col[c].hexdigest()}\n".encode())
    return final.hexdigest()


def data_pass(path: Path, num_cols: list[str]) -> dict:
    pf = pq.ParquetFile(path)
    stats = new_stats(num_cols)
    per_col = {c: hashlib.sha1() for c in META_COLS}
    crosstab: dict = {}
    for b in pf.iter_batches(batch_size=BATCH, columns=num_cols + META_COLS):
        df = b.to_pandas()
        update_stats(stats, df, num_cols)
        meta_update(per_col, crosstab, df)
    return {"rows": pf.metadata.num_rows, "stats": stats,
            "meta_sha1": meta_digest(per_col), "crosstab": crosstab}


def tabular_pass(path: Path, feat: list[str]) -> dict:
    pf = pq.ParquetFile(path)
    stats = new_stats(feat)
    per_col = {c: hashlib.sha1() for c in META_COLS}
    crosstab: dict = {}
    hashes, splits = [], []
    split_code = {"train": 0, "val": 1, "test": 2}
    for b in pf.iter_batches(batch_size=BATCH, columns=feat + META_COLS):
        df = b.to_pandas()
        update_stats(stats, df, feat)
        meta_update(per_col, crosstab, df)
        hashes.append(pd.util.hash_pandas_object(df[feat], index=False).to_numpy())
        splits.append(df["split"].astype(str).map(split_code).to_numpy(dtype=np.int8))
    h, s = np.concatenate(hashes), np.concatenate(splits)
    order = np.lexsort((s, h))
    hs, ss = h[order], s[order]
    cross = (hs[1:] == hs[:-1]) & (ss[1:] != ss[:-1])
    n_hash_multi_split = int(np.unique(hs[1:][cross]).size)
    return {"rows": pf.metadata.num_rows, "stats": stats,
            "meta_sha1": meta_digest(per_col), "crosstab": crosstab,
            "vectors_in_more_than_one_split": n_hash_multi_split,
            "unique_vectors": int(np.unique(h).size)}


def exact_test_train_overlap(path: Path, feat: list[str], sample_size: int, seed: int) -> dict:
    """Hash-free check: sample test rows, then compare them exactly (all
    feature columns, NaN equal to NaN) against every train row."""
    pf = pq.ParquetFile(path)
    n_test = 0
    for b in pf.iter_batches(batch_size=BATCH, columns=["split"]):
        n_test += int((b.column("split").to_pandas().astype(str) == "test").sum())
    p = min(1.0, sample_size / max(n_test, 1))
    rng = np.random.default_rng(seed)
    parts = []
    for b in pf.iter_batches(batch_size=BATCH, columns=feat + ["split"]):
        df = b.to_pandas()
        test = df[df["split"].astype(str) == "test"]
        keep = rng.random(len(test)) < p
        parts.append(test.loc[keep, feat])
    sample = pd.concat(parts, ignore_index=True).drop_duplicates()
    sample["_row"] = np.arange(len(sample))
    matched: set = set()
    for b in pf.iter_batches(batch_size=BATCH, columns=feat + ["split"]):
        df = b.to_pandas()
        train = df.loc[df["split"].astype(str) == "train", feat].drop_duplicates()
        m = sample.merge(train, on=feat, how="inner")
        matched.update(m["_row"].tolist())
    return {"test_rows_total": n_test, "sampled_distinct_test_vectors": int(len(sample)),
            "sampled_vectors_found_in_train": len(matched)}


# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #


def audit(dataset_name: str) -> dict:
    d = PROCESSED_ROOT / dataset_name
    raw_csv = RAW_CSV[dataset_name]
    feat = feature_cols(dataset_name)
    num_cols = [c for c, t in raw_dtypes(dataset_name).items()
                if t.startswith(("int", "float"))]
    data_p, tab_p = d / "data.parquet", d / "tabular.parquet"
    loader_m = json.loads((d / "manifest.json").read_text())
    tab_m = json.loads((d / "tabular_manifest.json").read_text())
    logger = make_logger(d / "audit.log")
    logger.info("=" * 72)
    logger.info("Audit for %s", dataset_name)
    logger.info("=" * 72)
    results: list[dict] = []

    def add(cid: str, name: str, ok: bool, detail: str = "") -> None:
        results.append({"check": cid, "name": name, "ok": bool(ok), "detail": detail})
        logger.info("  %s [%s] %s", "OK  " if ok else "FAIL", cid, name)
        if detail and not ok:
            logger.info("       %s", detail[:2000])

    t0 = time.time()
    sha = {"raw_csv": sha1_file(raw_csv), "data.parquet": sha1_file(data_p),
           "tabular.parquet": sha1_file(tab_p)}
    add("I1", "raw CSV SHA-1 matches loader manifest",
        sha["raw_csv"] == loader_m["input"]["sha1"])
    add("I2", "data.parquet SHA-1 matches loader manifest",
        sha["data.parquet"] == loader_m["outputs"]["parquet"]["sha1"])
    add("I3", "tabular.parquet SHA-1 matches tabular manifest",
        sha["tabular.parquet"] == tab_m["outputs"]["tabular"]["sha1"])

    logger.info("  reading raw CSV at full width ...")
    raw_stats, raw_rows = raw_pass(raw_csv, num_cols)
    logger.info("  reading data.parquet ...")
    dat = data_pass(data_p, num_cols)
    i5_bad = compare_stats(raw_stats, dat["stats"], num_cols,
                           merge_nonfinite=False, float_rtol=FLOAT_RTOL)
    if raw_rows != dat["rows"]:
        i5_bad.insert(0, f"row count raw {raw_rows} vs data {dat['rows']}")
    add("I5", "values of data.parquet equal the raw CSV (all numeric columns)",
        not i5_bad, "; ".join(i5_bad))

    logger.info("  reading tabular.parquet ...")
    tab = tabular_pass(tab_p, feat)
    logger.info("  exact test/train comparison on a sample ...")
    exact = exact_test_train_overlap(tab_p, feat, N5_SAMPLE, AUDIT_SEED)
    data_nonfinite = {c: dat["stats"][c]["nan"] + dat["stats"][c]["posinf"]
                      + dat["stats"][c]["neginf"] for c in feat}

    add("C1", "row counts equal in data and tabular",
        dat["rows"] == tab["rows"], f"data={dat['rows']} tabular={tab['rows']}")
    add("C2", "metadata SHA-1 identical in data and tabular",
        dat["meta_sha1"] == tab["meta_sha1"],
        f"data={dat['meta_sha1']} tabular={tab['meta_sha1']}")
    add("C3", "canonical x split crosstab identical in data and tabular",
        dat["crosstab"] == tab["crosstab"])

    tab_inf = sum(tab["stats"][c]["posinf"] + tab["stats"][c]["neginf"] for c in feat)
    add("N1", "tabular.parquet holds no Inf", tab_inf == 0, f"Inf found: {tab_inf}")
    n2_bad = compare_stats(dat["stats"], tab["stats"], feat,
                           merge_nonfinite=True, float_rtol=1e-9)
    add("N2", "tabular values equal normalised data.parquet values", not n2_bad,
        "; ".join(n2_bad))

    n5_ok = (tab["vectors_in_more_than_one_split"] == 0
             and exact["sampled_vectors_found_in_train"] == 0)
    add("N5", "no feature vector appears in more than one split", n5_ok,
        f"hashes in >1 split: {tab['vectors_in_more_than_one_split']}; "
        f"exact sample: {exact}")

    elapsed = time.time() - t0
    passed = sum(r["ok"] for r in results)
    failed = len(results) - passed
    logger.info("Audit complete: %d passed, %d failed (%.1fs)", passed, failed, elapsed)

    lines = [
        f"Audit report for {dataset_name}",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Elapsed:   {elapsed:.1f}s",
        f"Result:    {passed} passed, {failed} failed (out of {len(results)})",
        "",
        "Checks:",
    ]
    for r in results:
        lines.append(f"  [{r['check']}] {'OK  ' if r['ok'] else 'FAIL'}  {r['name']}")
        if r["detail"] and not r["ok"]:
            lines.append(f"        {r['detail'][:2000]}")
    lines += [
        "",
        "Measured values:",
        f"  rows: raw {raw_rows:,} | data {dat['rows']:,} | tabular {tab['rows']:,}",
        f"  metadata SHA-1 (data and tabular): {dat['meta_sha1']}",
        f"  unique feature vectors in tabular: {tab['unique_vectors']:,}",
        f"  feature vectors in more than one split: {tab['vectors_in_more_than_one_split']}",
        f"  exact test/train sample: {exact['sampled_distinct_test_vectors']:,} distinct "
        f"test vectors sampled of {exact['test_rows_total']:,} test rows; "
        f"found in train: {exact['sampled_vectors_found_in_train']}",
        f"  non-finite values per feature in data.parquet: "
        f"{ {c: n for c, n in data_nonfinite.items() if n} or 'none'}",
    ]
    (d / "audit_report.txt").write_text("\n".join(lines) + "\n")

    manifest = {
        "dataset": dataset_name,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(WORK)),
        "results": results,
        "measurements": {
            "file_sha1": sha,
            "row_counts": {"raw_csv": raw_rows, "data.parquet": dat["rows"],
                           "tabular.parquet": tab["rows"]},
            "metadata_sha1": {"data.parquet": dat["meta_sha1"],
                              "tabular.parquet": tab["meta_sha1"]},
            "crosstab": dat["crosstab"],
            "raw_column_stats": raw_stats,
            "nonfinite_per_feature_in_data": {c: n for c, n in data_nonfinite.items() if n},
            "unique_feature_vectors": tab["unique_vectors"],
            "vectors_in_more_than_one_split": tab["vectors_in_more_than_one_split"],
            "exact_test_train_sample": exact,
        },
        "elapsed_seconds": round(elapsed, 1),
        "passed": passed,
        "failed": failed,
        "total_checks": len(results),
    }
    (d / "audit_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if not args.dataset and not args.all:
        ap.error("either --dataset NAME or --all is required")
    failed = False
    for name in (DATASETS if args.all else [args.dataset]):
        failed |= audit(name)["failed"] > 0
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
