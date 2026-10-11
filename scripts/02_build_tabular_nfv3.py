"""02_build_tabular_nfv3.py — tabular representation of a loaded NF-v3 or
NF-v2 parquet.

For each dataset the script reads `data/processed/<dataset>/data.parquet`
(produced by `01_load_nfv3.py`) in batches and writes:

  - `tabular.parquet` — the features of `nfv3_config.feature_cols` (49 in
    NF-v3, 41 in NF-v2),
    normalised with `nfv3_config.normalise_features` (non-finite -> NaN,
    -0.0 -> 0.0; XGBoost handles NaN natively but not Inf), plus the
    metadata columns `Label`, `native_label`, `canonical_label`, `split`.
    These are exactly the values the split key was computed on.
  - `tabular_qc.txt` and `tabular_manifest.json`.

Usage:
  python scripts/02_build_tabular_nfv3.py --dataset NF-BoT-IoT-v3
  python scripts/02_build_tabular_nfv3.py --all
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
import pyarrow as pa
import pyarrow.parquet as pq

from nfv3_config import (
    DATASETS,
    META_COLS,
    PROCESSED_ROOT,
    WORK,
    feature_cols,
    normalise_features,
)

BATCH_SIZE = 500_000


def make_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"02_build_tabular::{log_path.parent.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def sha1_file(path: Path, block: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def meta_frame(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Label": df["Label"].to_numpy(dtype=np.int8),
        "native_label": df["native_label"].astype("string").to_numpy(),
        "canonical_label": df["canonical_label"].astype("string").to_numpy(),
        "split": df["split"].astype("string").to_numpy(),
    })


def process_dataset(dataset_name: str) -> None:
    d = PROCESSED_ROOT / dataset_name
    in_path = d / "data.parquet"
    tab_path = d / "tabular.parquet"
    log_path, qc_path = d / "tabular.log", d / "tabular_qc.txt"
    manifest_path = d / "tabular_manifest.json"
    logger = make_logger(log_path)
    logger.info("=" * 72)
    logger.info("Building tabular representation of %s", dataset_name)
    logger.info("=" * 72)
    if not in_path.exists():
        raise FileNotFoundError(f"{in_path} not found; run 01_load_nfv3.py first")

    t_start = time.time()
    in_sha1 = sha1_file(in_path)
    pf = pq.ParquetFile(in_path)
    total = pf.metadata.num_rows
    logger.info("input SHA-1 %s, %s rows", in_sha1, f"{total:,}")

    cols = feature_cols(dataset_name)
    inf_to_nan: dict = {}
    nan_after: dict = {}
    tab_writer = None
    processed = 0

    for batch in pf.iter_batches(batch_size=BATCH_SIZE,
                                 columns=cols + META_COLS):
        df = batch.to_pandas()
        for col in cols:
            if pd.api.types.is_float_dtype(df[col]):
                n_inf = int(np.isinf(df[col].to_numpy()).sum())
                if n_inf:
                    inf_to_nan[col] = inf_to_nan.get(col, 0) + n_inf

        feats = normalise_features(df, cols)
        for col, n in feats.isna().sum().items():
            if n:
                nan_after[col] = nan_after.get(col, 0) + int(n)
        tab = pd.concat([feats, meta_frame(df)], axis=1)
        tab_table = pa.Table.from_pandas(tab, preserve_index=False)
        if tab_writer is None:
            tab_writer = pq.ParquetWriter(tab_path, tab_table.schema, compression="snappy")
        tab_writer.write_table(tab_table)

        processed += len(df)
        logger.info("  %s / %s rows", f"{processed:,}", f"{total:,}")

    tab_writer.close()
    tab_sha1 = sha1_file(tab_path)

    lines = [
        f"Tabular QC report for {dataset_name}",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Input:  {in_path.relative_to(WORK)}  SHA-1 {in_sha1}",
        f"Rows processed: {processed:,}",
        f"Features: {len(cols)} (identifiers, absolute timestamps and labels excluded)",
        "",
        "Inf converted to NaN per column:",
    ]
    lines += [f"  {c:32s} {n:>12,}" for c, n in sorted(inf_to_nan.items())] or ["  none"]
    lines += ["", "NaN per column in tabular.parquet (source NaN + converted Inf):"]
    lines += [f"  {c:32s} {n:>12,}" for c, n in sorted(nan_after.items())] or ["  none"]
    lines += [
        "",
        f"tabular.parquet  {tab_sha1}  ({tab_path.stat().st_size/1024/1024:.1f} MB)",
    ]
    qc_path.write_text("\n".join(lines) + "\n")

    manifest = {
        "dataset": dataset_name,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(WORK)),
        "input": {"path": str(in_path.relative_to(WORK)), "sha1": in_sha1},
        "config": {
            "feature_cols": cols,
            "meta_cols": META_COLS,
            "normalisation": "float columns: -0.0 -> 0.0, non-finite -> NaN; "
                             "integer columns unchanged",
            "batch_size": BATCH_SIZE,
        },
        "outputs": {
            "tabular": {"path": str(tab_path.relative_to(WORK)), "sha1": tab_sha1,
                        "size_bytes": tab_path.stat().st_size},
            "qc_report": str(qc_path.relative_to(WORK)),
            "log": str(log_path.relative_to(WORK)),
        },
        "stats": {
            "rows_processed": processed,
            "inf_to_nan_conversions": inf_to_nan,
            "nan_per_feature": nan_after,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("tabular.parquet SHA-1 %s", tab_sha1)
    logger.info("done in %.1fs", time.time() - t_start)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if not args.dataset and not args.all:
        ap.error("either --dataset NAME or --all is required")
    for name in (DATASETS if args.all else [args.dataset]):
        process_dataset(name)


if __name__ == "__main__":
    main()
