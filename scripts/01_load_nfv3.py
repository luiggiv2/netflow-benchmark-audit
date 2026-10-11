"""01_load_nfv3.py — load a raw NF-v3 or NF-v2 CSV into a processed parquet.

For each dataset, the script:

  1. Reads the raw CSV from its BagIt bag with the dtypes of
     `nfv3_config.raw_dtypes` (NF-v3 dtypes; NF-v2 uses the same dtypes for
     its columns). pandas 3.x wraps silently on integer overflow,
     so those dtypes were checked against the raw maxima and the audit
     re-checks value fidelity against the raw CSV (check I5).
  2. Applies the CSE-CIC-IDS2018 typo correction `Infilteration -> Infiltration`
     on the `Attack` column (see `research/infiltration_typo_decision.md`).
  3. Left-joins `data/schemas/class_mapping.csv` (NF-v3) or
     `class_mapping_v2.csv` (NF-v2) on `(dataset, native_label)` to attach
     `canonical_label`. Labels missing from the mapping abort the run.
  4. Assigns `split` (train / val / test, 70 / 15 / 15) by a seeded hash of
     the normalised feature vector (49 features in NF-v3, 41 in NF-v2;
     `research/split_protocol.md`), so
     identical feature vectors always share a split. Raw values are stored
     unchanged in the parquet; the normalisation is used only for the key.
  5. Writes `data/processed/<dataset>/data.parquet` (snappy), a QC report and
     a JSON manifest.

Usage:
  python scripts/01_load_nfv3.py --dataset NF-ToN-IoT-v3
  python scripts/01_load_nfv3.py --all
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

from nfv3_config import (
    DATASETS,
    PROCESSED_ROOT,
    RAW_CSV,
    SPLIT_SEED,
    SPLIT_SEED_CONST,
    TRAIN_HIGH,
    VAL_HIGH,
    WORK,
    feature_cols,
    feature_row_hash,
    mapping_path,
    raw_dtypes,
    split_from_hash,
)

MIN_ROWS_WARNING = 1000
# A canonical class whose realised test share falls outside this band is
# flagged: grouping by feature vector can move heavily duplicated classes
# away from 15 %.
TEST_SHARE_BAND = (0.10, 0.20)

TYPO_FROM = "Infilteration"
TYPO_TO = "Infiltration"


def make_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("01_load_nfv3")
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


def load_class_mapping(path: Path) -> pd.DataFrame:
    m = pd.read_csv(path)
    m["canonical_label"] = m["canonical_label"].fillna("")
    return m


def apply_typo_correction(df: pd.DataFrame, logger: logging.Logger) -> int:
    n_before = int((df["Attack"] == TYPO_FROM).sum())
    if n_before > 0:
        df["Attack"] = df["Attack"].replace(TYPO_FROM, TYPO_TO)
        logger.info("applied typo correction: %s rows '%s' -> '%s'",
                    f"{n_before:,}", TYPO_FROM, TYPO_TO)
    else:
        logger.info("typo correction not needed (no '%s' rows)", TYPO_FROM)
    return n_before


def join_canonical(df: pd.DataFrame, mapping: pd.DataFrame, dataset_name: str,
                   logger: logging.Logger) -> pd.DataFrame:
    sub = mapping[mapping["dataset"] == dataset_name].copy()
    if sub.empty:
        raise RuntimeError(f"class_mapping.csv has no entries for '{dataset_name}'")
    known = set(sub["native_label"].astype(str))
    present = set(df["Attack"].astype(str).unique())
    missing = present - known
    if missing:
        raise RuntimeError(
            f"native labels present in the CSV but not in class_mapping.csv "
            f"for {dataset_name}: {sorted(missing)}"
        )
    unused = known - present
    if unused:
        logger.warning("class_mapping.csv lists labels absent from the CSV: %s",
                       sorted(unused))
    df["native_label"] = df["Attack"].astype(str)
    df = df.merge(sub[["native_label", "canonical_label"]], on="native_label",
                  how="left", validate="many_to_one")
    if df["canonical_label"].isna().any():
        raise RuntimeError("unexpected NaN in canonical_label after merge")
    return df


def duplicate_and_conflict_stats(h: np.ndarray, canonical: pd.Series) -> dict:
    """Exact-duplicate and conflicting-label statistics on the feature key."""
    n = len(h)
    n_unique = int(len(np.unique(h)))
    canon_mask = (canonical != "").to_numpy()
    d = pd.DataFrame({"h": h[canon_mask],
                      "c": canonical.to_numpy()[canon_mask]}).drop_duplicates()
    multi = d["h"].duplicated(keep=False)
    conflict_vecs = d.loc[multi, "h"].unique()
    rows_conflict = int(np.isin(h[canon_mask], conflict_vecs).sum())
    pairs = {}
    if len(conflict_vecs):
        combos = (d[d["h"].isin(conflict_vecs)].groupby("h")["c"]
                  .apply(lambda s: " / ".join(sorted(s))).value_counts())
        pairs = {k: int(v) for k, v in combos.items()}
    return {
        "rows": n,
        "unique_feature_vectors": n_unique,
        "duplicate_rows": n - n_unique,
        "canonical_rows": int(canon_mask.sum()),
        "conflicting_vectors": int(len(conflict_vecs)),
        "conflicting_rows": rows_conflict,
        "conflict_label_sets": pairs,
    }


def write_qc_report(dataset_name: str, df: pd.DataFrame, h: np.ndarray,
                    qc_path: Path, logger: logging.Logger, n_feat: int) -> dict:
    lines: list[str] = []
    lines.append(f"QC report for {dataset_name}")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Total rows: {len(df):,}")
    lines.append(f"Columns:    {len(df.columns)}")
    lines.append(f"Split: grouped by seeded hash of the normalised {n_feat}-feature "
                 f"vector, seed {SPLIT_SEED}, 70/15/15")
    lines.append("")

    # 1. Class distribution per split
    ct = pd.crosstab(df["canonical_label"], df["split"])
    for col in ("train", "val", "test"):
        if col not in ct.columns:
            ct[col] = 0
    ct = ct[["train", "val", "test"]]
    ct["total"] = ct.sum(axis=1)
    ct = ct.sort_values("total", ascending=False)

    lines.append("Rows per canonical class per split:")
    lines.append(f"  {'class':18s} {'train':>12s} {'val':>12s} {'test':>12s} "
                 f"{'total':>12s}  {'tr%':>6s} {'va%':>6s} {'te%':>6s}  status")
    warnings: list[str] = []
    for cls, row in ct.iterrows():
        tr, va, te, tot = (int(row[c]) for c in ("train", "val", "test", "total"))
        share = [x / tot if tot else 0.0 for x in (tr, va, te)]
        status = []
        if cls != "":
            if va < MIN_ROWS_WARNING:
                status.append(f"val<{MIN_ROWS_WARNING}")
            if te < MIN_ROWS_WARNING:
                status.append(f"test<{MIN_ROWS_WARNING}")
            if not (TEST_SHARE_BAND[0] <= share[2] <= TEST_SHARE_BAND[1]):
                status.append("test share outside 10-20%")
            if status:
                warnings.append(f"{cls}: {', '.join(status)} "
                                f"(val={va:,}, test={te:,}, test share={share[2]:.1%})")
        label = cls if cls else "(excluded)"
        lines.append(f"  {label:18s} {tr:>12,d} {va:>12,d} {te:>12,d} {tot:>12,d}  "
                     f"{share[0]:>6.1%} {share[1]:>6.1%} {share[2]:>6.1%}  "
                     f"{', '.join(status) or 'OK'}")
    lines.append("")
    if warnings:
        lines.append("WARNINGS:")
        lines.extend(f"  - {w}" for w in warnings)
    else:
        lines.append("No canonical class triggers a warning.")

    # 2. Duplicates and conflicting labels
    stats = duplicate_and_conflict_stats(h, df["canonical_label"].astype(str))
    lines.append("")
    lines.append(f"Exact duplicates on the normalised {n_feat}-feature vector:")
    lines.append(f"  unique feature vectors: {stats['unique_feature_vectors']:,} "
                 f"of {stats['rows']:,} rows")
    lines.append(f"  duplicate rows:         {stats['duplicate_rows']:,} "
                 f"({stats['duplicate_rows']/stats['rows']*100:.2f}%)")
    lines.append("  Duplicates stay within one split by construction of the key.")
    lines.append("")
    lines.append("Conflicting labels (identical vector, more than one canonical "
                 "label; kept in the data, irreducible error floor):")
    lines.append(f"  vectors: {stats['conflicting_vectors']:,}   rows: "
                 f"{stats['conflicting_rows']:,} "
                 f"({stats['conflicting_rows']/max(stats['canonical_rows'],1)*100:.2f}% "
                 f"of canonical rows)")
    for combo, k in list(stats["conflict_label_sets"].items())[:10]:
        lines.append(f"    {combo:40s} {k:>10,} vectors")

    # 3. Implementation sanity check (true by construction)
    per_hash = pd.DataFrame({"h": h, "s": df["split"].to_numpy()}).drop_duplicates()
    n_split_hashes = int(per_hash["h"].duplicated().sum())
    lines.append("")
    lines.append("Implementation sanity check (holds by construction; the audit "
                 "verifies it independently on tabular.parquet):")
    lines.append(f"  feature hashes assigned to more than one split: {n_split_hashes}")
    if n_split_hashes:
        raise RuntimeError("a feature hash was assigned to more than one split")

    # 4. Source-IP overlap (informative)
    ips = {sp: set(df.loc[df["split"] == sp, "IPV4_SRC_ADDR"])
           for sp in ("train", "val", "test")}
    lines.append("")
    lines.append("Source-IP overlap across splits (informative; expected given "
                 "the testbed structure):")
    for sp, s in ips.items():
        lines.append(f"  unique src IPs in {sp:5s}: {len(s):,}")
    lines.append(f"  shared train/test: {len(ips['train'] & ips['test']):,}")

    # 5. NaN and Inf per column (left as-is; handled by the serializer)
    lines.append("")
    lines.append("NaN and Inf per column (stored as-is; the serializer converts "
                 "Inf to NaN for the tabular representation):")
    nan_per_col = df.isna().sum()
    nan_per_col = nan_per_col[nan_per_col > 0]
    numeric = df.select_dtypes(include=["number"])
    inf_per_col = numeric.apply(lambda c: int(np.isinf(c.to_numpy()).sum())
                                if pd.api.types.is_float_dtype(c) else 0)
    inf_per_col = inf_per_col[inf_per_col > 0]
    if nan_per_col.empty and inf_per_col.empty:
        lines.append("  none")
    for col in sorted(set(nan_per_col.index) | set(inf_per_col.index)):
        lines.append(f"  {col:32s} NaN={int(nan_per_col.get(col, 0)):>12,} "
                     f"Inf={int(inf_per_col.get(col, 0)):>12,}")

    qc_path.write_text("\n".join(lines) + "\n")
    logger.info("QC report written to %s", qc_path)
    return {
        "class_counts": {
            str(cls): {c: int(ct.loc[cls, c]) for c in ("train", "val", "test", "total")}
            for cls in ct.index
        },
        "duplicates_and_conflicts": stats,
        "warnings": warnings,
    }


def process_dataset(dataset_name: str, raw_path: Path) -> None:
    out_dir = PROCESSED_ROOT / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "load.log"
    logger = make_logger(log_path)
    logger.info("=" * 72)
    logger.info("Processing %s", dataset_name)
    logger.info("Raw CSV: %s", raw_path)
    logger.info("=" * 72)
    if not raw_path.exists():
        raise FileNotFoundError(f"raw CSV not found: {raw_path}")

    t_start = time.time()
    raw_sha1 = sha1_file(raw_path)
    logger.info("raw CSV SHA-1: %s", raw_sha1)

    t0 = time.time()
    map_path = mapping_path(dataset_name)
    mapping = load_class_mapping(map_path)
    cols = feature_cols(dataset_name)
    df = pd.read_csv(raw_path, dtype=raw_dtypes(dataset_name))
    logger.info("loaded %s rows, %d columns in %.1fs",
                f"{len(df):,}", len(df.columns), time.time() - t0)

    n_typo = apply_typo_correction(df, logger)
    df = join_canonical(df, mapping, dataset_name, logger)
    n_canon = int((df["canonical_label"] != "").sum())
    logger.info("canonical class assigned to %s / %s rows (%.2f%%)",
                f"{n_canon:,}", f"{len(df):,}", 100.0 * n_canon / len(df))

    t0 = time.time()
    h = feature_row_hash(df, cols)
    df["split"] = split_from_hash(h)
    logger.info("split assigned from feature-vector hash in %.1fs", time.time() - t0)

    for col in ["Attack", "native_label", "canonical_label", "split",
                "IPV4_SRC_ADDR", "IPV4_DST_ADDR"]:
        df[col] = df[col].astype("category")

    parquet_path = out_dir / "data.parquet"
    t0 = time.time()
    df.to_parquet(parquet_path, engine="pyarrow", compression="snappy", index=False)
    logger.info("parquet written in %.1fs (%.1f MB)", time.time() - t0,
                parquet_path.stat().st_size / 1024 / 1024)
    parquet_sha1 = sha1_file(parquet_path)
    logger.info("parquet SHA-1: %s", parquet_sha1)

    qc_path = out_dir / "qc_report.txt"
    qc_summary = write_qc_report(dataset_name, df, h, qc_path, logger, len(cols))

    manifest = {
        "dataset": dataset_name,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(WORK)),
        "environment": {"pandas": pd.__version__, "numpy": np.__version__},
        "input": {"path": str(raw_path.relative_to(WORK)), "sha1": raw_sha1},
        "config": {
            "split_method": "grouped by pandas.util.hash_pandas_object of the "
                            f"normalised {len(cols)}-feature vector, mixed with the seed "
                            "constant through splitmix64, modulo 100",
            "split_seed": SPLIT_SEED,
            "split_seed_const": hex(int(SPLIT_SEED_CONST)),
            "train_high": TRAIN_HIGH,
            "val_high": VAL_HIGH,
            "class_mapping_path": str(map_path.relative_to(WORK)),
            "class_mapping_sha1": sha1_file(map_path),
        },
        "typo_correction": {"from": TYPO_FROM, "to": TYPO_TO, "rows_affected": n_typo},
        "outputs": {
            "parquet": {
                "path": str(parquet_path.relative_to(WORK)),
                "sha1": parquet_sha1,
                "size_bytes": parquet_path.stat().st_size,
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
            },
            "qc_report": str(qc_path.relative_to(WORK)),
            "log": str(log_path.relative_to(WORK)),
        },
        "qc_summary": qc_summary,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("manifest written; done in %.1fs", time.time() - t_start)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=DATASETS, help="dataset to load")
    ap.add_argument("--all", action="store_true", help="process every dataset")
    args = ap.parse_args()
    if not args.dataset and not args.all:
        ap.error("either --dataset NAME or --all is required")
    for name in (DATASETS if args.all else [args.dataset]):
        process_dataset(name, RAW_CSV[name])


if __name__ == "__main__":
    main()
