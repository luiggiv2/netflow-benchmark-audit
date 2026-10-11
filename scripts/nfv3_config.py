"""Shared configuration for the Paper B NetFlow pipeline (NF-v3 and NF-v2).

Loader, tabular builder and audit must agree on the dataset paths, the
dtypes, the feature columns, and the definition of the split key. They are
defined once here. NF-v3 uses the 49 features of FEATURE_COLS; NF-v2 lacks
the two timestamps and the eight IAT columns and uses the remaining 41
(`feature_cols`). Every NF-v3 default below is unchanged by the v2 support.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = WORK / "data/processed"
MAPPING_PATH = WORK / "data/schemas/class_mapping.csv"
MAPPING_PATH_V2 = WORK / "data/schemas/class_mapping_v2.csv"

RAW_CSV = {
    "NF-ToN-IoT-v3": WORK
    / "02934b58528a226b_NFV3DATA-A11964_A11964/data/NF-ToN-IoT-v3.csv",
    "NF-BoT-IoT-v3": WORK
    / "d509c9db7490cf92_NFV3DATA-A11964_A11964/data/NF-BoT-IoT-v3.csv",
    "NF-CSE-CIC-IDS2018-v3": WORK
    / "f78acbaa2afe1595_NFV3DATA-A11964_A11964/data/NF-CICIDS2018-v3.csv",
    "NF-ToN-IoT-v2": WORK
    / "9bafce9d380588c2_MOHANAD_A4706/data/NF-ToN-IoT-v2.csv",
    "NF-BoT-IoT-v2": WORK
    / "befb58edf3428167_MOHANAD_A4706/data/NF-BoT-IoT-v2.csv",
    "NF-CSE-CIC-IDS2018-v2": WORK
    / "b3427ed8ad063a09_MOHANAD_A4706/data/NF-CSE-CIC-IDS2018-v2.csv",
}
DATASETS = list(RAW_CSV)

# Dtypes applied when reading the raw CSVs. pandas 3.x wraps silently when a
# value does not fit an integer dtype, so every choice below was checked
# against the full-width maxima of the three raw files, and the audit
# re-checks value fidelity against the raw CSV on every run (check I5).
# DNS_TTL_ANSWER is int64: NF-BoT-IoT-v3 holds values up to 4,294,915,672.
NFV3_DTYPES = {
    "FLOW_START_MILLISECONDS": "int64",
    "FLOW_END_MILLISECONDS": "int64",
    "IPV4_SRC_ADDR": "string",
    "L4_SRC_PORT": "int32",
    "IPV4_DST_ADDR": "string",
    "L4_DST_PORT": "int32",
    "PROTOCOL": "int16",
    "L7_PROTO": "float32",
    "IN_BYTES": "int64",
    "IN_PKTS": "int32",
    "OUT_BYTES": "int64",
    "OUT_PKTS": "int32",
    "TCP_FLAGS": "int16",
    "CLIENT_TCP_FLAGS": "int16",
    "SERVER_TCP_FLAGS": "int16",
    "FLOW_DURATION_MILLISECONDS": "int32",
    "DURATION_IN": "int32",
    "DURATION_OUT": "int32",
    "MIN_TTL": "int16",
    "MAX_TTL": "int16",
    "LONGEST_FLOW_PKT": "int32",
    "SHORTEST_FLOW_PKT": "int32",
    "MIN_IP_PKT_LEN": "int32",
    "MAX_IP_PKT_LEN": "int32",
    "SRC_TO_DST_SECOND_BYTES": "float32",
    "DST_TO_SRC_SECOND_BYTES": "float32",
    "RETRANSMITTED_IN_BYTES": "int32",
    "RETRANSMITTED_IN_PKTS": "int32",
    "RETRANSMITTED_OUT_BYTES": "int32",
    "RETRANSMITTED_OUT_PKTS": "int32",
    "SRC_TO_DST_AVG_THROUGHPUT": "int64",
    "DST_TO_SRC_AVG_THROUGHPUT": "int64",
    "NUM_PKTS_UP_TO_128_BYTES": "int32",
    "NUM_PKTS_128_TO_256_BYTES": "int32",
    "NUM_PKTS_256_TO_512_BYTES": "int32",
    "NUM_PKTS_512_TO_1024_BYTES": "int32",
    "NUM_PKTS_1024_TO_1514_BYTES": "int32",
    "TCP_WIN_MAX_IN": "int32",
    "TCP_WIN_MAX_OUT": "int32",
    "ICMP_TYPE": "int32",
    "ICMP_IPV4_TYPE": "int32",
    "DNS_QUERY_ID": "int32",
    "DNS_QUERY_TYPE": "int32",
    "DNS_TTL_ANSWER": "int64",
    "FTP_COMMAND_RET_CODE": "float32",
    "SRC_TO_DST_IAT_MIN": "int32",
    "SRC_TO_DST_IAT_MAX": "int32",
    "SRC_TO_DST_IAT_AVG": "float32",
    "SRC_TO_DST_IAT_STDDEV": "float32",
    "DST_TO_SRC_IAT_MIN": "int32",
    "DST_TO_SRC_IAT_MAX": "int32",
    "DST_TO_SRC_IAT_AVG": "float32",
    "DST_TO_SRC_IAT_STDDEV": "float32",
    "Label": "int8",
    "Attack": "string",
}

# The 49 model features, in a fixed order. Identifiers (IPs), absolute
# timestamps and labels are excluded.
FEATURE_COLS: list[str] = [
    "L4_SRC_PORT", "L4_DST_PORT",
    "PROTOCOL", "L7_PROTO",
    "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS",
    "TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT",
    "MIN_TTL", "MAX_TTL",
    "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT",
    "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
    "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
    "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS",
    "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT",
    "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES",
    "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT",
    "ICMP_TYPE", "ICMP_IPV4_TYPE",
    "DNS_QUERY_ID", "DNS_QUERY_TYPE", "DNS_TTL_ANSWER",
    "FTP_COMMAND_RET_CODE",
    "SRC_TO_DST_IAT_MIN", "SRC_TO_DST_IAT_MAX",
    "SRC_TO_DST_IAT_AVG", "SRC_TO_DST_IAT_STDDEV",
    "DST_TO_SRC_IAT_MIN", "DST_TO_SRC_IAT_MAX",
    "DST_TO_SRC_IAT_AVG", "DST_TO_SRC_IAT_STDDEV",
]
assert len(FEATURE_COLS) == 49

# Columns that exist only in NF-v3.
V3_ONLY_COLS: list[str] = [
    "FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS",
    "SRC_TO_DST_IAT_MIN", "SRC_TO_DST_IAT_MAX",
    "SRC_TO_DST_IAT_AVG", "SRC_TO_DST_IAT_STDDEV",
    "DST_TO_SRC_IAT_MIN", "DST_TO_SRC_IAT_MAX",
    "DST_TO_SRC_IAT_AVG", "DST_TO_SRC_IAT_STDDEV",
]
FEATURE_COLS_V2: list[str] = [c for c in FEATURE_COLS if c not in V3_ONLY_COLS]
assert len(FEATURE_COLS_V2) == 41


def is_v2(dataset: str) -> bool:
    return dataset.endswith("-v2")


def feature_cols(dataset: str) -> list[str]:
    return FEATURE_COLS_V2 if is_v2(dataset) else FEATURE_COLS


# NF-v2 columns whose raw values exceed the float32 range (up to 8.9e304 in
# NF-ToN-IoT-v2); read as float64 so they are not turned into Inf.
V2_FLOAT64_COLS = ["SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES"]


def raw_dtypes(dataset: str) -> dict[str, str]:
    """Loader dtypes; NF-v2 uses the NF-v3 dtypes of its columns except
    V2_FLOAT64_COLS (ranges in data/exploration/nfv2_dtype_ranges.txt)."""
    if not is_v2(dataset):
        return NFV3_DTYPES
    return {c: ("float64" if c in V2_FLOAT64_COLS else t)
            for c, t in NFV3_DTYPES.items() if c not in V3_ONLY_COLS}


def mapping_path(dataset: str) -> Path:
    return MAPPING_PATH_V2 if is_v2(dataset) else MAPPING_PATH


META_COLS: list[str] = ["Label", "native_label", "canonical_label", "split"]

SPLIT_SEED = 42
TRAIN_HIGH = 70
VAL_HIGH = 85
# pandas.util.hash_pandas_object ignores its hash_key for numeric columns, so
# the seed enters through this 64-bit constant, mixed into the row hash before
# the bucket is taken (see split_from_hash).
SPLIT_SEED_CONST = np.uint64(
    int(hashlib.sha256(f"paperB-split-seed-{SPLIT_SEED}".encode()).hexdigest()[:16], 16)
)

HASH_BLOCK_ROWS = 2_000_000


def normalise_features(df: pd.DataFrame, cols: list[str] = FEATURE_COLS) -> pd.DataFrame:
    """Return the feature columns as they enter the tabular representation.

    Float columns: -0.0 becomes 0.0 and every non-finite value (NaN, +Inf,
    -Inf) becomes the canonical NaN, so numerically identical rows are also
    bitwise identical. Integer columns are returned unchanged.
    """
    out = {}
    for col in cols:
        s = df[col]
        if pd.api.types.is_float_dtype(s):
            arr = s.to_numpy(copy=True)
            arr = arr + arr.dtype.type(0.0)
            arr[~np.isfinite(arr)] = np.nan
            out[col] = arr
        else:
            out[col] = s.to_numpy()
    return pd.DataFrame(out, columns=cols)


def feature_row_hash(df: pd.DataFrame, cols: list[str] = FEATURE_COLS) -> np.ndarray:
    """uint64 hash of each row's normalised feature vector (49 features in
    NF-v3, 41 in NF-v2).

    Computed in blocks of HASH_BLOCK_ROWS rows to bound memory. Row hashes do
    not depend on the block boundaries because every feature column is numeric
    and index=False.
    """
    n = len(df)
    out = np.empty(n, dtype=np.uint64)
    for start in range(0, n, HASH_BLOCK_ROWS):
        stop = min(start + HASH_BLOCK_ROWS, n)
        block = normalise_features(df.iloc[start:stop], cols)
        out[start:stop] = pd.util.hash_pandas_object(block, index=False).to_numpy()
    return out


def _mix64(x: np.ndarray) -> np.ndarray:
    """splitmix64 finaliser (uint64 arithmetic wraps modulo 2**64)."""
    with np.errstate(over="ignore"):
        x = x ^ (x >> np.uint64(30))
        x = x * np.uint64(0xBF58476D1CE4E5B9)
        x = x ^ (x >> np.uint64(27))
        x = x * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return x


def split_from_hash(h: np.ndarray, seed_const: np.uint64 = SPLIT_SEED_CONST) -> np.ndarray:
    """Map row hashes to 'train' / 'val' / 'test' (70 / 15 / 15).

    Equal row hashes always map to the same split; the seed constant changes
    which split that is.
    """
    bucket = (_mix64(h ^ seed_const) % np.uint64(100)).astype(np.int16)
    split = np.full(len(h), "train", dtype=object)
    split[bucket >= TRAIN_HIGH] = "val"
    split[bucket >= VAL_HIGH] = "test"
    return split


# --------------------------------------------------------------------------- #
# Phase 4 (research/phase4_protocol.md): model feature sets and splits
# recomputed on the features each model sees.
# --------------------------------------------------------------------------- #

# Author decision 2026-10-02 (internal audit C1): no L4_SRC_PORT in the model
# features or in the split key; L4_DST_PORT is kept.
TIMESTAMP_COLS = ["FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS"]
PORT_COLS = ["L4_SRC_PORT", "L4_DST_PORT"]
IDENTITY_COLS = ["MIN_TTL", "MAX_TTL", "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT",
                 "DNS_QUERY_ID", "DNS_QUERY_TYPE", "DNS_TTL_ANSWER"]
# Fields that name the service or carry exporter information unrelated to the
# traffic (third reviewer report, major issue 1; author decisions 2026-10-10):
# L7_PROTO names the service of the server port; ICMP_TYPE / ICMP_IPV4_TYPE are
# non-zero on some TCP flows (data/exploration/r3m1_l7proto_icmp_check.txt).
SERVICE_COLS = ["L7_PROTO", "ICMP_TYPE", "ICMP_IPV4_TYPE"]


def model_feature_sets(dataset: str) -> dict[str, list[str]]:
    """Feature sets of the Phase 4 conditions (protocol §5).

    base  F48 (v3) / F40 (v2): all features except L4_SRC_PORT   (G0, R0, G-NC, G-ND)
    noport  base without L4_DST_PORT, F47 / F39                   (G-NP)
    noid  base without ports, TTL, TCP window and DNS, F40 / F32  (G-NI)
    ts    base plus both absolute timestamps, F50, NF-v3 only     (G-TS)
    """
    full = feature_cols(dataset)
    base = [c for c in full if c != "L4_SRC_PORT"]
    sets = {"base": base,
            "noport": [c for c in base if c not in PORT_COLS],
            "noid": [c for c in base if c not in PORT_COLS + IDENTITY_COLS]}
    if not is_v2(dataset):
        sets["ts"] = base + TIMESTAMP_COLS
    return sets


def seed_const(seed: int) -> np.uint64:
    """Split seed constant; seed 42 gives SPLIT_SEED_CONST (Phase 1)."""
    return np.uint64(int(hashlib.sha256(f"paperB-split-seed-{seed}".encode()).hexdigest()[:16], 16))


def grouped_split_codes(h: np.ndarray, seed: int) -> np.ndarray:
    """Grouped split from precomputed row hashes as int8 codes
    (0 train, 1 val, 2 test); same buckets as split_from_hash."""
    bucket = (_mix64(h ^ seed_const(seed)) % np.uint64(100)).astype(np.int16)
    return np.where(bucket < TRAIN_HIGH, 0, np.where(bucket < VAL_HIGH, 1, 2)).astype(np.int8)


def random_split_codes(n: int, seed: int) -> np.ndarray:
    """Row-level random 70/15/15 split as int8 codes (0 train, 1 val, 2 test)."""
    u = np.random.default_rng(seed).random(n)
    return np.where(u < 0.70, 0, np.where(u < 0.85, 1, 2)).astype(np.int8)


def tabular_row_hash(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Row hash of columns read from tabular.parquet, which already holds
    normalised features (as in 20_decision_test.py); extra integer columns
    such as the NF-v3 timestamps need no normalisation."""
    return pd.util.hash_pandas_object(df[cols], index=False).to_numpy()
