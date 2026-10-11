"""31_inspect_conflict_groups.py — RQ1: where do the largest label conflicts
come from? (PLAN-PAPER-B.md, P3.3b)

For each selected group (dataset + set of native labels), the script finds
the 41-feature vectors (key of script 30) that carry at least two labels of
the group, then reads the full records of those rows from `data.parquet` and
reports, per label: rows, top source / destination IPs, destination ports,
protocols, time span (NF-v3 only); the most frequent conflicting vectors with
their non-zero features; how many conflicting rows are full-record duplicates
(same IPs, ports, timestamps and features) labelled differently; and an
exact, hash-free re-check of a sample of conflicting vectors.

With --key 40 the vectors are keyed without L4_SRC_PORT (the features the
models see; reviewer point M1, 2026-10-06); full records still include it.

Usage:
  python scripts/31_inspect_conflict_groups.py --group cic3_ftp_slowhttp [--key 40]
  python scripts/31_inspect_conflict_groups.py --report [--key 40]
Outputs: data/audit/conflict_groups[_k40]/<group>.txt and
data/audit/conflict_groups[_k40]_report.txt (concatenation).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from nfv3_config import FEATURE_COLS_V2, PROCESSED_ROOT, WORK, is_v2, normalise_features

GROUPS = {
    "cic3_ftp_slowhttp": ("NF-CSE-CIC-IDS2018-v3", ["DoS_attacks-SlowHTTPTest", "FTP-BruteForce", "SSH-Bruteforce"]),
    "ton3_benign_scanning": ("NF-ToN-IoT-v3", ["Benign", "scanning"]),
    "ton3_benign_password": ("NF-ToN-IoT-v3", ["Benign", "password"]),
    "bot3_ddos_dos": ("NF-BoT-IoT-v3", ["DDoS", "DoS"]),
    "cic2_web_hoic": ("NF-CSE-CIC-IDS2018-v2", ["DDOS attack-HOIC", "Brute Force -Web", "Brute Force -XSS", "SQL Injection"]),
    # groups that appear with the 40-feature key (reviewer point M1, 2026-10-06)
    "ton3_benign_ddos": ("NF-ToN-IoT-v3", ["Benign", "ddos"]),
    "cic3_benign_infiltration": ("NF-CSE-CIC-IDS2018-v3", ["Benign", "Infiltration"]),
    "bot3_benign_recon": ("NF-BoT-IoT-v3", ["Benign", "Reconnaissance"]),
    "cic3_web_benign": ("NF-CSE-CIC-IDS2018-v3", ["Benign", "Brute_Force_-Web", "Brute_Force_-XSS", "SQL_Injection"]),
}
BATCH = 1_000_000
TOP = 5
EXACT_SAMPLE = 500
SEED = 2026
OUT = WORK / "data/audit"
KEY = FEATURE_COLS_V2  # vector key; set by --key
ID_COLS = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]
TIME_COLS = ["FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS"]


def key_hash(df: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(normalise_features(df, KEY), index=False).to_numpy()


def find_vectors(path, labels) -> np.ndarray:
    """Hashes of vectors that carry at least two labels of the group."""
    parts = []
    for b in pq.ParquetFile(path).iter_batches(batch_size=BATCH, columns=FEATURE_COLS_V2 + ["native_label"]):
        df = b.to_pandas()
        lab = df["native_label"].astype(str)
        m = lab.isin(labels).to_numpy()
        if m.any():
            parts.append(pd.DataFrame({"h": key_hash(df.loc[m]), "c": lab[m].to_numpy()}).drop_duplicates())
    pairs = pd.concat(parts).drop_duplicates()
    n = pairs.groupby("h")["c"].nunique()
    return n.index[n >= 2].to_numpy(dtype=np.uint64)


def collect_rows(path, labels, vecs, extra) -> pd.DataFrame:
    vset = np.sort(vecs)
    parts = []
    cols = list(dict.fromkeys(FEATURE_COLS_V2 + ["native_label"] + ID_COLS + extra))
    for b in pq.ParquetFile(path).iter_batches(batch_size=BATCH, columns=cols):
        df = b.to_pandas()
        lab = df["native_label"].astype(str)
        m = lab.isin(labels).to_numpy()
        if not m.any():
            continue
        sub = df.loc[m].copy()
        sub["h"] = key_hash(sub)
        sub = sub[np.isin(sub["h"].to_numpy(), vset)]
        sub["native_label"] = sub["native_label"].astype(str)
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def top(s: pd.Series, n=TOP) -> str:
    vc = s.astype(str).value_counts()
    tot = vc.sum()
    return ", ".join(f"{k} ({100 * v / tot:.1f} %)" for k, v in vc.head(n).items()) + (
        f" [+{len(vc) - n} more]" if len(vc) > n else "")


def fmt(x) -> str:
    x = float(x)
    return str(int(x)) if x.is_integer() else f"{x:.6g}"


def ts(ms) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def inspect(name: str) -> str:
    dataset, labels = GROUPS[name]
    path = PROCESSED_ROOT / dataset / "data.parquet"
    extra = [] if is_v2(dataset) else TIME_COLS
    vecs = find_vectors(path, labels)
    rows = collect_rows(path, labels, vecs, extra)
    L = [f"=== {name}: {dataset}, labels {labels}",
         f"vectors carrying >= 2 of these labels ({len(KEY)}-feature key): {len(vecs):,}; rows: {len(rows):,}", ""]
    for lab, g in rows.groupby("native_label"):
        L.append(f"[{lab}] rows {len(g):,}; distinct vectors {g['h'].nunique():,}")
        L.append(f"  src IP : {top(g['IPV4_SRC_ADDR'])}")
        L.append(f"  dst IP : {top(g['IPV4_DST_ADDR'])}")
        L.append(f"  dst port: {top(g['L4_DST_PORT'])}")
        L.append(f"  protocol: {top(g['PROTOCOL'])}; L7_PROTO: {top(g['L7_PROTO'], 3)}")
        if extra:
            L.append(f"  time   : {ts(g['FLOW_START_MILLISECONDS'].min())} to "
                     f"{ts(g['FLOW_END_MILLISECONDS'].max())} UTC")
    # most frequent conflicting vectors
    L.append("")
    L.append("Most frequent conflicting vectors (rows per label; non-zero features):")
    counts = rows.groupby(["h", "native_label"]).size().unstack(fill_value=0)
    counts["total"] = counts.sum(axis=1)
    first = rows.drop_duplicates("h").set_index("h")
    for h, r in counts.sort_values("total", ascending=False).head(TOP).iterrows():
        per = ", ".join(f"{c} {int(r[c]):,}" for c in counts.columns if c != "total" and r[c])
        v = first.loc[h, KEY]
        nz = ", ".join(f"{c}={fmt(v[c])}" for c in KEY if pd.notna(v[c]) and v[c] != 0)
        L.append(f"  - {per}")
        L.append(f"      {nz}")
    # coverage of the top vectors
    cum = counts["total"].sort_values(ascending=False).cumsum() / counts["total"].sum()
    L.append(f"  share of conflicting rows in the top 1 / 10 / 100 vectors: "
             f"{100 * cum.iloc[0]:.1f} % / {100 * cum.iloc[min(9, len(cum) - 1)]:.1f} % / "
             f"{100 * cum.iloc[min(99, len(cum) - 1)]:.1f} %")
    # full-record duplicates across labels
    rec_cols = [c for c in FEATURE_COLS_V2 + ID_COLS + extra if c in rows.columns]
    rec = pd.Series(pd.util.hash_pandas_object(rows[rec_cols], index=False).to_numpy())
    lab_per_rec = rows.assign(rec=rec).groupby("rec")["native_label"].nunique()
    multi = lab_per_rec.index[lab_per_rec > 1]
    n_rows_multi = int(rec.isin(multi).sum())
    L.append("")
    L.append(f"Full records (features + IPs{' + timestamps' if extra else ''}) carrying >1 label: "
             f"{len(multi):,} records, {n_rows_multi:,} rows ({100 * n_rows_multi / len(rows):.2f} % of group rows)")
    within = len(rows) - int(pd.DataFrame({"rec": rec, "lab": rows["native_label"].to_numpy()}).drop_duplicates().shape[0])
    L.append(f"Full-record duplicates within the same label (identical record repeated): "
             f"{within:,} rows ({100 * within / len(rows):.2f} % of group rows)")
    # exact re-check without hashing
    rng = np.random.default_rng(SEED)
    sample = rng.choice(vecs, size=min(EXACT_SAMPLE, len(vecs)), replace=False)
    sub = rows[rows["h"].isin(sample)]
    exact = normalise_features(sub, KEY)
    exact["native_label"] = sub["native_label"].to_numpy()
    n_lab = exact.drop_duplicates().groupby(KEY, dropna=False)["native_label"].nunique()
    L.append(f"Exact re-check (no hashing) on {len(sample)} sampled vectors: "
             f"{int((n_lab > 1).sum())} distinct exact vectors with >1 label, "
             f"{int((n_lab == 1).sum())} with a single label")
    L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--group", choices=list(GROUPS))
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--key", choices=["41", "40"], default="41")
    a = ap.parse_args()
    global KEY
    sfx = "" if a.key == "41" else "_k40"
    if a.key == "40":
        KEY = [c for c in FEATURE_COLS_V2 if c != "L4_SRC_PORT"]
    d = OUT / f"conflict_groups{sfx}"
    d.mkdir(parents=True, exist_ok=True)
    if a.group:
        text = inspect(a.group)
        (d / f"{a.group}.txt").write_text(text + "\n")
        print(text)
    if a.report:
        parts = [(d / f"{g}.txt").read_text() for g in GROUPS if (d / f"{g}.txt").exists()]
        (OUT / f"conflict_groups{sfx}_report.txt").write_text(
            f"RQ1: inspection of the largest conflict groups (generated "
            f"{datetime.now(timezone.utc).isoformat()})\n\n" + "\n".join(parts))


if __name__ == "__main__":
    main()
