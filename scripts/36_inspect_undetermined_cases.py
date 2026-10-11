"""36_inspect_undetermined_cases.py — resolve the two undetermined findings
of research/findings_classification.md (P3.7).

Case F1b, NF-CSE-CIC-IDS2018-v3 Infiltration duplicates. For the class's
exact duplicate records (every raw column, timestamps included): how many
copies each record has, how far apart the copies sit in the file, the days
they fall on, and the traffic they contain (ports, protocol, IPs), against
the non-duplicated records. FTP-BruteForce, whose records all appear twice,
is the control.

Case F2d, CIC web attacks (Brute Force -Web, Brute Force -XSS, SQL
Injection). In v2 (no timestamps): rows of the web attacker IP per label,
and full records that carry a web label and DDOS attack-HOIC. In v3: the
time windows of every label for the hosts that send web-attack rows, and,
for vectors that carry two of the web/benign labels, whether the rows of
each label fall in the same time window.

Usage: python scripts/36_inspect_undetermined_cases.py
Output: data/audit/undetermined_cases_report.txt
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from nfv3_config import FEATURE_COLS_V2, PROCESSED_ROOT, WORK, normalise_features

DERIVED = {"native_label", "canonical_label", "split"}
BATCH = 1_000_000
CIC3, CIC2 = "NF-CSE-CIC-IDS2018-v3", "NF-CSE-CIC-IDS2018-v2"
WEB3 = ["Brute_Force_-Web", "Brute_Force_-XSS", "SQL_Injection"]
WEB2 = ["Brute Force -Web", "Brute Force -XSS", "SQL Injection"]
HOIC3, HOIC2 = "DDOS_attack-HOIC", "DDOS attack-HOIC"


def load(dataset, labels=None, ips=None):
    """Rows (all raw columns) whose label is in `labels` or whose source IP is
    in `ips`, with their global row position."""
    pf = pq.ParquetFile(PROCESSED_ROOT / dataset / "data.parquet")
    cols = [c for c in pf.schema_arrow.names if c not in {"canonical_label", "split"}]
    parts, start = [], 0
    for b in pf.iter_batches(batch_size=BATCH, columns=cols):
        df = b.to_pandas()
        lab = df["native_label"].astype(str)
        m = np.zeros(len(df), dtype=bool)
        if labels:
            m |= lab.isin(labels).to_numpy()
        if ips:
            m |= df["IPV4_SRC_ADDR"].astype(str).isin(ips).to_numpy()
        if m.any():
            sub = df.loc[m].copy()
            sub["pos"] = np.flatnonzero(m) + start
            for c in sub.columns:
                if isinstance(sub[c].dtype, pd.CategoricalDtype):
                    sub[c] = sub[c].astype(str)
            parts.append(sub)
        start += len(df)
    return pd.concat(parts, ignore_index=True)


def benign_matching(dataset, hashes):
    """Benign rows whose 41-feature vector hash is in `hashes` (streamed)."""
    pf = pq.ParquetFile(PROCESSED_ROOT / dataset / "data.parquet")
    cols = FEATURE_COLS_V2 + ["native_label", "IPV4_SRC_ADDR", "FLOW_START_MILLISECONDS"]
    parts = []
    for b in pf.iter_batches(batch_size=BATCH, columns=cols):
        df = b.to_pandas()
        df = df[df["native_label"].astype(str) == "Benign"]
        h = pd.util.hash_pandas_object(normalise_features(df, FEATURE_COLS_V2), index=False).to_numpy()
        m = np.isin(h, hashes)
        if m.any():
            sub = df.loc[m].copy()
            sub["h"] = h[m]
            sub["native_label"] = "Benign"
            sub["IPV4_SRC_ADDR"] = sub["IPV4_SRC_ADDR"].astype(str)
            parts.append(sub)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols + ["h"])


def record_hash(df):
    raw = [c for c in df.columns if c not in DERIVED | {"pos"}]
    return pd.util.hash_pandas_object(df[raw], index=False).to_numpy()


def top(s, n=5):
    vc = s.astype(str).value_counts()
    return ", ".join(f"{k} ({100 * v / vc.sum():.1f} %)" for k, v in vc.head(n).items())


def day(ms):
    return pd.to_datetime(ms, unit="ms", utc=True).dt.strftime("%Y-%m-%d")


def hm(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def duplicates_case(L):
    for cls in ("Infiltration", "FTP-BruteForce"):
        df = load(CIC3, labels=[cls])
        df["rec"] = record_hash(df)
        mult = df.groupby("rec")["pos"].transform("size")
        L.append(f"[{cls}] rows {len(df):,}; distinct records {df['rec'].nunique():,}")
        mc = mult.value_counts().sort_index()
        L.append("  copies per record (rows): " + ", ".join(f"{k}x {v:,}" for k, v in mc.items()))
        dup = df[mult >= 2]
        if len(dup):
            g = dup.sort_values("pos").groupby("rec")["pos"]
            gaps = (g.max() - g.min()).to_numpy()
            L.append(f"  distance in the file between first and last copy (rows): min {gaps.min():,}, "
                     f"median {int(np.median(gaps)):,}, max {gaps.max():,}; "
                     f"most frequent {pd.Series(gaps).mode().iloc[0]:,} "
                     f"({100 * (gaps == pd.Series(gaps).mode().iloc[0]).mean():.1f} % of records)")
            first, last = int(df["pos"].min()), int(df["pos"].max())
            L.append(f"  class rows span file positions {first:,} to {last:,}")
        for name, part in (("duplicated", df[mult >= 2]), ("single", df[mult == 1])):
            if not len(part):
                continue
            L.append(f"  {name} rows {len(part):,}: days {top(day(part['FLOW_START_MILLISECONDS']), 4)}")
            L.append(f"    dst port {top(part['L4_DST_PORT'])}; protocol {top(part['PROTOCOL'], 3)}")
            L.append(f"    src IP {top(part['IPV4_SRC_ADDR'], 3)}; dst IP {top(part['IPV4_DST_ADDR'], 3)}")
            L.append(f"    packets per flow {top(part['IN_PKTS'] + part['OUT_PKTS'], 4)}")
        L.append("")


def web_case(L):
    # v2: no timestamps
    w2 = load(CIC2, labels=WEB2)
    att = w2["IPV4_SRC_ADDR"].value_counts().index[0]
    L.append(f"[v2] web-attack rows {len(w2):,}; main source IP {att} "
             f"({100 * (w2['IPV4_SRC_ADDR'] == att).mean():.1f} %)")
    a2 = load(CIC2, ips=[att])
    L.append(f"  all rows from {att} by label: " + ", ".join(
        f"{k} {v:,}" for k, v in a2["native_label"].value_counts().items()))
    a2["rec"] = record_hash(a2.drop(columns=["native_label", "Attack", "Label"]))
    labs = a2.groupby("rec")["native_label"].agg(lambda s: " / ".join(sorted(set(s))))
    multi = labs[labs.str.contains(" / ")]
    L.append(f"  records (all columns except labels) from {att} carrying >1 label: {len(multi):,}; label sets: "
             + "; ".join(f"{k} {v:,}" for k, v in multi.value_counts().head(5).items()))
    L.append("")
    # v3: timestamps
    w3 = load(CIC3, labels=WEB3)
    ips = w3["IPV4_SRC_ADDR"].value_counts()
    L.append(f"[v3] web-attack rows {len(w3):,}; source IPs " + ", ".join(
        f"{k} {v:,}" for k, v in ips.head(4).items()))
    a3 = load(CIC3, ips=list(ips.index[:4]))
    L.append("  time window per label for these source IPs (UTC):")
    for lab, g in a3.groupby("native_label"):
        days = top(day(g["FLOW_START_MILLISECONDS"]), 4)
        L.append(f"    {lab:24s} rows {len(g):>9,}  {hm(g['FLOW_START_MILLISECONDS'].min())} to "
                 f"{hm(g['FLOW_END_MILLISECONDS'].max())}  days: {days}")
    # vectors carrying two of the web/benign labels: same time window?
    cand = load(CIC3, labels=WEB3)
    cand["h"] = pd.util.hash_pandas_object(normalise_features(cand, FEATURE_COLS_V2), index=False).to_numpy()
    benign = benign_matching(CIC3, np.unique(cand["h"].to_numpy()))
    both = pd.concat([cand, benign], ignore_index=True)
    n_lab = both.groupby("h")["native_label"].nunique()
    conf = both[both["h"].isin(n_lab.index[n_lab > 1])]
    L.append(f"  vectors (41 features) carrying >1 of web labels / Benign: {conf['h'].nunique():,}; rows {len(conf):,}")
    if len(conf):
        spans = conf.groupby(["h", "native_label"])["FLOW_START_MILLISECONDS"].agg(["min", "max"]).reset_index()
        same_minute = 0
        for h, g in spans.groupby("h"):
            lo, hi = g["min"].max(), g["max"].min()
            same_minute += int(lo - hi <= 60_000)
        L.append(f"  of those vectors, rows of the different labels within 1 minute of each other: "
                 f"{same_minute:,} of {spans['h'].nunique():,}")
        L.append("  source IPs of the conflicting rows by label:")
        for lab, g in conf.groupby("native_label"):
            L.append(f"    {lab:24s} {len(g):>7,} rows; src {top(g['IPV4_SRC_ADDR'], 3)}; "
                     f"days {top(day(g['FLOW_START_MILLISECONDS']), 3)}")
    L.append("")


def main():
    L = ["Inspection of the undetermined findings F1b and F2d (P3.7)",
         f"Generated: {datetime.now(timezone.utc).isoformat()}", "",
         "== F1b: exact duplicate records in NF-CSE-CIC-IDS2018-v3 Infiltration (FTP-BruteForce as control)"]
    duplicates_case(L)
    L.append("== F2d: CIC web attacks")
    web_case(L)
    text = "\n".join(L) + "\n"
    (WORK / "data/audit/undetermined_cases_report.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
