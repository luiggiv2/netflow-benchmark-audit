"""n9_minority_composition.py — second reviewer report, point N9, 2026-10-08:
what are the minority rows of NF-v3 (rows whose label differs from the
majority label of their feature vector)? The abstract says "mostly flows of
attacker hosts labelled benign"; this script measures it.

Per NF-v3 dataset, on key 40 (the 40 features common to both releases without
L4_SRC_PORT, primary key of RQ1) and key 48 (all NF-v3 features without
L4_SRC_PORT, the features of the NF-v3 models), with the native labels:
  - minority rows (same definition and count as 30_audit_duplicates_conflicts.py:
    rows minus the majority count of each vector; check printed);
  - which rows are the minority: per vector, the majority label is the most
    frequent one (ties broken by label name; tied vectors counted);
  - share of minority rows labelled Benign, and of those, share whose source
    address also sends attack-labelled rows in the dataset (attacker hosts);
  - share labelled as an attack, and of those, share on Benign-majority
    vectors; share of minority rows with Benign as the minority or the
    majority label (an attack label against Benign), and from attacker hosts;
  - the most frequent (minority label -> majority label) pairs.

Usage: python scripts/exploration/n9_minority_composition.py
Output: data/exploration/n9_minority_composition.txt
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nfv3_config import FEATURE_COLS_V2, PROCESSED_ROOT, WORK, feature_cols, tabular_row_hash  # noqa: E402

DATASETS = ["NF-ToN-IoT-v3", "NF-BoT-IoT-v3", "NF-CSE-CIC-IDS2018-v3"]
BENIGN = "Benign"
BATCH = 1_000_000
TOP = 8


def keys(ds: str) -> dict[str, list[str]]:
    return {"40": [c for c in FEATURE_COLS_V2 if c != "L4_SRC_PORT"],
            "48": [c for c in feature_cols(ds) if c != "L4_SRC_PORT"]}


def analyse(ds: str) -> list[str]:
    meta = pq.read_table(PROCESSED_ROOT / ds / "data.parquet", columns=["native_label", "IPV4_SRC_ADDR"])
    lab = meta.column("native_label").to_pandas().astype(str).astype("category")
    ip = meta.column("IPV4_SRC_ADDR").to_pandas().astype("category")
    names = list(lab.cat.categories)
    codes = lab.cat.codes.to_numpy().astype(np.int64)
    ben = names.index(BENIGN)
    att_ips = set(ip[(codes != ben)].unique())
    from_att = ip.isin(att_ips).to_numpy()
    stored = json.loads((WORK / "data/audit/dupconf" / f"{ds}.json").read_text())["keys"]
    L = [f"== {ds}: {len(codes):,} rows; {len(att_ips):,} source addresses send attack-labelled rows"]
    for k, cols in keys(ds).items():
        pf = pq.ParquetFile(PROCESSED_ROOT / ds / "tabular.parquet")
        h = np.concatenate([tabular_row_hash(b.to_pandas(), cols) for b in pf.iter_batches(batch_size=BATCH,
                                                                                            columns=cols)])
        _, inv = np.unique(h, return_inverse=True)
        del h
        nl = len(names)
        pair, cnt = np.unique(inv * nl + codes, return_counts=True)
        vec, lb = pair // nl, pair % nl
        # majority label per vector: highest count, ties broken by label name (code order)
        o = np.lexsort((lb, -cnt, vec))
        first = np.r_[True, vec[o][1:] != vec[o][:-1]]
        maj = np.empty(vec.max() + 1, dtype=np.int64)
        maj[vec[o][first]] = lb[o][first]
        top_cnt = np.zeros(vec.max() + 1, dtype=np.int64)
        top_cnt[vec[o][first]] = cnt[o][first]
        n_labels = np.bincount(vec, minlength=len(maj))
        second = np.r_[False, vec[o][1:] == vec[o][:-1]] & np.r_[False, first[:-1]]
        ties = int(np.sum(cnt[o][second] == top_cnt[vec[o][second]]))
        row_maj = maj[inv]
        mino = codes != row_maj
        n_min = int(mino.sum())
        ok = n_min == stored[k]["native"]["minority_rows"]
        mb = mino & (codes == ben)
        ma = mino & (codes != ben)
        L.append(f"  key {k} ({len(cols)} features): minority rows {n_min:,} ({100 * n_min / len(codes):.2f} % of rows)"
                 f" [matches script 30: {ok}]; conflicting vectors {int(np.sum(n_labels > 1)):,}, of which with a tied"
                 f" majority {ties:,}")
        tied_vec = np.zeros(len(maj), dtype=bool)
        tied_vec[vec[o][second][cnt[o][second] == top_cnt[vec[o][second]]]] = True
        mt = mino & tied_vec[inv]
        L.append(f"    minority rows on tied vectors (label depends on the tie-break): {int(mt.sum()):,}"
                 f" ({100 * mt.sum() / max(n_min, 1):.1f} %); of the non-tied minority rows, labelled Benign"
                 f" {100 * np.sum(mino & ~tied_vec[inv] & (codes == ben)) / max(n_min - mt.sum(), 1):.1f} %")
        L.append(f"    labelled Benign: {int(mb.sum()):,} ({100 * mb.sum() / max(n_min, 1):.1f} % of minority rows);"
                 f" from attacker hosts {100 * np.sum(mb & from_att) / max(mb.sum(), 1):.1f} %")
        L.append(f"    labelled as an attack: {int(ma.sum()):,} ({100 * ma.sum() / max(n_min, 1):.1f} %);"
                 f" on Benign-majority vectors {100 * np.sum(ma & (row_maj == ben)) / max(ma.sum(), 1):.1f} %")
        mixed = mino & ((codes == ben) | (row_maj == ben))
        L.append(f"    with Benign as the minority or the majority label: {int(mixed.sum()):,}"
                 f" ({100 * mixed.sum() / max(n_min, 1):.1f} % of minority rows); from attacker hosts"
                 f" {100 * np.sum(mixed & from_att) / max(mixed.sum(), 1):.1f} %")
        pr, pc = np.unique(codes[mino] * nl + row_maj[mino], return_counts=True)
        top = np.argsort(-pc)[:TOP]
        L.append("    top pairs (minority label -> majority label): " + "; ".join(
            f"{names[pr[i] // nl]} -> {names[pr[i] % nl]} {pc[i]:,} ({100 * pc[i] / max(n_min, 1):.1f} %)" for i in top))
    return L + [""]


def main() -> None:
    L = ["N9 check: composition of the minority rows of NF-v3 (native labels)",
         f"Generated: {datetime.now(timezone.utc).isoformat()}",
         "Minority row: label differs from the majority label of its feature vector. Attacker hosts: source",
         "addresses that send at least one attack-labelled row in the same dataset.", ""]
    for ds in DATASETS:
        part = analyse(ds)
        L += part
        print("\n".join(part), flush=True)
    out = WORK / "data/exploration/n9_minority_composition.txt"
    out.write_text("\n".join(L) + "\n")
    print(f"written {out}")


if __name__ == "__main__":
    main()
