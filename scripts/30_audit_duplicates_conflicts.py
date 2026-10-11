"""30_audit_duplicates_conflicts.py — RQ1: exact duplicates and conflicting
labels per class, NF-v2 vs NF-v3 (PLAN-PAPER-B.md, P3.3).

For each dataset the feature vector is keyed by a 64-bit hash of its
normalised features read from `tabular.parquet`:
  - key "41": the 41 features common to NF-v2 and NF-v3 (all datasets);
  - key "49": the full NF-v3 feature set (NF-v3 only; must reproduce the
    loader QC figures, which the report checks);
  - keys "40" and "48": the same sets without L4_SRC_PORT, the features the
    Phase 4 models see (added 2026-10-06, reviewer point M1; "40" is the
    primary key for comparing releases, "41" shows what the source port hides).

Definitions (per key):
  duplicate rows       rows minus distinct vectors
  intra-class dups     rows of class c minus distinct vectors within class c
  conflicting vector   a vector that carries more than one label
  conflict rows        all rows of conflicting vectors (every label)
  minority rows        rows whose label is not the majority label of their
                       vector (the rows any model must misclassify; added
                       2026-10-06, reviewer point M1)
Conflicts are computed on native labels (all rows) and on canonical labels
(rows with a canonical class only, as in the loader QC).

Usage:
  python scripts/30_audit_duplicates_conflicts.py --dataset NF-ToN-IoT-v3
  python scripts/30_audit_duplicates_conflicts.py --report
Outputs: data/audit/dupconf/<dataset>.json and
data/audit/duplicates_conflicts_report.txt.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from nfv3_config import (DATASETS, FEATURE_COLS, FEATURE_COLS_V2, PROCESSED_ROOT,
                         WORK, feature_cols, is_v2)

BATCH = 1_000_000
TOP_SETS = 10
OUT = WORK / "data/audit"
TESTBEDS = [("ToN", "NF-ToN-IoT-v2", "NF-ToN-IoT-v3"),
            ("BoT", "NF-BoT-IoT-v2", "NF-BoT-IoT-v3"),
            ("CIC", "NF-CSE-CIC-IDS2018-v2", "NF-CSE-CIC-IDS2018-v3")]


def encode(s: pd.Series, codes: dict) -> np.ndarray:
    s = s.astype(str)
    for v in s.unique():
        codes.setdefault(v, len(codes))
    return s.map(codes).to_numpy(dtype=np.int16)


def conflict_stats(h: np.ndarray, lab: np.ndarray, names: list[str]) -> dict:
    hc = pd.DataFrame({"h": h, "c": lab})
    cnt = hc.value_counts(sort=False)
    minority = int(len(h) - cnt.groupby(level=0).max().sum())
    pairs = hc.drop_duplicates()
    nlab = pairs.groupby("h", sort=False)["c"].transform("size").to_numpy()
    conf_pairs = pairs.loc[nlab > 1]
    conf_h = conf_pairs["h"].unique()
    in_conf = np.isin(h, conf_h)
    sets = {}
    if len(conf_pairs):
        cp = conf_pairs.assign(n=conf_pairs["c"].map(dict(enumerate(names)))).sort_values(["h", "n"])
        sets = cp.groupby("h", sort=False)["n"].agg(" / ".join).value_counts()
        sets = {k: int(v) for k, v in sets.head(TOP_SETS).items()}
    per_class = {}
    distinct_in_class = pairs["c"].value_counts()
    for i, name in enumerate(names):
        m = lab == i
        rows = int(m.sum())
        if not rows:
            continue
        distinct = int(distinct_in_class.get(i, 0))
        per_class[name] = {"rows": rows, "intra_dup_rows": rows - distinct,
                           "conflict_rows": int((m & in_conf).sum())}
    return {"rows": int(len(h)), "conflicting_vectors": int(len(conf_h)),
            "conflict_rows": int(in_conf.sum()), "minority_rows": minority, "top_label_sets": sets,
            "per_class": per_class}


def analyse(dataset: str) -> dict:
    t0 = time.time()
    keys = {"41": FEATURE_COLS_V2} if is_v2(dataset) else {"49": FEATURE_COLS, "41": FEATURE_COLS_V2}
    for k, cols in list(keys.items()):
        keys[str(int(k) - 1)] = [c for c in cols if c != "L4_SRC_PORT"]
    pf = pq.ParquetFile(PROCESSED_ROOT / dataset / "tabular.parquet")
    hs = {k: [] for k in keys}
    nat, can = [], []
    cn, cc = {}, {}
    for b in pf.iter_batches(batch_size=BATCH,
                             columns=feature_cols(dataset) + ["native_label", "canonical_label"]):
        df = b.to_pandas()
        for k, cols in keys.items():
            hs[k].append(pd.util.hash_pandas_object(df[cols], index=False).to_numpy())
        nat.append(encode(df["native_label"], cn))
        can.append(encode(df["canonical_label"].fillna(""), cc))
    nat, can = np.concatenate(nat), np.concatenate(can)
    nat_names = [k for k, _ in sorted(cn.items(), key=lambda x: x[1])]
    can_names = [k for k, _ in sorted(cc.items(), key=lambda x: x[1])]
    canon_mask = can != cc.get("", -1)
    out = {"dataset": dataset, "generated_utc": datetime.now(timezone.utc).isoformat(),
           "keys": {}}
    for k in keys:
        h = np.concatenate(hs[k])
        distinct = int(np.unique(h).size)
        out["keys"][k] = {
            "n_features": len(keys[k]), "rows": int(len(h)), "distinct_vectors": distinct,
            "duplicate_rows": int(len(h) - distinct),
            "native": conflict_stats(h, nat, nat_names),
            "canonical": conflict_stats(h[canon_mask], can[canon_mask], can_names),
        }
    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


def norm(name: str) -> str:
    return name.lower().replace("_", " ")


def pct(a, b):
    return f"{100 * a / b:6.2f}" if b else "     -"


def section(res: dict, k: str, kf: str) -> list[str]:
    """Dataset-level and per-class tables for key k (v3 key kf in brackets)."""
    L = []
    L.append(f"Dataset level, key {k} (v3 key {kf} in brackets):")
    L.append(f"  {'dataset':24s} {'rows':>12s} {'dup %':>7s} {'nat conf vec':>13s} {'nat conf %':>10s} "
             f"{'can conf vec':>13s} {'can conf %':>10s} {'nat min %':>9s} {'can min %':>9s}")
    for ds, r in sorted(res.items()):
        x = r["keys"][k]
        line = (f"  {ds:24s} {x['rows']:>12,} {pct(x['duplicate_rows'], x['rows']):>7s} "
                f"{x['native']['conflicting_vectors']:>13,} {pct(x['native']['conflict_rows'], x['rows']):>10s} "
                f"{x['canonical']['conflicting_vectors']:>13,} "
                f"{pct(x['canonical']['conflict_rows'], x['canonical']['rows']):>10s} "
                f"{pct(x['native']['minority_rows'], x['rows']):>9s} "
                f"{pct(x['canonical']['minority_rows'], x['canonical']['rows']):>9s}")
        if kf in r["keys"]:
            y = r["keys"][kf]
            line += (f"   [{kf}: dup {pct(y['duplicate_rows'], y['rows']).strip()} %, "
                     f"nat conf {y['native']['conflicting_vectors']:,} vec, "
                     f"{pct(y['native']['conflict_rows'], y['rows']).strip()} %, "
                     f"nat min {pct(y['native']['minority_rows'], y['rows']).strip()} %, "
                     f"can min {pct(y['canonical']['minority_rows'], y['canonical']['rows']).strip()} %]")
        L.append(line)
    for tb, v2, v3 in TESTBEDS:
        if v2 not in res or v3 not in res:
            continue
        a, b = res[v2]["keys"][k]["native"], res[v3]["keys"][k]["native"]
        L += ["", f"{tb}: per native class, key {k} (intra-class duplicates and conflict rows, % of class rows)",
              f"  {'class':26s} {'v2 rows':>11s} {'v2 dup%':>8s} {'v2 conf%':>9s} | "
              f"{'v3 rows':>11s} {'v3 dup%':>8s} {'v3 conf%':>9s}"]
        pa_ = {norm(k): v for k, v in a["per_class"].items()}
        pb_ = {norm(k): v for k, v in b["per_class"].items()}
        for c in sorted(set(pa_) | set(pb_), key=lambda c: -(pb_.get(c, {}).get("rows", 0))):
            x, y = pa_.get(c), pb_.get(c)
            fx = (f"{x['rows']:>11,} {pct(x['intra_dup_rows'], x['rows']):>8s} "
                  f"{pct(x['conflict_rows'], x['rows']):>9s}") if x else f"{'-':>11s} {'-':>8s} {'-':>9s}"
            fy = (f"{y['rows']:>11,} {pct(y['intra_dup_rows'], y['rows']):>8s} "
                  f"{pct(y['conflict_rows'], y['rows']):>9s}") if y else f"{'-':>11s} {'-':>8s} {'-':>9s}"
            L.append(f"  {c:26s} {fx} | {fy}")
        for lbl, d in (("v2", a), ("v3", b)):
            L.append(f"  top native label sets in {lbl} (vectors): " + (
                "; ".join(f"{k} {v:,}" for k, v in d["top_label_sets"].items()) or "none"))
    L.append("")
    return L


def report() -> str:
    res = {}
    for f in (OUT / "dupconf").glob("*.json"):
        r = json.loads(f.read_text())
        res[r["dataset"]] = r
    L = ["RQ1: exact duplicates and conflicting labels, NF-v2 vs NF-v3",
         f"Generated: {datetime.now(timezone.utc).isoformat()}",
         "Key: 64-bit hash of the normalised feature vector. 41 = features common to",
         "v2 and v3; 49 = full NF-v3 set; 40 and 48 = the same without L4_SRC_PORT (the",
         "features the models see; 40 is the primary key for v2 vs v3). Conflict rows = all rows of vectors that carry",
         "more than one label. Percentages are of the rows in scope: 'dup %' and",
         "'nat conf %' over all rows; 'can conf %' over the rows that have a canonical",
         "class (classes without one, such as ToN xss, mitm and ransomware, are left out).",
         "'min %' = minority rows: label differs from the majority label of the vector.", ""]
    # consistency with the loader QC (49-feature key, canonical conflicts)
    L.append("Consistency check against the loader QC (NF-v3, key 49; NF-v2, key 41):")
    for ds, r in sorted(res.items()):
        k = "41" if is_v2(ds) else "49"
        qc = json.loads((PROCESSED_ROOT / ds / "manifest.json").read_text())["qc_summary"]["duplicates_and_conflicts"]
        x = r["keys"][k]
        ok = (x["duplicate_rows"] == qc["duplicate_rows"]
              and x["canonical"]["conflicting_vectors"] == qc["conflicting_vectors"]
              and x["canonical"]["conflict_rows"] == qc["conflicting_rows"])
        L.append(f"  {ds:24s} {'OK' if ok else 'MISMATCH'} (duplicates {x['duplicate_rows']:,}; "
                 f"canonical conflicts {x['canonical']['conflicting_vectors']:,} vectors)")
    L.append("")
    for key, full in (("40", "48"), ("41", "49")):
        L += section(res, key, full)
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    (OUT / "dupconf").mkdir(parents=True, exist_ok=True)
    if a.dataset:
        r = analyse(a.dataset)
        (OUT / "dupconf" / f"{a.dataset}.json").write_text(json.dumps(r, indent=2))
        print(f"{a.dataset}: done in {r['elapsed_s']}s")
    if a.report:
        text = report()
        (OUT / "duplicates_conflicts_report.txt").write_text(text)
        print(text)


if __name__ == "__main__":
    main()
