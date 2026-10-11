"""34_shortcut_probes.py — RQ2: which simple rules separate each attack class
from benign traffic? (PLAN-PAPER-B.md, P3.5)

For each native attack class of each NF dataset (v2 and v3), decision trees of
depth 1, 2 and 3 are trained on class vs benign rows of the grouped `train`
split and evaluated on the grouped `test` split. That split is keyed on the
full feature vector of each version (49 features in NF-v3, 41 in NF-v2), so
rows identical on the probe features alone (41 in NF-v3, or fewer in sets B
and C) can fall on both sides; trees of depth 1-3 cannot exploit this
(internal audit, finding B3). The sampling seed (SEED = 2026) is fixed in
this file and not written to the JSON outputs (finding A1). Samples are
balanced: at most TRAIN_CAP rows per label in train and TEST_CAP in test, so
macro F1 measures separability independently of
class prevalence; each class is paired with the same number of benign rows
(train and test separately). Three feature sets:

  A  the 41 features common to NF-v2 and NF-v3
  B  A without the two ports
  C  B without testbed-identity features: TTL (MIN_TTL, MAX_TTL), TCP window
     (TCP_WIN_MAX_IN, TCP_WIN_MAX_OUT) and DNS fields (DNS_QUERY_ID,
     DNS_QUERY_TYPE, DNS_TTL_ANSWER)

The depth-1 rule (feature, threshold, side predicted as the class) and the
depth-2 paths are recorded. Values outside the float32 range (NF-v2 rate
columns) are clipped to it, because the trees work in float32; NaN is handled
natively by the trees.

Re-keyed variant (--rekey; Phase 4 block A, research/phase4_protocol.md §2):
set A drops L4_SRC_PORT (40 features; author decision 2026-10-02, internal
audit C1), and each set uses its own grouped split keyed on the features it
sees (seed 42), so no probe vector crosses splits. Rows are sampled per label
before the split with probability min(1, max(TRAIN_CAP / 0.70, TEST_CAP /
0.15) / n), then capped at TRAIN_CAP / TEST_CAP per label when balancing.
Outputs go to separate files; the Phase 3 outputs are not touched.

Common-split variant (--common; author decision 2026-10-05, P4.6): sets as in
--rekey (A without L4_SRC_PORT), but one grouped split for all three sets,
keyed on the 40 features of set A (seed 42), so that A, B and C are scored
on the same test rows. Near-duplicates of the smaller sets B and C may cross
that split; depth 1-3 trees cannot exploit them (internal audit, B3).

Usage:
  python scripts/34_shortcut_probes.py --dataset NF-BoT-IoT-v3 [--rekey | --common]
  python scripts/34_shortcut_probes.py --report [--rekey]
Outputs: data/audit/shortcut_probes[_rekeyed]/<dataset>.json and
data/audit/shortcut_probes[_rekeyed]_report.txt
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
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

from nfv3_config import (DATASETS, FEATURE_COLS_V2, PROCESSED_ROOT, WORK,
                         grouped_split_codes, tabular_row_hash)

BENIGN = "Benign"
SEED = 2026
BATCH = 1_000_000
TRAIN_CAP, TEST_CAP = 200_000, 100_000
PORTS = ["L4_SRC_PORT", "L4_DST_PORT"]
IDENTITY = ["MIN_TTL", "MAX_TTL", "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT",
            "DNS_QUERY_ID", "DNS_QUERY_TYPE", "DNS_TTL_ANSWER"]
SETS = {"A": FEATURE_COLS_V2,
        "B": [c for c in FEATURE_COLS_V2 if c not in PORTS],
        "C": [c for c in FEATURE_COLS_V2 if c not in PORTS + IDENTITY]}
SETS_REKEY = {"A": [c for c in FEATURE_COLS_V2 if c != "L4_SRC_PORT"],
              "B": SETS["B"], "C": SETS["C"]}
REKEY_SEED = 42
F32 = float(np.finfo(np.float32).max)
OUT = WORK / "data/audit"
REKEY = False  # set by --rekey or --common
COMMON = False  # set by --common


def sets():
    return SETS_REKEY if REKEY else SETS


def out_tag():
    if COMMON:
        return "shortcut_probes_common"
    return "shortcut_probes_rekeyed" if REKEY else "shortcut_probes"


def sample_rekey(dataset):
    """Per-label Bernoulli sample over all rows, then one grouped split per
    feature set, keyed on that set (columns split_A, split_B, split_C)."""
    path = PROCESSED_ROOT / dataset / "tabular.parquet"
    pf = pq.ParquetFile(path)
    counts = Counter()
    for b in pf.iter_batches(batch_size=BATCH, columns=["native_label"]):
        counts.update(b.column("native_label").to_pandas().astype(str))
    target = max(TRAIN_CAP / 0.70, TEST_CAP / 0.15)
    prob = {k: min(1.0, target / n) for k, n in counts.items()}
    rng = np.random.default_rng(SEED)
    names = {0: "train", 1: "val", 2: "test"}
    parts = []
    for b in pf.iter_batches(batch_size=BATCH, columns=FEATURE_COLS_V2 + ["native_label"]):
        df = b.to_pandas()
        lab = df["native_label"].astype(str)
        keep = rng.random(len(df)) < lab.map(prob).to_numpy()
        sub = df.loc[keep].copy()
        sub["native_label"] = lab[keep].to_numpy()
        for k, cols in SETS_REKEY.items():
            codes = grouped_split_codes(tabular_row_hash(sub, cols), REKEY_SEED)
            sub[f"split_{k}"] = pd.Series(codes).map(names).to_numpy()
        parts.append(sub)
    data = pd.concat(parts, ignore_index=True)
    X = data[FEATURE_COLS_V2].to_numpy(dtype=np.float64)
    X = np.where(np.isfinite(X), np.clip(X, -F32, F32), np.nan)
    data[FEATURE_COLS_V2] = X
    return data, {(lab, "all"): n for lab, n in counts.items()}


def sample(dataset):
    """Two passes over tabular.parquet: counts per (label, split), then a
    seeded Bernoulli sample reaching about the caps."""
    path = PROCESSED_ROOT / dataset / "tabular.parquet"
    pf = pq.ParquetFile(path)
    counts = Counter()
    for b in pf.iter_batches(batch_size=BATCH, columns=["native_label", "split"]):
        df = b.to_pandas()
        counts.update(zip(df["native_label"].astype(str), df["split"].astype(str)))
    cap = {"train": TRAIN_CAP, "test": TEST_CAP}
    prob = {k: min(1.0, cap.get(k[1], 0) / n) for k, n in counts.items()}
    rng = np.random.default_rng(SEED)
    parts = []
    for b in pf.iter_batches(batch_size=BATCH, columns=FEATURE_COLS_V2 + ["native_label", "split"]):
        df = b.to_pandas()
        lab, sp = df["native_label"].astype(str), df["split"].astype(str)
        p = np.array([prob[k] for k in zip(lab, sp)])
        keep = (rng.random(len(df)) < p) & sp.isin(["train", "test"]).to_numpy()
        sub = df.loc[keep].copy()
        sub["native_label"], sub["split"] = lab[keep].to_numpy(), sp[keep].to_numpy()
        parts.append(sub)
    data = pd.concat(parts, ignore_index=True)
    X = data[FEATURE_COLS_V2].to_numpy(dtype=np.float64)
    X = np.where(np.isfinite(X), np.clip(X, -F32, F32), np.nan)
    data[FEATURE_COLS_V2] = X
    return data, counts


def fmt(x):
    return str(int(x)) if float(x).is_integer() else f"{x:.6g}"


def rules(tree, feats, max_depth):
    """Paths to leaves that predict the class (label 1), with leaf support."""
    t = tree.tree_
    out = []

    def walk(node, path):
        if t.children_left[node] == -1:
            v = t.value[node][0]
            if v.argmax() == 1:
                frac = v[1] / v.sum()
                out.append({"path": " and ".join(path) or "(all)",
                            "class_share": round(float(frac), 4),
                            "samples": int(t.n_node_samples[node])})
            return
        f, thr = feats[t.feature[node]], t.threshold[node]
        walk(t.children_left[node], path + [f"{f} <= {fmt(thr)}"])
        walk(t.children_right[node], path + [f"{f} > {fmt(thr)}"])

    walk(0, [])
    return sorted(out, key=lambda r: -r["samples"])


def balance(d, cls, rng, cap=None):
    """Equal numbers of class and benign rows (the smaller side sets the size,
    at most cap)."""
    c, b = d[d["native_label"] == cls], d[d["native_label"] == BENIGN]
    n = min(len(c), len(b), cap or len(c))
    c = c.iloc[rng.choice(len(c), size=n, replace=False)]
    b = b.iloc[rng.choice(len(b), size=n, replace=False)]
    return pd.concat([b, c])


def probe(data, cls):
    d = data[data["native_label"].isin([BENIGN, cls])]
    res = {"sets": {}}
    for name, feats in sets().items():
        col = "split_A" if COMMON else f"split_{name}" if REKEY else "split"
        cap = (TRAIN_CAP, TEST_CAP) if REKEY else (None, None)
        rng = np.random.default_rng(SEED)
        tr = balance(d[d[col] == "train"], cls, rng, cap[0])
        te = balance(d[d[col] == "test"], cls, rng, cap[1])
        ytr = (tr["native_label"] == cls).to_numpy().astype(int)
        yte = (te["native_label"] == cls).to_numpy().astype(int)
        rows = {"train_rows": {"benign": int((ytr == 0).sum()), "class": int(ytr.sum())},
                "test_rows": {"benign": int((yte == 0).sum()), "class": int(yte.sum())}}
        if REKEY:
            rows_by_set = res.setdefault("rows_by_set", {})
            rows_by_set[name] = rows
        else:
            res.update(rows)
        if ytr.sum() < 10 or yte.sum() < 10:
            res["note"] = "too few class rows in train or test"
            res["sets"] = {}
            return res
        r = {}
        for depth in (1, 2, 3):
            m = DecisionTreeClassifier(max_depth=depth, random_state=SEED)
            m.fit(tr[feats].to_numpy(), ytr)
            r[f"f1_d{depth}"] = float(f1_score(yte, m.predict(te[feats].to_numpy()), average="macro"))
            if depth <= 2:
                r[f"rules_d{depth}"] = rules(m, feats, depth)[:3]
        res["sets"][name] = r
    return res


def run(dataset):
    t0 = time.time()
    data, counts = sample_rekey(dataset) if REKEY else sample(dataset)
    classes = sorted({lab for lab, _ in counts} - {BENIGN})
    out = {"dataset": dataset, "generated_utc": datetime.now(timezone.utc).isoformat(),
           "caps": {"train": TRAIN_CAP, "test": TEST_CAP},
           "feature_sets": sets(), "classes": {}}
    for c in classes:
        r = probe(data, c)
        out["classes"][c] = r
        if "note" in r:
            print(f"  {c:28s} {r['note']}")
            continue
        s = r["sets"]
        top = s["A"]["rules_d1"][0]["path"] if s["A"]["rules_d1"] else "-"
        print(f"  {c:28s} d1 A/B/C {s['A']['f1_d1']:.2f} {s['B']['f1_d1']:.2f} {s['C']['f1_d1']:.2f}"
              f"  d3 A/B/C {s['A']['f1_d3']:.2f} {s['B']['f1_d3']:.2f} {s['C']['f1_d3']:.2f}  [{top}]", flush=True)
    out["elapsed_s"] = round(time.time() - t0, 1)
    if REKEY:
        out["split"] = (f"one grouped split keyed on set A (40 features), seed {REKEY_SEED}" if COMMON
                        else f"grouped per feature set, seed {REKEY_SEED}")
    d = OUT / out_tag()
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{dataset}.json").write_text(json.dumps(out, indent=2))
    print(f"{dataset}: done in {out['elapsed_s']}s")


def report():
    res = {}
    for f in sorted((OUT / out_tag()).glob("*.json")):
        r = json.loads(f.read_text())
        res[r["dataset"]] = r
    a_desc = ("A = 40 features (41 common without L4_SRC_PORT); one grouped split for all sets,\n"
              "keyed on the 40 features of A (seed 42)") if COMMON else (
              "A = 40 features (41 common without L4_SRC_PORT); each set has its own grouped split\n"
              "keyed on its features (seed 42)") if REKEY else "A = 41 features"
    L = ["RQ2: shallow decision trees, attack class vs benign (grouped split, balanced samples)",
         f"Generated: {datetime.now(timezone.utc).isoformat()}",
         f"Macro F1 on the grouped test split. Sets: {a_desc}; B = A without ports;",
         "C = B without TTL, TCP window and DNS fields. Rule = depth-1 split predicting the class.", ""]
    summary = {}
    for ds, r in res.items():
        L.append(f"== {ds}")
        L.append(f"  {'class':28s} {'d1 A':>5s} {'d1 B':>5s} {'d1 C':>5s} {'d3 A':>5s} {'d3 C':>5s}  depth-1 rule A | depth-1 rule C")
        for c, x in sorted(r["classes"].items()):
            if "note" in x:
                L.append(f"  {c:28s} {x['note']}")
                continue
            s = x["sets"]
            ra = s["A"]["rules_d1"][0]["path"] if s["A"]["rules_d1"] else "-"
            rc = s["C"]["rules_d1"][0]["path"] if s["C"]["rules_d1"] else "-"
            L.append(f"  {c:28s} {s['A']['f1_d1']:5.2f} {s['B']['f1_d1']:5.2f} {s['C']['f1_d1']:5.2f} "
                     f"{s['A']['f1_d3']:5.2f} {s['C']['f1_d3']:5.2f}  {ra} | {rc}")
            for k in ("A", "C"):
                for d in (1, 3):
                    summary.setdefault((ds.endswith("-v2"), k, d), []).append(s[k][f"f1_d{d}"])
            feat = ra.split(" ")[0]
            summary.setdefault(("feat", ds.endswith("-v2")), Counter())[feat] += 1
        L.append("")
    L.append("Summary (classes with macro F1 >= 0.95 / classes probed):")
    for v2 in (True, False):
        tag = "NF-v2" if v2 else "NF-v3"
        parts = []
        for k in ("A", "C"):
            for d in (1, 3):
                xs = summary.get((v2, k, d), [])
                parts.append(f"{k} depth {d}: {sum(x >= 0.95 for x in xs)}/{len(xs)}")
        L.append(f"  {tag}: " + "; ".join(parts))
        L.append(f"  {tag} depth-1 features (set A): " + ", ".join(
            f"{f} {n}" for f, n in summary.get(("feat", v2), Counter()).most_common()))
    text = "\n".join(L) + "\n"
    (OUT / f"{out_tag()}_report.txt").write_text(text)
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--rekey", action="store_true", help="Phase 4 re-keyed variant")
    ap.add_argument("--common", action="store_true", help="re-keyed sets, one split keyed on set A")
    a = ap.parse_args()
    global REKEY, COMMON
    REKEY = a.rekey or a.common
    COMMON = a.common
    if a.dataset:
        run(a.dataset)
    if a.report:
        report()


if __name__ == "__main__":
    main()
