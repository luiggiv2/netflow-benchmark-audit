"""42_in_domain.py — P4.2 (block B of research/phase4_protocol.md): in-domain
evaluation under the Phase 4 conditions.

For one dataset, every condition × task × model × seed is trained on a
capped sample of the train split and evaluated on the whole test split.

Conditions (protocol §5; feature sets from nfv3_config.model_feature_sets):
  G0    grouped split on the base set (no L4_SRC_PORT), data as published
  R0    row-level random 70/15/15 split, base set
  G-NC  G0 without conflicting rows (identical base vector, more than one
        label of the task), removed from train and test before the split
  G-NP  grouped split keyed on, and trained with, base without L4_DST_PORT
  G-NI  same with base without ports, TTL, TCP window and DNS fields
  G-ND  G0 after reducing release-level duplicate records (identical in all
        55 raw columns, IPs, ports and timestamps included) to one copy (NF-v3)
  G-TS  base plus both absolute timestamps, split keyed on the 50 features
        (NF-v3); timestamps enter the model shifted by the dataset minimum
        (as Luay et al. 2026) so float32 keeps sub-second resolution
Tasks: BIN (Label: benign vs every attack row) and CAN (canonical classes;
rows outside the canonical taxonomy dropped). Models: XGBoost and random
forest with library defaults (author decision 1, §11), except XGBoost
max_delta_step=1 (decision of 2026-10-06, see make_model). Seeds 1-5: each seed
re-draws the split (grouped: seed constant; random: generator), the training
sample and the model seed.

Training sample (decision 2): from the train split, every row of a canonical
class (rows outside the taxonomy form one group) with up to CAP rows, CAP
random rows of larger groups. The validation split is not used.

Metrics on the whole test split: macro-F1 and per-class F1 (BIN also the F1
of the attack class); per native class, the share of rows predicted
correctly; a 95 % CI per run from a Poisson bootstrap over test vector groups
(rows sharing the condition's feature vector are resampled together, B
resamples; groups with the same label-by-prediction counts are pooled, which
is exact for the Poisson bootstrap); a deduplicated view (each distinct test
vector once, majority label); the conflict ceiling (score of predicting the
majority label of every test vector group). Native classes with no test or
no training rows under a grouped split are flagged as degenerate (decision 8).

Usage:
  python scripts/42_in_domain.py --dataset NF-BoT-IoT-v3 [--seeds 1] [--tag smoke]
  python scripts/42_in_domain.py --report [--tag smoke]
Outputs: data/eval/in_domain[_<tag>]/<dataset>.jsonl (one line per run) and
data/eval/in_domain[_<tag>]_report.txt; log in results/eval/.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from nfv3_config import (DATASETS, PROCESSED_ROOT, TIMESTAMP_COLS, WORK, grouped_split_codes,
                         is_v2, model_feature_sets, random_split_codes, tabular_row_hash)

CAP = 300_000
SEEDS = [1, 2, 3, 4, 5]
B = 1000
BATCH = 1_000_000
CHUNK = 1_000_000
CLIP = 1e30
NJOBS = os.cpu_count()
EXCLUDED = "(excluded)"
RAW_EXCLUDE = {"native_label", "canonical_label", "split"}
OUT = WORK / "data/eval"


def log(msg):
    rss = psutil.Process().memory_info().rss / 2**30
    print(f"{datetime.now().strftime('%H:%M:%S')} [{rss:4.1f} GB] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load(dataset):
    """Feature matrix (float32, all columns of every key), row hashes per key,
    labels, and the release-level duplicate mask (NF-v3)."""
    sets = model_feature_sets(dataset)
    v3 = not is_v2(dataset)
    tab_cols = sets["base"]
    cols = tab_cols + (TIMESTAMP_COLS if v3 else [])
    tab = pq.ParquetFile(PROCESSED_ROOT / dataset / "tabular.parquet")
    n = tab.metadata.num_rows
    X = np.empty((n, len(cols)), dtype=np.float32)
    ts = None
    if v3:
        ts = pq.read_table(PROCESSED_ROOT / dataset / "data.parquet", columns=TIMESTAMP_COLS + ["Label"])
        ts = ts.to_pandas()
    hashes = {k: np.empty(n, dtype=np.uint64) for k in sets}
    label, canon, native = [], [], []
    canon_ids, native_ids = {}, {}
    off = 0
    for b in tab.iter_batches(batch_size=BATCH, columns=tab_cols + ["Label", "native_label", "canonical_label"]):
        df = b.to_pandas()
        sl = slice(off, off + len(df))
        if v3:
            if not (ts["Label"].to_numpy()[sl] == df["Label"].to_numpy()).all():
                raise RuntimeError("data.parquet and tabular.parquet rows are not aligned")
            for c in TIMESTAMP_COLS:
                df[c] = ts[c].to_numpy()[sl]
        for k, kc in sets.items():
            hashes[k][sl] = tabular_row_hash(df, kc)
        A = df[cols].to_numpy(dtype=np.float64)
        if v3:
            A[:, -2:] -= float(ts[TIMESTAMP_COLS[0]].min())
        X[sl] = np.where(np.isfinite(A), np.clip(A, -CLIP, CLIP), np.nan)
        label.append(df["Label"].to_numpy().astype(np.int8))
        c = df["canonical_label"].astype(object).fillna("").astype(str).replace("", EXCLUDED)
        canon.append(encode(c, canon_ids))
        native.append(encode(df["native_label"].astype(str), native_ids))
        off += len(df)
    del ts
    data = {"X": X, "cols": cols, "sets": sets, "hashes": hashes, "label": np.concatenate(label),
            "canon": np.concatenate(canon), "native": np.concatenate(native),
            "canon_names": list(canon_ids), "native_names": list(native_ids)}
    if v3:
        data["dup"] = release_duplicates(dataset, n)
    return data


def encode(series, ids):
    """Map strings to int8 codes, extending ids (insertion order) as needed."""
    for v in series.unique():
        ids.setdefault(v, len(ids))
    return series.map(ids).to_numpy().astype(np.int8)


def release_duplicates(dataset, n):
    """True for every copy of a raw record after its first occurrence."""
    pf = pq.ParquetFile(PROCESSED_ROOT / dataset / "data.parquet")
    raw = [c for c in pf.schema_arrow.names if c not in RAW_EXCLUDE]
    h = np.empty(n, dtype=np.uint64)
    off = 0
    for b in pf.iter_batches(batch_size=BATCH, columns=raw):
        df = b.to_pandas()
        h[off:off + len(df)] = pd.util.hash_pandas_object(df, index=False).to_numpy()
        off += len(df)
    _, first = np.unique(h, return_index=True)
    dup = np.ones(n, dtype=bool)
    dup[first] = False
    return dup


def conflict_mask(h, y, keep):
    """Rows (within keep) whose vector carries more than one label of y."""
    idx = np.flatnonzero(keep)
    hh, yy = h[idx], y[idx]
    order = np.lexsort((yy, hh))
    hs, ys = hh[order], yy[order]
    new_pair = np.ones(len(hs), dtype=bool)
    new_pair[1:] = (hs[1:] != hs[:-1]) | (ys[1:] != ys[:-1])
    uh, inv = np.unique(hs, return_inverse=True)
    n_labels = np.bincount(inv, weights=new_pair)
    out = np.zeros(len(h), dtype=bool)
    out[idx[order]] = n_labels[inv] > 1
    return out


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def f1_from_confusion(C):
    """Per-class F1 from confusion counts C[..., true, pred]."""
    tp = np.diagonal(C, axis1=-2, axis2=-1)
    p, t = C.sum(-2), C.sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(p + t > 0, 2 * tp / (p + t), np.nan)
    return f1


def group_metrics(h, y, yhat, k, rng):
    """Bootstrap CI over vector groups, deduplicated view and conflict ceiling."""
    uh, g = np.unique(h, return_inverse=True)
    G = len(uh)
    # per-group label counts and the group's prediction (constant within a group)
    lab_counts = np.zeros((G, k), dtype=np.int64)
    np.add.at(lab_counts, (g, y), 1)
    pred = np.zeros(G, dtype=np.int64)
    pred[g] = yhat
    # pool groups with the same (label counts, prediction): Poisson weights add up
    key = np.concatenate([lab_counts, pred[:, None]], axis=1)
    types, mult = np.unique(key, axis=0, return_counts=True)
    tc, tp_ = types[:, :k], types[:, k]
    contrib = np.zeros((len(types), k, k))
    contrib[np.arange(len(types))[:, None], np.arange(k)[None, :], tp_[:, None]] = tc
    contrib = contrib.reshape(len(types), k * k)
    present = lab_counts.sum(0) > 0  # classes with test rows, as in the macro-F1
    scores = []
    for start in range(0, B, 100):
        w = rng.poisson(mult, size=(min(100, B - start), len(types)))
        C = (w @ contrib).reshape(-1, k, k)
        scores.append(np.nanmean(f1_from_confusion(C)[:, present], axis=1))
    scores = np.concatenate(scores)
    maj = lab_counts.argmax(1)
    dedup = f1_score(maj, pred, average="macro", labels=np.unique(maj))
    ceil_C = np.zeros((k, k))
    np.add.at(ceil_C, (np.repeat(np.arange(k)[None, :], G, 0).ravel(),
                       np.repeat(maj[:, None], k, 1).ravel()), lab_counts.ravel())
    ceiling = float(np.nanmean(f1_from_confusion(ceil_C)[present]))
    return {"ci95": [round(float(np.percentile(scores, 2.5)), 5), round(float(np.percentile(scores, 97.5)), 5)],
            "test_groups": int(G), "dedup_macro_f1": round(float(dedup), 5),
            "conflict_ceiling_macro_f1": round(ceiling, 5),
            "test_rows_in_conflict_groups": int(lab_counts[(lab_counts > 0).sum(1) > 1].sum())}


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #

def conditions(v3):
    c = {"G0": ("base", "grouped", None), "R0": ("base", "random", None),
         "G-NC": ("base", "grouped", "noconflict"), "G-NP": ("noport", "grouped", None),
         "G-NI": ("noid", "grouped", None)}
    if v3:
        c |= {"G-ND": ("base", "grouped", "nodup"), "G-TS": ("ts", "grouped", None)}
    return c


def make_model(name, seed):
    if name == "xgb":
        # max_delta_step=1 (author decision 2026-10-06, reviewer point M4): with library
        # defaults the multiclass fit diverged on NF-CSE-CIC-IDS2018-v3 CAN (training
        # mlogloss 0.027 at round 49, 30.2 at round 99); default-setting runs are kept in
        # data/eval/*_xgbdefault as a sensitivity analysis
        return xgb.XGBClassifier(tree_method="hist", max_delta_step=1, n_jobs=NJOBS, random_state=seed)
    return RandomForestClassifier(n_jobs=NJOBS, random_state=seed)


def train_sample(split, canon, keep, rng):
    take = np.zeros(len(split), dtype=bool)
    tr = keep & (split == 0)
    for c in np.unique(canon[tr]):
        idx = np.flatnonzero(tr & (canon == c))
        take[idx if len(idx) <= CAP else rng.choice(idx, CAP, replace=False)] = True
    return take


def run(dataset, seeds, tag):
    t0 = time.time()
    log(f"{dataset}: loading")
    d = load(dataset)
    n = len(d["label"])
    X, cols, sets = d["X"], d["cols"], d["sets"]
    excl = d["canon_names"].index(EXCLUDED) if EXCLUDED in d["canon_names"] else -1
    can_names = sorted(c for c in d["canon_names"] if c != EXCLUDED)
    can_map = np.full(len(d["canon_names"]), -1, dtype=np.int8)
    for i, c in enumerate(d["canon_names"]):
        if c != EXCLUDED:
            can_map[i] = can_names.index(c)
    y_can = can_map[d["canon"]]
    log(f"{dataset}: loaded {n:,} rows in {time.time() - t0:.0f}s")

    out_dir = OUT / (f"in_domain_{tag}" if tag else "in_domain")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset}.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["condition"], r["task"], r["model"], r["seed"]))

    for cond, (key, kind, filt) in conditions(not is_v2(dataset)).items():
        feat = sets[key]
        fidx = [cols.index(c) for c in feat]
        h = d["hashes"][key]
        for task in ("BIN", "CAN"):
            y = d["label"].astype(np.int64) if task == "BIN" else y_can.astype(np.int64)
            names = ["Benign", "Attack"] if task == "BIN" else can_names
            keep = np.ones(n, dtype=bool) if task == "BIN" else (d["canon"] != excl)
            info = {}
            if filt == "noconflict":
                cm = conflict_mask(h, y, keep)
                info["conflict_rows_removed"] = int(cm.sum())
                keep &= ~cm
            elif filt == "nodup":
                info["duplicate_records_removed"] = int((d["dup"] & keep).sum())
                keep &= ~d["dup"]
            for seed in seeds:
                for model_name in ("xgb", "rf"):
                    if (cond, task, model_name, seed) in done:
                        continue
                    t1 = time.time()
                    split = grouped_split_codes(h, seed) if kind == "grouped" else random_split_codes(n, seed)
                    rng = np.random.default_rng(seed)
                    take = train_sample(split, d["canon"], keep, rng)
                    test = keep & (split == 2)
                    present = np.unique(y[take])
                    enc = np.full(len(names), -1)
                    enc[present] = np.arange(len(present))
                    model = make_model(model_name, seed)
                    model.fit(X[np.ix_(np.flatnonzero(take), fidx)], enc[y[take]])
                    ti = np.flatnonzero(test)
                    yhat = np.concatenate([present[model.predict(X[ti[i:i + CHUNK]][:, fidx])]
                                           for i in range(0, len(ti), CHUNK)])
                    yt = y[ti]
                    labels = np.arange(len(names))
                    per = f1_score(yt, yhat, labels=labels, average=None, zero_division=0)
                    in_test = np.bincount(yt, minlength=len(names)) > 0
                    res = {"dataset": dataset, "condition": cond, "task": task, "model": model_name,
                           "seed": seed, "features": len(feat), "split": kind, **info,
                           "train_rows": int(take.sum()), "test_rows": int(len(ti)),
                           "classes_in_train": [names[i] for i in present],
                           "macro_f1": round(float(per[in_test].mean()), 5),
                           "per_class_f1": {names[i]: (round(float(per[i]), 5) if in_test[i] else None)
                                            for i in labels}}
                    if task == "BIN":
                        res["attack_f1"] = round(float(per[1]), 5)
                    res |= group_metrics(h[ti], yt, yhat, len(names), np.random.default_rng(seed))
                    nat, nat_tr = d["native"][ti], d["native"][take]
                    res["native"] = {}
                    for j, nm in enumerate(d["native_names"]):
                        m = nat == j
                        n_tr = int((nat_tr == j).sum())
                        if task == "CAN" and not (keep & (d["native"] == j)).any():
                            continue
                        res["native"][nm] = {"test_rows": int(m.sum()), "train_rows": n_tr,
                                             "correct_share": (round(float((yhat[m] == yt[m]).mean()), 5)
                                                               if m.any() else None),
                                             "degenerate": bool(kind == "grouped" and (m.sum() == 0 or n_tr == 0))}
                    if kind == "random":
                        tr_h = np.unique(h[keep & (split == 0)])
                        res["test_rows_with_vector_in_train"] = round(float(np.isin(h[ti], tr_h).mean()), 5)
                    res["seconds"] = round(time.time() - t1, 1)
                    with out_path.open("a") as f:
                        f.write(json.dumps(res) + "\n")
                    log(f"{cond:5s} {task} {model_name:3s} s{seed}  macro-F1 {res['macro_f1']:.4f} "
                        f"CI {res['ci95']}  train {res['train_rows']:,} test {res['test_rows']:,}  {res['seconds']}s")
                    del model
    log(f"{dataset}: done in {time.time() - t0:.0f}s")


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def report(tag):
    """Median and range over seeds (protocol §12). Per-seed differences against
    G0 are paired only for G-NC and G-ND, which share the base key and seed with
    G0 (same assignment of the remaining vectors); the other conditions change
    the split and are compared through their seed distributions only."""
    out_dir = OUT / (f"in_domain_{tag}" if tag else "in_domain")
    rows = []
    for f in sorted(out_dir.glob("*.jsonl")):
        rows += [json.loads(x) for x in f.read_text().splitlines()]
    agg = defaultdict(list)
    for r in rows:
        agg[(r["dataset"], r["task"], r["model"], r["condition"])].append(r)
    paired = ("G-NC", "G-ND")
    L = ["P4.2 in-domain evaluation (research/phase4_protocol.md, block B; report format of §12)",
         f"Generated: {datetime.now(timezone.utc).isoformat()}",
         "Macro-F1 on the whole test split: median [min, max] over seeds (n). paired = median [min, max]",
         "of the per-seed difference with G0 in points, only for G-NC and G-ND (same key and seed as G0).",
         "Conditions with another key or split (R0, G-NP, G-NI, G-TS) change the test set: compare their",
         "distributions, not differences. dedup = deduplicated view, ceiling = conflict ceiling (medians).",
         "BIN also shows the attack-class F1 (median). G-TS keys on the timestamps and behaves almost like a",
         "random split: compare it with R0.", ""]
    order = ["G0", "R0", "G-NC", "G-ND", "G-NP", "G-NI", "G-TS"]

    def mm(xs, scale=1.0, fmt="{:.4f}"):
        xs = np.asarray(xs) * scale
        return f"{fmt.format(np.median(xs))} [{fmt.format(xs.min())}, {fmt.format(xs.max())}]"

    for ds in DATASETS:
        if not any(k[0] == ds for k in agg):
            continue
        L.append(f"== {ds}")
        for task in ("BIN", "CAN"):
            for model in ("xgb", "rf"):
                g0 = {r["seed"]: r["macro_f1"] for r in agg.get((ds, task, model, "G0"), [])}
                L.append(f"  {task} {model}: {'cond':5s} {'n':>2s} {'macro-F1 median [min, max]':>27s} "
                         f"{'paired vs G0 (points)':>26s} {'dedup':>6s} {'ceiling':>7s}"
                         + (f" {'attackF1':>8s}" if task == "BIN" else ""))
                for c in order:
                    rs = agg.get((ds, task, model, c))
                    if not rs:
                        continue
                    pr = (mm([r["macro_f1"] - g0[r["seed"]] for r in rs], 100, "{:+.2f}")
                          if c in paired and g0 else "")
                    line = (f"           {c:5s} {len(rs):2d} {mm([r['macro_f1'] for r in rs]):>27s} {pr:>26s} "
                            f"{np.median([r['dedup_macro_f1'] for r in rs]):6.3f} "
                            f"{np.median([r['conflict_ceiling_macro_f1'] for r in rs]):7.3f}")
                    if task == "BIN":
                        line += f" {np.median([r['attack_f1'] for r in rs]):8.4f}"
                    L.append(line)
        for c in ("R0", "G-NC", "G-ND"):
            rs = agg.get((ds, "CAN", "xgb", c), [])
            if not rs:
                continue
            r = rs[0]
            extra = {k: r[k] for k in ("test_rows_with_vector_in_train", "conflict_rows_removed",
                                       "duplicate_records_removed") if k in r}
            L.append(f"  {c} (CAN, seed {r['seed']}): {extra}")
        degen = sorted({nm for (dd, _, _, c), rs in agg.items() if dd == ds
                        for r in rs for nm, v in r["native"].items() if v["degenerate"]})
        if degen:
            L.append(f"  degenerate native classes in some grouped run: {', '.join(degen)}")
        L.append("")
    text = "\n".join(L) + "\n"
    (OUT / f"{out_dir.name}_report.txt").write_text(text)
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--tag", default="")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.dataset:
        run(a.dataset, a.seeds, a.tag)
    if a.report:
        report(a.tag)


if __name__ == "__main__":
    main()
