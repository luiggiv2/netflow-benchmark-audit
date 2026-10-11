"""44_cross_dataset.py — P4.3 (block C of research/phase4_protocol.md, design
in §14): cross-dataset evaluation within each NF version.

Step 1, cache (--cache DATASET): the loader of 42_in_domain.py is run once per
dataset and its float32 matrix, row hashes and label codes are saved under
results/eval/cache/<dataset>/ (ignored by git). Later steps read the matrix
memory-mapped, in chunks, so three datasets fit in 16 GB. The values are the
ones P4.2 used.

Step 2, check (--check): a P4.2 run (NF-BoT-IoT-v3, G0, BIN, XGBoost, seed 1)
is recomputed from the cache with the same functions; its macro-F1 must equal
the stored one exactly.

Step 3, runs (--source DATASET): for every condition (G0, R0, G-NC, G-NP,
G-NI), task (BIN; CORE = Benign, DoS, DDoS) and seed (1-5), XGBoost and RF are
trained on the capped sample of the source train split (as in P4.2) and
scored on (a) the source test split (in-domain reference, so Δ is paired by
run) and (b) all rows of each other dataset of the same version. For G-NC the
source conflicts are removed before training and each target is scored as
published and without its own conflicts. Extended pairs (exploratory): CAN
models under G0, scored on the target rows of the canonical classes shared
with the source. Metrics per target: macro-F1, per-class F1 (BIN also the
attack F1), Poisson group-bootstrap CI over target vector groups, the
deduplicated view and the target conflict ceiling.

Split-controlled conditions (second reviewer report, point N3; author
authorisation 2026-10-08): G-NP0 and G-NI0 train without the destination port
or without the identity features, as G-NP and G-NI do, but on the grouped split
of G0 (keyed on the base features), so their training and test rows are those
of G0 and only the columns change. Run with --conditions G-NP0 G-NI0 and a tag;
the stored P4.3 results are not touched.

G-NIS0 (third reviewer report, major issue 1; author authorisation
2026-10-10): as G-NI0, and also without the service fields L7_PROTO,
ICMP_TYPE and ICMP_IPV4_TYPE (nfv3_config.SERVICE_COLS). Its feature set
("noidsvc") is not in the cache: the row hashes of that set are computed once
from tabular.parquet with the same function as the cached sets and saved as
results/eval/cache/<dataset>/hash_noidsvc.npy; the same pass recomputes the
"noid" hashes and stops unless they equal the cached ones (row alignment).

Usage:
  python scripts/44_cross_dataset.py --cache NF-BoT-IoT-v3
  python scripts/44_cross_dataset.py --check
  python scripts/44_cross_dataset.py --source NF-BoT-IoT-v3 [--seeds 1] [--tag smoke]
  python scripts/44_cross_dataset.py --source NF-BoT-IoT-v3 --conditions G-NP0 G-NI0 --tag g0split
  python scripts/44_cross_dataset.py --source NF-BoT-IoT-v3 --conditions G-NIS0 --tag g0split_nosvc
  python scripts/44_cross_dataset.py --report [--tag smoke]
Outputs: data/eval/cross[_<tag>]/<source>.jsonl and data/eval/cross[_<tag>]_report.txt
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import f1_score

from nfv3_config import (DATASETS, PROCESSED_ROOT, SERVICE_COLS, WORK, grouped_split_codes, is_v2,
                         random_split_codes, tabular_row_hash)

SEEDS = [1, 2, 3, 4, 5]
CORE = ["Benign", "DoS", "DDoS"]
CHUNK = 1_000_000
B = 1000
CACHE = WORK / "results/eval/cache"
OUT = WORK / "data/eval"

_spec = importlib.util.spec_from_file_location("in_domain", WORK / "scripts/42_in_domain.py")
P42 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P42)
log = P42.log
EXCLUDED = P42.EXCLUDED


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #

def build_cache(dataset):
    t0 = time.time()
    d = P42.load(dataset)
    out = CACHE / dataset
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "X.npy", d["X"])
    for k, h in d["hashes"].items():
        np.save(out / f"hash_{k}.npy", h)
    for k in ("label", "canon", "native"):
        np.save(out / f"{k}.npy", d[k])
    meta = {"cols": d["cols"], "sets": d["sets"], "canon_names": d["canon_names"],
            "native_names": d["native_names"], "rows": int(len(d["label"]))}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    log(f"{dataset}: cache written in {time.time() - t0:.0f}s")


def open_cache(dataset):
    c = CACHE / dataset
    meta = json.loads((c / "meta.json").read_text())
    d = {"X": np.load(c / "X.npy", mmap_mode="r"), **meta}
    d["hashes"] = {k: np.load(c / f"hash_{k}.npy") for k in meta["sets"]}
    for k in ("label", "canon", "native"):
        d[k] = np.load(c / f"{k}.npy")
    return d


# --------------------------------------------------------------------------- #
# Labels, groups and metrics
# --------------------------------------------------------------------------- #

def task_labels(d, task, names=None):
    """(y, keep, class names) for a task; y is -1 outside keep."""
    if task == "BIN":
        return d["label"].astype(np.int64), np.ones(len(d["label"]), dtype=bool), ["Benign", "Attack"]
    canon = np.array(d["canon_names"], dtype=object)[d["canon"]]
    names = names or (CORE if task == "CORE" else sorted(c for c in d["canon_names"] if c != EXCLUDED))
    idx = {c: i for i, c in enumerate(names)}
    y = np.array([idx.get(c, -1) for c in d["canon_names"]], dtype=np.int64)[d["canon"]]
    del canon
    return y, y >= 0, names


def predict(model, X, rows, fidx, present):
    out = np.empty(len(rows), dtype=np.int64)
    for i in range(0, len(rows), CHUNK):
        r = rows[i:i + CHUNK]
        block = X[r[0]:r[-1] + 1] if len(r) and r[-1] - r[0] + 1 == len(r) else X[r]
        out[i:i + CHUNK] = present[model.predict(np.ascontiguousarray(block[:, fidx]))]
    return out


def predict_table(model, X, present):
    return np.concatenate([present[model.predict(X[i:i + CHUNK])] for i in range(0, len(X), CHUNK)])


def group_metrics(g, y, yhat, k, rng):
    """Bootstrap CI over vector groups, dedup view and conflict ceiling.
    g: group index per row (0..G-1). Groups with one label are pooled by
    (label, prediction, size); the few multi-label groups are kept as is."""
    G = int(g.max()) + 1
    size = np.bincount(g, minlength=G)
    lab_counts = np.bincount(g * k + y, minlength=G * k).reshape(G, k)
    pred = np.zeros(G, dtype=np.int64)
    pred[g] = yhat
    used = size > 0
    n_lab = (lab_counts > 0).sum(1)
    present = lab_counts.sum(0) > 0
    single = used & (n_lab == 1)
    lab1 = lab_counts[single].argmax(1)
    code = (size[single].astype(np.int64) * k + lab1) * k + pred[single]
    ucode, mult1 = np.unique(code, return_counts=True)
    s_size, rest = ucode // (k * k), ucode % (k * k)
    s_lab, s_pred = rest // k, rest % k
    multi = np.flatnonzero(used & (n_lab > 1))
    T = len(ucode) + len(multi)
    contrib = np.zeros((T, k * k))
    contrib[np.arange(len(ucode)), s_lab * k + s_pred] = s_size
    for j, gi in enumerate(multi):
        contrib[len(ucode) + j, np.arange(k) * k + pred[gi]] = lab_counts[gi]
    mult = np.concatenate([mult1, np.ones(len(multi), dtype=np.int64)])
    scores = []
    for start in range(0, B, 100):
        w = rng.poisson(mult, size=(min(100, B - start), T)).astype(np.float64)
        C = (w @ contrib).reshape(-1, k, k)
        scores.append(np.nanmean(P42.f1_from_confusion(C)[:, present], axis=1))
    scores = np.concatenate(scores)
    maj = lab_counts[used].argmax(1)
    dedup = f1_score(maj, pred[used], average="macro", labels=np.unique(maj))
    ceil_C = np.zeros((k, k))
    np.add.at(ceil_C, (np.tile(np.arange(k), used.sum()), np.repeat(maj, k)), lab_counts[used].ravel())
    return {"ci95": [round(float(np.percentile(scores, 2.5)), 5), round(float(np.percentile(scores, 97.5)), 5)],
            "groups": int(used.sum()), "dedup_macro_f1": round(float(dedup), 5),
            "conflict_ceiling_macro_f1": round(float(np.nanmean(P42.f1_from_confusion(ceil_C)[present])), 5),
            "rows_in_conflict_groups": int(lab_counts[n_lab > 1].sum())}


def score(y, yhat, names):
    labels = np.arange(len(names))
    per = f1_score(y, yhat, labels=labels, average=None, zero_division=0)
    in_y = np.bincount(y, minlength=len(names)) > 0
    r = {"rows": int(len(y)), "macro_f1": round(float(per[in_y].mean()), 5),
         "per_class_f1": {names[i]: (round(float(per[i]), 5) if in_y[i] else None) for i in labels}}
    if names == ["Benign", "Attack"]:
        r["attack_f1"] = round(float(per[1]), 5)
    return r


class Groups:
    """Group index of every row of a dataset for a key, computed once."""

    def __init__(self):
        self.cache = {}

    def get(self, name, d, key):
        if (name, key) not in self.cache:
            _, inv = np.unique(d["hashes"][key], return_inverse=True)
            self.cache[(name, key)] = inv.astype(np.int64)
        return self.cache[(name, key)]


def target_table(t, key, fidx_t, task, names):
    """One row per distinct target vector (under the model's features): its
    feature values, label counts and whether it carries several labels.
    Identical vectors receive identical predictions, so scoring the table
    weighted by the counts equals scoring every row (checked in --equiv)."""
    y, keep, _ = task_labels(t, task, names)
    if task == "CAN":  # extended pairs: rows of the classes shared with the source
        keep &= np.isin(y, names_shared(t, names))
    rows = np.flatnonzero(keep)
    _, first, inv = np.unique(t["hashes"][key][rows], return_index=True, return_inverse=True)
    k = len(names)
    lab_counts = np.bincount(inv * k + y[rows], minlength=len(first) * k).reshape(len(first), k)
    order = np.argsort(rows[first])
    rep = rows[first][order]
    X = np.empty((len(rep), len(fidx_t)), dtype=np.float32)
    for i in range(0, len(rep), CHUNK):
        X[i:i + CHUNK] = t["X"][rep[i:i + CHUNK]][:, fidx_t]
    return {"X": X, "lab_counts": lab_counts[order], "rows": int(len(rows))}


def names_shared(t, names):
    return [i for i, c in enumerate(names) if c in t["canon_names"]]


def table_metrics(lab_counts, pred, names, rng, labels=None):
    """Row-weighted scores, group bootstrap, dedup view and conflict ceiling
    from a table of label counts per distinct vector and its prediction."""
    k = len(names)
    G = len(pred)
    C = np.zeros((k, k))
    np.add.at(C, (np.tile(np.arange(k), G), np.repeat(pred, k)), lab_counts.ravel())
    present = C.sum(1) > 0 if labels is None else np.isin(np.arange(k), labels)
    f1 = P42.f1_from_confusion(C)
    res = {"rows": int(C.sum()), "macro_f1": round(float(np.nanmean(np.where(present, f1, np.nan))), 5),
           "per_class_f1": {names[i]: (round(float(f1[i]), 5) if present[i] else None) for i in range(k)}}
    if names == ["Benign", "Attack"]:
        res["attack_f1"] = round(float(f1[1]), 5)
    n_lab = (lab_counts > 0).sum(1)
    size = lab_counts.sum(1)
    single = n_lab == 1
    lab1 = lab_counts[single].argmax(1)
    code = (size[single].astype(np.int64) * k + lab1) * k + pred[single]
    ucode, mult1 = np.unique(code, return_counts=True)
    s_size, rest = ucode // (k * k), ucode % (k * k)
    multi = np.flatnonzero(n_lab > 1)
    T = len(ucode) + len(multi)
    contrib = np.zeros((T, k * k))
    contrib[np.arange(len(ucode)), rest] = s_size
    if len(multi):
        contrib[len(ucode) + np.arange(len(multi))[:, None], np.arange(k)[None, :] * k + pred[multi][:, None]] = \
            lab_counts[multi]
    mult = np.concatenate([mult1, np.ones(len(multi), dtype=np.int64)])
    scores = []
    for start in range(0, B, 100):
        w = rng.poisson(mult, size=(min(100, B - start), T)).astype(np.float64)
        Cb = (w @ contrib).reshape(-1, k, k)
        scores.append(np.nanmean(P42.f1_from_confusion(Cb)[:, present], axis=1))
    scores = np.concatenate(scores)
    maj = lab_counts.argmax(1)
    Cd = np.zeros((k, k))
    np.add.at(Cd, (maj, pred), 1)
    Cc = np.zeros((k, k))
    np.add.at(Cc, (np.tile(np.arange(k), G), np.repeat(maj, k)), lab_counts.ravel())
    pres_d = Cd.sum(1) > 0
    res |= {"ci95": [round(float(np.percentile(scores, 2.5)), 5), round(float(np.percentile(scores, 97.5)), 5)],
            "groups": int(G),
            "dedup_macro_f1": round(float(np.nanmean(P42.f1_from_confusion(Cd)[pres_d])), 5),
            "conflict_ceiling_macro_f1": round(float(np.nanmean(P42.f1_from_confusion(Cc)[present])), 5),
            "rows_in_conflict_groups": int(size[multi].sum())}
    return res


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #

CONDITIONS = {"G0": ("base", "grouped", None), "R0": ("base", "random", None),
              "G-NC": ("base", "grouped", "noconflict"), "G-NP": ("noport", "grouped", None),
              "G-NI": ("noid", "grouped", None)}
# split-controlled variants (N3): (features, split kind, filter, key of the grouped split)
SPLIT_CONTROLLED = {"G-NP0": ("noport", "grouped", None, "base"), "G-NI0": ("noid", "grouped", None, "base"),
                    "G-NIS0": ("noidsvc", "grouped", None, "base")}
ALL_CONDITIONS = CONDITIONS | SPLIT_CONTROLLED
# feature sets not in the cache: name -> (parent set, columns removed from it)
DERIVED_SETS = {"noidsvc": ("noid", SERVICE_COLS)}


def ensure_derived(d, dataset, name):
    """Add a derived feature set and its row hashes to an opened cache."""
    if name in d["sets"] or name not in DERIVED_SETS:
        return
    parent, drop = DERIVED_SETS[name]
    feats = [c for c in d["sets"][parent] if c not in drop]
    path = CACHE / dataset / f"hash_{name}.npy"
    if not path.exists():
        pf = pq.ParquetFile(PROCESSED_ROOT / dataset / "tabular.parquet")
        h = np.empty(pf.metadata.num_rows, dtype=np.uint64)
        off = 0
        for b in pf.iter_batches(batch_size=CHUNK, columns=d["sets"][parent]):
            df = b.to_pandas()
            sl = slice(off, off + len(df))
            if not np.array_equal(tabular_row_hash(df, d["sets"][parent]), d["hashes"][parent][sl]):
                raise RuntimeError(f"{dataset}: recomputed '{parent}' hashes differ from the cache")
            h[sl] = tabular_row_hash(df, feats)
            off += len(df)
        np.save(path, h)
        log(f"{dataset}: '{name}' hashes written ({len(feats)} features; '{parent}' hashes match the cache)")
    d["sets"][name] = feats
    d["hashes"][name] = np.load(path)


def version_of(ds):
    return [x for x in DATASETS if is_v2(x) == is_v2(ds)]


def eval_target(model, present, tname, t, key, fidx_t, task, names, groups, seed, clean):
    """Score a model on all rows of a target (and without target conflicts if clean)."""
    y, keep, _ = task_labels(t, task, names)
    rows = np.flatnonzero(keep)
    yhat = predict(model, t["X"], rows, fidx_t, present)
    yt = y[rows]
    gi = groups.get(tname, t, key)[rows]
    _, g = np.unique(gi, return_inverse=True)
    res = score(yt, yhat, names) | group_metrics(g, yt, yhat, len(names), np.random.default_rng(seed))
    if clean:
        cm = P42.conflict_mask(t["hashes"][key], y, keep)[rows]
        res["no_target_conflicts"] = score(yt[~cm], yhat[~cm], names) | {"rows_removed": int(cm.sum())}
    return res


def run_source(src, seeds, tag, conds=None):
    t0 = time.time()
    out_dir = OUT / (f"cross_{tag}" if tag else "cross")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{src}.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["condition"], r["task"], r["model"], r["seed"]))
    a = open_cache(src)
    targets = {n: open_cache(n) for n in version_of(src) if n != src}
    n = len(a["label"])
    conds = conds or list(CONDITIONS)
    for c in conds:
        ensure_derived(a, src, ALL_CONDITIONS[c][0])
        for tname, t in targets.items():
            ensure_derived(t, tname, ALL_CONDITIONS[c][0])
    jobs = [(c, task) for c in conds for task in ("BIN", "CORE")] + ([("G0", "EXT")] if "G0" in conds else [])
    for cond, task in jobs:
        key, kind, filt = ALL_CONDITIONS[cond][:3]
        split_key = ALL_CONDITIONS[cond][3] if len(ALL_CONDITIONS[cond]) > 3 else key
        feat = a["sets"][key]
        fidx = [a["cols"].index(c) for c in feat]
        h = a["hashes"][key]
        h_split = a["hashes"][split_key]
        y, keep, names = task_labels(a, "CAN" if task == "EXT" else task)
        info = {}
        if filt == "noconflict":
            cm = P42.conflict_mask(h, y, keep)
            info["source_conflict_rows_removed"] = int(cm.sum())
            keep = keep & ~cm
        tables = {}
        for tname, t in targets.items():
            if task == "EXT" and len(names_shared(t, names)) < 2:
                tables[tname] = None
                continue
            tables[tname] = target_table(t, key, [t["cols"].index(c) for c in feat],
                                         "CAN" if task == "EXT" else task, names)
        log(f"{cond} {task}: target tables " + ", ".join(
            f"{k2.split('-')[1]} {v['X'].shape[0]:,} vectors / {v['rows']:,} rows" for k2, v in tables.items() if v))
        for seed in seeds:
            for model_name in ("xgb", "rf"):
                if (cond, task, model_name, seed) in done:
                    continue
                t1 = time.time()
                split = grouped_split_codes(h_split, seed) if kind == "grouped" else random_split_codes(n, seed)
                take = P42.train_sample(split, a["canon"], keep, np.random.default_rng(seed))
                present = np.unique(y[take])
                enc = np.full(len(names), -1)
                enc[present] = np.arange(len(present))
                tr_rows = np.flatnonzero(take)
                model = P42.make_model(model_name, seed)
                model.fit(np.ascontiguousarray(a["X"][tr_rows][:, fidx]), enc[y[take]])
                te_rows = np.flatnonzero(keep & (split == 2))
                yhat = predict(model, a["X"], te_rows, fidx, present)
                res = {"source": src, "condition": cond, "task": task, "model": model_name, "seed": seed,
                       "features": len(feat), "split": kind, "split_key": split_key, **info,
                       "train_rows": int(take.sum()), "test_rows": int(len(te_rows)),
                       "classes_in_train": [names[i] for i in present],
                       "in_domain": score(y[te_rows], yhat, names), "targets": {}}
                for tname in targets:
                    tab = tables[tname]
                    if tab is None:
                        continue
                    pred = predict_table(model, tab["X"], present)
                    rng = np.random.default_rng(seed)
                    if task == "EXT":
                        res["targets"][tname] = table_metrics(tab["lab_counts"], pred, names, rng,
                                                              labels=names_shared(targets[tname], names))
                        res["targets"][tname]["shared_classes"] = [names[i] for i in
                                                                   names_shared(targets[tname], names)]
                        continue
                    r = table_metrics(tab["lab_counts"], pred, names, rng)
                    if filt == "noconflict":
                        single = (tab["lab_counts"] > 0).sum(1) == 1
                        nc = table_metrics(tab["lab_counts"][single], pred[single], names, rng)
                        r["no_target_conflicts"] = {k2: nc[k2] for k2 in ("rows", "macro_f1", "per_class_f1")
                                                    if k2 in nc} | {"rows_removed": r["rows"] - nc["rows"]}
                    r["delta_points"] = round(100 * (res["in_domain"]["macro_f1"] - r["macro_f1"]), 3)
                    res["targets"][tname] = r
                res["seconds"] = round(time.time() - t1, 1)
                with out_path.open("a") as f:
                    f.write(json.dumps(res) + "\n")
                tg = " ".join(f"{k.split('-')[1]} {v['macro_f1']:.3f}" for k, v in res["targets"].items())
                log(f"{cond:5s} {task:4s} {model_name:3s} s{seed}  in-domain {res['in_domain']['macro_f1']:.3f}  "
                    f"targets {tg}  {res['seconds']}s")
                del model
    log(f"{src}: done in {time.time() - t0:.0f}s")


# --------------------------------------------------------------------------- #
# Check and report
# --------------------------------------------------------------------------- #

def check():
    """Recompute a P4.2 run from the cache; the macro-F1 must match exactly."""
    ds, seed = "NF-BoT-IoT-v3", 1
    stored = [json.loads(x) for x in (OUT / "in_domain" / f"{ds}.jsonl").read_text().splitlines()]
    ref = next(r for r in stored if (r["condition"], r["task"], r["model"], r["seed"]) == ("G0", "BIN", "xgb", seed))
    d = open_cache(ds)
    feat = d["sets"]["base"]
    fidx = [d["cols"].index(c) for c in feat]
    y, keep, names = task_labels(d, "BIN")
    split = grouped_split_codes(d["hashes"]["base"], seed)
    take = P42.train_sample(split, d["canon"], keep, np.random.default_rng(seed))
    present = np.unique(y[take])
    model = P42.make_model("xgb", seed)
    model.fit(np.ascontiguousarray(d["X"][np.flatnonzero(take)][:, fidx]), y[take])
    te = np.flatnonzero(split == 2)
    yhat = predict(model, d["X"], te, fidx, present)
    m = score(y[te], yhat, names)["macro_f1"]
    ok = m == ref["macro_f1"] and int(take.sum()) == ref["train_rows"] and len(te) == ref["test_rows"]
    text = (f"P4.3 cache check ({datetime.now(timezone.utc).isoformat()}): {ds} G0 BIN xgb seed {seed}\n"
            f"  stored P4.2: macro-F1 {ref['macro_f1']}, train {ref['train_rows']:,}, test {ref['test_rows']:,}\n"
            f"  from cache:  macro-F1 {m}, train {int(take.sum()):,}, test {len(te):,}\n"
            f"  IDENTICAL: {ok}\n")
    (OUT / "cross_cache_check.txt").write_text(text)
    print(text)
    if not ok:
        raise SystemExit(1)


def equiv():
    """Table-based scoring must equal row-based scoring (eval_target) exactly."""
    src, tgt, seed = "NF-BoT-IoT-v3", "NF-ToN-IoT-v3", 1
    a, t = open_cache(src), open_cache(tgt)
    feat = a["sets"]["base"]
    fidx = [a["cols"].index(c) for c in feat]
    fidx_t = [t["cols"].index(c) for c in feat]
    lines = [f"P4.3 table vs row scoring ({datetime.now(timezone.utc).isoformat()}): {src} -> {tgt}, seed {seed}"]
    ok = True
    for task in ("BIN", "CORE"):
        y, keep, names = task_labels(a, task)
        split = grouped_split_codes(a["hashes"]["base"], seed)
        take = P42.train_sample(split, a["canon"], keep, np.random.default_rng(seed))
        present = np.unique(y[take])
        enc = np.full(len(names), -1)
        enc[present] = np.arange(len(present))
        model = P42.make_model("xgb", seed)
        model.fit(np.ascontiguousarray(a["X"][np.flatnonzero(take)][:, fidx]), enc[y[take]])
        row = eval_target(model, present, tgt, t, "base", fidx_t, task, names, Groups(), seed, clean=True)
        tab = target_table(t, "base", fidx_t, task, names)
        pred = predict_table(model, tab["X"], present)
        tb = table_metrics(tab["lab_counts"], pred, names, np.random.default_rng(seed))
        single = (tab["lab_counts"] > 0).sum(1) == 1
        nc = table_metrics(tab["lab_counts"][single], pred[single], names, np.random.default_rng(seed))
        same = (row["macro_f1"] == tb["macro_f1"] and row["rows"] == tb["rows"]
                and row["dedup_macro_f1"] == tb["dedup_macro_f1"]
                and row["conflict_ceiling_macro_f1"] == tb["conflict_ceiling_macro_f1"]
                and row["no_target_conflicts"]["macro_f1"] == nc["macro_f1"])
        ok &= same
        lines.append(f"  {task}: rows macro-F1 {row['macro_f1']} table {tb['macro_f1']}; dedup {row['dedup_macro_f1']} "
                     f"/ {tb['dedup_macro_f1']}; ceiling {row['conflict_ceiling_macro_f1']} / "
                     f"{tb['conflict_ceiling_macro_f1']}; no conflicts {row['no_target_conflicts']['macro_f1']} / "
                     f"{nc['macro_f1']}; CI rows {row['ci95']} table {tb['ci95']}; identical: {same}")
    lines.append(f"  IDENTICAL: {ok}")
    text = "\n".join(lines) + "\n"
    (OUT / "cross_table_check.txt").write_text(text)
    print(text)
    if not ok:
        raise SystemExit(1)


def report(tag):
    out_dir = OUT / (f"cross_{tag}" if tag else "cross")
    rows = []
    for f in sorted(out_dir.glob("*.jsonl")):
        rows += [json.loads(x) for x in f.read_text().splitlines()]
    agg = defaultdict(list)
    for r in rows:
        for t, v in r["targets"].items():
            agg[(r["source"], t, r["task"], r["model"], r["condition"])].append((r, v))

    def mm(xs, fmt="{:.3f}"):
        xs = np.asarray(xs)
        return f"{fmt.format(np.median(xs))} [{fmt.format(xs.min())}, {fmt.format(xs.max())}]"

    L = ["P4.3 cross-dataset evaluation (research/phase4_protocol.md, block C, §14)",
         f"Generated: {datetime.now(timezone.utc).isoformat()}",
         "Target = all rows of the other dataset of the same version. Values: median [min, max] over seeds.",
         "in-domain = same model on the source test split; delta = in-domain minus target, points, per run.",
         "G-NC: target also scored without its own conflicts (nc). EXT: CAN model under G0, shared classes.", ""]
    for src in DATASETS:
        for tgt in DATASETS:
            keys = [k for k in agg if k[0] == src and k[1] == tgt]
            if not keys:
                continue
            L.append(f"== {src} -> {tgt}")
            for task in ("BIN", "CORE", "EXT"):
                for model in ("xgb", "rf"):
                    for c in CONDITIONS:
                        rv = agg.get((src, tgt, task, model, c))
                        if not rv:
                            continue
                        line = (f"  {task:4s} {model:3s} {c:5s} n={len(rv)}  in-domain "
                                f"{mm([r['in_domain']['macro_f1'] for r, _ in rv])}  target "
                                f"{mm([v['macro_f1'] for _, v in rv])}")
                        if task != "EXT":
                            line += f"  delta {mm([v['delta_points'] for _, v in rv], '{:+.1f}')}"
                        if "no_target_conflicts" in rv[0][1]:
                            line += f"  nc {mm([v['no_target_conflicts']['macro_f1'] for _, v in rv])}"
                        L.append(line)
            L.append("")
    text = "\n".join(L) + "\n"
    (OUT / f"{out_dir.name}_report.txt").write_text(text)
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", choices=DATASETS)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--equiv", action="store_true")
    ap.add_argument("--source", choices=DATASETS)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--tag", default="")
    ap.add_argument("--conditions", nargs="+", choices=list(ALL_CONDITIONS), default=None)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.cache:
        build_cache(a.cache)
    if a.check:
        check()
    if a.equiv:
        equiv()
    if a.source:
        run_source(a.source, a.seeds, a.tag, a.conditions)
    if a.report:
        report(a.tag)


if __name__ == "__main__":
    main()
