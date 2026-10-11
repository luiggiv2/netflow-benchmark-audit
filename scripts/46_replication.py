"""46_replication.py — P4.4 (block D of research/phase4_protocol.md §6, §11
decisions 5-6): replication of Layeghy and Portmann (2023), Comput. Electr.
Eng. 108:108692, on the three NF-v2 datasets, and how the published result
changes under the controls of this study.

Their setting (Section 4, Tables 3, 4 and 6 of the paper): binary task
(Benign vs Attack), 39 features (both IPs and both ports removed), a
class-stratified sample of 1,000,000 flows per dataset, Random Forest with
scikit-learn defaults except ccp_alpha = 0.001, validation split 0.3, F1 of
the attack class; cross-domain: the model of one dataset evaluated on the
others. "No. of folds 5" is read as five repetitions (seeds 1-5).

Steps (on each seed's sample; rows kept in file order):
  L0  as published: row-level random 70/30 split
  L0k 5-fold cross-validation without shuffling (StratifiedKFold(5, shuffle=False),
      what cross_val_score(cv=5) does for a classifier); scores averaged over folds
  L0t the first 70 % of the sample for training, the last 30 % for test
      (L0k and L0t added by the author's decision of 2026-10-04, protocol §16)
  L1  grouped 70/30 split keyed on the 39 features (vectors never cross)
  L2  L1 without conflicting rows (identical 39-feature vector, both labels)
  L3  L1 with FLOW_DURATION_MILLISECONDS recovered (F8: every non-zero value
      is 2^32/1000 ms minus the duration; data/exploration/f8_duration_check.txt),
      in source and targets
  L4  L1 without FLOW_DURATION_MILLISECONDS (38 features), source and targets
Each model is also scored on the 1,000,000-flow samples of the other two
datasets (same seed), as in their Table 6.

Usage:
  python scripts/46_replication.py --dataset NF-ToN-IoT-v2 [--seeds 1] [--tag smoke]
  python scripts/46_replication.py --report [--tag smoke]
Outputs: data/eval/replication[_<tag>]/<dataset>.jsonl and
data/eval/replication[_<tag>]_report.txt
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

from nfv3_config import PROCESSED_ROOT, WORK, grouped_split_codes

V2 = ["NF-BoT-IoT-v2", "NF-CSE-CIC-IDS2018-v2", "NF-ToN-IoT-v2"]
SEEDS = [1, 2, 3, 4, 5]
SAMPLE = 1_000_000
CCP_ALPHA = 0.001
WRAP_MS = 4_294_967  # floor(2^32 / 1000)
DUR = "FLOW_DURATION_MILLISECONDS"
OUT = WORK / "data/eval"
# Layeghy and Portmann (2023): Table 4 (RF, in-domain) and Table 6 (RF, cross-domain), F1 of the attack class, %
PUBLISHED_IN = {"NF-BoT-IoT-v2": 99.82, "NF-CSE-CIC-IDS2018-v2": 95.44, "NF-ToN-IoT-v2": 77.33}
PUBLISHED_CROSS = {("NF-CSE-CIC-IDS2018-v2", "NF-BoT-IoT-v2"): 12.91, ("NF-ToN-IoT-v2", "NF-BoT-IoT-v2"): 46.75,
                   ("NF-BoT-IoT-v2", "NF-CSE-CIC-IDS2018-v2"): 22.32, ("NF-ToN-IoT-v2", "NF-CSE-CIC-IDS2018-v2"): 4.50,
                   ("NF-BoT-IoT-v2", "NF-ToN-IoT-v2"): 85.50, ("NF-CSE-CIC-IDS2018-v2", "NF-ToN-IoT-v2"): 30.28}

_spec = importlib.util.spec_from_file_location("cross", WORK / "scripts/44_cross_dataset.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)
log = C.log


def load(ds):
    d = C.open_cache(ds)
    d["feat39"] = d["sets"]["noport"]
    assert len(d["feat39"]) == 39
    dur = pq.read_table(PROCESSED_ROOT / ds / "data.parquet", columns=[DUR]).column(0).to_numpy()
    d["dur_recovered"] = np.where(dur > 0, WRAP_MS - dur, 0).astype(np.float32)
    return d


def stratified_sample(d, seed):
    """Proportional sample of SAMPLE rows by native class (sorted row ids)."""
    rng = np.random.default_rng(seed)
    nat = d["native"]
    counts = np.bincount(nat)
    quota = np.floor(counts / counts.sum() * SAMPLE).astype(int)
    quota[np.argmax(counts)] += SAMPLE - quota.sum()
    rows = [rng.choice(np.flatnonzero(nat == c), q, replace=False) for c, q in enumerate(quota) if q > 0]
    return np.sort(np.concatenate(rows))


def matrix(d, rows, step):
    feats = d["feat39"]
    X = np.ascontiguousarray(d["X"][rows][:, [d["cols"].index(c) for c in feats]])
    j = feats.index(DUR)
    if step == "L3":
        X[:, j] = d["dur_recovered"][rows]
    if step == "L4":
        X = np.delete(X, j, axis=1)
    return X


def scores(y, p):
    return {"attack_f1": round(100 * float(f1_score(y, p, pos_label=1, zero_division=0)), 3),
            "macro_f1": round(100 * float(f1_score(y, p, average="macro", zero_division=0)), 3),
            "precision": round(100 * float(precision_score(y, p, zero_division=0)), 3),
            "recall": round(100 * float(recall_score(y, p, zero_division=0)), 3),
            "rows": int(len(y)), "attack_share": round(float(np.mean(y)), 5)}


def run(ds, seeds, tag):
    t0 = time.time()
    out_dir = OUT / (f"replication_{tag}" if tag else "replication")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ds}.jsonl"
    done = set()
    if out_path.exists():
        done = {(r["step"], r["seed"]) for r in map(json.loads, out_path.read_text().splitlines())}
    data = {x: load(x) for x in V2}
    a = data[ds]
    for seed in seeds:
        samples = {x: stratified_sample(data[x], seed) for x in V2}
        rows = samples[ds]
        y = a["label"][rows].astype(int)
        h = a["hashes"]["noport"][rows]
        rng = np.random.default_rng(seed)
        random_split = np.where(rng.random(len(rows)) < 0.70, 0, 1)
        grouped = np.where(grouped_split_codes(h, seed) == 0, 0, 1)
        conflict = C.P42.conflict_mask(h, y, np.ones(len(rows), dtype=bool))
        n = len(rows)
        ordered = np.where(np.arange(n) < int(0.70 * n), 0, 1)
        folds = [te_idx for _, te_idx in StratifiedKFold(5, shuffle=False).split(np.zeros(n), y)]
        targets = {tgt: (matrix(data[tgt], samples[tgt], "L0"), data[tgt]["label"][samples[tgt]].astype(int))
                   for tgt in V2 if tgt != ds}
        for step in ("L0", "L0k", "L0t", "L1", "L2", "L3", "L4"):
            if (step, seed) in done:
                continue
            t1 = time.time()
            keep = ~conflict if step == "L2" else np.ones(n, dtype=bool)
            X = matrix(a, rows, step)
            if step == "L0k":
                parts = []
                for te_idx in folds:
                    te = np.zeros(n, dtype=bool)
                    te[te_idx] = True
                    parts.append((~te, te))
            else:
                split = {"L0": random_split, "L0t": ordered}.get(step, grouped)
                parts = [(keep & (split == 0), keep & (split == 1))]
            fold_res = []
            for tr, te in parts:
                model = RandomForestClassifier(ccp_alpha=CCP_ALPHA, n_jobs=-1, random_state=seed)
                model.fit(X[tr], y[tr])
                fr = {"train_rows": int(tr.sum()), "test_rows": int(te.sum()),
                      "test_rows_with_vector_in_train": round(float(np.isin(h[te], np.unique(h[tr])).mean()), 5),
                      "in_domain": scores(y[te], model.predict(X[te])), "cross": {}}
                for tgt, (Xt, yt) in targets.items():
                    if step in ("L3", "L4"):
                        Xt = matrix(data[tgt], samples[tgt], step)
                    fr["cross"][tgt] = scores(yt, model.predict(Xt))
                fold_res.append(fr)
                del model
            avg = lambda ds_: {k: round(float(np.mean([d[k] for d in ds_])), 3)  # noqa: E731
                               for k in ("attack_f1", "macro_f1", "precision", "recall")}
            res = {"dataset": ds, "step": step, "seed": seed, "features": int(X.shape[1]),
                   "train_rows": fold_res[0]["train_rows"], "test_rows": fold_res[0]["test_rows"],
                   "conflict_rows_in_sample": int(conflict.sum()),
                   "test_rows_with_vector_in_train": round(float(np.mean(
                       [f["test_rows_with_vector_in_train"] for f in fold_res])), 5),
                   "in_domain": avg([f["in_domain"] for f in fold_res]),
                   "published_in_domain_attack_f1": PUBLISHED_IN[ds],
                   "cross": {tgt: avg([f["cross"][tgt] for f in fold_res])
                             | {"published_attack_f1": PUBLISHED_CROSS[(ds, tgt)]} for tgt in targets}}
            if len(fold_res) > 1:
                res["folds"] = fold_res
            res["seconds"] = round(time.time() - t1, 1)
            with out_path.open("a") as f:
                f.write(json.dumps(res) + "\n")
            log(f"{ds} {step} s{seed}: attack F1 {res['in_domain']['attack_f1']:.2f} "
                f"(published {PUBLISHED_IN[ds]}); cross " + " ".join(
                    f"{k.split('-')[1]} {v['attack_f1']:.2f}" for k, v in res["cross"].items())
                + f"  {res['seconds']}s")
    log(f"{ds}: done in {time.time() - t0:.0f}s")


def report(tag):
    out_dir = OUT / (f"replication_{tag}" if tag else "replication")
    rows = []
    for f in sorted(out_dir.glob("*.jsonl")):
        rows += [json.loads(x) for x in f.read_text().splitlines()]
    agg = defaultdict(list)
    for r in rows:
        agg[(r["dataset"], r["step"])].append(r)

    def mm(xs):
        xs = np.asarray(xs)
        return f"{np.median(xs):6.2f} [{xs.min():6.2f}, {xs.max():6.2f}]"

    L = ["P4.4 replication of Layeghy and Portmann (2023), RF, binary, 39 features, 1 M stratified sample",
         f"Generated: {datetime.now(timezone.utc).isoformat()}",
         "F1 of the attack class in %, median [min, max] over seeds. L0 as published (random 70/30); L0k 5-fold",
         "CV without shuffling (mean over folds); L0t first 70 % / last 30 % in file order; L1 grouped",
         "70/30; L2 L1 without conflicting rows; L3 L1 with the F8 duration recovered; L4 L1 without the duration.",
         "leak = share of test rows whose 39-feature vector occurs in train. pub = published value.", ""]
    for ds in V2:
        if not any(k[0] == ds for k in agg):
            continue
        L.append(f"== {ds} (published in-domain {PUBLISHED_IN[ds]})")
        for step in ("L0", "L0k", "L0t", "L1", "L2", "L3", "L4"):
            rs = agg.get((ds, step))
            if not rs:
                continue
            line = (f"  {step} n={len(rs)} in-domain {mm([r['in_domain']['attack_f1'] for r in rs])} "
                    f"macro {np.median([r['in_domain']['macro_f1'] for r in rs]):6.2f} "
                    f"leak {np.median([r['test_rows_with_vector_in_train'] for r in rs]):.3f}")
            if step == "L2":
                line += f" conflicts {int(np.median([r['conflict_rows_in_sample'] for r in rs])):,}"
            L.append(line)
            for tgt in V2:
                if tgt == ds:
                    continue
                L.append(f"       -> {tgt:24s} {mm([r['cross'][tgt]['attack_f1'] for r in rs])} "
                         f"(pub {PUBLISHED_CROSS[(ds, tgt)]:6.2f})")
        L.append("")
    text = "\n".join(L) + "\n"
    (OUT / f"{out_dir.name}_report.txt").write_text(text)
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=V2)
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
