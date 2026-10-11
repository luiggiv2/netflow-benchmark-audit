"""59_split_checksums.py — checksums of every partition used in the article
(third reviewer report, minor 19; author decision 2026-10-10, option a).

Section 4.6 and the data availability statement promise "a checksum of every
partition used (so that a reproduction can verify its splits)". Every
partition is a deterministic function of the published CSV and the code; this
script recomputes each one with the functions of the script that used it and
records, for every run configuration, the SHA-256 of the sorted 0-based row
positions in the published CSV (the row order kept by 01_load_nfv3.py), as
little-endian uint64, of the training rows and of the test rows. Each
recomputed partition is checked against the row counts stored with the runs.

  in_domain    42_in_domain.py: 7 conditions (5 in NF-v2) x BIN, CAN x seeds 1-5;
               the SHAP models of 47_shap.py use the same partitions
  cross        44_cross_dataset.py: source training and test rows of G0, R0,
               G-NC, G-NP, G-NI, EXT and of the repetitions on the split of G0
               (G-NP0, G-NI0, G-NIS0) x seeds 1-5; targets use every row of the task
  replication  46_replication.py: the stratified sample of each seed and the
               training and test rows of every step (L0k: fold 1 of 5)
  probes       34_shortcut_probes.py --common and exploration/r3m7_probes_seeds.py:
               the Bernoulli sample (seed 2026) and its train and test rows under
               the split of set A for seeds 42 and 1-5 (per-class balancing
               happens after this partition)

Reads the cache of 44_cross_dataset.py (results/eval/cache) and, for the
duplicate mask of G-ND and the probe sample, the processed parquet files.
Usage: .venv/bin/python scripts/59_split_checksums.py   (about 15-25 min)
Outputs: data/paper/split_checksums.json and data/paper/split_checksums.txt
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from nfv3_config import DATASETS, WORK, grouped_split_codes, is_v2, random_split_codes


def module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, WORK / path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


C = module("cross_dataset", "scripts/44_cross_dataset.py")
P42 = C.P42
R = module("replication", "scripts/46_replication.py")
PR = module("probes", "scripts/34_shortcut_probes.py")

SEEDS = [1, 2, 3, 4, 5]
PROBE_SEEDS = [42, 1, 2, 3, 4, 5]
OUT = WORK / "data/paper"
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def sha(rows: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(rows, dtype="<u8").tobytes()).hexdigest()


def stored(path, key) -> dict:
    """(configuration) -> (train_rows, test_rows) from a jsonl of runs; every model of a configuration must agree."""
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        r = json.loads(line)
        # the first cross-dataset runs store the source test rows only as in_domain.rows
        k, v = key(r), (r["train_rows"], r["test_rows"] if "test_rows" in r else r["in_domain"]["rows"])
        if out.setdefault(k, v) != v:
            raise RuntimeError(f"{path.name}: runs of {k} disagree on row counts")
    return out


def entry(take: np.ndarray, test: np.ndarray, ref, **ids) -> dict:
    tr, te = np.flatnonzero(take), np.flatnonzero(test)
    e = {**ids, "train_rows": int(len(tr)), "test_rows": int(len(te)), "train_sha256": sha(tr), "test_sha256": sha(te)}
    e["matches_stored_counts"] = None if ref is None else (ref == (e["train_rows"], e["test_rows"]))
    return e


def in_domain(ds: str, d: dict) -> list[dict]:
    """The partition code of 42_in_domain.run, without fitting."""
    n = len(d["label"])
    excl = d["canon_names"].index(P42.EXCLUDED) if P42.EXCLUDED in d["canon_names"] else -1
    can_names = sorted(c for c in d["canon_names"] if c != P42.EXCLUDED)
    can_map = np.full(len(d["canon_names"]), -1, dtype=np.int8)
    for i, c in enumerate(d["canon_names"]):
        if c != P42.EXCLUDED:
            can_map[i] = can_names.index(c)
    y_can = can_map[d["canon"]]
    ref = stored(WORK / f"data/eval/in_domain/{ds}.jsonl", lambda r: (r["condition"], r["task"], r["seed"]))
    out = []
    for cond, (key, kind, filt) in P42.conditions(not is_v2(ds)).items():
        h = d["hashes"][key]
        for task in ("BIN", "CAN"):
            y = d["label"].astype(np.int64) if task == "BIN" else y_can.astype(np.int64)
            keep = np.ones(n, dtype=bool) if task == "BIN" else (d["canon"] != excl)
            if filt == "noconflict":
                keep &= ~P42.conflict_mask(h, y, keep)
            elif filt == "nodup":
                keep &= ~d["dup"]
            for seed in SEEDS:
                split = grouped_split_codes(h, seed) if kind == "grouped" else random_split_codes(n, seed)
                take = P42.train_sample(split, d["canon"], keep, np.random.default_rng(seed))
                out.append(entry(take, keep & (split == 2), ref.get((cond, task, seed)), dataset=ds, condition=cond,
                                 task=task, seed=seed, split_key=key, split=kind))
    return out


def cross(ds: str, a: dict) -> list[dict]:
    """The partition code of 44_cross_dataset.run_source, without fitting."""
    n = len(a["label"])
    ref = {}
    for sub in ("cross", "cross_g0split", "cross_g0split_nosvc"):
        ref |= stored(WORK / f"data/eval/{sub}/{ds}.jsonl", lambda r: (r["condition"], r["task"], r["seed"]))
    out = []
    jobs = [(c, t) for c in C.ALL_CONDITIONS for t in ("BIN", "CORE")] + [("G0", "EXT")]
    for cond, task in jobs:
        spec = C.ALL_CONDITIONS[cond]
        key, kind, filt = spec[:3]
        split_key = spec[3] if len(spec) > 3 else key
        C.ensure_derived(a, ds, key)
        h, h_split = a["hashes"][key], a["hashes"][split_key]
        y, keep, _ = C.task_labels(a, "CAN" if task == "EXT" else task)
        if filt == "noconflict":
            keep = keep & ~P42.conflict_mask(h, y, keep)
        for seed in SEEDS:
            split = grouped_split_codes(h_split, seed) if kind == "grouped" else random_split_codes(n, seed)
            take = P42.train_sample(split, a["canon"], keep, np.random.default_rng(seed))
            out.append(entry(take, keep & (split == 2), ref.get((cond, task, seed)), dataset=ds, condition=cond,
                             task=task, seed=seed, split_key=split_key, split=kind))
    return out


def replication() -> list[dict]:
    """The sample and partition code of 46_replication.run, without fitting."""
    out = []
    data = {x: R.load(x) for x in R.V2}
    for ds in R.V2:
        path = WORK / f"data/eval/replication/{ds}.jsonl"
        if not path.exists():
            continue
        runs = [json.loads(x) for x in path.read_text().splitlines()]
        ref = {(r["step"], r["seed"]): (r["train_rows"], r["test_rows"]) for r in runs}
        a = data[ds]
        for seed in sorted({r["seed"] for r in runs}):
            rows = R.stratified_sample(a, seed)
            n = len(rows)
            y = a["label"][rows].astype(int)
            h = a["hashes"]["noport"][rows]
            rng = np.random.default_rng(seed)
            random_split = np.where(rng.random(n) < 0.70, 0, 1)
            grouped = np.where(grouped_split_codes(h, seed) == 0, 0, 1)
            conflict = P42.conflict_mask(h, y, np.ones(n, dtype=bool))
            ordered = np.where(np.arange(n) < int(0.70 * n), 0, 1)
            fold1 = next(iter(R.StratifiedKFold(5, shuffle=False).split(np.zeros(n), y)))[1]
            for step in sorted({r["step"] for r in runs if r["seed"] == seed}):
                if step == "L0k":
                    te = np.zeros(n, dtype=bool)
                    te[fold1] = True
                    tr = ~te
                else:
                    keep = ~conflict if step == "L2" else np.ones(n, dtype=bool)
                    split = {"L0": random_split, "L0t": ordered}.get(step, grouped)
                    tr, te = keep & (split == 0), keep & (split == 1)
                take, test = np.zeros(len(a["label"]), dtype=bool), np.zeros(len(a["label"]), dtype=bool)
                take[rows[tr]], test[rows[te]] = True, True
                e = entry(take, test, ref.get((step, seed)), dataset=ds, step=step, seed=seed)
                e["sample_rows"], e["sample_sha256"] = int(n), sha(rows)
                out.append(e)
    return out


def probes(ds: str) -> list[dict]:
    """The Bernoulli sample of 34_shortcut_probes.sample_rekey (and of
    r3m7_probes_seeds.sample_seeds), with row positions, and the split of set A."""
    pf = pq.ParquetFile(PR.PROCESSED_ROOT / ds / "tabular.parquet")
    counts = Counter()
    for b in pf.iter_batches(batch_size=PR.BATCH, columns=["native_label"]):
        counts.update(b.column("native_label").to_pandas().astype(str))
    target = max(PR.TRAIN_CAP / 0.70, PR.TEST_CAP / 0.15)
    prob = {k: min(1.0, target / n) for k, n in counts.items()}
    rng = np.random.default_rng(PR.SEED)
    A = PR.SETS_REKEY["A"]
    pos, hashes, labels, off = [], [], [], 0
    for b in pf.iter_batches(batch_size=PR.BATCH, columns=PR.FEATURE_COLS_V2 + ["native_label"]):
        df = b.to_pandas()
        lab = df["native_label"].astype(str)
        keep = rng.random(len(df)) < lab.map(prob).to_numpy()
        sub = df.loc[keep]
        pos.append(off + np.flatnonzero(keep))
        hashes.append(PR.tabular_row_hash(sub, A))
        labels.append(lab[keep].to_numpy())
        off += len(df)
    pos, h, lab = np.concatenate(pos), np.concatenate(hashes), np.concatenate(labels)
    # check: the per-class balanced row counts of the stored seed-42 probes
    js = json.loads((WORK / f"data/audit/shortcut_probes_common/{ds}.json").read_text())
    out = []
    for seed in PROBE_SEEDS:
        codes = grouped_split_codes(h, seed)
        ok = None
        if seed == 42:
            ok = True
            for cls, res in js["classes"].items():
                rb = res.get("rows_by_set", {}).get("A")
                if not rb:
                    continue
                for part, code, cap in (("train_rows", 0, PR.TRAIN_CAP), ("test_rows", 2, PR.TEST_CAP)):
                    nc = int(np.sum((lab == cls) & (codes == code)))
                    nb = int(np.sum((lab == PR.BENIGN) & (codes == code)))
                    ok &= rb[part]["class"] == min(nc, nb, cap)
        out.append({"dataset": ds, "seed": seed, "sample_rows": int(len(pos)), "sample_sha256": sha(pos),
                    "train_rows": int(np.sum(codes == 0)), "test_rows": int(np.sum(codes == 2)),
                    "train_sha256": sha(pos[codes == 0]), "test_sha256": sha(pos[codes == 2]),
                    "matches_stored_counts": ok})
    return out


def provenance() -> dict:
    def git(*a):
        return subprocess.run(["git", *a], cwd=WORK, capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)data/paper")),
            "script_sha1": hashlib.sha1((WORK / "scripts/59_split_checksums.py").read_bytes()).hexdigest(),
            "generated_utc": datetime.now(timezone.utc).isoformat()}


def main() -> None:
    res = {"in_domain": [], "cross": [], "replication": [], "probes": []}
    for ds in DATASETS:
        a = C.open_cache(ds)
        if not is_v2(ds):
            log(f"{ds}: duplicate mask")
            a["dup"] = P42.release_duplicates(ds, len(a["label"]))
        log(f"{ds}: in-domain partitions")
        res["in_domain"] += in_domain(ds, a)
        log(f"{ds}: cross-dataset partitions")
        res["cross"] += cross(ds, a)
        del a
        log(f"{ds}: probe sample")
        res["probes"] += probes(ds)
    log("replication")
    res["replication"] = replication()
    prov = provenance()
    definition = ("SHA-256 of the sorted 0-based row positions in the published CSV, as little-endian uint64; "
                  "matches_stored_counts compares the recomputed training and test row counts with those stored "
                  "with the runs (probes: per-class balanced counts of the seed-42 probes; null where no run "
                  "stores them)")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "split_checksums.json").write_text(json.dumps({"provenance": prov, "definition": definition, **res},
                                                         indent=1) + "\n")
    L = ["Partition checksums (third reviewer report, minor 19)", f"provenance: {json.dumps(prov)}",
         f"definition: {definition}", ""]
    for k, v in res.items():
        m = Counter(str(e["matches_stored_counts"]) for e in v)
        L.append(f"  {k}: {len(v)} partitions; matches stored counts: {dict(m)}")
    bad = [e for v in res.values() for e in v if e["matches_stored_counts"] is False]
    L += ["", f"partitions whose recomputed counts differ from the stored runs: {len(bad)}"]
    L += [f"  {json.dumps({k: e[k] for k in e if not k.endswith('sha256')})}" for e in bad[:20]]
    (OUT / "split_checksums.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
