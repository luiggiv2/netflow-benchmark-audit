"""47_shap.py — P4.5 (block E of research/phase4_protocol.md, design §18):
SHAP explanations of the Phase 4 models (RQ4).

Models are refitted with the P4.2/P4.3 procedure (same splits, capped
training samples and seeds) from the cache of 44_cross_dataset.py. Feature
importance = mean |SHAP| over the explained rows (and over output classes for
multiclass models).

  XGBoost  conditions G0, G-NP, G-NI (and G-TS in NF-v3), tasks BIN and CAN,
           20,000 test rows with equal numbers per class; G0 BIN and CORE
           models also explained on 20,000 rows of each other dataset of the
           same version.
  RF       condition G0, tasks BIN, CAN and CORE, without SHAP (option A,
           protocol §18 amendment: exact TreeSHAP on the default forests takes
           about 1.6 s per row): impurity importance and permutation importance
           (decrease of macro-F1 on 5,000 test rows, equal per class, 5 repeats);
           used for Q1-Q2 and compared with the XGBoost SHAP ranking.

Q4 (criterion fixed in §18 before any SHAP value was computed): CORE models
of ToN and CIC explained on NF-BoT-IoT DoS and DDoS rows; features ranked by
the mean signed SHAP toward the Benign output; supported if >= 3 of the top-5
Benign-pushing features are reply features, refuted if <= 1, inconclusive
with 2. The same quantity is computed on the source's own DoS/DDoS test rows
for comparison.

Usage:
  python scripts/47_shap.py --dataset NF-ToN-IoT-v3 [--seeds 1] [--tag smoke]
  python scripts/47_shap.py --report [--tag smoke]
Outputs: data/eval/shap[_<tag>]/<dataset>.jsonl and data/eval/shap[_<tag>]_report.txt
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations

import numpy as np
import shap
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance

from nfv3_config import DATASETS, IDENTITY_COLS, PORT_COLS, TIMESTAMP_COLS, WORK, grouped_split_codes, is_v2

SEEDS = [1, 2, 3, 4, 5]
ROWS = {"xgb": 20_000, "rf": 5_000}
PERM_REPEATS = 5
REPLY = {"OUT_PKTS", "OUT_BYTES", "DURATION_OUT", "SERVER_TCP_FLAGS"}  # plus every DST_TO_SRC_* column
SHORTCUTS = {"ports": set(PORT_COLS), "identity": set(IDENTITY_COLS), "timestamps": set(TIMESTAMP_COLS)}
OUT = WORK / "data/eval"

_spec = importlib.util.spec_from_file_location("cross", WORK / "scripts/44_cross_dataset.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)
P42, log = C.P42, C.log


def is_reply(f):
    return f in REPLY or f.startswith("DST_TO_SRC_")


def conditions(ds):
    c = {"G0": "base", "G-NP": "noport", "G-NI": "noid"}
    if not is_v2(ds):
        c["G-TS"] = "ts"
    return c


def balanced_rows(y, mask, n, rng):
    """Up to n rows from mask with equal numbers per class present."""
    classes = np.unique(y[mask])
    per = int(np.ceil(n / max(len(classes), 1)))
    out = []
    for c in classes:
        idx = np.flatnonzero(mask & (y == c))
        out.append(idx if len(idx) <= per else rng.choice(idx, per, replace=False))
    return np.sort(np.concatenate(out)) if out else np.array([], dtype=np.int64)


def shap_array(model, X):
    """XGBoost SHAP values as (rows, features, outputs); a binary model gives
    one output (log-odds of class 1). RF is not explained with SHAP (§18)."""
    sv = np.asarray(shap.TreeExplainer(model).shap_values(X, check_additivity=False))
    return sv[:, :, None] if sv.ndim == 2 else sv


def importance(sv, feat):
    imp = np.abs(sv).mean(axis=(0, 2))
    return {f: round(float(v), 7) for f, v in zip(feat, imp)}


def run(ds, seeds, tag):
    t0 = time.time()
    out_dir = OUT / (f"shap_{tag}" if tag else "shap")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ds}.jsonl"
    done = set()
    if out_path.exists():
        done = {(r["condition"], r["task"], r["model"], r["seed"]) for r in map(json.loads, out_path.read_text().splitlines())}
    a = C.open_cache(ds)
    targets = {n: C.open_cache(n) for n in C.version_of(ds) if n != ds}
    n = len(a["label"])
    jobs = [(c, t) for c in conditions(ds) for t in ("BIN", "CAN")] + [("G0", "CORE")]
    for cond, task in jobs:
        key = conditions(ds)[cond]
        feat = a["sets"][key]
        fidx = [a["cols"].index(c) for c in feat]
        h = a["hashes"][key]
        y, keep, names = C.task_labels(a, task)
        models = ("xgb", "rf") if cond == "G0" else ("xgb",)
        for seed in seeds:
            split = grouped_split_codes(h, seed)
            take = P42.train_sample(split, a["canon"], keep, np.random.default_rng(seed))
            present = np.unique(y[take])
            enc = np.full(len(names), -1)
            enc[present] = np.arange(len(present))
            Xtr = np.ascontiguousarray(a["X"][np.flatnonzero(take)][:, fidx])
            for mname in models:
                if (cond, task, mname, seed) in done:
                    continue
                t1 = time.time()
                model = P42.make_model(mname, seed)
                model.fit(Xtr, enc[y[take]])
                rng = np.random.default_rng(10_000 + seed)
                rows = balanced_rows(y, keep & (split == 2), ROWS[mname], rng)
                Xte = np.ascontiguousarray(a["X"][rows][:, fidx])
                res = {"dataset": ds, "condition": cond, "task": task, "model": mname, "seed": seed,
                       "features": feat, "classes": [names[i] for i in present], "rows_test": int(len(rows)),
                       "targets": {}}
                if mname == "rf":
                    pi = permutation_importance(model, Xte, enc[y[rows]], scoring="f1_macro",
                                                n_repeats=PERM_REPEATS, random_state=seed, n_jobs=1)
                    res["importance_kind"] = "permutation (macro-F1 decrease)"
                    res["importance_test"] = {f: round(float(v), 7) for f, v in zip(feat, pi.importances_mean)}
                    res["impurity_importance"] = {f: round(float(v), 7) for f, v in zip(feat, model.feature_importances_)}
                    res["seconds"] = round(time.time() - t1, 1)
                    with out_path.open("a") as f:
                        f.write(json.dumps(res) + "\n")
                    top = sorted(res["importance_test"], key=lambda k: -res["importance_test"][k])[:3]
                    log(f"{ds} {cond:5s} {task:4s} rf  s{seed}: permutation top {', '.join(top)}  {res['seconds']}s")
                    del model
                    continue
                sv = shap_array(model, Xte)
                res["importance_kind"] = "mean |SHAP|"
                res["importance_test"] = importance(sv, feat)
                benign = int(np.flatnonzero(present == names.index("Benign"))[0]) if task == "CORE" else None
                if task == "CORE":
                    attack = np.isin(y[rows], [names.index("DoS"), names.index("DDoS")])
                    res["benign_push_source"] = benign_push(sv, attack, benign, feat)
                explain_targets = cond == "G0" and task in ("BIN", "CORE")
                for tname, t in targets.items():
                    if not explain_targets:
                        continue
                    ty, tkeep, _ = C.task_labels(t, task, names)
                    trows = balanced_rows(ty, tkeep, ROWS[mname], rng)
                    tsv = shap_array(model, np.ascontiguousarray(t["X"][trows][:, [t["cols"].index(c) for c in feat]]))
                    entry = {"rows": int(len(trows)), "importance": importance(tsv, feat)}
                    if task == "CORE":
                        tatt = np.isin(ty[trows], [names.index("DoS"), names.index("DDoS")])
                        entry["benign_push"] = benign_push(tsv, tatt, benign, feat)
                        Xt = np.ascontiguousarray(t["X"][trows[tatt]][:, [t["cols"].index(c) for c in feat]])
                        entry["dos_ddos_predicted_benign"] = round(float(np.mean(present[model.predict(Xt)]
                                                                                 == names.index("Benign"))), 4)
                    res["targets"][tname] = entry
                res["seconds"] = round(time.time() - t1, 1)
                with out_path.open("a") as f:
                    f.write(json.dumps(res) + "\n")
                top = sorted(res["importance_test"], key=lambda k: -res["importance_test"][k])[:3]
                log(f"{ds} {cond:5s} {task:4s} {mname:3s} s{seed}: top {', '.join(top)}; "
                    f"targets {len(res['targets'])}  {res['seconds']}s")
                del model
    log(f"{ds}: done in {time.time() - t0:.0f}s")


def benign_push(sv, rows_mask, benign, feat):
    """Mean signed SHAP toward the Benign output on the given rows, Q4 ranking."""
    if rows_mask.sum() == 0:
        return None
    push = sv[rows_mask][:, :, benign].mean(axis=0)
    order = np.argsort(-push)
    top5 = [feat[i] for i in order[:5]]
    pos = push[push > 0]
    reply_mass = float(push[[is_reply(f) for f in feat]].clip(min=0).sum() / pos.sum()) if pos.sum() > 0 else None
    n_reply = sum(is_reply(f) for f in top5)
    verdict = "supported" if n_reply >= 3 else "refuted" if n_reply <= 1 else "inconclusive"
    return {"rows": int(rows_mask.sum()), "top5": top5, "top5_values": [round(float(push[i]), 6) for i in order[:5]],
            "reply_in_top5": n_reply, "verdict": verdict,
            "reply_share_of_positive_push": None if reply_mass is None else round(reply_mass, 4)}


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def topk(imp, k):
    return [f for f, _ in sorted(imp.items(), key=lambda x: -x[1])[:k]]


def jaccard(a, b):
    return len(set(a) & set(b)) / len(set(a) | set(b))


def spearman(i1, i2):
    shared = sorted(set(i1) & set(i2))
    return float(spearmanr([i1[f] for f in shared], [i2[f] for f in shared]).statistic)


def report(tag):
    out_dir = OUT / (f"shap_{tag}" if tag else "shap")
    rows = []
    for f in sorted(out_dir.glob("*.jsonl")):
        rows += [json.loads(x) for x in f.read_text().splitlines()]
    R = defaultdict(dict)
    for r in rows:
        R[(r["dataset"], r["condition"], r["task"], r["model"])][r["seed"]] = r
    med = lambda xs: f"{np.median(xs):.2f} [{min(xs):.2f}, {max(xs):.2f}]" if xs else "-"  # noqa: E731
    L = ["P4.5 SHAP explanations (research/phase4_protocol.md §18)",
         f"Generated: {datetime.now(timezone.utc).isoformat()}",
         "Importance = mean |SHAP| on test rows (equal per class). Values: median [min, max] over seeds.", ""]
    L.append("Q1. Share of the top-5 / top-10 features that are ports, identity features or timestamps")
    for (ds, cond, task, m), s in sorted(R.items()):
        sh = {g: ([sum(f in fs for f in topk(r["importance_test"], 5)) / 5 for r in s.values()],
                  [sum(f in fs for f in topk(r["importance_test"], 10)) / 10 for r in s.values()])
              for g, fs in SHORTCUTS.items()}
        first = topk(next(iter(s.values()))["importance_test"], 5)
        L.append(f"  {ds:22s} {cond:5s} {task:4s} {m:3s} n={len(s)}  " + "  ".join(
            f"{g} top5 {med(v5)} top10 {med(v10)}" for g, (v5, v10) in sh.items() if g != "timestamps" or cond == "G-TS")
                 + f"  | seed {min(s)} top5: {', '.join(first)}")
    L.append("")
    L.append("Q2. Rank stability: across seeds (pairs) and against G0 (same seed); Jaccard@10, Spearman")
    for (ds, cond, task, m), s in sorted(R.items()):
        imps = [r["importance_test"] for r in s.values()]
        jac = [jaccard(topk(x, 10), topk(z, 10)) for x, z in combinations(imps, 2)]
        spr = [spearman(x, z) for x, z in combinations(imps, 2)]
        line = f"  {ds:22s} {cond:5s} {task:4s} {m:3s} seeds: J@10 {med(jac)}  rho {med(spr)}"
        g0 = R.get((ds, "G0", task, m))
        if cond != "G0" and g0:
            pj, ps = [], []
            for sd, r in s.items():
                if sd in g0:
                    i0 = {f: v for f, v in g0[sd]["importance_test"].items() if f in r["importance_test"]}
                    pj.append(jaccard(topk(i0, 10), topk(r["importance_test"], 10)))
                    ps.append(spearman(i0, r["importance_test"]))
            line += f"  | vs G0: J@10 {med(pj)}  rho {med(ps)}"
        L.append(line)
    L.append("")
    L.append("Q2b. Same seed, G0: XGBoost mean |SHAP| ranking vs RF permutation ranking (Jaccard@10, Spearman)")
    for (ds, cond, task, m), s in sorted(R.items()):
        if m != "rf":
            continue
        xg = R.get((ds, cond, task, "xgb"), {})
        pj = [jaccard(topk(xg[sd]["importance_test"], 10), topk(r["importance_test"], 10)) for sd, r in s.items() if sd in xg]
        ps = [spearman(xg[sd]["importance_test"], r["importance_test"]) for sd, r in s.items() if sd in xg]
        L.append(f"  {ds:22s} {cond:5s} {task:4s} J@10 {med(pj)}  rho {med(ps)}")
    L.append("")
    L.append("Q3. Same G0 model explained on its test rows and on each other dataset (Jaccard@10, Spearman)")
    for (ds, cond, task, m), s in sorted(R.items()):
        if cond != "G0":
            continue
        tg = defaultdict(lambda: ([], []))
        for r in s.values():
            for t, e in r["targets"].items():
                tg[t][0].append(jaccard(topk(r["importance_test"], 10), topk(e["importance"], 10)))
                tg[t][1].append(spearman(r["importance_test"], e["importance"]))
        for t, (j, sp) in sorted(tg.items()):
            L.append(f"  {ds:22s} {task:4s} {m:3s} -> {t:22s} J@10 {med(j)}  rho {med(sp)}")
    L.append("")
    L.append("Q4. NF-BoT-IoT DoS/DDoS rows explained by ToN and CIC CORE models: verdict per seed (criterion §18),")
    L.append("    reply features in the top-5 Benign-pushing features, share of positive Benign push from reply features")
    for (ds, cond, task, m), s in sorted(R.items()):
        if task != "CORE":
            continue
        for t in sorted({t for r in s.values() for t in r["targets"] if "BoT" in t}):
            bp = [r["targets"][t]["benign_push"] for r in s.values() if t in r["targets"]]
            verdicts = [b["verdict"] for b in bp]
            maj = max(set(verdicts), key=verdicts.count)
            pred_b = [r["targets"][t]["dos_ddos_predicted_benign"] for r in s.values() if t in r["targets"]]
            src = [r["benign_push_source"] for r in s.values() if r.get("benign_push_source")]
            L.append(f"  {ds:22s} {m:3s} -> {t}: verdicts {verdicts} (majority: {maj}); reply in top5 "
                     f"{[b['reply_in_top5'] for b in bp]}; reply share {med([b['reply_share_of_positive_push'] for b in bp])}; "
                     f"DoS/DDoS predicted Benign {med(pred_b)}")
            first = next(iter(s.values()))
            L.append(f"      seed {first['seed']} top-5 toward Benign on BoT: {first['targets'][t]['benign_push']['top5']}")
            if src:
                L.append(f"      same model on its own DoS/DDoS test rows: reply in top5 {[b['reply_in_top5'] for b in src]}; "
                         f"top-5 {src[0]['top5']}")
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
