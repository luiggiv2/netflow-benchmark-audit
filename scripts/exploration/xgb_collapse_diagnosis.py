"""xgb_collapse_diagnosis.py — reviewer report 2026-10-06, point M4: why does
XGBoost (library defaults) collapse on NF-CSE-CIC-IDS2018-v3 CAN in some seeds
(G0 seed 3 macro-F1 0.088) while random forest on the same rows does not?

Reproduces one P4.2 run exactly (scripts/42_in_domain.py: same loader, split,
training sample, label encoding and model) and then checks:
  1. the reproduced test macro-F1 and per-class F1;
  2. the macro-F1 on the training sample itself (a collapse there means the
     fit diverged, not that the test data differ);
  3. predicted class distribution and non-finite probabilities;
  4. per-feature magnitude in the training sample;
  5. training mlogloss per boosting round on the training sample;
  6. variants: features clipped to |x| <= 1e9, max_delta_step=1, and the
     exact tree method, to locate the cause.

Usage: python scripts/exploration/xgb_collapse_diagnosis.py [--dataset D] [--task CAN|BIN]
         [--condition G0] [--seed 3]
Output: data/exploration/xgb_collapse_diagnosis[_<dataset>_<task>_<condition>_s<seed>].txt
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("in_domain", ROOT / "scripts/42_in_domain.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

DEFAULT = ("NF-CSE-CIC-IDS2018-v3", "CAN", "G0", 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT[0])
    ap.add_argument("--task", default=DEFAULT[1], choices=["CAN", "BIN"])
    ap.add_argument("--condition", default=DEFAULT[2])
    ap.add_argument("--seed", type=int, default=DEFAULT[3])
    a = ap.parse_args()
    DATASET = a.dataset
    L = [f"XGBoost collapse diagnosis: {DATASET} {a.task} {a.condition} seed {a.seed}",
         f"Generated: {datetime.now(timezone.utc).isoformat()}  xgboost {xgb.__version__}", ""]

    d = m.load(DATASET)
    n = len(d["label"])
    X, cols, sets = d["X"], d["cols"], d["sets"]
    excl = d["canon_names"].index(m.EXCLUDED) if m.EXCLUDED in d["canon_names"] else -1
    names = sorted(c for c in d["canon_names"] if c != m.EXCLUDED)
    can_map = np.full(len(d["canon_names"]), -1, dtype=np.int8)
    for i, c in enumerate(d["canon_names"]):
        if c != m.EXCLUDED:
            can_map[i] = names.index(c)
    y = can_map[d["canon"]].astype(np.int64)
    keep = d["canon"] != excl
    if a.task == "BIN":
        names, y, keep = ["Benign", "Attack"], d["label"].astype(np.int64), np.ones(n, dtype=bool)

    key, kind, filt = m.conditions(not m.is_v2(DATASET))[a.condition]
    if filt is not None:
        raise SystemExit("conditions with row filters are not supported here")
    feat = sets[key]
    fidx = [cols.index(c) for c in feat]
    h = d["hashes"][key]
    split = m.grouped_split_codes(h, a.seed) if kind == "grouped" else m.random_split_codes(n, a.seed)
    rng = np.random.default_rng(a.seed)
    take = m.train_sample(split, d["canon"], keep, rng)
    test = keep & (split == 2)
    present = np.unique(y[take])
    enc = np.full(len(names), -1)
    enc[present] = np.arange(len(present))
    Xtr = X[np.ix_(np.flatnonzero(take), fidx)]
    ytr = enc[y[take]]
    ti = np.flatnonzero(test)
    Xte = X[ti][:, fidx]
    yte = y[ti]
    labels = np.arange(len(names))
    L.append(f"train rows {len(ytr):,}, test rows {len(yte):,}, features {len(feat)}")
    L.append("train class counts: " + ", ".join(f"{names[c]} {int((y[take] == c).sum()):,}" for c in present))
    L.append("")

    def evaluate(tag, model, Xa, Xb):
        p_tr = present[model.predict(Xa)]
        p_te = present[model.predict(Xb)]
        f_tr = f1_score(y[take], p_tr, labels=present, average=None, zero_division=0)
        f_te = f1_score(yte, p_te, labels=labels, average=None, zero_division=0)
        in_te = np.bincount(yte, minlength=len(names)) > 0
        proba = model.predict_proba(Xb[:200_000])
        L.append(f"-- {tag}")
        L.append(f"   train macro-F1 {f_tr.mean():.4f}  test macro-F1 {f_te[in_te].mean():.4f}")
        L.append("   test per-class F1: " + ", ".join(f"{names[i]} {f_te[i]:.3f}" for i in labels if in_te[i]))
        L.append("   test predicted share: " + ", ".join(
            f"{names[c]} {(p_te == c).mean():.3f}" for c in present))
        L.append(f"   non-finite probabilities in first 200k test rows: {int((~np.isfinite(proba)).sum())}")
        return model

    base = evaluate("reproduction (library defaults)", m.make_model("xgb", a.seed).fit(Xtr, ytr), Xtr, Xte)
    L.append("")

    L.append("feature magnitude in the training sample (columns with max |x| >= 1e9 or NaN):")
    for j, c in enumerate(feat):
        col = Xtr[:, j]
        nan = int(np.isnan(col).sum())
        mx = float(np.nanmax(np.abs(col))) if nan < len(col) else float("nan")
        if mx >= 1e9 or nan:
            L.append(f"   {c:32s} max|x| {mx:.3e}  NaN {nan:,}")
    L.append("")

    ev = {}
    mdl = m.make_model("xgb", a.seed)
    metric = "mlogloss" if a.task == "CAN" else "logloss"
    mdl.set_params(eval_metric=metric)
    sub = np.random.default_rng(0).choice(len(ytr), min(200_000, len(ytr)), replace=False)
    mdl.fit(Xtr, ytr, eval_set=[(Xtr[sub], ytr[sub])], verbose=False)
    ev = mdl.evals_result()["validation_0"][metric]
    L.append(f"training {metric} by round (200k-row subsample of the training sample):")
    L.append("   " + ", ".join(f"r{r}:{ev[r]:.4g}" for r in (0, 1, 2, 5, 10, 20, 50, 99) if r < len(ev)))
    finite = [v for v in ev if np.isfinite(v)]
    L.append(f"   min {min(finite):.4g} at round {int(np.nanargmin(ev))}; non-finite rounds {len(ev) - len(finite)}")
    L.append("")

    Xtr_c = np.clip(Xtr, -1e9, 1e9)
    Xte_c = np.clip(Xte, -1e9, 1e9)
    evaluate("features clipped to |x| <= 1e9", m.make_model("xgb", a.seed).fit(Xtr_c, ytr), Xtr_c, Xte_c)
    mdl = m.make_model("xgb", a.seed)
    mdl.set_params(max_delta_step=1)
    evaluate("max_delta_step=1", mdl.fit(Xtr, ytr), Xtr, Xte)
    mdl = m.make_model("xgb", a.seed)
    mdl.set_params(tree_method="exact")
    evaluate("tree_method=exact", mdl.fit(Xtr, ytr), Xtr, Xte)
    del base

    run = (DATASET, a.task, a.condition, a.seed)
    suffix = "" if run == DEFAULT else "_" + "_".join(map(str, run))
    out = ROOT / f"data/exploration/xgb_collapse_diagnosis{suffix}.txt"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
