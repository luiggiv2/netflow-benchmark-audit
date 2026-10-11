"""rf_shap_cost.py — reviewer point (e), 2026-10-06: why the random forest is
explained with permutation importance and not with exact TreeSHAP (Section
4.5: "about 100,000 nodes", "1.6 s per row"; until now only in
research/phase4_protocol.md, P4.5 amendment).

Rebuilds the RF of NF-ToN-IoT-v3, G0, BIN, seed 1 exactly as 47_shap.py
(cache of 44_cross_dataset.py, same split, training sample and model), then
reports the number of nodes and the depth per tree and times exact TreeSHAP
(shap.TreeExplainer) on ROWS balanced test rows.

Usage: python scripts/exploration/rf_shap_cost.py
Output: data/exploration/rf_shap_cost.txt
"""

from __future__ import annotations

import importlib.util
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import shap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
_spec = importlib.util.spec_from_file_location("shapmod", ROOT / "scripts/47_shap.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)

DS, COND, TASK, SEED, ROWS = "NF-ToN-IoT-v3", "G0", "BIN", 1, 20


def main() -> None:
    C, P42 = S.C, S.P42
    a = C.open_cache(DS)
    key = S.conditions(DS)[COND]
    feat = a["sets"][key]
    fidx = [a["cols"].index(c) for c in feat]
    y, keep, names = C.task_labels(a, TASK)
    split = S.grouped_split_codes(a["hashes"][key], SEED)
    take = P42.train_sample(split, a["canon"], keep, np.random.default_rng(SEED))
    present = np.unique(y[take])
    enc = np.full(len(names), -1)
    enc[present] = np.arange(len(present))
    Xtr = np.ascontiguousarray(a["X"][np.flatnonzero(take)][:, fidx])
    t0 = time.time()
    model = P42.make_model("rf", SEED).fit(Xtr, enc[y[take]])
    fit_s = time.time() - t0
    nodes = np.array([e.tree_.node_count for e in model.estimators_])
    depth = np.array([e.tree_.max_depth for e in model.estimators_])
    rows = S.balanced_rows(y, keep & (split == 2), ROWS, np.random.default_rng(10_000 + SEED))
    Xte = np.ascontiguousarray(a["X"][rows][:, fidx])
    t0 = time.time()
    shap.TreeExplainer(model).shap_values(Xte)
    per_row = (time.time() - t0) / len(rows)
    L = [f"RF TreeSHAP cost: {DS} {COND} {TASK} seed {SEED} (model as in 47_shap.py)",
         f"Generated: {datetime.now(timezone.utc).isoformat()}  shap {shap.__version__}", "",
         f"training rows {len(Xtr):,}, features {len(feat)}, trees {len(nodes)}, fit {fit_s:.0f} s",
         f"nodes per tree: median {np.median(nodes):,.0f} (min {nodes.min():,}, max {nodes.max():,})",
         f"depth per tree: median {np.median(depth):.0f} (min {depth.min()}, max {depth.max()})",
         f"exact TreeSHAP on {len(rows)} test rows: {per_row:.2f} s per row "
         f"(5,000 rows: {per_row * 5000 / 60:.0f} min; the XGBoost runs explain 20,000 rows per dataset)"]
    out = ROOT / "data/exploration/rf_shap_cost.txt"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
