"""n5_threshold_check.py — second reviewer report, point N5 / question 3,
2026-10-08: is the collapse into NF-BoT-IoT a matter of the decision
threshold, or do the models give the BoT floods almost no attack probability?

Models: the CORE G0 models (XGBoost and RF) of the ToN and CIC sources (v2 and
v3), seeds 1-5, rebuilt as in m8_reply_check.py part b (cache of
44_cross_dataset.py, same split, training sample and model seed), applied to
the same balanced sample of NF-BoT-IoT core rows (50,000 per class, seed 7).

For each source, model and seed: attack probability = 1 - P(Benign) on the
BoT DoS/DDoS rows and, as the false-alarm reference, on the BoT Benign rows;
its median and 99th percentile, and the share of rows at or above thresholds
0.5, 0.25, 0.1 and 0.01 (the share of rows a lower threshold would flag as
attack). Threshold-free summary (added 2026-10-08): ROC AUC of p for BoT DoS/DDoS
against BoT Benign rows (0.5 = no separation). Fixed false-alarm view (added 2026-10-08): the
threshold t set at the 95th and 90th percentile of p on the BoT Benign rows, the share of BoT DoS/DDoS
rows with p > t (detected) and the share of Benign rows with p > t (actual false alarms; below the
target when p has ties). Medians [min, max] over seeds.

Usage: python scripts/exploration/n5_threshold_check.py
Output: data/exploration/n5_threshold_check.txt
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
_spec = importlib.util.spec_from_file_location("m8", ROOT / "scripts/exploration/m8_reply_check.py")
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)
S, C, P42 = M.S, M.C, M.P42

THRESHOLDS = [0.5, 0.25, 0.1, 0.01]
FALSE_ALARMS = [0.05, 0.10]
CHUNK = 50_000


def stats(p: np.ndarray) -> list[float]:
    return [float(np.median(p)), float(np.percentile(p, 99))] + [float(np.mean(p >= t)) for t in THRESHOLDS]


def main() -> None:
    L = ["N5 threshold check: attack probability of the CORE G0 models on NF-BoT-IoT core rows",
         f"Generated: {datetime.now(timezone.utc).isoformat()}",
         f"Balanced target sample ({M.TARGET_ROWS:,} rows per class, seed 7), as in m8_reply_check.py part b.",
         "p = 1 - P(Benign). Columns: median and 99th percentile of p; share of rows with p >= "
         + ", ".join(str(t) for t in THRESHOLDS) + ". Median [min, max] over seeds.", ""]
    cols = ["median p", "p99"] + [f">= {t}" for t in THRESHOLDS]
    fmt = lambda v: f"{np.median(v):.3f} [{min(v):.3f}, {max(v):.3f}]"
    summary = []
    for src in M.SOURCES:
        a, t = C.open_cache(src), C.open_cache(M.bot_of(src))
        res = {}
        for seed in M.SEEDS:
            feat, y, keep, names, split, take, present, enc = M.setup(src, seed, a)
            ty, tkeep, _ = C.task_labels(t, "CORE", names)
            trows = S.balanced_rows(ty, tkeep, 3 * M.TARGET_ROWS, np.random.default_rng(7))
            groups = {"DoS/DDoS": np.isin(ty[trows], [names.index("DoS"), names.index("DDoS")]),
                      "Benign": ty[trows] == names.index("Benign")}
            Xtr = np.ascontiguousarray(a["X"][np.flatnonzero(take)][:, [a["cols"].index(c) for c in feat]])
            Xt = np.ascontiguousarray(t["X"][trows][:, [t["cols"].index(c) for c in feat]])
            bcol = int(np.flatnonzero(present == names.index("Benign"))[0])
            for m in ("xgb", "rf"):
                model = P42.make_model(m, seed).fit(Xtr, enc[y[take]])
                pb = np.concatenate([model.predict_proba(Xt[i:i + CHUNK])[:, bcol] for i in range(0, len(Xt), CHUNK)])
                p = 1.0 - pb
                for g, mask in groups.items():
                    res.setdefault((m, g), []).append(stats(p[mask]))
                both = groups["DoS/DDoS"] | groups["Benign"]
                res.setdefault((m, "auc"), []).append(float(roc_auc_score(groups["DoS/DDoS"][both], p[both])))
                pf, pb_ = p[groups["DoS/DDoS"]], p[groups["Benign"]]
                for fa in FALSE_ALARMS:
                    thr = float(np.quantile(pb_, 1 - fa, method="higher"))
                    res.setdefault((m, fa), []).append((float(np.mean(pf > thr)), float(np.mean(pb_ > thr)), thr))
                C.log(f"{src} s{seed} {m}: DoS/DDoS median p {np.median(p[groups['DoS/DDoS']]):.4f}, "
                      f"share p >= 0.1 {np.mean(p[groups['DoS/DDoS']] >= 0.1):.3f}")
        L.append(f"== {src} -> {M.bot_of(src)}")
        for m in ("xgb", "rf"):
            for g in ("DoS/DDoS", "Benign"):
                v = np.array(res[(m, g)])
                L.append(f"  {m:3s} {g:8s} " + " | ".join(f"{c} {fmt(v[:, i])}" for i, c in enumerate(cols)))
            L.append(f"  {m:3s} ROC AUC DoS/DDoS vs Benign {fmt(res[(m, 'auc')])}  per seed "
                     + " ".join(f"{x:.3f}" for x in res[(m, 'auc')]))
            for fa in FALSE_ALARMS:
                v = np.array(res[(m, fa)])
                L.append(f"  {m:3s} threshold at Benign p{100 * (1 - fa):.0f}: detected {fmt(v[:, 0])} | false alarms "
                         f"{fmt(v[:, 1])} | threshold " + f"{np.median(v[:, 2]):.2g} [{v[:, 2].min():.2g}, {v[:, 2].max():.2g}]")
        L.append("")
        for m in ("xgb", "rf"):
            summary.append((f"{src} {m}", np.median(res[(m, "auc")]),
                            *[np.median([x[0] for x in res[(m, fa)]]) for fa in FALSE_ALARMS],
                            *[np.median([x[2] for x in res[(m, fa)]]) for fa in FALSE_ALARMS]))
    sm = np.array([x[1:] for x in summary])
    L.append("Summary over the eight source and model pairs (per-pair medians over seeds):")
    L.append(f"  ROC AUC: {sm[:, 0].min():.3f} to {sm[:, 0].max():.3f}")
    for i, fa in enumerate(FALSE_ALARMS):
        d, thr = sm[:, 1 + i], sm[:, 1 + len(FALSE_ALARMS) + i]
        L.append(f"  false alarms {fa:.0%}: floods detected {d.min():.3f} to {d.max():.3f}, median {np.median(d):.3f},"
                 f" pairs <= 0.004: {int(np.sum(d <= 0.004))}; threshold {thr.min():.2g} to {thr.max():.2g}")
    out = ROOT / "data/exploration/n5_threshold_check.txt"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
