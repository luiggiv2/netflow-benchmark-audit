"""n1_reply_proxy_check.py — second reviewer report, point N1, 2026-10-07:
does a feature outside the reply sets of 47_shap.py and m8_reply_check.py
encode the missing reply?

Part 1 (all rows of the six datasets, from the cache of 44_cross_dataset.py,
i.e. the values the models see): 2x2 table of OUT_PKTS == 0 against
MIN_IP_PKT_LEN == 0; among flows with a reply, share with MIN_IP_PKT_LEN ==
SHORTEST_FLOW_PKT; share with MAX_IP_PKT_LEN == LONGEST_FLOW_PKT; share whose
packet-size bins (NUM_PKTS_*) add up to IN_PKTS + OUT_PKTS and to IN_PKTS.

Part 2 (no new model): the Q4 verdicts recounted from the stored top-5
Benign-pushing features (data/eval/shap/*.jsonl, CORE G0 XGBoost, NF-BoT-IoT
DoS/DDoS rows) with four reply sets: original, original + MIN_IP_PKT_LEN,
expanded (m8_reply_check.py), expanded + MIN_IP_PKT_LEN.

Part 3: how well each feature kept by the "no-reply" retraining of
m8_reply_check.py part b (2026-10-07 run) separates flows without a reply (OUT_PKTS == 0)
from flows with one: ROC AUC of the raw value, reported as max(AUC, 1 - AUC),
on a balanced CORE sample (Benign, DoS, DDoS; 100,000 rows per class, seed 7)
of each source dataset and of NF-BoT-IoT. Non-finite values (NF-BoT-IoT-v3
rates) are ranked above every finite value.

Usage: python scripts/exploration/n1_reply_proxy_check.py
Output: data/exploration/n1_reply_proxy_check.txt
"""

from __future__ import annotations

import importlib.util
import json
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
S, C = M.S, M.C

ALL = ["NF-ToN-IoT-v2", "NF-BoT-IoT-v2", "NF-CSE-CIC-IDS2018-v2", "NF-ToN-IoT-v3", "NF-BoT-IoT-v3",
       "NF-CSE-CIC-IDS2018-v3"]
BINS = ["NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES", "NUM_PKTS_256_TO_512_BYTES",
        "NUM_PKTS_512_TO_1024_BYTES", "NUM_PKTS_1024_TO_1514_BYTES"]
# expanded reply set of m8_reply_check.py as run on 2026-10-07 (before MIN_IP_PKT_LEN was added to it)
EXTRA_M8 = {"TCP_FLAGS", "TCP_WIN_MAX_OUT", "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS"}
CHUNK = 2_000_000
PER_CLASS = 100_000
TOP = 10


def expanded_m8(f: str) -> bool:
    return S.is_reply(f) or f in EXTRA_M8


def part1() -> list[str]:
    L = ["== Part 1: exact relations on all rows (cache values)"]
    for ds in ALL:
        a = C.open_cache(ds)
        X, cols = a["X"], a["cols"]
        ix = {c: cols.index(c) for c in ["OUT_PKTS", "IN_PKTS", "MIN_IP_PKT_LEN", "SHORTEST_FLOW_PKT",
                                         "MAX_IP_PKT_LEN", "LONGEST_FLOW_PKT", *BINS]}
        n = t00 = t01 = t10 = t11 = reply_eq = maxeq = bins_all = bins_in = 0
        for i in range(0, X.shape[0], CHUNK):
            B = np.asarray(X[i:i + CHUNK])
            out0 = B[:, ix["OUT_PKTS"]] == 0
            min0 = B[:, ix["MIN_IP_PKT_LEN"]] == 0
            n += len(B)
            t11 += int(np.sum(out0 & min0))
            t10 += int(np.sum(out0 & ~min0))
            t01 += int(np.sum(~out0 & min0))
            t00 += int(np.sum(~out0 & ~min0))
            reply_eq += int(np.sum(~out0 & (B[:, ix["MIN_IP_PKT_LEN"]] == B[:, ix["SHORTEST_FLOW_PKT"]])))
            maxeq += int(np.sum(B[:, ix["MAX_IP_PKT_LEN"]] == B[:, ix["LONGEST_FLOW_PKT"]]))
            bsum = B[:, [ix[b] for b in BINS]].astype(np.float64).sum(axis=1)
            pin = B[:, ix["IN_PKTS"]].astype(np.float64)
            bins_all += int(np.sum(bsum == pin + B[:, ix["OUT_PKTS"]].astype(np.float64)))
            bins_in += int(np.sum(bsum == pin))
        reply = t00 + t01
        L += [f"  {ds}: {n:,} rows; OUT_PKTS == 0 in {t11 + t10:,} ({(t11 + t10) / n:.2%})",
              f"    OUT_PKTS == 0 and MIN_IP_PKT_LEN == 0: {t11:,} | OUT_PKTS == 0 and MIN_IP_PKT_LEN > 0: {t10:,}"
              f" | OUT_PKTS > 0 and MIN_IP_PKT_LEN == 0: {t01:,} | OUT_PKTS > 0 and MIN_IP_PKT_LEN > 0: {t00:,}",
              f"    with a reply, MIN_IP_PKT_LEN == SHORTEST_FLOW_PKT: {reply_eq:,} of {reply:,}"
              f" ({reply_eq / max(reply, 1):.4%})",
              f"    MAX_IP_PKT_LEN == LONGEST_FLOW_PKT: {maxeq:,} of {n:,} ({maxeq / n:.4%})",
              f"    sum of NUM_PKTS_* bins == IN_PKTS + OUT_PKTS: {bins_all / n:.4%}; == IN_PKTS: {bins_in / n:.4%}"]
    return L + [""]


def part2() -> list[str]:
    sets = {"original": S.is_reply,
            "original + MIN_IP_PKT_LEN": lambda f: S.is_reply(f) or f == "MIN_IP_PKT_LEN",
            "expanded": expanded_m8,
            "expanded + MIN_IP_PKT_LEN": lambda f: expanded_m8(f) or f == "MIN_IP_PKT_LEN"}
    L = ["== Part 2: Q4 verdicts recounted from the stored top-5 (CORE G0 XGBoost -> NF-BoT-IoT DoS/DDoS)",
         "  verdict: supported if >= 3 reply features in the top-5, refuted if <= 1, inconclusive with 2;"
         " majority over seeds"]
    totals = {k: {"supported": 0, "refuted": 0, "inconclusive": 0} for k in sets}
    for src in M.SOURCES:
        rows = {}
        for line in open(ROOT / f"data/eval/shap/{src}.jsonl"):
            r = json.loads(line)
            if r["model"] == "xgb" and r["condition"] == "G0" and r["task"] == "CORE":
                rows[r["seed"]] = r["targets"][M.bot_of(src)]["benign_push"]["top5"]
        L.append(f"  {src}")
        for name, fn in sets.items():
            counts = [sum(map(fn, rows[s])) for s in M.SEEDS]
            verdicts = [M.verdict(c) for c in counts]
            for v in verdicts:
                totals[name][v] += 1
            major = max(set(verdicts), key=verdicts.count)
            L.append(f"    {name:26s} counts {counts} -> {verdicts}  majority: {major}"
                     f" ({verdicts.count(major)} of {len(verdicts)})")
    L.append("  Totals over the 20 seeds:")
    for name, t in totals.items():
        L.append(f"    {name:26s} refuted {t['refuted']}, inconclusive {t['inconclusive']}, supported {t['supported']}")
    return L + [""]


def auc_sep(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64)
    fin = np.isfinite(x)
    if not fin.all():
        top = x[fin].max() if fin.any() else 0.0
        x = np.where(fin, x, top + 1.0)
    if np.all(x == x[0]):
        return 0.5
    auc = roc_auc_score(y, x)
    return max(auc, 1.0 - auc)


def part3() -> list[str]:
    L = ["== Part 3: separation of no-reply flows (OUT_PKTS == 0) by the features of the no-reply retraining",
         f"  balanced CORE sample, {PER_CLASS:,} rows per class (seed 7); max(AUC, 1 - AUC); top {TOP}"]
    for ds in ALL:
        a = C.open_cache(ds)
        feat = a["sets"]["base"]
        kept = [f for f in feat if not expanded_m8(f)]
        y, keep, names = C.task_labels(a, "CORE")
        rows = S.balanced_rows(y, keep, 3 * PER_CLASS, np.random.default_rng(7))
        X = np.asarray(a["X"][rows])
        noreply = X[:, a["cols"].index("OUT_PKTS")] == 0
        by_class = ", ".join(f"{names[c]} {np.mean(noreply[y[rows] == c]):.1%}" for c in np.unique(y[rows]))
        res = sorted(((auc_sep(X[:, a["cols"].index(f)], noreply), f) for f in kept), reverse=True)
        L.append(f"  {ds} ({len(kept)} kept features of {len(feat)}); no reply: {by_class}")
        L.append("    " + "; ".join(f"{f} {v:.3f}" for v, f in res[:TOP]))
    return L + [""]


def main() -> None:
    L = ["N1 check: features that encode the missing reply (second reviewer report, 2026-10-07)",
         f"Generated: {datetime.now(timezone.utc).isoformat()}", ""]
    for part in (part1, part2, part3):
        L += part()
        print("\n".join(L[-40:]), flush=True)
    out = ROOT / "data/exploration/n1_reply_proxy_check.txt"
    out.write_text("\n".join(L) + "\n")
    print(f"written {out}")


if __name__ == "__main__":
    main()
