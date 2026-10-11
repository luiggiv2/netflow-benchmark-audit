"""m8_reply_check.py — reviewer point M8, 2026-10-07: robustness of the Q4
refutation (the collapse into NF-BoT-IoT is not driven by the missing reply).

Models: the CORE G0 XGBoost models of 47_shap.py for the ToN and CIC sources
(v2 and v3), seeds 1-5, rebuilt exactly (cache of 44_cross_dataset.py, same
split, training sample, model and explained rows).

Part a (sensitivity of the criterion): the Q4 ranking (mean signed SHAP
toward Benign on the NF-BoT-IoT DoS/DDoS rows) is scored with the original
reply set of 47_shap.py and with an expanded set that adds TCP_FLAGS (OR of
both directions, so it encodes whether the server answered),
TCP_WIN_MAX_OUT, RETRANSMITTED_OUT_* and MIN_IP_PKT_LEN (zero exactly when
OUT_PKTS is zero; added 2026-10-08 after reviewer point N1,
data/exploration/n1_reply_proxy_check.txt). The original-set verdicts must
equal the stored ones (check printed). The chance level is the expected
number of reply features in a random top-5.

Part b (intervention): the same CORE models (XGBoost and RF) are retrained
without the expanded reply set and compared with the G0 models on one
balanced sample of NF-BoT-IoT core rows (TARGET_ROWS per class): share of
DoS/DDoS rows predicted Benign and CORE macro-F1. Natural comparison (N1,
2026-10-08): the same share is given separately for the BoT DoS/DDoS rows
with a reply (OUT_PKTS > 0) and without one, with their size and TCP share.

Usage: python scripts/exploration/m8_reply_check.py --part a|b
Output: data/exploration/m8_reply_check_<part>.txt
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
_spec = importlib.util.spec_from_file_location("shapmod", ROOT / "scripts/47_shap.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)
C, P42 = S.C, S.P42

SOURCES = ["NF-ToN-IoT-v2", "NF-CSE-CIC-IDS2018-v2", "NF-ToN-IoT-v3", "NF-CSE-CIC-IDS2018-v3"]
SEEDS = [1, 2, 3, 4, 5]
EXTRA = {"TCP_FLAGS", "TCP_WIN_MAX_OUT", "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS", "MIN_IP_PKT_LEN"}
TARGET_ROWS = 50_000


def expanded(f: str) -> bool:
    return S.is_reply(f) or f in EXTRA


def verdict(n: int) -> str:
    return "supported" if n >= 3 else "refuted" if n <= 1 else "inconclusive"


def bot_of(src: str) -> str:
    return "NF-BoT-IoT-" + src[-2:]


def setup(src: str, seed: int, a: dict):
    feat = a["sets"]["base"]
    y, keep, names = C.task_labels(a, "CORE")
    split = S.grouped_split_codes(a["hashes"]["base"], seed)
    take = P42.train_sample(split, a["canon"], keep, np.random.default_rng(seed))
    present = np.unique(y[take])
    enc = np.full(len(names), -1)
    enc[present] = np.arange(len(present))
    return feat, y, keep, names, split, take, present, enc


def part_a() -> list[str]:
    stored = {}
    for src in SOURCES:
        for line in open(ROOT / f"data/eval/shap/{src}.jsonl"):
            r = json.loads(line)
            if r["model"] == "xgb" and r["condition"] == "G0" and r["task"] == "CORE":
                stored[(src, r["seed"])] = r["targets"][bot_of(src)]["benign_push"]
    L = ["M8a: Q4 criterion with the original and the expanded reply set (CORE G0 XGBoost -> NF-BoT-IoT DoS/DDoS)",
         f"Generated: {datetime.now(timezone.utc).isoformat()}",
         "Expanded set = original (OUT_PKTS, OUT_BYTES, DURATION_OUT, SERVER_TCP_FLAGS, DST_TO_SRC_*) + "
         + ", ".join(sorted(EXTRA)), ""]
    match = 0
    for src in SOURCES:
        a = C.open_cache(src)
        targets = {n: C.open_cache(n) for n in C.version_of(src) if n != src}
        feat = a["sets"]["base"]
        n_orig, n_exp = sum(map(S.is_reply, feat)), sum(map(expanded, feat))
        L.append(f"== {src} ({len(feat)} features; reply features: original {n_orig}, expanded {n_exp}; chance in a "
                 f"random top-5: {5 * n_orig / len(feat):.2f} / {5 * n_exp / len(feat):.2f})")
        for seed in SEEDS:
            feat, y, keep, names, split, take, present, enc = setup(src, seed, a)
            Xtr = np.ascontiguousarray(a["X"][np.flatnonzero(take)][:, [a["cols"].index(c) for c in feat]])
            model = P42.make_model("xgb", seed).fit(Xtr, enc[y[take]])
            rng = np.random.default_rng(10_000 + seed)
            S.balanced_rows(y, keep & (split == 2), S.ROWS["xgb"], rng)  # same rng sequence as 47_shap.py
            benign = int(np.flatnonzero(present == names.index("Benign"))[0])
            for tname, t in targets.items():
                ty, tkeep, _ = C.task_labels(t, "CORE", names)
                trows = S.balanced_rows(ty, tkeep, S.ROWS["xgb"], rng)
                if tname != bot_of(src):
                    continue
                tatt = np.isin(ty[trows], [names.index("DoS"), names.index("DDoS")])
                Xt = np.ascontiguousarray(t["X"][trows][:, [t["cols"].index(c) for c in feat]])
                sv = S.shap_array(model, Xt)
                push = sv[tatt][:, :, benign].mean(axis=0)
                top5 = [feat[i] for i in np.argsort(-push)[:5]]
                pos = push.clip(min=0).sum()
                share = lambda fn: float(push[[fn(f) for f in feat]].clip(min=0).sum() / pos)
                no, ne = sum(map(S.is_reply, top5)), sum(map(expanded, top5))
                ok = stored[(src, seed)]["verdict"] == verdict(no) and stored[(src, seed)]["top5"] == top5
                match += ok
                L.append(f"  seed {seed}: top-5 {top5}")
                L.append(f"          original: {no} in top-5 -> {verdict(no):12s} share {share(S.is_reply):.2f}"
                         f" | expanded: {ne} in top-5 -> {verdict(ne):12s} share {share(expanded):.2f}"
                         f"  [stored verdict reproduced: {ok}]")
        L.append("")
    L.append(f"Stored original-set results reproduced in {match} of {len(SOURCES) * len(SEEDS)} runs")
    return L


def part_b() -> list[str]:
    L = ["M8b: CORE models retrained without the expanded reply set, evaluated on NF-BoT-IoT core rows",
         f"Generated: {datetime.now(timezone.utc).isoformat()}",
         f"Same balanced target sample for both variants ({TARGET_ROWS:,} rows per class, seed 7); share of "
         "DoS/DDoS rows predicted Benign and CORE macro-F1; median [min, max] over seeds.",
         "Natural comparison: share predicted Benign among the DoS/DDoS rows with a reply (OUT_PKTS > 0) and "
         "without one.", ""]
    md = lambda v: f"{np.median(v):.3f} [{min(v):.3f}, {max(v):.3f}]"
    for src in SOURCES:
        a, t = C.open_cache(src), C.open_cache(bot_of(src))
        res = {}
        for seed in SEEDS:
            feat, y, keep, names, split, take, present, enc = setup(src, seed, a)
            ty, tkeep, _ = C.task_labels(t, "CORE", names)
            trows = S.balanced_rows(ty, tkeep, 3 * TARGET_ROWS, np.random.default_rng(7))
            tatt = np.isin(ty[trows], [names.index("DoS"), names.index("DDoS")])
            replied = t["X"][trows][:, t["cols"].index("OUT_PKTS")] > 0
            if seed == SEEDS[0]:
                tcp = t["X"][trows][:, t["cols"].index("PROTOCOL")] == 6
                parts = []
                for cname in ("DoS", "DDoS"):
                    r = ty[trows] == names.index(cname)
                    parts.append(f"{cname} {int(np.sum(r & replied)):,} with reply / {int(np.sum(r & ~replied)):,} "
                                 f"without")
                L_head = (f"  target DoS/DDoS rows: {'; '.join(parts)}; TCP share with reply "
                          f"{np.mean(tcp[tatt & replied]):.1%}, without {np.mean(tcp[tatt & ~replied]):.1%}")
            for variant, fs in (("G0", feat), ("no-reply", [f for f in feat if not expanded(f)])):
                Xtr = np.ascontiguousarray(a["X"][np.flatnonzero(take)][:, [a["cols"].index(c) for c in fs]])
                Xt = np.ascontiguousarray(t["X"][trows][:, [t["cols"].index(c) for c in fs]])
                for m in ("xgb", "rf"):
                    t0 = time.time()
                    model = P42.make_model(m, seed).fit(Xtr, enc[y[take]])
                    pred = present[model.predict(Xt)]
                    ben = pred == names.index("Benign")
                    res.setdefault((variant, m), []).append(
                        (float(np.mean(ben[tatt])),
                         float(f1_score(ty[trows], pred, labels=np.unique(ty[trows]), average="macro")),
                         len(fs), time.time() - t0,
                         float(np.mean(ben[tatt & replied])), float(np.mean(ben[tatt & ~replied]))))
                    C.log(f"{src} s{seed} {variant:8s} {m}: predicted Benign {res[(variant, m)][-1][0]:.3f} "
                          f"(reply {res[(variant, m)][-1][4]:.3f}, no reply {res[(variant, m)][-1][5]:.3f}) "
                          f"macro-F1 {res[(variant, m)][-1][1]:.3f} ({res[(variant, m)][-1][3]:.0f}s)")
        L.append(f"== {src} -> {bot_of(src)}")
        L.append(L_head)
        for (variant, m), v in sorted(res.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            L.append(f"  {m:3s} {variant:8s} ({v[0][2]} features): DoS/DDoS predicted Benign {md([x[0] for x in v])}"
                     f"  macro-F1 {md([x[1] for x in v])}")
            L.append(f"      with reply {md([x[4] for x in v])} | without reply {md([x[5] for x in v])}")
        L.append("")
    return L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["a", "b"], required=True)
    p = ap.parse_args().part
    L = part_a() if p == "a" else part_b()
    out = ROOT / f"data/exploration/m8_reply_check_{p}.txt"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
