"""45_inspect_cross.py — P4.3b: traceable summary and inspection of the P4.3
cross-dataset results (research/phase4_protocol.md §14).

Part 1, tables from data/eval/cross/*.jsonl (no training): median target
macro-F1 per direction, task, model and condition; counts of directions in
which a condition is better, similar or worse than G0 on the target (median
difference of at least SIMILAR); the in-domain baseline and the degradation
(delta) measured against the grouped (G0) and the random (R0) source split.
Part 2, confusion matrices of the CORE task for the four directions into
NF-BoT-IoT (v2 and v3; sources ToN and CIC), refitted for seed 1 with the
P4.3 functions (XGBoost and RF, condition G0).
Part 3, what DoS and DDoS flows look like in each dataset: share of
one-packet flows, flows without a reply (OUT_PKTS = 0), protocol mix, the
most common destination ports, median packets and bytes; and the share of
NF-BoT-IoT DoS/DDoS rows whose exact feature vector (base key) also occurs
with the label Benign, DoS or DDoS in the other datasets of the same version.

Part 4, robustness of the P4.3 claims (no training; added 2026-10-04 after
the author's review): sign tests of each condition against G0 on the target
over the 24 direction x model cases, at thresholds 0.01, 0.02 and 0.05 and
with strict signs; the same counts without the cases at the floor (G0 target
median macro-F1 below FLOOR); and the CORE per-class F1 into NF-BoT-IoT for
every seed, condition and model.

Usage: python scripts/45_inspect_cross.py [--part 1 2 3 4]
Outputs: data/eval/cross_inspection.txt and .json (parts 1-3);
data/eval/cross_robustness.txt and .json (part 4)
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

from nfv3_config import DATASETS, WORK, grouped_split_codes

SIMILAR = 0.02
THRESHOLDS = [0.0, 0.01, 0.02, 0.05]
FLOOR = 0.05
OUT = WORK / "data/eval"
CONDS = ["G0", "R0", "G-NC", "G-NP", "G-NI"]

_spec = importlib.util.spec_from_file_location("cross", WORK / "scripts/44_cross_dataset.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)


def ab(d):
    return d.replace("NF-", "").replace("-IoT", "").replace("CSE-CIC-IDS2018", "CIC")


def part1():
    R = defaultdict(list)
    for f in glob.glob(str(OUT / "cross" / "*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            for t, v in r["targets"].items():
                R[(r["source"], t, r["task"], r["model"], r["condition"])].append((r, v))
    med = lambda xs: float(np.median(xs))  # noqa: E731
    L, res = [], {"tables": [], "counts": {}}
    L.append("Part 1a. Median target macro-F1 over seeds by condition; in = median in-domain macro-F1 (G0)")
    for task in ("BIN", "CORE"):
        L.append(f"  {task}: {'direction':22s} {'model':5s} {'in G0':>6s} " + " ".join(f"{c:>6s}" for c in CONDS))
        dirs = sorted({(s, t) for (s, t, tk, _, _) in R if tk == task}, key=lambda x: (x[0][-2:], x))
        for s, t in dirs:
            for m in ("xgb", "rf"):
                vals = [med([v["macro_f1"] for _, v in R[(s, t, task, m, c)]]) for c in CONDS]
                ind = med([r["in_domain"]["macro_f1"] for r, _ in R[(s, t, task, m, "G0")]])
                res["tables"].append({"task": task, "source": s, "target": t, "model": m, "in_domain_G0": ind,
                                      "target_median": dict(zip(CONDS, vals))})
                L.append(f"        {ab(s) + ' -> ' + ab(t):22s} {m:5s} {ind:6.3f} " + " ".join(f"{x:6.3f}" for x in vals))
    L.append("")
    L.append(f"Part 1b. Condition vs G0 on the target (median over seeds; similar = |difference| < {SIMILAR})")
    for task in ("BIN", "CORE"):
        for c in CONDS[1:]:
            cnt = {"better": 0, "similar": 0, "worse": 0}
            for (s, t, tk, m, cc), rv in R.items():
                if tk != task or cc != c:
                    continue
                d = med([v["macro_f1"] for _, v in rv]) - med([v["macro_f1"] for _, v in R[(s, t, tk, m, "G0")]])
                cnt["better" if d >= SIMILAR else "worse" if d <= -SIMILAR else "similar"] += 1
            res["counts"][f"{task} {c}"] = cnt
            L.append(f"  {task:4s} {c:5s}: better {cnt['better']:2d}, similar {cnt['similar']:2d}, worse {cnt['worse']:2d} "
                     f"(of {sum(cnt.values())} direction x model)")
    L.append("")
    L.append("Part 1c. Degradation (in-domain minus target, points; medians over seeds) against the grouped (G0) and")
    L.append("the random (R0) source split; R0 - G0 = how much the random baseline adds to the reported degradation")
    res["delta"] = []
    for task in ("BIN", "CORE"):
        L.append(f"  {task}: {'direction':22s} {'model':5s} {'delta G0':>9s} {'delta R0':>9s} {'R0-G0':>7s}")
        dirs = sorted({(s, t) for (s, t, tk, _, _) in R if tk == task}, key=lambda x: (x[0][-2:], x))
        for s, t in dirs:
            for m in ("xgb", "rf"):
                dg = med([v["delta_points"] for _, v in R[(s, t, task, m, "G0")]])
                dr = med([v["delta_points"] for _, v in R[(s, t, task, m, "R0")]])
                res["delta"].append({"task": task, "source": s, "target": t, "model": m,
                                     "delta_G0": dg, "delta_R0": dr})
                L.append(f"        {ab(s) + ' -> ' + ab(t):22s} {m:5s} {dg:+9.1f} {dr:+9.1f} {dr - dg:+7.1f}")
    diffs = [x["delta_R0"] - x["delta_G0"] for x in res["delta"]]
    L.append(f"  R0 - G0 over all {len(diffs)} cases: median {np.median(diffs):+.1f}, min {min(diffs):+.1f}, "
             f"max {max(diffs):+.1f}; positive in {sum(d > 0 for d in diffs)}")
    return L, res


def part2():
    L, res = ["Part 2. CORE confusion matrices into NF-BoT-IoT (condition G0, seed 1, refitted); rows = true class,"
              " columns = predicted (Benign, DoS, DDoS)"], []
    for src, tgt in (("NF-ToN-IoT-v3", "NF-BoT-IoT-v3"), ("NF-CSE-CIC-IDS2018-v3", "NF-BoT-IoT-v3"),
                     ("NF-ToN-IoT-v2", "NF-BoT-IoT-v2"), ("NF-CSE-CIC-IDS2018-v2", "NF-BoT-IoT-v2")):
        a, t = C.open_cache(src), C.open_cache(tgt)
        feat = a["sets"]["base"]
        fidx = [a["cols"].index(x) for x in feat]
        y, keep, names = C.task_labels(a, "CORE")
        split = grouped_split_codes(a["hashes"]["base"], 1)
        take = C.P42.train_sample(split, a["canon"], keep, np.random.default_rng(1))
        present = np.unique(y[take])
        enc = np.full(len(names), -1)
        enc[present] = np.arange(len(present))
        Xtr = np.ascontiguousarray(a["X"][np.flatnonzero(take)][:, fidx])
        tab = C.target_table(t, "base", [t["cols"].index(x) for x in feat], "CORE", names)
        for mname in ("xgb", "rf"):
            m = C.P42.make_model(mname, 1)
            m.fit(Xtr, enc[y[take]])
            pred = C.predict_table(m, tab["X"], present)
            M = np.zeros((3, 3), dtype=np.int64)
            np.add.at(M, (np.tile(np.arange(3), len(pred)), np.repeat(pred, 3)), tab["lab_counts"].ravel())
            share = (M[:, 0] / M.sum(1)).round(4).tolist()
            res.append({"source": src, "target": tgt, "model": mname, "classes": names,
                        "confusion": M.tolist(), "share_predicted_benign": share})
            L.append(f"  {ab(src)} -> {ab(tgt)} {mname}: " + "; ".join(
                f"{nm} {M[i].tolist()} (Benign {share[i]:.1%})" for i, nm in enumerate(names)))
            del m
    return L, res


def part3():
    L = ["Part 3. DoS and DDoS flows per dataset (canonical classes; Benign for reference)"]
    res = {"profiles": [], "overlap": []}
    feats = ["IN_PKTS", "OUT_PKTS", "IN_BYTES", "OUT_BYTES", "PROTOCOL", "L4_DST_PORT"]
    L.append(f"  {'dataset':12s} {'class':7s} {'rows':>11s} {'1 pkt':>6s} {'no reply':>8s} {'TCP':>6s} {'UDP':>6s} "
             f"{'med pkts':>8s} {'med bytes':>9s}  top destination ports (share)")
    for ds in DATASETS:
        d = C.open_cache(ds)
        canon = np.array(d["canon_names"], dtype=object)[d["canon"]]
        idx = {f: d["cols"].index(f) for f in feats}
        for cl in ("Benign", "DoS", "DDoS"):
            rows = np.flatnonzero(canon == cl)
            if not len(rows):
                continue
            rng = np.random.default_rng(0)
            s = np.sort(rng.choice(rows, min(len(rows), 2_000_000), replace=False))
            X = d["X"][s][:, [idx[f] for f in feats]]
            inp, outp, inb, proto, port = X[:, 0], X[:, 1], X[:, 2], X[:, 4], X[:, 5]
            ports, pc = np.unique(port, return_counts=True)
            top = ", ".join(f"{int(p)} {c / len(s):.0%}" for p, c in sorted(zip(ports, pc), key=lambda z: -z[1])[:3])
            prof = {"dataset": ds, "class": cl, "rows": int(len(rows)), "sampled": int(len(s)),
                    "one_packet": float(np.mean(inp + outp <= 1)), "no_reply": float(np.mean(outp == 0)),
                    "tcp": float(np.mean(proto == 6)), "udp": float(np.mean(proto == 17)),
                    "median_packets": float(np.median(inp + outp)), "median_in_bytes": float(np.median(inb)),
                    "top_dst_ports": top}
            res["profiles"].append(prof)
            L.append(f"  {ab(ds):12s} {cl:7s} {len(rows):11,d} {prof['one_packet']:6.1%} {prof['no_reply']:8.1%} "
                     f"{prof['tcp']:6.1%} {prof['udp']:6.1%} {prof['median_packets']:8.0f} {prof['median_in_bytes']:9.0f}  {top}")
    L.append("")
    L.append("  Exact-vector overlap (base key): share of NF-BoT-IoT DoS/DDoS rows whose vector also occurs in another")
    L.append("  dataset of the same version with the given canonical label")
    for v in ("v3", "v2"):
        bot = C.open_cache(f"NF-BoT-IoT-{v}")
        bcan = np.array(bot["canon_names"], dtype=object)[bot["canon"]]
        for other in (f"NF-ToN-IoT-{v}", f"NF-CSE-CIC-IDS2018-{v}"):
            o = C.open_cache(other)
            ocan = np.array(o["canon_names"], dtype=object)[o["canon"]]
            oh = o["hashes"]["base"]
            sets = {cl: np.unique(oh[ocan == cl]) for cl in ("Benign", "DoS", "DDoS")}
            for cl in ("DoS", "DDoS"):
                bh = bot["hashes"]["base"][bcan == cl]
                ov = {lab: float(np.isin(bh, s).mean()) for lab, s in sets.items()}
                res["overlap"].append({"bot": f"NF-BoT-IoT-{v}", "class": cl, "other": other, "share_in": ov})
                L.append(f"  BoT-{v} {cl:5s} rows with a vector seen in {ab(other):7s}: "
                         + ", ".join(f"as {lab} {x:.2%}" for lab, x in ov.items()))
    return L, res


def load_runs():
    R = defaultdict(list)
    for f in glob.glob(str(OUT / "cross" / "*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            for t, v in r["targets"].items():
                R[(r["source"], t, r["task"], r["model"], r["condition"])].append((r, v))
    return R


def part4():
    from scipy.stats import binomtest
    R = load_runs()
    med = lambda rv: float(np.median([v["macro_f1"] for _, v in rv]))  # noqa: E731
    L = ["Part 4. Robustness of the P4.3 claims",
         "Sign test (two-sided binomial, p = 0.5) of better vs worse cases against G0 on the target; a case is a",
         "direction x model (median over seeds). threshold 0 = strict sign; 'all' = 24 cases, 'off floor' = cases",
         f"whose G0 target median macro-F1 is at least {FLOOR}.", ""]
    res = {"sign_tests": [], "collapse_into_bot": []}
    L.append(f"  {'task':4s} {'cond':5s} {'cases':9s} {'thr':>5s} {'n':>3s} {'better':>6s} {'similar':>7s} {'worse':>5s} {'p':>9s}")
    for task in ("BIN", "CORE"):
        for c in CONDS[1:]:
            for scope in ("all", "off floor"):
                diffs = []
                for (s_, t, tk, m, cc), rv in R.items():
                    if tk != task or cc != c:
                        continue
                    g0 = med(R[(s_, t, tk, m, "G0")])
                    if scope == "off floor" and g0 < FLOOR:
                        continue
                    diffs.append(med(rv) - g0)
                for thr in THRESHOLDS:
                    if thr == 0:
                        b = sum(d > 0 for d in diffs); w = sum(d < 0 for d in diffs)
                    else:
                        b = sum(d >= thr for d in diffs); w = sum(d <= -thr for d in diffs)
                    sim = len(diffs) - b - w
                    pv = binomtest(b, b + w, 0.5).pvalue if b + w else float("nan")
                    res["sign_tests"].append({"task": task, "condition": c, "scope": scope, "threshold": thr,
                                              "cases": len(diffs), "better": b, "similar": sim, "worse": w,
                                              "p_value": pv})
                    L.append(f"  {task:4s} {c:5s} {scope:9s} {thr:5.2f} {len(diffs):3d} {b:6d} {sim:7d} {w:5d} {pv:9.2g}")
    L.append("")
    L.append("  CORE per-class F1 into NF-BoT-IoT, every seed: max over seeds of DoS and DDoS F1, and Benign F1 range")
    for (s_, t, tk, m, c), rv in sorted(R.items()):
        if tk != "CORE" or "BoT" not in t:
            continue
        dos = [v["per_class_f1"]["DoS"] for _, v in rv]
        ddos = [v["per_class_f1"]["DDoS"] for _, v in rv]
        ben = [v["per_class_f1"]["Benign"] for _, v in rv]
        res["collapse_into_bot"].append({"source": s_, "target": t, "model": m, "condition": c, "seeds": len(rv),
                                         "max_dos_f1": max(dos), "max_ddos_f1": max(ddos),
                                         "benign_f1": [min(ben), max(ben)]})
        L.append(f"    {ab(s_) + ' -> ' + ab(t):22s} {m:3s} {c:5s} n={len(rv)}  max DoS F1 {max(dos):.4f}  "
                 f"max DDoS F1 {max(ddos):.4f}  Benign F1 {min(ben):.4f}-{max(ben):.4f}")
    return L, res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--part", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--robustness", action="store_true", help="run part 4 only")
    a = ap.parse_args()
    if a.robustness:
        lines, res = part4()
        stamp = datetime.now(timezone.utc).isoformat()
        text = "\n".join([f"P4.3b robustness ({stamp})", ""] + lines) + "\n"
        (OUT / "cross_robustness.txt").write_text(text)
        (OUT / "cross_robustness.json").write_text(json.dumps({"generated_utc": stamp, **res}, indent=2))
        print(text)
        return
    out = {"generated_utc": datetime.now(timezone.utc).isoformat()}
    L = ["P4.3b inspection of the cross-dataset results", f"Generated: {out['generated_utc']}", ""]
    for p, fn in ((1, part1), (2, part2), (3, part3)):
        if p in a.part:
            lines, res = fn()
            L += lines + [""]
            out[f"part{p}"] = res
    text = "\n".join(L) + "\n"
    (OUT / "cross_inspection.txt").write_text(text)
    (OUT / "cross_inspection.json").write_text(json.dumps(out, indent=2, default=str))
    print(text)


if __name__ == "__main__":
    main()
