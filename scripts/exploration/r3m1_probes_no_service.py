"""r3m1_probes_no_service.py — third reviewer report, major issue 1, 2026-10-10
(plan task P6.1b, block 1; author decisions of 2026-10-10): how much of the
RQ2 set-C separability comes from L7_PROTO and the ICMP fields?

Set C of the probes drops ports, TTL, TCP window and DNS fields, but keeps
L7_PROTO (the application label, which names the service of the server port,
data/exploration/r3m1_l7proto_icmp_check.txt) and ICMP_TYPE / ICMP_IPV4_TYPE
(non-zero on some TCP flows, concentrated in some classes). Set D is set C
without these three fields (34 features).

The probe sample, the common split (keyed on the 40 features of set A, seed
42) and the balanced train/test rows are rebuilt exactly as in
34_shortcut_probes.py --common; the stored depth-1 and depth-3 scores of sets
A and C must be reproduced (check printed). Per attack class: macro-F1 of
depth-1 and depth-3 trees for A, C and D on all test rows; the depth-1 rule of
C and D; and, as in n2_setc_leak_check.py, the depth-3 scores of A, C and D on
the test rows whose set-D vector has no copy in the training sample (rows
without a set-D copy have no set-C copy either). Classes left with fewer than
ten class or benign test rows without a copy are marked n/a.

Usage: python scripts/exploration/r3m1_probes_no_service.py
Output: data/exploration/r3m1_probes_no_service.txt
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
_spec = importlib.util.spec_from_file_location("probes", ROOT / "scripts/34_shortcut_probes.py")
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)
P.REKEY = P.COMMON = True

ORDER = ["NF-ToN-IoT-v2", "NF-BoT-IoT-v2", "NF-CSE-CIC-IDS2018-v2", "NF-ToN-IoT-v3", "NF-BoT-IoT-v3",
         "NF-CSE-CIC-IDS2018-v3"]
SERVICE = ["L7_PROTO", "ICMP_TYPE", "ICMP_IPV4_TYPE"]
A, C = P.SETS_REKEY["A"], P.SETS_REKEY["C"]
D = [c for c in C if c not in SERVICE]
MIN_ROWS = 10
OUT = ROOT / "data/exploration/r3m1_probes_no_service.txt"


def fit(tr, ytr, feats, depth):
    return DecisionTreeClassifier(max_depth=depth, random_state=P.SEED).fit(tr[feats].to_numpy(), ytr)


def rule1(m, feats) -> str:
    r = P.rules(m, feats, 1)
    return r[0]["path"] if r else "-"


def check(dataset: str) -> tuple[list[str], list[dict]]:
    stored = json.loads((P.OUT / "shortcut_probes_common" / f"{dataset}.json").read_text())["classes"]
    data, counts = P.sample_rekey(dataset)
    classes = sorted({lab for lab, _ in counts} - {P.BENIGN})
    L, rows = [f"== {dataset}"], []
    L.append(f"  {'class':28s} {'d1 A':>5s} {'d1 C':>5s} {'d1 D':>5s} {'d3 A':>5s} {'d3 C':>5s} {'d3 D':>5s}"
             f" {'D copy':>6s} | no copy: {'rows':>7s} {'A':>5s} {'C':>5s} {'D':>5s} | depth-1 rule C | depth-1 rule D")
    for cls in classes:
        d = data[data["native_label"].isin([P.BENIGN, cls])]
        rng = np.random.default_rng(P.SEED)
        tr = P.balance(d[d["split_A"] == "train"], cls, rng, P.TRAIN_CAP)
        te = P.balance(d[d["split_A"] == "test"], cls, rng, P.TEST_CAP)
        ytr = (tr["native_label"] == cls).to_numpy().astype(int)
        yte = (te["native_label"] == cls).to_numpy().astype(int)
        if ytr.sum() < MIN_ROWS or yte.sum() < MIN_ROWS:
            L.append(f"  {cls:28s} too few class rows in train or test")
            continue
        f1, pred3, rule = {}, {}, {}
        for name, feats in (("A", A), ("C", C), ("D", D)):
            for depth in (1, 3):
                m = fit(tr, ytr, feats, depth)
                p = m.predict(te[feats].to_numpy())
                f1[(name, depth)] = float(f1_score(yte, p, average="macro"))
                if depth == 3:
                    pred3[name] = p
                else:
                    rule[name] = rule1(m, feats)
        st = stored[cls]["sets"]
        ok = all(abs(f1[(k, dp)] - st[k][f"f1_d{dp}"]) < 1e-12 for k in ("A", "C") for dp in (1, 3))
        copy = np.isin(P.tabular_row_hash(te, D), list(set(P.tabular_row_hash(tr, D))))
        keep = ~copy
        n0, n1 = int(np.sum(keep & (yte == 0))), int(np.sum(keep & (yte == 1)))
        nc = {}
        if min(n0, n1) >= MIN_ROWS:
            nc = {k: float(f1_score(yte[keep], pred3[k][keep], average="macro")) for k in ("A", "C", "D")}
            tail = f"{int(keep.sum()):7,d} {nc['A']:5.2f} {nc['C']:5.2f} {nc['D']:5.2f}"
        else:
            tail = f"{int(keep.sum()):7,d}   n/a (benign {n0}, class {n1})"
        L.append(f"  {cls:28s} {f1[('A', 1)]:5.2f} {f1[('C', 1)]:5.2f} {f1[('D', 1)]:5.2f} {f1[('A', 3)]:5.2f}"
                 f" {f1[('C', 3)]:5.2f} {f1[('D', 3)]:5.2f} {copy.mean():6.1%} | no copy: {tail} | {rule['C']} | "
                 f"{rule['D']}{'' if ok else '  [stored value NOT reproduced]'}")
        rows.append({"ds": dataset, "cls": cls, "f1": f1, "nc": nc, "copy": float(copy.mean()), "ok": ok,
                     "rule_c": rule["C"], "rule_d": rule["D"]})
    return L + [""], rows


def summary(allrows: list[dict]) -> list[str]:
    L = ["Summary (classes with macro-F1 >= 0.95 / classes scored; losses = depth-3 macro-F1 difference)"]
    for v in ("v2", "v3"):
        r = [x for x in allrows if x["ds"].endswith(v)]
        cnt = "; ".join(f"{k} depth {dp}: {sum(x['f1'][(k, dp)] >= 0.95 for x in r)}/{len(r)}"
                        for k in ("A", "C", "D") for dp in (1, 3))
        L.append(f"  NF-{v}: {cnt}")
        for a, b in (("A", "C"), ("A", "D"), ("C", "D")):
            loss = np.array([x["f1"][(a, 3)] - x["f1"][(b, 3)] for x in r])
            worst = max(r, key=lambda x: x["f1"][(a, 3)] - x["f1"][(b, 3)])
            L.append(f"    {a} - {b}, all test rows: median {np.median(loss):.3f}, max {loss.max():.3f} "
                     f"({worst['ds']} {worst['cls']}, {worst['f1'][(a, 3)]:.2f} -> {worst['f1'][(b, 3)]:.2f}); "
                     f">= 0.05 in {int(np.sum(loss >= 0.05))} of {len(r)}")
        q = [x for x in r if x["nc"]]
        for a, b in (("A", "C"), ("A", "D")):
            loss = np.array([x["nc"][a] - x["nc"][b] for x in q])
            worst = max(q, key=lambda x: x["nc"][a] - x["nc"][b])
            L.append(f"    {a} - {b}, test rows without a set-D copy ({len(q)} classes scorable): median "
                     f"{np.median(loss):.3f}, max {loss.max():.3f} ({worst['ds']} {worst['cls']}, "
                     f"{worst['nc'][a]:.2f} -> {worst['nc'][b]:.2f}); >= 0.05 in {int(np.sum(loss >= 0.05))} of {len(q)}")
        L.append(f"    share of test rows with a set-D copy in training: median {np.median([x['copy'] for x in r]):.1%},"
                 f" max {max(x['copy'] for x in r):.1%}")
    n = len(allrows)
    big = [x for x in allrows if x["f1"][("C", 3)] - x["f1"][("D", 3)] >= 0.05]
    L.append(f"  All {n} classes: C - D loss >= 0.05 (depth 3, all test rows) in {len(big)}: "
             + ("; ".join(f"{x['ds']} {x['cls']} {x['f1'][('C', 3)]:.2f} -> {x['f1'][('D', 3)]:.2f}" for x in big)
                or "none"))
    svc = [x for x in allrows if x["rule_c"].split(" ")[0] in SERVICE]
    L.append(f"  Classes whose depth-1 rule in C uses L7_PROTO or an ICMP field: {len(svc)}; their depth-1 score C -> D: "
             + "; ".join(f"{x['ds']} {x['cls']} {x['f1'][('C', 1)]:.2f} -> {x['f1'][('D', 1)]:.2f} ({x['rule_d']})"
                         for x in svc))
    L.append(f"Stored common-split values (sets A and C, depth 1 and 3) reproduced in {sum(x['ok'] for x in allrows)}"
             f" of {n} classes")
    return L


def main() -> None:
    L = ["Major issue 1 (third reviewer report): RQ2 probes without L7_PROTO and the ICMP fields",
         f"Generated: {datetime.now(timezone.utc).isoformat()}",
         "Balanced samples and common split as in 34_shortcut_probes.py --common (keyed on the 40 features of set A,",
         f"seed 42). A = {len(A)} features; C = A without ports, TTL, TCP window and DNS fields ({len(C)});",
         f"D = C without {', '.join(SERVICE)} ({len(D)}). 'D copy': share of test rows whose set-D vector is in the",
         "training sample; 'no copy': depth-3 scores on the test rows without such a copy.", ""]
    allrows = []
    for ds in ORDER:
        part, rows = check(ds)
        L += part
        allrows += rows
        print("\n".join(part), flush=True)
    L += summary(allrows)
    OUT.write_text("\n".join(L) + "\n")
    print("\n".join(summary(allrows)))
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
