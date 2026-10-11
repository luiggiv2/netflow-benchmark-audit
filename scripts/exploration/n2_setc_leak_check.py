"""n2_setc_leak_check.py — second reviewer report, point N2, 2026-10-08: do the
RQ2 probes of feature set C gain from near-copies that cross the common split?

The common split of 34_shortcut_probes.py --common is keyed on the 40 features
of set A. Set C drops ports, TTL, TCP window and DNS fields, so rows that
differ only in those fields share a set-C vector and may sit in train and
test. The probe docstring argues that trees of depth 1-3 cannot exploit this;
this script measures it.

The probe sample and balanced train/test rows are rebuilt exactly as in
34_shortcut_probes.py --common (same seeds; the depth-3 set-C score on all
test rows must equal the stored one, check printed). Per attack class:
  - share of test rows (class and benign) whose set-C vector appears in the
    balanced training sample the tree saw, and in the whole train partition
    of the sample;
  - the same share for set A (0 in the partition by construction);
  - depth-3 macro-F1 of sets A and C on all test rows and on the test rows
    without a set-C copy in the training sample, and the A - C loss on both.
Classes left with fewer than ten class or benign test rows without a copy are
marked n/a.

Usage: python scripts/exploration/n2_setc_leak_check.py
Output: data/exploration/n2_setc_leak_check.txt
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
MIN_ROWS = 10


def d3(tr, ytr, te, yte, feats):
    m = DecisionTreeClassifier(max_depth=3, random_state=P.SEED).fit(tr[feats].to_numpy(), ytr)
    return m.predict(te[feats].to_numpy())


def check(dataset: str) -> tuple[list[str], list[tuple]]:
    stored = json.loads((P.OUT / "shortcut_probes_common" / f"{dataset}.json").read_text())["classes"]
    data, counts = P.sample_rekey(dataset)
    classes = sorted({lab for lab, _ in counts} - {P.BENIGN})
    A, C = P.SETS_REKEY["A"], P.SETS_REKEY["C"]
    L, rows = [f"== {dataset}"], []
    L.append(f"  {'class':28s} {'test':>7s} {'C copy tr':>9s} {'C copy part':>11s} {'A copy tr':>9s}"
             f"   all: A    C   A-C | no copy: rows    A    C   A-C")
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
        hc_tr = set(P.tabular_row_hash(tr, C))
        hc_part = set(P.tabular_row_hash(d[d["split_A"] == "train"], C))
        hc_te = P.tabular_row_hash(te, C)
        ha_tr = set(P.tabular_row_hash(tr, A))
        copy_tr = np.isin(hc_te, list(hc_tr))
        copy_part = np.isin(hc_te, list(hc_part))
        copy_a = np.isin(P.tabular_row_hash(te, A), list(ha_tr))
        pa, pc = d3(tr, ytr, te, yte, A), d3(tr, ytr, te, yte, C)
        fa = f1_score(yte, pa, average="macro")
        fc = f1_score(yte, pc, average="macro")
        ok = abs(fc - stored[cls]["sets"]["C"]["f1_d3"]) < 1e-12 and abs(fa - stored[cls]["sets"]["A"]["f1_d3"]) < 1e-12
        keep = ~copy_tr
        n0, n1 = int(np.sum(keep & (yte == 0))), int(np.sum(keep & (yte == 1)))
        if min(n0, n1) >= MIN_ROWS:
            ga = f1_score(yte[keep], pa[keep], average="macro")
            gc = f1_score(yte[keep], pc[keep], average="macro")
            tail = f"{int(keep.sum()):12,d} {ga:5.2f} {gc:5.2f} {ga - gc:+5.2f}"
        else:
            ga = gc = None
            tail = f"{int(keep.sum()):12,d}   n/a (benign {n0}, class {n1})"
        L.append(f"  {cls:28s} {len(te):7,d} {copy_tr.mean():9.1%} {copy_part.mean():11.1%} {copy_a.mean():9.1%}"
                 f"   {fa:5.2f} {fc:5.2f} {fa - fc:+5.2f} | {tail}{'' if ok else '  [stored value NOT reproduced]'}")
        rows.append((dataset, cls, fa, fc, ga, gc, float(copy_tr.mean()), ok))
    return L + [""], rows


def main() -> None:
    L = ["N2 check: set-C probes on the common split (keyed on set A) with and without set-C copies in training",
         f"Generated: {datetime.now(timezone.utc).isoformat()}",
         "Depth-3 trees, balanced samples as in 34_shortcut_probes.py --common. 'C copy tr': share of test rows whose",
         "set-C vector is in the training sample; 'C copy part': in the whole train partition of the probe sample;",
         "'A copy tr': same for set A. 'no copy': test rows without a set-C copy in the training sample.", ""]
    allrows = []
    for ds in ORDER:
        part, rows = check(ds)
        L += part
        allrows += rows
        print("\n".join(part), flush=True)
    L.append("Summary (depth 3; classes with macro-F1 >= 0.95 / classes scored; A - C loss median and max)")
    for v in ("v2", "v3"):
        r = [x for x in allrows if x[0].endswith(v)]
        q = [x for x in r if x[4] is not None]
        L.append(f"  NF-{v}: all test rows: A {sum(x[2] >= 0.95 for x in r)}/{len(r)}, C {sum(x[3] >= 0.95 for x in r)}/{len(r)};"
                 f" loss median {np.median([x[2] - x[3] for x in r]):.3f}, max {max(x[2] - x[3] for x in r):.3f}")
        L.append(f"         no-copy rows ({len(q)} classes scorable): A {sum(x[4] >= 0.95 for x in q)}/{len(q)},"
                 f" C {sum(x[5] >= 0.95 for x in q)}/{len(q)}; loss median {np.median([x[4] - x[5] for x in q]):.3f},"
                 f" max {max(x[4] - x[5] for x in q):.3f}; share of test rows with a set-C copy: median"
                 f" {np.median([x[6] for x in r]):.1%}, max {max(x[6] for x in r):.1%}")
    L.append(f"Stored common-split values reproduced in {sum(x[7] for x in allrows)} of {len(allrows)} classes")
    out = ROOT / "data/exploration/n2_setc_leak_check.txt"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L[-6:]))
    print(f"written {out}")


if __name__ == "__main__":
    main()
