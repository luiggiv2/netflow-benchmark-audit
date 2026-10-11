"""r3m7_probes_seeds.py — third reviewer report, major issue 7, 2026-10-10
(plan task P6.1b, block 5; author decision of 2026-10-10, option b): how much
do the RQ2 probe results depend on the single grouped split (seed 42)?

The stored probes (34_shortcut_probes.py --common, data/audit/
shortcut_probes_common*) and their extensions (n2_setc_leak_check.py,
r3m1_probes_no_service.py) use one grouped split keyed on the 40 features of
set A with seed 42. The other probe versions on file (phase 3, rekeyed) differ
in the split key, not in the seed, so they do not measure the variation
between draws. Here the probe sample is rebuilt exactly as in
34_shortcut_probes.py --common (same Bernoulli sample, SEED 2026), and the
split of set A is drawn with seed 42 and with the five split seeds of Phase 4
(1-5); nothing else changes. Seed 42 must reproduce the stored depth-1 and
depth-3 scores of sets A and C (check printed). Stored files are not modified.

Per split seed and attack class (balanced train/test rows as in the probe
script): macro-F1 of depth-1 and depth-3 trees for sets A, C and D (D = C
without L7_PROTO, ICMP_TYPE, ICMP_IPV4_TYPE, as in r3m1_probes_no_service.py),
the depth-1 rule of A, the share of test rows whose set-C (and set-D) vector
occurs in the training sample, and the depth-3 scores on the test rows without
a set-C copy (A, C; as in n2_setc_leak_check.py) and without a set-D copy (A,
C, D; as in r3m1_probes_no_service.py). Classes left with fewer than ten class
rows in train or test are not probed; no-copy scores need ten rows of each
label.

Usage: python scripts/exploration/r3m7_probes_seeds.py
Output: data/exploration/r3m7_probes_seeds.txt and .json
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
_spec = importlib.util.spec_from_file_location("probes", ROOT / "scripts/34_shortcut_probes.py")
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)
P.REKEY = P.COMMON = True
from nfv3_config import grouped_split_codes  # noqa: E402

ORDER = ["NF-ToN-IoT-v2", "NF-BoT-IoT-v2", "NF-CSE-CIC-IDS2018-v2", "NF-ToN-IoT-v3", "NF-BoT-IoT-v3",
         "NF-CSE-CIC-IDS2018-v3"]
SPLIT_SEEDS = [42, 1, 2, 3, 4, 5]
SERVICE = ["L7_PROTO", "ICMP_TYPE", "ICMP_IPV4_TYPE"]
A, C = P.SETS_REKEY["A"], P.SETS_REKEY["C"]
D = [c for c in C if c not in SERVICE]
SETS = {"A": A, "C": C, "D": D}
MIN_ROWS = 10
OUT = ROOT / "data/exploration/r3m7_probes_seeds.txt"
# classes named in Section 5.2 (05:102) and in the approved block-1 wording
NAMED = [("NF-CSE-CIC-IDS2018-v2", "Infiltration"), ("NF-CSE-CIC-IDS2018-v3", "Infiltration"),
         ("NF-ToN-IoT-v3", "scanning"), ("NF-ToN-IoT-v2", "scanning"), ("NF-CSE-CIC-IDS2018-v2", "DDoS attacks-LOIC-HTTP"),
         ("NF-ToN-IoT-v3", "Backdoor"), ("NF-ToN-IoT-v2", "password"), ("NF-CSE-CIC-IDS2018-v2", "SSH-Bruteforce"),
         ("NF-CSE-CIC-IDS2018-v3", "SSH-Bruteforce"), ("NF-CSE-CIC-IDS2018-v2", "FTP-BruteForce"),
         ("NF-CSE-CIC-IDS2018-v3", "FTP-BruteForce")]


def sample_seeds(dataset: str) -> tuple[pd.DataFrame, Counter]:
    """P.sample_rekey with the split of set A drawn for every seed of SPLIT_SEEDS
    (columns split_A_<seed>); the Bernoulli sample and the row hashes are those of
    the probe script, so seed 42 gives its split_A."""
    pf = pq.ParquetFile(P.PROCESSED_ROOT / dataset / "tabular.parquet")
    counts = Counter()
    for b in pf.iter_batches(batch_size=P.BATCH, columns=["native_label"]):
        counts.update(b.column("native_label").to_pandas().astype(str))
    target = max(P.TRAIN_CAP / 0.70, P.TEST_CAP / 0.15)
    prob = {k: min(1.0, target / n) for k, n in counts.items()}
    rng = np.random.default_rng(P.SEED)
    names = {0: "train", 1: "val", 2: "test"}
    parts = []
    for b in pf.iter_batches(batch_size=P.BATCH, columns=P.FEATURE_COLS_V2 + ["native_label"]):
        df = b.to_pandas()
        lab = df["native_label"].astype(str)
        keep = rng.random(len(df)) < lab.map(prob).to_numpy()
        sub = df.loc[keep].copy()
        sub["native_label"] = lab[keep].to_numpy()
        h = P.tabular_row_hash(sub, A)
        for s in SPLIT_SEEDS:
            sub[f"split_A_{s}"] = pd.Series(grouped_split_codes(h, s)).map(names).to_numpy()
        parts.append(sub)
    data = pd.concat(parts, ignore_index=True)
    X = data[P.FEATURE_COLS_V2].to_numpy(dtype=np.float64)
    X = np.where(np.isfinite(X), np.clip(X, -P.F32, P.F32), np.nan)
    data[P.FEATURE_COLS_V2] = X
    return data, counts


def probe(d: pd.DataFrame, cls: str, col: str) -> dict | None:
    rng = np.random.default_rng(P.SEED)
    tr = P.balance(d[d[col] == "train"], cls, rng, P.TRAIN_CAP)
    te = P.balance(d[d[col] == "test"], cls, rng, P.TEST_CAP)
    ytr = (tr["native_label"] == cls).to_numpy().astype(int)
    yte = (te["native_label"] == cls).to_numpy().astype(int)
    if ytr.sum() < MIN_ROWS or yte.sum() < MIN_ROWS:
        return None
    f1, pred3, rule = {}, {}, "-"
    for name, feats in SETS.items():
        for depth in (1, 3):
            m = DecisionTreeClassifier(max_depth=depth, random_state=P.SEED).fit(tr[feats].to_numpy(), ytr)
            p = m.predict(te[feats].to_numpy())
            f1[f"{name}{depth}"] = float(f1_score(yte, p, average="macro"))
            if depth == 3:
                pred3[name] = p
            elif name == "A":
                r = P.rules(m, feats, 1)
                rule = r[0]["path"] if r else "-"
    out = {"f1": f1, "rule_a": rule, "test_rows": int(len(te))}
    for key, feats, sets in (("C", C, ("A", "C")), ("D", D, ("A", "C", "D"))):
        copy = np.isin(P.tabular_row_hash(te, feats), list(set(P.tabular_row_hash(tr, feats))))
        keep = ~copy
        n0, n1 = int(np.sum(keep & (yte == 0))), int(np.sum(keep & (yte == 1)))
        out[f"copy_{key}"] = float(copy.mean())
        out[f"nocopy_{key}"] = ({k: float(f1_score(yte[keep], pred3[k][keep], average="macro")) for k in sets}
                                if min(n0, n1) >= MIN_ROWS else None)
    return out


def rng_str(vals: list[float], fmt: str = "{:.2f}") -> str:
    return (fmt.format(min(vals)) + "-" + fmt.format(max(vals))) if vals else "n/a"


def main() -> None:
    res = {}  # (ds, seed, cls) -> probe output
    checks = []
    for ds in ORDER:
        data, counts = sample_seeds(ds)
        stored = json.loads((P.OUT / "shortcut_probes_common" / f"{ds}.json").read_text())["classes"]
        classes = sorted({lab for lab in counts} - {P.BENIGN})
        for s in SPLIT_SEEDS:
            col = f"split_A_{s}"
            for cls in classes:
                d = data[data["native_label"].isin([P.BENIGN, cls])]
                r = probe(d, cls, col)
                res[(ds, s, cls)] = r
                if s == 42:
                    st = stored.get(cls, {}).get("sets")
                    if r is None or not st:
                        checks.append((r is None) == (not st or "A" not in st))
                    else:
                        checks.append(all(abs(r["f1"][f"{k}{dp}"] - st[k][f"f1_d{dp}"]) < 1e-12
                                          for k in ("A", "C") for dp in (1, 3)))
            print(f"{ds} split seed {s}: {sum(res[(ds, s, c)] is not None for c in classes)} classes probed", flush=True)
        del data

    L = ["Major issue 7 (third reviewer report): RQ2 probes over six grouped splits",
         f"Generated: {datetime.now(timezone.utc).isoformat()}",
         "Probe sample and balanced rows as in 34_shortcut_probes.py --common; split of set A (40 features) drawn with",
         f"seeds {SPLIT_SEEDS} (42 = stored split; 1-5 = the split seeds of Phase 4). A = {len(A)} features; C = A",
         f"without ports, TTL, TCP window and DNS fields ({len(C)}); D = C without {', '.join(SERVICE)} ({len(D)}).",
         "Scores: macro-F1 of the class against Benign on the balanced test rows. 'no copy C/D': test rows whose",
         "set-C/set-D vector has no copy in the training sample.",
         f"Seed 42 reproduces the stored common-split values (A and C, depth 1 and 3) in {sum(checks)} of {len(checks)} classes",
         ""]
    summ = {}
    L.append("Part 1. Per split seed (classes with depth-3 macro-F1 >= 0.95 / classes probed)")
    for v in ("v2", "v3"):
        L.append(f"  NF-{v}")
        for s in SPLIT_SEEDS:
            rows = [(k, r) for k, r in res.items() if k[0].endswith(v) and k[1] == s and r is not None]
            n = len(rows)
            cnt = {f"{k}{dp}": sum(r["f1"][f"{k}{dp}"] >= 0.95 for _, r in rows) for k in "ACD" for dp in (1, 3)}
            lac = np.array([r["f1"]["A3"] - r["f1"]["C3"] for _, r in rows])
            lcd = np.array([r["f1"]["C3"] - r["f1"]["D3"] for _, r in rows])
            nc = [r for _, r in rows if r["nocopy_C"]]
            lnc = np.array([r["nocopy_C"]["A"] - r["nocopy_C"]["C"] for r in nc])
            nca = sum(r["nocopy_C"]["A"] >= 0.95 for r in nc)
            leak = float(np.median([r["copy_C"] for _, r in rows]))
            feats = Counter(r["rule_a"].split(" ")[0] for _, r in rows)
            summ[(v, s)] = {"n": n, **cnt, "lac_med": float(np.median(lac)), "lac_max": float(lac.max()),
                            "lac_ge05": int(np.sum(lac >= 0.05)), "lcd_med": float(np.median(lcd)),
                            "lcd_ge05": int(np.sum(lcd >= 0.05)), "nc_n": len(nc), "nc_A": int(nca),
                            "lnc_med": float(np.median(lnc)), "lnc_max": float(lnc.max()),
                            "lnc_ge05": int(np.sum(lnc >= 0.05)), "leak_med": leak,
                            "win": feats.get("TCP_WIN_MAX_IN", 0), "port": feats.get("L4_DST_PORT", 0)}
            x = summ[(v, s)]
            L.append(f"    seed {s:2d}: probed {n}; A d1 {x['A1']}, A d3 {x['A3']}, C d1 {x['C1']}, C d3 {x['C3']},"
                     f" D d3 {x['D3']}; A-C loss median {x['lac_med']:.3f}, max {x['lac_max']:.3f}, >=0.05 in"
                     f" {x['lac_ge05']}; C-D loss median {x['lcd_med']:.3f}, >=0.05 in {x['lcd_ge05']}")
            L.append(f"             set-C copy share median {leak:.1%}; no copy C ({len(nc)} scorable): A >= 0.95 in"
                     f" {nca}, A-C loss median {x['lnc_med']:.3f}, max {x['lnc_max']:.3f}, >= 0.05 in {x['lnc_ge05']};"
                     f" depth-1 rule A: TCP_WIN_MAX_IN {x['win']}, L4_DST_PORT {x['port']}")
    L.append("")
    L.append("Part 2. Range over the six split seeds (and over the five Phase-4 seeds 1-5)")
    for v in ("v2", "v3"):
        for key, lab in (("n", "classes probed"), ("A1", "A depth 1 >= 0.95"), ("A3", "A depth 3 >= 0.95"),
                         ("C1", "C depth 1 >= 0.95"), ("C3", "C depth 3 >= 0.95"), ("D3", "D depth 3 >= 0.95"),
                         ("lac_med", "A-C loss median"), ("lac_max", "A-C loss max"), ("lac_ge05", "A-C loss >= 0.05"),
                         ("lnc_max", "A-C loss max, no copy C"), ("lnc_ge05", "A-C loss >= 0.05, no copy C"),
                         ("nc_A", "A >= 0.95, no copy C"), ("leak_med", "set-C copy share median"),
                         ("win", "depth-1 TCP_WIN_MAX_IN"), ("port", "depth-1 L4_DST_PORT")):
            all6 = [summ[(v, s)][key] for s in SPLIT_SEEDS]
            p4 = [summ[(v, s)][key] for s in SPLIT_SEEDS if s != 42]
            f = "{:.1%}" if key == "leak_med" else ("{:.3f}" if isinstance(all6[0], float) else "{}")
            L.append(f"  NF-{v} {lab:28s}: all six {rng_str(all6, f)} | seeds 1-5 {rng_str(p4, f)} | seed 42 "
                     + f.format(summ[(v, 42)][key]))
    L.append("")
    L.append("Part 3. Classes named in the text: depth-3 macro-F1 per split seed (42, 1, 2, 3, 4, 5)")
    for ds, cls in NAMED:
        rs = [res.get((ds, s, cls)) for s in SPLIT_SEEDS]
        def row(fn):  # noqa: E306
            return " ".join(f"{fn(r):5.2f}" if r is not None and fn(r) is not None else "  n/a" for r in rs)
        L.append(f"  {ds} {cls}")
        for k in "ACD":
            L.append(f"    {k} all test rows : {row(lambda r, k=k: r['f1'][k + '3'])}")
        L.append(f"    A no copy C     : {row(lambda r: r['nocopy_C']['A'] if r['nocopy_C'] else None)}")
        L.append(f"    C no copy C     : {row(lambda r: r['nocopy_C']['C'] if r['nocopy_C'] else None)}")
        L.append(f"    D no copy D     : {row(lambda r: r['nocopy_D']['D'] if r['nocopy_D'] else None)}")
        L.append("    depth-1 rule A  : " + " | ".join(r["rule_a"] if r else "n/a" for r in rs))
    L.append("")
    L.append("Part 4. Per class: depth-3 macro-F1 of set A, range over the six split seeds (classes probed in all six)")
    for ds in ORDER:
        cl = sorted({k[2] for k in res if k[0] == ds})
        for cls in cl:
            rs = [res[(ds, s, cls)] for s in SPLIT_SEEDS]
            if any(r is None for r in rs):
                L.append(f"  {ds:22s} {cls:28s} probed in {sum(r is not None for r in rs)} of 6 seeds")
                continue
            a = [r["f1"]["A3"] for r in rs]
            c = [r["f1"]["C3"] for r in rs]
            L.append(f"  {ds:22s} {cls:28s} A {rng_str(a)} | C {rng_str(c)} | seed 42 A {a[0]:.2f} C {c[0]:.2f}")
    OUT.write_text("\n".join(L) + "\n")
    OUT.with_suffix(".json").write_text(json.dumps(
        {"split_seeds": SPLIT_SEEDS, "reproduced_seed42": [sum(checks), len(checks)],
         "summary": {f"{v} {s}": x for (v, s), x in summ.items()},
         "classes": {f"{ds} | {s} | {cls}": r for (ds, s, cls), r in res.items()}}, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
