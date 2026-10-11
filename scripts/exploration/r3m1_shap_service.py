"""r3m1_shap_service.py — third reviewer report, major issue 1, 2026-10-10
(plan task P6.1b, block 1; author decisions of 2026-10-10): Figure 9 and Q1
with L7_PROTO and the ICMP fields counted together with the destination port,
and whether the XGBoost models without the port (G-NP, G-NI) use them instead.

Reads the stored SHAP importances (data/eval/shap/*.jsonl, XGBoost, mean
|SHAP| on the source test rows); no model is refitted. The G-NP and G-NI
models are those of the protocol conditions (same splits and seeds as P4.2
and P4.3).

Part 1, Figure 9 (top-5 features, G0 and G-TS, BIN and CAN, mean over the
five seeds). Original grouping (ports, identity, timestamps, other) and new
grouping, each checked against data/paper/rq4_figures.json when that file
holds it (old grouping up to commit 071b59d, new grouping after Figure 9 was
regrouped on 2026-10-10); new grouping with "ports and
service" = L4_DST_PORT, L7_PROTO, ICMP_TYPE, ICMP_IPV4_TYPE. Median over the
12 G0 cells of each share, and the per-dataset values.

Part 2, per condition (G0, G-NP, G-NI; BIN and CAN): median over seeds of the
share of the top-5 and top-10 features that are L7_PROTO or ICMP fields, of
the share of the total mean |SHAP| they carry, and the rank of L7_PROTO and
of ICMP_TYPE (1 = most important).

Part 3, top-1 features: G0 cells (BIN, CAN, CORE) and G-NP / G-NI cells whose
top-1 feature is the same in every seed, and which it is.

Usage: python scripts/exploration/r3m1_shap_service.py
Output: data/exploration/r3m1_shap_service.txt
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nfv3_config import IDENTITY_COLS, PORT_COLS, TIMESTAMP_COLS, WORK  # noqa: E402

ORDER = ["NF-ToN-IoT-v2", "NF-BoT-IoT-v2", "NF-CSE-CIC-IDS2018-v2", "NF-ToN-IoT-v3", "NF-BoT-IoT-v3",
         "NF-CSE-CIC-IDS2018-v3"]
SERVICE = ["L7_PROTO", "ICMP_TYPE", "ICMP_IPV4_TYPE"]
OUT = WORK / "data/exploration/r3m1_shap_service.txt"
FIG = WORK / "data/paper/rq4_figures.json"


def cat_old(f: str) -> str:
    if f in PORT_COLS:
        return "ports"
    if f in IDENTITY_COLS:
        return "identity"
    if f in TIMESTAMP_COLS:
        return "timestamps"
    return "other"


def cat_new(f: str) -> str:
    return "ports+service" if f in PORT_COLS or f in SERVICE else cat_old(f)


def ranked(imp: dict) -> list[str]:
    return sorted(imp, key=imp.get, reverse=True)


def load() -> dict:
    R = defaultdict(dict)
    for ds in ORDER:
        for line in open(WORK / f"data/eval/shap/{ds}.jsonl"):
            r = json.loads(line)
            if r["model"] == "xgb":
                R[(r["dataset"], r["condition"], r["task"])][r["seed"]] = r["importance_test"]
    return R


def comp(seeds: dict, cat, cats: list[str], k: int = 5) -> np.ndarray:
    return np.mean([[sum(cat(f) == c for f in ranked(imp)[:k]) / k for c in cats] for imp in seeds.values()], axis=0)


def part1(R) -> list[str]:
    old_c = ["ports", "identity", "timestamps", "other"]
    new_c = ["ports+service", "identity", "timestamps", "other"]
    stored = json.loads(FIG.read_text())["F9"]
    L = ["== Part 1: Figure 9 (top-5, mean over seeds); old grouping | new grouping (ports+service = L4_DST_PORT,",
         "   L7_PROTO, ICMP_TYPE, ICMP_IPV4_TYPE)"]
    ok = 0
    g0 = []
    cells = [(ds, "G0", t) for ds in ORDER for t in ("BIN", "CAN")] + \
            [(ds, "G-TS", t) for ds in ORDER if ds.endswith("v3") for t in ("BIN", "CAN")]
    for ds, cond, task in cells:
        o = comp(R[(ds, cond, task)], cat_old, old_c)
        n = comp(R[(ds, cond, task)], cat_new, new_c)
        o10 = comp(R[(ds, cond, task)], cat_old, old_c, 10)
        n10 = comp(R[(ds, cond, task)], cat_new, new_c, 10)
        st = stored[f"{ds} {cond} {task}"]
        cats, vals = (old_c, o) if "ports" in st else (["ports and service", *new_c[1:]], n)
        same = all(abs(round(float(v), 3) - st[c]) < 1e-9 for c, v in zip(cats, vals))
        ok += same
        L.append(f"  {ds:22s} {cond:4s} {task}: " + ", ".join(f"{c} {v:.2f}" for c, v in zip(old_c, o)) + " | "
                 + ", ".join(f"{c} {v:.2f}" for c, v in zip(new_c, n)) + ("" if same else "  [NOT equal to stored F9]"))
        if cond == "G0":
            g0.append((ds, task, o, n, o10, n10))
    which = "Old" if "ports" in next(iter(stored.values())) else "New"
    L.append(f"  {which}-grouping values equal to {FIG.relative_to(WORK)}: {ok} of {len(cells)} cells")
    for label, idx in (("top-5", (2, 3)), ("top-10", (4, 5))):
        po = [x[idx[0]][0] + x[idx[0]][1] for x in g0]
        pn = [x[idx[1]][0] + x[idx[1]][1] for x in g0]
        so = [x[idx[0]][0] for x in g0]
        sn = [x[idx[1]][0] for x in g0]
        L.append(f"  G0, 12 cells, {label}: ports + identity median {np.median(po):.2f} (range {min(po):.2f}-{max(po):.2f})"
                 f" -> ports + service + identity median {np.median(pn):.2f} (range {min(pn):.2f}-{max(pn):.2f}); "
                 f"ports alone median {np.median(so):.2f} -> ports + service median {np.median(sn):.2f}")
    for base in ("NF-ToN-IoT", "NF-BoT-IoT", "NF-CSE-CIC-IDS2018"):
        xs = [x for x in g0 if x[0].startswith(base + "-")]
        po = [x[2][0] + x[2][1] for x in xs]
        pn = [x[3][0] + x[3][1] for x in xs]
        L.append(f"  G0 {base} (4 cells, top-5): ports + identity {min(po):.2f}-{max(po):.2f} -> "
                 f"ports + service + identity {min(pn):.2f}-{max(pn):.2f}")
    return L + [""]


def part2(R) -> list[str]:
    L = ["== Part 2: L7_PROTO and ICMP fields per condition (XGBoost; median over seeds [min, max])",
         f"  {'cell':32s} {'svc top5':>9s} {'svc top10':>10s} {'svc mass':>18s} {'rank L7_PROTO':>18s} {'rank ICMP_TYPE':>18s}"]
    med = lambda xs: f"{np.median(xs):.2f} [{min(xs):.2f}, {max(xs):.2f}]"  # noqa: E731
    medi = lambda xs: f"{np.median(xs):.0f} [{min(xs)}, {max(xs)}]"  # noqa: E731
    summ = defaultdict(list)
    for ds in ORDER:
        for task in ("BIN", "CAN"):
            for cond in ("G0", "G-NP", "G-NI"):
                s = R[(ds, cond, task)]
                t5, t10, mass, r7, ri = [], [], [], [], []
                for imp in s.values():
                    rk = ranked(imp)
                    t5.append(sum(f in SERVICE for f in rk[:5]) / 5)
                    t10.append(sum(f in SERVICE for f in rk[:10]) / 10)
                    tot = sum(imp.values())
                    mass.append(sum(imp.get(f, 0.0) for f in SERVICE) / tot if tot else 0.0)
                    r7.append(rk.index("L7_PROTO") + 1)
                    ri.append(rk.index("ICMP_TYPE") + 1)
                L.append(f"  {ds:22s} {task} {cond:4s} {np.median(t5):9.2f} {np.median(t10):10.2f} {med(mass):>18s} "
                         f"{medi(r7):>18s} {medi(ri):>18s}")
                summ[(cond, "t5")].append(np.median(t5))
                summ[(cond, "mass")].append(np.median(mass))
                summ[(cond, "r7")].append(np.median(r7))
    L.append("  Over the 12 cells (medians of the per-cell medians):")
    for cond in ("G0", "G-NP", "G-NI"):
        L.append(f"    {cond:4s}: service share of top-5 {np.median(summ[(cond, 't5')]):.2f} (max "
                 f"{max(summ[(cond, 't5')]):.2f}); service share of SHAP mass {np.median(summ[(cond, 'mass')]):.3f} (max "
                 f"{max(summ[(cond, 'mass')]):.3f}); rank of L7_PROTO {np.median(summ[(cond, 'r7')]):.1f}")
    return L + [""]


def part3(R) -> list[str]:
    L = ["== Part 3: top-1 feature identical in every seed"]
    for cond, tasks in (("G0", ("BIN", "CAN", "CORE")), ("G-NP", ("BIN", "CAN")), ("G-NI", ("BIN", "CAN"))):
        stable = Counter()
        n = 0
        lines = []
        for ds in ORDER:
            for task in tasks:
                s = R.get((ds, cond, task))
                if not s:
                    continue
                n += 1
                tops = {ranked(imp)[0] for imp in s.values()}
                if len(tops) == 1:
                    f = next(iter(tops))
                    stable[f] += 1
                    lines.append(f"{ds} {task} {f}")
        L.append(f"  {cond}: {n} cells; same top-1 in every seed in {sum(stable.values())}: "
                 + ", ".join(f"{f} {k}" for f, k in stable.most_common()))
        L.append("    " + "; ".join(lines))
    return L + [""]


def main() -> None:
    R = load()
    L = ["Major issue 1 (third reviewer report): SHAP rankings with L7_PROTO and the ICMP fields grouped with the port",
         f"Generated: {datetime.now(timezone.utc).isoformat()}",
         "XGBoost, mean |SHAP| on the source test rows (data/eval/shap/*.jsonl); five seeds per cell.",
         f"Service fields: {', '.join(SERVICE)}.", ""]
    L += part1(R) + part2(R) + part3(R)
    OUT.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
