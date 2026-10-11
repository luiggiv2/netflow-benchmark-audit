"""Compare the two versions of the RQ2 shallow-tree probes (34_shortcut_probes.py):
Phase 3 (stored grouped split keyed on all features, set A = 41 common
features) and the Phase 4 block A re-run (--rekey: set A = 40 features
without L4_SRC_PORT, each set split on its own features). Read-only; the
author asked for this comparison before choosing the version used in the
manuscript (P4.6, 2026-10-05).

Reported per version (NF-v2, NF-v3): classes with macro F1 >= 0.95 at depth 1
and 3 for sets A and C; median and maximum loss A - C at depth 3 (the
quantity behind F4); depth-1 features of set A; per-class values and the
classes whose value moves by at least DIFF between versions.

Usage: python scripts/exploration/compare_shortcut_probes.py
Output: data/exploration/shortcut_probes_comparison.txt
"""
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nfv3_config import WORK  # noqa: E402

DIFF = 0.03
VERSIONS = {"phase3": WORK / "data/audit/shortcut_probes", "rekeyed": WORK / "data/audit/shortcut_probes_rekeyed",
            "common": WORK / "data/audit/shortcut_probes_common"}


def load(d):
    return {json.loads(f.read_text())["dataset"]: json.loads(f.read_text()) for f in sorted(d.glob("*.json"))}


def summary(res, v2):
    out = {"n": 0, "a1": 0, "a3": 0, "c1": 0, "c3": 0, "loss": [], "feat": Counter(), "skipped": []}
    for ds, r in res.items():
        if ds.endswith("-v2") != v2:
            continue
        for c, x in r["classes"].items():
            if "note" in x:
                out["skipped"].append(f"{ds} {c}")
                continue
            s = x["sets"]
            out["n"] += 1
            out["a1"] += s["A"]["f1_d1"] >= 0.95
            out["a3"] += s["A"]["f1_d3"] >= 0.95
            out["c1"] += s["C"]["f1_d1"] >= 0.95
            out["c3"] += s["C"]["f1_d3"] >= 0.95
            out["loss"].append(s["A"]["f1_d3"] - s["C"]["f1_d3"])
            rule = s["A"]["rules_d1"][0]["path"] if s["A"]["rules_d1"] else "-"
            out["feat"][rule.split(" ")[0]] += 1
    return out


def main():
    R = {k: load(v) for k, v in VERSIONS.items()}
    L = ["RQ2 shallow-tree probes: Phase 3 version vs re-keyed version (Phase 4 block A)",
         f"Generated: {datetime.now(timezone.utc).isoformat()}",
         "phase3: set A = 41 common features, stored split keyed on all features (L4_SRC_PORT included).",
         "rekeyed: set A = 40 features (no L4_SRC_PORT), each set split on its own features. Set C is the same",
         "feature set in both (no ports, TTL, TCP window, DNS); only its split differs.",
         "common (added 2026-10-05): sets as in rekeyed, one split for all sets keyed on the 40 features of A.", ""]
    for v2 in (True, False):
        tag = "NF-v2" if v2 else "NF-v3"
        L.append(f"== {tag}")
        for k in R:
            s = summary(R[k], v2)
            loss = np.array(s["loss"])
            L.append(f"  {k:8s} probed {s['n']:2d}: >=0.95 A d1 {s['a1']}/{s['n']}, A d3 {s['a3']}/{s['n']}, "
                     f"C d1 {s['c1']}/{s['n']}, C d3 {s['c3']}/{s['n']}; loss A-C at d3 median {np.median(loss):.3f}, "
                     f"max {loss.max():.3f}")
            L.append(f"           depth-1 features (A): " + ", ".join(f"{f} {n}" for f, n in s["feat"].most_common()))
            if s["skipped"]:
                L.append(f"           not probed (too few rows): {', '.join(s['skipped'])}")
        L.append("")
    L.append(f"Per-class depth-3 macro F1 of sets A and C, and A - C, per version (phase3 / rekeyed / common);")
    L.append(f"* = the common-split value of A or C differs from phase3 by >= {DIFF}")
    for ds in sorted(R["phase3"]):
        L.append(f"  {ds}")
        for c in sorted(R["phase3"][ds]["classes"]):
            vals, ok = [], True
            for k in R:
                x = R[k][ds]["classes"].get(c, {"note": "missing"})
                if "note" in x:
                    vals.append(None)
                else:
                    vals.append((x["sets"]["A"]["f1_d3"], x["sets"]["C"]["f1_d3"]))
            fmt = lambda v: "   n/a           " if v is None else f"{v[0]:.2f} {v[1]:.2f} {v[0] - v[1]:+.2f}"  # noqa: E731
            p0, cm = vals[0], vals[-1]
            flag = "*" if p0 and cm and max(abs(cm[0] - p0[0]), abs(cm[1] - p0[1])) >= DIFF else " "
            L.append(f"   {flag}{c:28s} " + " | ".join(fmt(v) for v in vals))
    out = WORK / "data/exploration/shortcut_probes_comparison.txt"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
