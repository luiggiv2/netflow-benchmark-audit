"""xgb_default_sensitivity.py — second reviewer report, point N8, 2026-10-08:
the main XGBoost-dependent results with the library-default XGBoost (the
sensitivity runs kept in data/eval/*_xgbdefault*) and with max_delta_step=1
(the setting of the paper), side by side. The random forest runs are the same
in both.

  In-domain (P4.2, 12 XGBoost cells): G0 cells with a seed range >= 0.05 and the
    median range; cells where the R0 median exceeds the G0 median; G-TS minus R0
    for the six NF-v3 cells.
  Cross-dataset (P4.3): BIN sign counts of G-NP and G-NI against G0 (case =
    direction x model, median over seeds; strict and 0.02); CORE combinations
    (ToN and CIC sources, models, conditions) where the DoS and DDoS F1 on
    NF-BoT-IoT stay below 0.01 in every seed.
  SHAP (P4.5): median over G0 XGBoost cells of the per-cell median share of
    ports and identity features in the top-5; Q4 verdicts (criterion of §18,
    original reply set) over the 20 seeds; median Spearman between the source
    and target rankings of the same G0 model (BIN, CORE).

Usage: python scripts/exploration/xgb_default_sensitivity.py
Output: data/exploration/xgb_default_sensitivity.txt
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from nfv3_config import IDENTITY_COLS, PORT_COLS  # noqa: E402

DATASETS = ["NF-ToN-IoT-v2", "NF-BoT-IoT-v2", "NF-CSE-CIC-IDS2018-v2", "NF-ToN-IoT-v3", "NF-BoT-IoT-v3",
            "NF-CSE-CIC-IDS2018-v3"]
SETTINGS = {"default": "_xgbdefault", "max_delta_step=1": ""}
SEEDS = [1, 2, 3, 4, 5]
SHORT = set(PORT_COLS) | set(IDENTITY_COLS)


def jsonl(d: str, ds: str) -> list[dict]:
    return [json.loads(x) for x in open(ROOT / f"data/eval/{d}/{ds}.jsonl")]


def in_domain(tag: str) -> list[str]:
    cells = defaultdict(dict)
    for ds in DATASETS:
        for r in jsonl(f"in_domain{tag}", ds):
            if r["model"] == "xgb":
                cells[(ds, r["task"])].setdefault(r["condition"], []).append(r["macro_f1"])
    rng = [max(c["G0"]) - min(c["G0"]) for c in cells.values()]
    r0 = sum(np.median(c["R0"]) > np.median(c["G0"]) for c in cells.values())
    ts = [f"{ds.split('-')[1]}-{ds[-2:]} {t} {100 * (np.median(c['G-TS']) - np.median(c['R0'])):+.1f}"
          for (ds, t), c in sorted(cells.items()) if "G-TS" in c]
    return [f"    G0 seed range >= 0.05 in {sum(x >= 0.05 for x in rng)} of {len(rng)} XGBoost cells "
            f"(median range {np.median(rng):.3f}); R0 median above G0 in {r0} of {len(cells)}",
            "    G-TS minus R0 (points): " + "; ".join(ts)]


def cross(tag: str) -> list[str]:
    R = {}
    for ds in DATASETS:
        for r in jsonl(f"cross{tag}", ds):
            R[(ds, r["task"], r["model"], r["condition"], r["seed"])] = r
    out = []
    for c in ("G-NP", "G-NI"):
        diffs = []
        for ds in DATASETS:
            for m in ("xgb", "rf"):
                for t in R[(ds, "BIN", m, "G0", 1)]["targets"]:
                    g0 = np.median([R[(ds, "BIN", m, "G0", s)]["targets"][t]["macro_f1"] for s in SEEDS])
                    cv = np.median([R[(ds, "BIN", m, c, s)]["targets"][t]["macro_f1"] for s in SEEDS])
                    diffs.append(cv - g0)
        out.append(f"    BIN {c}: strict worse {sum(d < 0 for d in diffs)}, better {sum(d > 0 for d in diffs)} of "
                   f"{len(diffs)}; at 0.02 worse {sum(d <= -0.02 for d in diffs)}, better {sum(d >= 0.02 for d in diffs)}")
    n = low = 0
    for ds in DATASETS:
        if "BoT" in ds:
            continue
        bot = "NF-BoT-IoT-" + ds[-2:]
        for m in ("xgb", "rf"):
            for c in ("G0", "R0", "G-NC", "G-NP", "G-NI"):
                f = [R[(ds, "CORE", m, c, s)]["targets"][bot]["per_class_f1"] for s in SEEDS]
                n += 1
                low += all(x["DoS"] < 0.01 and x["DDoS"] < 0.01 for x in f)
    out.append(f"    CORE into NF-BoT-IoT: DoS and DDoS F1 < 0.01 in every seed in {low} of {n} combinations")
    return out


def shap(tag: str) -> list[str]:
    cell_share, verdicts = defaultdict(list), Counter()
    for ds in DATASETS:
        for r in jsonl(f"shap{tag}", ds):
            if r["model"] != "xgb":
                continue
            if r["condition"] == "G0" and r["task"] in ("BIN", "CAN"):
                imp = r["importance_test"]
                imp = dict(imp) if isinstance(imp, dict) else {k: v for k, v in imp}
                top5 = sorted(imp, key=imp.get, reverse=True)[:5]
                cell_share[(ds, r["task"])].append(sum(f in SHORT for f in top5) / 5)
            if r["condition"] == "G0" and r["task"] == "CORE" and "BoT" not in ds:
                verdicts[r["targets"]["NF-BoT-IoT-" + ds[-2:]]["benign_push"]["verdict"]] += 1
    share = np.median([np.median(v) for v in cell_share.values()])
    rep = (ROOT / f"data/eval/shap{tag}_report.txt").read_text()
    q3 = rep.split("Q3.")[1].split("Q4.")[0]
    rho = defaultdict(list)
    for task, val in re.findall(r" (BIN|CORE) +xgb -> \S+ +J@10 [\d.]+ \[[^\]]+\]  rho ([\d.]+)", q3):
        rho[task].append(float(val))
    return [f"    ports and identity in the G0 top-5: median {100 * share:.0f} % over {len(cell_share)} cells",
            f"    Q4 verdicts (original reply set): {dict(verdicts)}",
            f"    source vs target Spearman, median: BIN {np.median(rho['BIN']):.2f}, CORE {np.median(rho['CORE']):.2f}"]


def main() -> None:
    L = ["Sensitivity of the main XGBoost results to the setting: library default against max_delta_step=1",
         f"Generated: {datetime.now(timezone.utc).isoformat()}",
         "Sources: data/eval/in_domain*, cross*, shap* (default: *_xgbdefault). Random forest runs are the same.", ""]
    for name, tag in SETTINGS.items():
        L.append(f"== XGBoost {name}")
        L += ["  in-domain"] + in_domain(tag) + ["  cross-dataset"] + cross(tag) + ["  SHAP"] + shap(tag) + [""]
    out = ROOT / "data/exploration/xgb_default_sensitivity.txt"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
