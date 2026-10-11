"""58_table_s6.py — Supplementary Table S6: the main XGBoost-dependent results
with the library-default XGBoost (sensitivity runs, data/eval/*_xgbdefault)
and with max_delta_step=1 (the setting of the paper). Second reviewer report,
point N8a; the same quantities as scripts/exploration/xgb_default_sensitivity.py.
Third reviewer report, major issue 6 (2026-10-10): cross-dataset rows labelled
with their split; SHAP row with the service fields (SERVICE_COLS) counted with
ports and identity, and seed means per cell as in Figure 9 and Section 5.4.

Outputs: data/paper/table_s6.{json,txt} and manuscript/tables/S6_xgb_setting.md
Run from the workspace root: .venv/bin/python scripts/58_table_s6.py
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np

from nfv3_config import IDENTITY_COLS, PORT_COLS, SERVICE_COLS, WORK

OUT_DATA = WORK / "data/paper"
OUT_TAB = WORK / "manuscript/tables"
DATASETS = ["NF-ToN-IoT-v2", "NF-BoT-IoT-v2", "NF-CSE-CIC-IDS2018-v2", "NF-ToN-IoT-v3", "NF-BoT-IoT-v3",
            "NF-CSE-CIC-IDS2018-v3"]
SETTINGS = {"default": "_xgbdefault", "mds1": ""}
SEEDS = [1, 2, 3, 4, 5]
SHORT = set(PORT_COLS) | set(SERVICE_COLS) | set(IDENTITY_COLS)


def provenance() -> dict:
    def git(*a):
        return subprocess.run(["git", *a], cwd=WORK, capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)data/paper",
                         ":(exclude)manuscript/figures", ":(exclude)manuscript/tables")),  # own outputs ignored
            "script_sha1": hashlib.sha1((WORK / "scripts/58_table_s6.py").read_bytes()).hexdigest(),
            "generated_utc": datetime.now(timezone.utc).isoformat()}


def jsonl(d: str, ds: str) -> list[dict]:
    return [json.loads(x) for x in open(WORK / f"data/eval/{d}/{ds}.jsonl")]


def values(tag: str) -> dict:
    v = {}
    cells = defaultdict(dict)
    for ds in DATASETS:
        for r in jsonl(f"in_domain{tag}", ds):
            if r["model"] == "xgb":
                cells[(ds, r["task"])].setdefault(r["condition"], []).append(r["macro_f1"])
    rng = [max(c["G0"]) - min(c["G0"]) for c in cells.values()]
    v["g0_unstable"] = [int(sum(x >= 0.05 for x in rng)), len(rng)]
    v["r0_above_g0"] = [int(sum(np.median(c["R0"]) > np.median(c["G0"]) for c in cells.values())), len(cells)]
    ts = {(ds, t): 100 * (np.median(c["G-TS"]) - np.median(c["R0"])) for (ds, t), c in cells.items() if "G-TS" in c}
    v["gts_cic_can"] = round(ts.pop(("NF-CSE-CIC-IDS2018-v3", "CAN")), 1)
    v["gts_other"] = [round(min(ts.values()), 1), round(max(ts.values()), 1)]
    R = {}
    for ds in DATASETS:
        for r in jsonl(f"cross{tag}", ds):
            R[(ds, r["task"], r["model"], r["condition"], r["seed"])] = r
    for c in ("G-NP", "G-NI"):
        diffs = []
        for ds in DATASETS:
            for m in ("xgb", "rf"):
                for t in R[(ds, "BIN", m, "G0", 1)]["targets"]:
                    g0 = np.median([R[(ds, "BIN", m, "G0", s)]["targets"][t]["macro_f1"] for s in SEEDS])
                    cv = np.median([R[(ds, "BIN", m, c, s)]["targets"][t]["macro_f1"] for s in SEEDS])
                    diffs.append(cv - g0)
        v[f"bin_{c}"] = [int(sum(d < 0 for d in diffs)), int(sum(d > 0 for d in diffs)), len(diffs)]
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
    v["collapse_bot"] = [low, n]
    share, verdicts = defaultdict(list), Counter()
    for ds in DATASETS:
        for r in jsonl(f"shap{tag}", ds):
            if r["model"] != "xgb" or r["condition"] != "G0":
                continue
            if r["task"] in ("BIN", "CAN"):
                imp = dict(r["importance_test"])
                top5 = sorted(imp, key=imp.get, reverse=True)[:5]
                share[(ds, r["task"])].append(sum(f in SHORT for f in top5) / 5)
            if r["task"] == "CORE" and "BoT" not in ds:
                verdicts[r["targets"]["NF-BoT-IoT-" + ds[-2:]]["benign_push"]["verdict"]] += 1
    v["ports_service_identity_top5"] = round(100 * float(np.median([np.mean(x) for x in share.values()])))
    v["q4"] = [verdicts["refuted"], verdicts["inconclusive"], verdicts["supported"]]
    q3 = (WORK / f"data/eval/shap{tag}_report.txt").read_text().split("Q3.")[1].split("Q4.")[0]
    rho = defaultdict(list)
    for task, val in re.findall(r" (BIN|CORE) +xgb -> \S+ +J@10 [\d.]+ \[[^\]]+\]  rho ([\d.]+)", q3):
        rho[task].append(float(val))
    v["rho_src_tgt"] = [round(float(np.median(rho["BIN"])), 2), round(float(np.median(rho["CORE"])), 2)]
    return v


def pts(x: float) -> str:
    return f"{x:+.1f}".replace("-", "−")


def main() -> None:
    res = {k: values(tag) for k, tag in SETTINGS.items()}
    d, m = res["default"], res["mds1"]
    rows = [
        ("In-domain: G0 cells with a seed range of 0.05 or more", f"{d['g0_unstable'][0]} of {d['g0_unstable'][1]}",
         f"{m['g0_unstable'][0]} of {m['g0_unstable'][1]}"),
        ("In-domain: cells with R0 above G0", f"{d['r0_above_g0'][0]} of {d['r0_above_g0'][1]}",
         f"{m['r0_above_g0'][0]} of {m['r0_above_g0'][1]}"),
        ("In-domain: G-TS minus R0, NF-CSE-CIC-IDS2018-v3 CAN (points)", pts(d["gts_cic_can"]), pts(m["gts_cic_can"])),
        ("In-domain: G-TS minus R0, other NF-v3 cells (points)",
         f"{pts(d['gts_other'][0])} to {pts(d['gts_other'][1])}", f"{pts(m['gts_other'][0])} to {pts(m['gts_other'][1])}"),
        ("Cross-dataset BIN, without destination port (split of the condition): worse / better",
         f"{d['bin_G-NP'][0]} / {d['bin_G-NP'][1]}", f"{m['bin_G-NP'][0]} / {m['bin_G-NP'][1]}"),
        ("Cross-dataset BIN, without identity features (split of the condition): worse / better",
         f"{d['bin_G-NI'][0]} / {d['bin_G-NI'][1]}", f"{m['bin_G-NI'][0]} / {m['bin_G-NI'][1]}"),
        ("CORE into NF-BoT-IoT: DoS and DDoS F1 below 0.01 in every seed",
         f"{d['collapse_bot'][0]} of {d['collapse_bot'][1]}", f"{m['collapse_bot'][0]} of {m['collapse_bot'][1]}"),
        ("SHAP: ports, service and identity features in the G0 top-5 (median of seed means)",
         f"{d['ports_service_identity_top5']} %", f"{m['ports_service_identity_top5']} %"),
        ("SHAP Q4 (original reply set): refuted / inconclusive seeds", f"{d['q4'][0]} / {d['q4'][1]}",
         f"{m['q4'][0]} / {m['q4'][1]}"),
        ("SHAP: source vs target Spearman, BIN / CORE (median)", f"{d['rho_src_tgt'][0]:.2f} / {d['rho_src_tgt'][1]:.2f}",
         f"{m['rho_src_tgt'][0]:.2f} / {m['rho_src_tgt'][1]:.2f}"),
    ]
    prov = provenance()
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    (OUT_DATA / "table_s6.json").write_text(json.dumps({"provenance": prov, "values": res}, indent=2))
    L = ["Table S6: main XGBoost results with the library default and with max_delta_step=1",
         f"provenance: {json.dumps(prov)}", ""] + [f"  {a}: default {b}; max_delta_step=1 {c}" for a, b, c in rows]
    (OUT_DATA / "table_s6.txt").write_text("\n".join(L) + "\n")
    md = ["**Table S6.** Main results with the two XGBoost settings.", "",
          "| Result | Library default | max_delta_step = 1 |", "|---|---|---|"]
    md += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    md += ["", "*Note.* Within datasets and for SHAP, XGBoost runs only (12 dataset and task cells); the "
           "cross-dataset counts cover the 24 direction and model pairs of both models, the random-forest pairs "
           "being the same with both settings. Worse / better: pairs whose median target macro-F1 over seeds is "
           "below / above G0. The setting of the "
           "paper is max_delta_step = 1 (Section 4.4); with the library default the multiclass fit diverged on the "
           "canonical task of NF-CSE-CIC-IDS2018-v3. The cross-dataset rows use the split keyed on the reduced "
           "features; the repetitions on the split of G0 (Section 5.3.2) were run only with max_delta_step = 1."]
    (OUT_TAB / "S6_xgb_setting.md").write_text("\n".join(md) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
