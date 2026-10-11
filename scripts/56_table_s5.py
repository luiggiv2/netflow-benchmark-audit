"""Table S5 of the manuscript (reviewer point M6, 2026-10-06): the uncertainty
and reference values promised in Section 4.4, for the G0 runs.

For every in-domain cell (dataset x task x model) and every cross-dataset
cell (direction x task x model) of condition G0 it reports, over the five
seeds: the median macro-F1 and its range; the median width of the per-run
95 % Poisson-bootstrap CI (vector groups resampled together); the median
deduplicated macro-F1 (each distinct test vector once, majority label) and
the median per-seed difference deduplicated minus row-level; and, in-domain,
the conflict ceiling. The summary compares the within-run CI with the spread
over seeds.

Reads only data/eval/in_domain/*.jsonl and data/eval/cross/*.jsonl. Writes the
numbers to data/paper/table_s5.json and .txt and the table to
manuscript/tables/S5_ci_dedup.md.
Run: `.venv/bin/python scripts/56_table_s5.py`
"""
from __future__ import annotations

import hashlib
import json
import statistics as st
import subprocess
from datetime import datetime, timezone

from nfv3_config import WORK

OUT_DATA = WORK / "data/paper"
OUT_TAB = WORK / "manuscript/tables"
ORDER = ["NF-ToN-IoT-v3", "NF-BoT-IoT-v3", "NF-CSE-CIC-IDS2018-v3",
         "NF-ToN-IoT-v2", "NF-BoT-IoT-v2", "NF-CSE-CIC-IDS2018-v2"]
MODELS = ["xgb", "rf"]
MODEL_NAME = {"xgb": "XGBoost", "rf": "RF"}


def provenance() -> dict:
    def git(*a):
        return subprocess.run(["git", *a], cwd=WORK, capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)data/paper",
                         ":(exclude)manuscript/figures", ":(exclude)manuscript/tables")),  # own outputs ignored
            "script_sha1": hashlib.sha1((WORK / "scripts/56_table_s5.py").read_bytes()).hexdigest(),
            "generated_utc": datetime.now(timezone.utc).isoformat()}


def short(ds: str) -> str:
    v = ds[-2:]
    return {"ToN": "ToN", "BoT": "BoT", "CSE": "CIC"}[ds.split("-")[1]] + "-" + v


def summarise(runs: list[dict]) -> dict:
    f1 = [r["macro_f1"] for r in runs]
    width = [r["ci95"][1] - r["ci95"][0] for r in runs]
    dedup = [r["dedup_macro_f1"] for r in runs]
    out = {"n": len(runs), "macro_f1_median": st.median(f1), "macro_f1_min": min(f1), "macro_f1_max": max(f1),
           "seed_range": max(f1) - min(f1), "ci_width_median": st.median(width),
           "dedup_median": st.median(dedup),
           "dedup_minus_rows_median": st.median(d - f for d, f in zip(dedup, f1))}
    ceil = [r["conflict_ceiling_macro_f1"] for r in runs if "conflict_ceiling_macro_f1" in r]
    if ceil:
        out["conflict_ceiling_median"] = st.median(ceil)
    return out


def load_in_domain() -> dict:
    cells = {}
    for ds in ORDER:
        for line in open(WORK / f"data/eval/in_domain/{ds}.jsonl"):
            r = json.loads(line)
            if r["condition"] == "G0":
                cells.setdefault((ds, r["task"], r["model"]), []).append(r)
    return {k: summarise(v) for k, v in cells.items()}


def load_cross() -> dict:
    cells = {}
    for ds in ORDER:
        for line in open(WORK / f"data/eval/cross/{ds}.jsonl"):
            r = json.loads(line)
            if r["condition"] != "G0" or r["task"] not in ("BIN", "CORE"):  # EXT is not reported
                continue
            for tgt, v in r["targets"].items():
                cells.setdefault((ds, tgt, r["task"], r["model"]), []).append(v)
    return {k: summarise(v) for k, v in cells.items()}


def overall(cells: dict) -> dict:
    v = list(cells.values())
    return {"cells": len(v),
            "ci_width_median": st.median(c["ci_width_median"] for c in v),
            "ci_width_max": max(c["ci_width_median"] for c in v),
            "seed_range_median": st.median(c["seed_range"] for c in v),
            "seed_range_above_ci": sum(c["seed_range"] > c["ci_width_median"] for c in v),
            "dedup_minus_rows_median": st.median(c["dedup_minus_rows_median"] for c in v),
            "dedup_minus_rows_min": min(c["dedup_minus_rows_median"] for c in v),
            "dedup_minus_rows_max": max(c["dedup_minus_rows_median"] for c in v),
            "abs_dedup_minus_rows_ge_005": sum(abs(c["dedup_minus_rows_median"]) >= 0.05 for c in v)}


def f2(x: float) -> str:
    return f"{x:.2f}"


def pts(x: float) -> str:
    return f"{100 * x:+.1f}"


def main() -> None:
    ind, cross = load_in_domain(), load_cross()
    oi, oc = overall(ind), overall(cross)
    rec = {"provenance": provenance(),
           "in_domain": {"|".join(k): v for k, v in ind.items()}, "in_domain_overall": oi,
           "cross": {"|".join(k): v for k, v in cross.items()}, "cross_overall": oc}
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    (OUT_DATA / "table_s5.json").write_text(json.dumps(rec, indent=1))

    p = rec["provenance"]
    L = [f"Table S5 (reviewer M6). Commit {p['commit']}{' (dirty)' if p['dirty'] else ''}",
         "G0 runs; medians over 5 seeds. CI width = upper minus lower bound of the per-run 95 % Poisson",
         "bootstrap CI (vector groups resampled together). dedup - rows = median per-seed difference.", ""]
    for name, o in (("In-domain", oi), ("Cross-dataset (all target rows)", oc)):
        L.append(f"{name}: {o['cells']} cells; median CI width {o['ci_width_median']:.3f} (max "
                 f"{o['ci_width_max']:.3f}); median seed range {o['seed_range_median']:.3f}; seed range wider "
                 f"than the median CI in {o['seed_range_above_ci']} of {o['cells']} cells; dedup - rows median "
                 f"{pts(o['dedup_minus_rows_median'])} points (min {pts(o['dedup_minus_rows_min'])}, max "
                 f"{pts(o['dedup_minus_rows_max'])}); |dedup - rows| >= 5 points in "
                 f"{o['abs_dedup_minus_rows_ge_005']} cells")
    L += ["", "In-domain cells:"]
    for (ds, task, model), c in sorted(ind.items(), key=lambda kv: (ORDER.index(kv[0][0]), kv[0][1], kv[0][2])):
        L.append(f"  {ds:24s} {task} {model:3s} macro-F1 {c['macro_f1_median']:.4f} [{c['macro_f1_min']:.4f}, "
                 f"{c['macro_f1_max']:.4f}]  CI width {c['ci_width_median']:.4f}  dedup {c['dedup_median']:.4f} "
                 f"({pts(c['dedup_minus_rows_median'])})  ceiling {c['conflict_ceiling_median']:.3f}")
    L += ["", "Cross-dataset cells:"]
    for (src, tgt, task, model), c in sorted(cross.items(), key=lambda kv: (ORDER.index(kv[0][0]), kv[0][1],
                                                                             kv[0][2], kv[0][3])):
        L.append(f"  {src:24s} -> {tgt:24s} {task:4s} {model:3s} macro-F1 {c['macro_f1_median']:.4f} "
                 f"[{c['macro_f1_min']:.4f}, {c['macro_f1_max']:.4f}]  CI width {c['ci_width_median']:.4f}  "
                 f"dedup {c['dedup_median']:.4f} ({pts(c['dedup_minus_rows_median'])})")
    (OUT_DATA / "table_s5.txt").write_text("\n".join(L) + "\n")

    hdr = ["Dataset", "Task", "Model", "Macro-F1 (range)", "CI width", "Dedup.", "Dedup. − rows", "Ceiling"]
    rows = []
    for ds in ORDER:
        for task in ("BIN", "CAN"):
            for m in MODELS:
                c = ind[(ds, task, m)]
                rows.append([short(ds), task, MODEL_NAME[m],
                             f"{f2(c['macro_f1_median'])} ({f2(c['macro_f1_min'])}–{f2(c['macro_f1_max'])})",
                             f"{c['ci_width_median']:.3f}", f2(c["dedup_median"]),
                             pts(c["dedup_minus_rows_median"]).replace("-", "−"), f2(c["conflict_ceiling_median"])])
    hdr_x = ["Trained on", "Tested on", "Task", "Model", "Macro-F1 (range)", "CI width", "Dedup.", "Dedup. − rows"]
    rows_x = []
    for src in ORDER:
        tgts = sorted({k[1] for k in cross if k[0] == src}, key=ORDER.index)
        for tgt in tgts:
            for task in ("BIN", "CORE"):
                for m in MODELS:
                    c = cross[(src, tgt, task, m)]
                    rows_x.append([short(src), short(tgt), task, MODEL_NAME[m],
                                   f"{f2(c['macro_f1_median'])} ({f2(c['macro_f1_min'])}–{f2(c['macro_f1_max'])})",
                                   f"{c['ci_width_median']:.3f}", f2(c["dedup_median"]),
                                   pts(c["dedup_minus_rows_median"]).replace("-", "−")])
    md = ["**Table S5.** Uncertainty and reference values of the G0 runs.", "",
          "*(a) In-domain.*", "", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    md += ["", "*(b) Cross-dataset, all rows of the target dataset (for CORE, its Benign, DoS and DDoS rows).*", "",
           "| " + " | ".join(hdr_x) + " |", "|" + "---|" * len(hdr_x)]
    md += ["| " + " | ".join(r) + " |" for r in rows_x]
    md += ["", "*Note.* Medians over five seeds. Macro-F1 range: minimum and maximum over seeds. CI width: "
           "width of the per-run 95 % Poisson-bootstrap interval, with the rows of one feature vector resampled "
           "together. Dedup.: macro-F1 counting each distinct test vector once with its majority label; "
           "Dedup. − rows: median per-seed difference, in points. Ceiling: conflict ceiling of the test split, "
           "computed on the features each model sees (48 in NF-v3, 40 in NF-v2)."]
    (OUT_TAB / "S5_ci_dedup.md").write_text("\n".join(md) + "\n")
    print("\n".join(L[:7]))


if __name__ == "__main__":
    main()
