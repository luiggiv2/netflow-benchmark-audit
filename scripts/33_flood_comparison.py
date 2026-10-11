"""33_flood_comparison.py — compare the heuristics of Flood et al. (2024) on
the parent datasets (Table 8, transcribed in data/audit/flood_table8_original.csv)
with the reimplementation on NF-v2 and NF-v3 (script 32). PLAN-PAPER-B.md, P3.4c.

Output: data/audit/flood_comparison_report.txt and flood_comparison.csv.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from nfv3_config import WORK

AUD = WORK / "data/audit"
DATASETS = {"ToN_IoT": ("NF-ToN-IoT-v2", "NF-ToN-IoT-v3"),
            "Bot_IoT": ("NF-BoT-IoT-v2", "NF-BoT-IoT-v3"),
            "CSE-CIC-IDS2018": ("NF-CSE-CIC-IDS2018-v2", "NF-CSE-CIC-IDS2018-v3")}
METRICS = ["PDD", "TC", "WL", "HDF", "UGT"]


def nf_values(res: dict, label: str) -> dict:
    c = res["classes"].get(label)
    if c is None:
        return {m: None for m in METRICS} | {"HDF_np": None, "HDF_feat": None, "WL20k": None,
                                             "WLbal": None, "rows": None, "rows_raw": None}
    return {"PDD": c["pdd_tc"]["pdd"], "TC": c["pdd_tc"]["tc"], "WL": c["wl_500"]["mean"],
            "HDF": c["hdf"]["hdf"], "UGT": c["ugt"], "HDF_np": c["hdf"]["hdf_no_ports"],
            "HDF_feat": c["hdf"]["feature"], "WL20k": c["wl_big"]["mean"],
            "WLbal": c["wl_balanced"]["mean"], "rows": c["rows_dedup"],
            "rows_raw": raw_rows(res["dataset"], label)}


def raw_rows(dataset: str, label: str):
    r = json.loads((AUD / "dupconf" / f"{dataset}.json").read_text())
    return r["keys"]["41"]["native"]["per_class"].get(label, {}).get("rows")


def f(x):
    return "   -" if x is None else f"{x:4.2f}"


def main():
    t8 = pd.read_csv(AUD / "flood_table8_original.csv")
    res = {}
    for v2, v3 in DATASETS.values():
        for ds in (v2, v3):
            res[ds] = json.loads((AUD / "flood_heuristics" / f"{ds}.json").read_text())
    rows = []
    for _, r in t8.iterrows():
        v2, v3 = DATASETS[r["original_dataset"]]
        a, b = nf_values(res[v2], r["nf_label_v2"]), nf_values(res[v3], r["nf_label_v3"])
        row = {"dataset": r["original_dataset"], "class": r["flood_class"]}
        for m in METRICS:
            row[f"{m}_orig"], row[f"{m}_v2"], row[f"{m}_v3"] = r[m], a[m], b[m]
        for k in ("HDF_np", "HDF_feat", "WL20k", "WLbal", "rows", "rows_raw"):
            row[f"{k}_v2"], row[f"{k}_v3"] = a[k], b[k]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(AUD / "flood_comparison.csv", index=False)

    L = ["Flood et al. (2024) heuristics: parent datasets (Table 8) vs NF-v2 vs NF-v3",
         f"Generated: {datetime.now(timezone.utc).isoformat()}",
         "Columns per metric: orig / v2 / v3. WL = 500-flow reference (5-seed mean);",
         "HDF over 41 features (ports included); HDF-np = without ports. See",
         "research/flood_heuristics_spec.md for parameters and deviations.", ""]
    for ds in DATASETS:
        sub = df[df["dataset"] == ds]
        L.append(f"== {ds}")
        L.append(f"  {'class':14s} {'rows v2':>10s} {'rows v3':>10s} | " +
                 " | ".join(f"{m:^14s}" for m in METRICS) + " | HDF-np v2/v3 | WLbal v2/v3 | HDF feature v3")
        for _, r in sub.iterrows():
            cells = " | ".join(f"{f(r[m + '_orig'])} {f(r[m + '_v2'])} {f(r[m + '_v3'])}" for m in METRICS)
            rv2 = f"{int(r['rows_v2']):,}" if pd.notna(r["rows_v2"]) else "-"
            rv3 = f"{int(r['rows_v3']):,}" if pd.notna(r["rows_v3"]) else "-"
            L.append(f"  {r['class']:14s} {rv2:>10s} {rv3:>10s} | {cells} | "
                     f"{f(r['HDF_np_v2'])} {f(r['HDF_np_v3'])}  | {f(r['WLbal_v2'])} {f(r['WLbal_v3'])}   | "
                     f"{r['HDF_feat_v3']}")
        L.append("")
    # summary
    L.append("Summary over the 27 classes (mean; count of classes >= 0.95 for PDD, TC, HDF;"
             " >= 0.10 for WL and UGT):")
    for m in METRICS:
        thr = 0.95 if m in ("PDD", "TC", "HDF") else 0.10
        parts = []
        for v in ("orig", "v2", "v3"):
            s = pd.to_numeric(df[f"{m}_{v}"], errors="coerce").dropna()
            parts.append(f"{v} {s.mean():.2f} ({int((s >= thr).sum())}/{len(s)})")
        L.append(f"  {m:4s}: " + ";  ".join(parts))
    s2 = pd.to_numeric(df["HDF_np_v2"], errors="coerce").dropna()
    s3 = pd.to_numeric(df["HDF_np_v3"], errors="coerce").dropna()
    L.append(f"  HDF without ports: v2 {s2.mean():.2f} ({int((s2 >= 0.95).sum())}/{len(s2)});  "
             f"v3 {s3.mean():.2f} ({int((s3 >= 0.95).sum())}/{len(s3)})")
    w2 = pd.to_numeric(df["WL20k_v2"], errors="coerce").dropna()
    w3 = pd.to_numeric(df["WL20k_v3"], errors="coerce").dropna()
    L.append(f"  WL with a 20,000-flow reference: v2 {w2.mean():.2f};  v3 {w3.mean():.2f}")
    b2 = pd.to_numeric(df["WLbal_v2"], errors="coerce").dropna()
    b3 = pd.to_numeric(df["WLbal_v3"], errors="coerce").dropna()
    L.append(f"  WL with a class-balanced reference: v2 {b2.mean():.2f} ({int((b2 >= 0.10).sum())}/{len(b2)});  "
             f"v3 {b3.mean():.2f} ({int((b3 >= 0.10).sum())}/{len(b3)})")
    L += ["", "Exact duplicate records (all raw columns, label included; v2 has no timestamps):",
          f"  {'dataset':16s} {'class':14s} {'v2 raw':>11s} {'v2 dup %':>9s} {'v3 raw':>11s} {'v3 dup %':>9s}"]
    for _, r in df.iterrows():
        cells = []
        for v in ("v2", "v3"):
            raw, ded = r[f"rows_raw_{v}"], r[f"rows_{v}"]
            cells.append(f"{int(raw):>11,} {100 * (raw - ded) / raw:>8.1f}%" if pd.notna(raw) and pd.notna(ded)
                         else f"{'-':>11s} {'-':>9s}")
        L.append(f"  {r['dataset']:16s} {r['class']:14s} {cells[0]} {cells[1]}")
    (AUD / "flood_comparison_report.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
