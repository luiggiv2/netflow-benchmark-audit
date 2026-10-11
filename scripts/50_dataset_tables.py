"""Tables 2 and 3 of the manuscript (P5.4): the six NF datasets and the
canonical class taxonomy.

Reads only `native_label`, `canonical_label` and `Label` from
`data/processed/<dataset>/data.parquet` and the class mappings in
`data/schemas/`. Writes the counts to `data/paper/dataset_tables.json` and
`.txt` (traceable source of every number) and the formatted tables to
`manuscript/tables/T2_datasets.*` and `T3_taxonomy.*` (CSV, LaTeX, Markdown).

Run from the workspace root: `.venv/bin/python scripts/50_dataset_tables.py`.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone

import pyarrow.parquet as pq

from nfv3_config import WORK, feature_cols, is_v2

OUT_DATA = WORK / "data/paper"
OUT_TAB = WORK / "manuscript/tables"
ORDER = ["NF-ToN-IoT-v3", "NF-BoT-IoT-v3", "NF-CSE-CIC-IDS2018-v3",
         "NF-ToN-IoT-v2", "NF-BoT-IoT-v2", "NF-CSE-CIC-IDS2018-v2"]
PARENT = {"ToN": "ToN-IoT", "BoT": "Bot-IoT", "CSE": "CSE-CIC-IDS2018"}
CANON = ["Benign", "DoS", "DDoS", "Reconnaissance", "CredentialAttack", "Bot_or_Backdoor", "Injection"]
# Tasks each canonical class enters (fourth reviewer report, B10, 2026-10-11): no pair or extended
# cross-dataset result is reported, so the column gives the tasks instead of the comparison panels.
PANEL = {"Benign": "BIN, CAN, CORE", "DoS": "BIN, CAN, CORE", "DDoS": "BIN, CAN, CORE", "Reconnaissance": "BIN, CAN",
         "CredentialAttack": "BIN, CAN", "Bot_or_Backdoor": "BIN, CAN", "Injection": "BIN, CAN"}


def provenance() -> dict:
    def git(*a):
        return subprocess.run(["git", *a], cwd=WORK, capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)data/paper",
                         ":(exclude)manuscript/figures", ":(exclude)manuscript/tables")),  # own outputs ignored
            "script_sha1": hashlib.sha1((WORK / "scripts/50_dataset_tables.py").read_bytes()).hexdigest(),
            "generated_utc": datetime.now(timezone.utc).isoformat()}


def short(ds: str) -> str:
    return ds.replace("NF-", "").split("-")[0]


def counts(ds: str) -> dict:
    t = pq.read_table(WORK / f"data/processed/{ds}/data.parquet",
                      columns=["native_label", "canonical_label", "Label"]).to_pandas()
    native = t["native_label"].value_counts().to_dict()
    canon = t["canonical_label"].fillna("").value_counts().to_dict()
    return {"rows": int(len(t)), "native_classes": len(native),
            "native_counts": {k: int(v) for k, v in sorted(native.items())},
            "canonical_counts": {k: int(v) for k, v in canon.items() if k},
            "excluded_rows": int(canon.get("", 0)),
            "benign_rows": int((t["Label"] == 0).sum()),
            "model_features": len(feature_cols(ds)),
            "netflow_fields": len([c for c in pq.read_schema(WORK / f"data/processed/{ds}/data.parquet").names
                                   if c not in ("Label", "Attack", "native_label", "canonical_label", "split")])}


def mapping() -> dict:
    m: dict = {}
    for f in ["class_mapping.csv", "class_mapping_v2.csv"]:
        for r in csv.DictReader(open(WORK / "data/schemas" / f)):
            m.setdefault(r["dataset"], []).append((r["native_label"], r["canonical_label"]))
    return m


def write_table(stem, header, rows, title, note, colspec):
    with open(OUT_TAB / f"{stem}.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    esc = lambda x: str(x).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
    tex = [r"\begin{table}[t]", r"\centering", r"\caption{" + title + ".}", r"\label{tab:" + stem.split("_")[1] + "}",
           r"\footnotesize", r"\begin{tabular}{" + colspec + "}", r"\toprule", " & ".join(map(esc, header)) + r" \\",
           r"\midrule"] + [" & ".join(map(esc, r)) + r" \\" for r in rows] + [
           r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
           r"\parbox{\linewidth}{\footnotesize \textit{Note.} " + esc(note) + "}", r"\end{table}"]
    (OUT_TAB / f"{stem}.tex").write_text("\n".join(tex) + "\n")
    md = [f"**Table {stem[1]}.** {title}.", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(map(str, r)) + " |" for r in rows] + ["", f"*Note.* {note}"]
    (OUT_TAB / f"{stem}.md").write_text("\n".join(md) + "\n")


def main() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    data = {ds: counts(ds) for ds in ORDER}
    out = {"provenance": provenance(), "datasets": data}
    (OUT_DATA / "dataset_tables.json").write_text(json.dumps(out, indent=1))

    lines = [f"Dataset tables (P5.4). Commit {out['provenance']['commit']}"
             f"{' (dirty)' if out['provenance']['dirty'] else ''}", ""]
    t2 = []
    for ds in ORDER:
        d = data[ds]
        v = "v2" if is_v2(ds) else "v3"
        name = ds.replace(f"-{v}", "")
        t2.append([v, name, PARENT[short(ds)], f"{d['rows']:,}", d["native_classes"],
                   f"{100 * d['benign_rows'] / d['rows']:.1f}", d["netflow_fields"], d["model_features"],
                   len(d["canonical_counts"])])
        lines.append(f"{ds}: rows {d['rows']:,}; native classes {d['native_classes']}; benign "
                     f"{d['benign_rows']:,} ({100 * d['benign_rows'] / d['rows']:.2f} %); excluded from taxonomy "
                     f"{d['excluded_rows']:,}")
        lines += [f"    {k}: {n:,}" for k, n in d["native_counts"].items()]
    (OUT_DATA / "dataset_tables.txt").write_text("\n".join(lines) + "\n")

    write_table(
        "T2_datasets",
        ["Release", "Dataset", "Parent dataset", "Flows", "Classes", "Benign (%)", "Fields", "Features used",
         "Canonical classes"],
        t2, "The six NetFlow datasets audited",
        "One row per released file. Flows: records in the file. Classes: attack labels plus Benign as released. "
        "Fields: NetFlow fields in the file, excluding the two label columns. Features used: fields kept for "
        "the audit after removing IP addresses and absolute timestamps; the models of Section 4 and the main "
        "duplicate key of the audit also drop the source port. Canonical classes: classes of the dataset that enter the shared taxonomy of Table 3. "
        "The v3 files add two timestamps and eight inter-arrival statistics to the v2 fields.",
        "llp{2.4cm}rrrrrr")

    m = mapping()
    t3 = []
    for c in CANON:
        row = [c]
        for key in ["NF-ToN-IoT-v3", "NF-BoT-IoT-v3", "NF-CSE-CIC-IDS2018-v3"]:
            labs = [n for n, cl in m[key] if cl == c]
            row.append("; ".join(labs) if labs else "none")
        row.append(PANEL[c])
        t3.append(row)
    excl = {k: [n for n, cl in m[k] if not cl] for k in ["NF-ToN-IoT-v3", "NF-BoT-IoT-v3", "NF-CSE-CIC-IDS2018-v3"]}
    t3.append(["(not mapped)"] + ["; ".join(excl[k]) for k in excl] + ["BIN"])
    write_table(
        "T3_taxonomy", ["Canonical class", "NF-ToN-IoT", "NF-BoT-IoT", "NF-CSE-CIC-IDS2018", "Tasks"], t3,
        "Canonical class taxonomy",
        "Native labels as written in the v3 files; the v2 files use the same classes with different spelling "
        "(case, spaces instead of underscores). Tasks (Section 4.4): BIN, binary; CAN, canonical classes within each "
        "dataset; CORE, Benign, DoS and DDoS, the only classes present in all three datasets and the cross-dataset "
        "multiclass task. Labels in the last row have no counterpart in another dataset; they stay in the data, count as "
        "attacks in the binary task and are left out of the canonical and core tasks. Infiltration is spelled "
        "Infilteration in both CIC files.",
        "lp{3.2cm}p{2.2cm}p{4.2cm}l")
    print(lines[0])


if __name__ == "__main__":
    main()
