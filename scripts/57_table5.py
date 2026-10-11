"""Table 5 of the manuscript (findings of the audit by origin), rendered from
its single editable source `manuscript/tables/T5_findings.csv` (added
2026-10-07, block 5 of the reviewer response, so that the CSV, LaTeX and
Markdown versions and the copy in Section 5.1 cannot drift apart).

The values in the CSV come from `research/findings_classification.md`
(decisions 1-15) and the files it cites under `data/audit/` and
`data/exploration/`; conflict vectors and repeated rows are counted on the 40
features the models see (key 40 of 30_audit_duplicates_conflicts.py).

Run: `.venv/bin/python scripts/57_table5.py`
Writes manuscript/tables/T5_findings.{md,tex} and replaces Table 5 in
manuscript/sections/05-results.md.
"""
from __future__ import annotations

import csv
import re

from nfv3_config import WORK

TAB = WORK / "manuscript/tables"
SECTION = WORK / "manuscript/sections/05-results.md"
TITLE = "Findings of the audit by origin"
NOTE = ("How to read: findings are grouped by where they arise, using the rules of Section 4.2. Introduced by a "
        "release: present in one release only, or changed by that release, with a cause in flow export, labelling or "
        "file assembly. "
        "Release-dependent: the releases differ, but the cause cannot be placed in one of them. Release "
        "documentation: errors in labels, web pages, publications or feature descriptions; v2 errors that remain in "
        "v3 are noted. NetFlow export: present in both releases and caused by how NetFlow fields are computed. "
        "Inherited from the parent: reported by Flood et al. for the parent dataset or caused by the testbed itself; "
        "masked means absent from NF-v2 although present in the parent and in NF-v3. Confidence: high when measured "
        "in both releases and supported by the parent data or a clear mechanism; medium when one link is missing; "
        "low when the evidence is indirect. Vectors and repeated rows are counted on the 40 features common to both "
        "releases without the source port (the features of the NF-v2 models). Percentages are shares of rows of the dataset or class indicated; n.a.: not "
        "applicable.")


def tex_esc(x: str) -> str:
    return (x.replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
            .replace("^", "\\^{}").replace("–", "--"))


def main() -> None:
    rows = list(csv.DictReader(open(TAB / "T5_findings.csv")))
    md = [f"**Table 5.** {TITLE}.", "", "| ID | Finding | NF-v2 | NF-v3 | Confidence |", "|---|---|---|---|---|"]
    tex = [r"\begin{table*}[t]", r"\centering", r"\caption{" + TITLE + ".}", r"\label{tab:findings}",
           r"\footnotesize", r"\begin{tabular}{lp{5.6cm}p{3.4cm}p{3.6cm}l}", r"\toprule",
           r"ID & Finding & NF-v2 & NF-v3 & Confidence \\", r"\midrule"]
    group = None
    for r in rows:
        if r["origin"] != group:
            if group is not None:
                tex.append(r"\midrule")
            group = r["origin"]
            md.append(f"| **{group}** | | | | |")
            tex.append(r"\multicolumn{5}{l}{\textit{" + tex_esc(group) + r"}} \\")
        cells = [r["id"], r["finding"], r["nf_v2"], r["nf_v3"], r["confidence"]]
        md.append("| " + " | ".join(cells) + " |")
        tex.append(" & ".join(map(tex_esc, cells)) + r" \\")
    md += ["", f"*Note.* {NOTE}"]
    tex += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
            r"\parbox{\textwidth}{\footnotesize \textit{Note.} " + tex_esc(NOTE) + "}", r"\end{table*}"]
    text = "\n".join(md) + "\n"
    (TAB / "T5_findings.md").write_text(text)
    (TAB / "T5_findings.tex").write_text("\n".join(tex) + "\n")
    if not SECTION.exists():  # public archive: the manuscript sections are not included
        print(f"Table 5: {len(rows)} entries; written to tables/")
        return
    sec = SECTION.read_text()
    new, n = re.subn(r"\*\*Table 5\.\*\*.*?\n\*Note\.\*[^\n]*\n", lambda _: text, sec, count=1, flags=re.S)
    if n != 1:
        raise SystemExit("Table 5 not found in Section 5")
    SECTION.write_text(new)
    print(f"Table 5: {len(rows)} entries; written to tables/ and Section 5.1")


if __name__ == "__main__":
    main()
