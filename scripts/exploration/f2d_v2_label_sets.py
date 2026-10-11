"""f2d_v2_label_sets.py — second reviewer report, minor 1 (first report minor
13), 2026-10-08: the F2d (v2) count. 36_inspect_undetermined_cases.py reports
2,887 records from the main web-attack source of NF-CSE-CIC-IDS2018-v2 that
carry more than one label, but lists only the five most frequent label sets
(2,876 records). This script lists every label set, with the same records and
hashing as script 36, and counts the records that carry a web-attack label
and DDOS attack-HOIC.

Usage: python scripts/exploration/f2d_v2_label_sets.py
Output: data/exploration/f2d_v2_label_sets.txt
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
_spec = importlib.util.spec_from_file_location("s36", ROOT / "scripts/36_inspect_undetermined_cases.py")
U = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(U)


def main() -> None:
    w2 = U.load(U.CIC2, labels=U.WEB2)
    att = w2["IPV4_SRC_ADDR"].value_counts().index[0]
    a2 = U.load(U.CIC2, ips=[att])
    a2["rec"] = U.record_hash(a2.drop(columns=["native_label", "Attack", "Label"]))
    sets = a2.groupby("rec")["native_label"].agg(lambda s: " / ".join(sorted(set(s))))
    multi = sets[sets.str.contains(" / ")]
    rows = a2[a2["rec"].isin(multi.index)]
    web_hoic = multi[multi.str.contains("DDOS attack-HOIC") & multi.apply(lambda x: any(w in x for w in U.WEB2))]
    L = ["F2d (v2): label sets of the records carrying more than one label (all columns except the labels)",
         f"Generated: {datetime.now(timezone.utc).isoformat()}",
         f"Source address {att} (main source of the web-attack rows of {U.CIC2}); same records as script 36.", "",
         f"records with more than one label: {len(multi):,} (rows {len(rows):,})",
         f"records with a web-attack label and DDOS attack-HOIC: {len(web_hoic):,} "
         f"(rows {int(a2['rec'].isin(web_hoic.index).sum()):,})", "", "all label sets (records):"]
    L += [f"  {k}: {v:,}" for k, v in multi.value_counts().items()]
    out = ROOT / "data/exploration/f2d_v2_label_sets.txt"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
