"""replication_gaps.py — second reviewer report, point N7, 2026-10-08: the
difference between each published value of Layeghy and Portmann (2023) and its
replication under L0 (their setting as read here), read from
data/eval/replication_report.txt, with the label at a stated tolerance:
reproduced if the L0 median lies within TOL points of the published value.

Usage: python scripts/exploration/replication_gaps.py
Output: data/exploration/replication_gaps.txt
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOL = 2.0
NUM = r"([\d.]+)"


def main() -> None:
    text = (ROOT / "data/eval/replication_report.txt").read_text()
    rows = []
    for block in text.split("\n== ")[1:]:
        src = block.split()[0]
        pub_in = float(re.search(r"published in-domain " + NUM, block).group(1))
        l0 = block.split("\n  L0 ")[1].split("\n  L0k ")[0]
        rows.append((src, "in-domain", pub_in, float(re.search(r"in-domain +" + NUM, l0).group(1))))
        for tgt, val, pub in re.findall(r"-> (\S+) +" + NUM + r" \[[^\]]+\] \(pub +" + NUM + r"\)", l0):
            rows.append((src, tgt, float(pub), float(val)))
    L = ["Replication of Layeghy and Portmann (2023): published value against L0 (attack-class F1, %, median over seeds)",
         f"Generated: {datetime.now(timezone.utc).isoformat()}",
         f"Source: data/eval/replication_report.txt (L0 rows). Reproduced: |L0 - published| <= {TOL} points.", "",
         f"  {'trained on':24s} {'tested on':24s} {'published':>9s} {'L0':>7s} {'L0 - pub':>9s}  label"]
    for src, tgt, pub, val in rows:
        gap = val - pub
        L.append(f"  {src:24s} {tgt:24s} {pub:9.2f} {val:7.2f} {gap:+9.2f}  "
                 f"{'reproduced' if abs(gap) <= TOL else 'not reproduced'}")
    gaps = [abs(v - p) for _, _, p, v in rows]
    rep = [r for r, g in zip(rows, gaps) if g <= TOL]
    L += ["", f"Reproduced: {len(rep)} of {len(rows)}; absolute gaps of the others: "
          + ", ".join(f"{g:.2f}" for g in sorted(g for g in gaps if g > TOL))]
    for kind, sel in (("in-domain", lambda r: r[1] == "in-domain"), ("cross-dataset", lambda r: r[1] != "in-domain")):
        sub = [r for r in rows if sel(r)]
        L.append(f"  {kind}: reproduced {sum(abs(v - p) <= TOL for _, _, p, v in sub)} of {len(sub)}")
    out = ROOT / "data/exploration/replication_gaps.txt"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
