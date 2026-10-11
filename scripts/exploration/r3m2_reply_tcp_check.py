"""r3m2_reply_tcp_check.py — third reviewer report, major issue 2, 2026-10-10
(plan task P6.1b, block 2; author decision of 2026-10-09): the natural
comparison of m8_reply_check.py part b restricted to TCP floods.

In NF-BoT-IoT the DoS/DDoS flows that received a reply are all TCP, while
only 41-47 % of those without one are, so the natural comparison (share
predicted Benign with and without a reply) mixes the reply with the protocol.
Here the CORE G0 models of m8_reply_check.py part b (ToN and CIC sources, v2
and v3, XGBoost and RF, seeds 1-5) are rebuilt exactly and scored on the same
balanced NF-BoT-IoT sample (50,000 rows per class, seed 7). The stored
all-protocol values of part b (G0 rows of data/exploration/m8_reply_check_b.txt)
must be reproduced (check printed). Per source and model, share of DoS/DDoS
rows predicted Benign among: TCP flows with a reply, TCP flows without one,
and non-TCP flows (none of which has a reply), with their sizes; median [min,
max] over seeds. A summary gives, for each group, the range of these medians
over the eight source x model pairs.

Usage: python scripts/exploration/r3m2_reply_tcp_check.py
Output: data/exploration/r3m2_reply_tcp_check.txt
"""

from __future__ import annotations

import importlib.util
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
_spec = importlib.util.spec_from_file_location("m8", ROOT / "scripts/exploration/m8_reply_check.py")
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)
S, C, P42 = M.S, M.C, M.P42

STORED = ROOT / "data/exploration/m8_reply_check_b.txt"
OUT = ROOT / "data/exploration/r3m2_reply_tcp_check.txt"


def md(v) -> str:
    return f"{np.median(v):.3f} [{min(v):.3f}, {max(v):.3f}]"


def stored_g0() -> dict:
    """(source, model) -> 'with reply ... | without reply ...' text of the G0 rows of part b."""
    out, src, model = {}, None, None
    for line in STORED.read_text().splitlines():
        if line.startswith("== "):
            src = line[3:].split(" -> ")[0]
        m = re.match(r"\s+(xgb|rf)\s+(G0|no-reply)\s", line)
        if m:
            model = (m.group(1), m.group(2))
        if line.strip().startswith("with reply") and model and model[1] == "G0":
            out[(src, model[0])] = line.strip()
    return out


def main() -> None:
    ref = stored_g0()
    L = ["Major issue 2 (third reviewer report): natural comparison of the reply restricted to TCP floods",
         f"Generated: {datetime.now(timezone.utc).isoformat()}",
         f"CORE G0 models of m8_reply_check.py part b, rebuilt; same NF-BoT-IoT sample ({M.TARGET_ROWS:,} rows per "
         "class, seed 7). Share of DoS/DDoS rows predicted Benign; median [min, max] over seeds.", ""]
    ok = n = 0
    meds = {}
    for src in M.SOURCES:
        a, t = C.open_cache(src), C.open_cache(M.bot_of(src))
        res = {}
        head = None
        for seed in M.SEEDS:
            feat, y, keep, names, split, take, present, enc = M.setup(src, seed, a)
            ty, tkeep, _ = C.task_labels(t, "CORE", names)
            trows = S.balanced_rows(ty, tkeep, 3 * M.TARGET_ROWS, np.random.default_rng(7))
            tatt = np.isin(ty[trows], [names.index("DoS"), names.index("DDoS")])
            Xt_all = t["X"][trows]
            replied = Xt_all[:, t["cols"].index("OUT_PKTS")] > 0
            tcp = Xt_all[:, t["cols"].index("PROTOCOL")] == 6
            groups = {"TCP, reply": tatt & tcp & replied, "TCP, no reply": tatt & tcp & ~replied,
                      "non-TCP": tatt & ~tcp, "all, reply": tatt & replied, "all, no reply": tatt & ~replied}
            if head is None:
                head = ("  target DoS/DDoS rows: " + "; ".join(f"{k} {int(v.sum()):,}" for k, v in groups.items()
                                                              if not k.startswith("all"))
                        + f"; non-TCP with a reply: {int(np.sum(tatt & ~tcp & replied))}")
            Xtr = np.ascontiguousarray(a["X"][np.flatnonzero(take)][:, [a["cols"].index(c) for c in feat]])
            Xt = np.ascontiguousarray(Xt_all[:, [t["cols"].index(c) for c in feat]])
            for m in ("xgb", "rf"):
                model = P42.make_model(m, seed).fit(Xtr, enc[y[take]])
                ben = present[model.predict(Xt)] == names.index("Benign")
                for k, g in groups.items():
                    res.setdefault((m, k), []).append(float(np.mean(ben[g])) if g.any() else float("nan"))
                C.log(f"{src} s{seed} {m}: TCP reply {res[(m, 'TCP, reply')][-1]:.3f}, TCP no reply "
                      f"{res[(m, 'TCP, no reply')][-1]:.3f}, non-TCP {res[(m, 'non-TCP')][-1]:.3f}")
        L.append(f"== {src} -> {M.bot_of(src)}")
        L.append(head)
        for m in ("xgb", "rf"):
            again = f"with reply {md(res[(m, 'all, reply')])} | without reply {md(res[(m, 'all, no reply')])}"
            same = again == ref.get((src, m))
            ok += same
            n += 1
            for k in ("TCP, reply", "TCP, no reply", "non-TCP", "all, reply", "all, no reply"):
                meds.setdefault(k, []).append(float(np.median(res[(m, k)])))
            L.append(f"  {m:3s} G0: TCP with reply {md(res[(m, 'TCP, reply')])} | TCP without reply "
                     f"{md(res[(m, 'TCP, no reply')])} | non-TCP (no reply) {md(res[(m, 'non-TCP')])}"
                     + ("" if same else f"  [all-protocol values differ from stored: {again}]"))
        L.append("")
    L.append("Summary: range of the per-pair medians (8 source x model pairs) of the share predicted Benign")
    for k, v in meds.items():
        L.append(f"  {k:14s} {min(v):.3f} to {max(v):.3f}")
    L.append(f"Stored all-protocol values of m8_reply_check_b.txt (G0) reproduced in {ok} of {n} source x model pairs")
    OUT.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
