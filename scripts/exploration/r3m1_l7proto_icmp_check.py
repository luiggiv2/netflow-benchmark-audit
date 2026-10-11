"""r3m1_l7proto_icmp_check.py — third reviewer report, major issue 1, 2026-10-09
(plan task P6.1b, block 1): does L7_PROTO copy the destination port, and does
ICMP_TYPE carry information on flows that are not ICMP?

Read-only, all rows of the six datasets, raw values from tabular.parquet.

Part 1, L7_PROTO and L4_DST_PORT. For all flows, flows without a reply
(OUT_PKTS == 0) and flows with one: share with L7_PROTO == 0 (unknown);
port -> L7 = share of rows whose L7_PROTO is the most frequent value for
their destination port (full value, and integer part = master protocol);
baseline = share of the most frequent L7_PROTO value without the port;
L7 -> port = among rows with L7_PROTO != 0, share whose destination port is
the most frequent port for their L7_PROTO value. Then the most frequent
destination ports with their L7_PROTO values, and the reverse. Second pass:
rows with L7_PROTO != 0 whose destination port is not the most frequent port
of their L7_PROTO value, and how many of them have that port as source port
(L4_SRC_PORT is not a model feature).

Part 2, probe rules. Every class of data/audit/shortcut_probes_common_report.txt
whose depth-1 rule uses L4_DST_PORT, L7_PROTO or an ICMP field in set A, or
L7_PROTO or an ICMP field in set C: share of the class rows and of the Benign
rows on the class side of each rule, and share of rows on which both rules
agree. The probes were fitted on a balanced sample of one grouped split; here
the rules are only re-applied to all rows. A summary at the end counts, over
all classes of the report, the depth-1 rules that use L7_PROTO or an ICMP
field in each set.

Part 3, ICMP_TYPE and ICMP_IPV4_TYPE by protocol (ICMP = 1 or 58). On TCP and
UDP flows with ICMP_TYPE != 0: share with a zero low byte, share equal to a
port, share whose high byte equals the high byte of a port, distinct values,
and the classes that carry them.

Usage: python scripts/exploration/r3m1_l7proto_icmp_check.py
Output: data/exploration/r3m1_l7proto_icmp_check.txt
"""

from __future__ import annotations

import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nfv3_config import DATASETS, PROCESSED_ROOT, WORK  # noqa: E402

PROBES = WORK / "data/audit/shortcut_probes_common_report.txt"
OUT = WORK / "data/exploration/r3m1_l7proto_icmp_check.txt"
BATCH = 2_000_000
BENIGN = "Benign"
ICMP_PROTOS = (1, 58)
ICMP_FIELDS = {"ICMP_TYPE", "ICMP_IPV4_TYPE"}
TOP_PORTS, TOP_L7 = 8, 8
BASE_COLS = ["L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL", "L7_PROTO", "ICMP_TYPE", "ICMP_IPV4_TYPE", "OUT_PKTS",
             "native_label"]
RULE_RE = re.compile(r"^  (.+?)\s+(\d\.\d\d)\s+(\d\.\d\d)\s+(\d\.\d\d)\s+(\d\.\d\d)\s+(\d\.\d\d)\s+"
                     r"(\S+) (<=|>) (\S+) \| (\S+) (<=|>) (\S+)\s*$")


def probe_rules() -> tuple[dict[str, list[dict]], list[tuple[str, str, str, str]]]:
    """Classes whose depth-1 rule in set A or set C touches the port, L7_PROTO or ICMP;
    and (dataset, class, feature A, feature C) for every class of the report."""
    out: dict[str, list[dict]] = {}
    every: list[tuple[str, str, str, str]] = []
    ds = None
    for line in PROBES.read_text().splitlines():
        if line.startswith("== "):
            ds = line[3:].strip()
            continue
        m = RULE_RE.match(line)
        if not (m and ds):
            continue
        cls, fa, oa, ta, fc, oc, tc = m.group(1), *m.group(7, 8, 9, 10, 11, 12)
        every.append((ds, cls, fa, fc))
        if fa in {"L4_DST_PORT", "L7_PROTO"} | ICMP_FIELDS or fc in {"L7_PROTO"} | ICMP_FIELDS:
            out.setdefault(ds, []).append({"cls": cls, "d1A": m.group(2), "d1C": m.group(4),
                                           "A": (fa, oa, float(ta)), "C": (fc, oc, float(tc))})
    return out, every


def rule_summary(every: list[tuple[str, str, str, str]]) -> list[str]:
    service = {"L7_PROTO"} | ICMP_FIELDS
    L = [f"== Summary of the depth-1 probe rules ({len(every)} classes in {PROBES.relative_to(WORK)})"]
    for k, name in ((2, "set A"), (3, "set C")):
        hit = [e for e in every if e[k] in service]
        L.append(f"  {name}: {len(hit)} of {len(every)} rules use L7_PROTO or an ICMP field: "
                 + "; ".join(f"{e[0]} {e[1]} ({e[k]})" for e in hit))
    port_a = [e for e in every if e[2] == "L4_DST_PORT"]
    to_svc = [e for e in port_a if e[3] in service]
    L.append(f"  set A rules on L4_DST_PORT: {len(port_a)}; their set-C rule uses L7_PROTO or an ICMP field in "
             f"{len(to_svc)}: " + "; ".join(f"{d} {c} ({fc})" for d, c, _, fc in to_svc))
    return L


def off_port_source(ds: str, allc: pd.Series) -> str:
    """Rows with L7_PROTO != 0 off the most frequent destination port of their value."""
    known = allc[allc.index.get_level_values("l7") != 0.0]
    modal = known.groupby(level="l7").idxmax().map(lambda t: t[0])
    off = src = n = 0
    pf = pq.ParquetFile(PROCESSED_ROOT / ds / "tabular.parquet")
    for b in pf.iter_batches(batch_size=BATCH, columns=["L4_SRC_PORT", "L4_DST_PORT", "L7_PROTO"]):
        l7 = b.column("L7_PROTO").to_numpy()
        k = l7 != 0
        mp = modal.to_numpy()[modal.index.get_indexer(l7[k])]
        o = b.column("L4_DST_PORT").to_numpy()[k] != mp
        n += int(k.sum())
        off += int(o.sum())
        src += int((b.column("L4_SRC_PORT").to_numpy()[k][o] == mp[o]).sum())
    return (f"  L7 != 0 rows off the most frequent destination port of their L7_PROTO value: {off:,} "
            f"({100 * off / n:.1f} % of {n:,}); of these, source port = that port {100 * src / max(off, 1):.1f} %")


def side(x: np.ndarray, op: str, thr: float) -> np.ndarray:
    return x <= thr if op == "<=" else x > thr


def modal_acc(counts: pd.Series, key: str) -> float:
    """Share of rows equal to the most frequent value within each group of `key`."""
    if counts.empty:
        return float("nan")
    return float(counts.groupby(level=key).max().sum() / counts.sum())


def part1_block(cnt: pd.Series, label: str) -> list[str]:
    """cnt: row counts indexed by (port, l7)."""
    n = int(cnt.sum())
    if n == 0:
        return [f"  {label:10s} rows 0"]
    l7 = cnt.groupby(level="l7").sum()
    unk = float(l7.get(0.0, 0) / n)
    master = cnt.copy()
    master.index = pd.MultiIndex.from_arrays([cnt.index.get_level_values("port"),
                                              np.floor(cnt.index.get_level_values("l7"))], names=["port", "l7"])
    master = master.groupby(level=["port", "l7"]).sum()
    known = cnt[cnt.index.get_level_values("l7") != 0.0]
    return [f"  {label:10s} rows {n:>11,}  L7=0 {100 * unk:5.1f} %  port->L7 {100 * modal_acc(cnt, 'port'):5.1f} %"
            f" (master {100 * modal_acc(master, 'port'):5.1f} %)  baseline {100 * l7.max() / n:5.1f} %"
            f"  L7->port {100 * modal_acc(known, 'l7'):5.1f} % (L7 != 0, {int(known.sum()):,} rows)"]


def top_tables(cnt: pd.Series) -> list[str]:
    n = cnt.sum()
    L = [f"  top destination ports (share of rows; L7_PROTO values within the port):"]
    by_port = cnt.groupby(level="port").sum().sort_values(ascending=False)
    for p in by_port.index[:TOP_PORTS]:
        g = cnt.xs(p, level="port").sort_values(ascending=False)
        vals = ", ".join(f"{v:g} {100 * c / g.sum():.1f} %" for v, c in g.head(3).items())
        L.append(f"    port {int(p):>5d}  {100 * by_port[p] / n:5.1f} %  L7: {vals}")
    L.append(f"  top L7_PROTO values != 0 (share of rows; destination ports within the value):")
    by_l7 = cnt.groupby(level="l7").sum().drop(0.0, errors="ignore").sort_values(ascending=False)
    for v in by_l7.index[:TOP_L7]:
        g = cnt.xs(v, level="l7").sort_values(ascending=False)
        ports = ", ".join(f"{int(p)} {100 * c / g.sum():.1f} %" for p, c in g.head(3).items())
        L.append(f"    L7 {v:>8g}  {100 * by_l7[v] / n:5.1f} %  ports: {ports}  ({len(g):,} distinct ports)")
    return L


def scan(ds: str, rules: list[dict]) -> list[str]:
    feats = sorted({r[k][0] for r in rules for k in ("A", "C")} - set(BASE_COLS))
    pf = pq.ParquetFile(PROCESSED_ROOT / ds / "tabular.parquet")
    pairs: list[pd.Series] = []
    proto_rows: dict[str, list[int]] = {}
    nz_parts: list[pd.DataFrame] = []
    cls_rows: dict[str, int] = {}
    rule_acc = {(r["cls"], who): np.zeros(4, dtype=np.int64) for r in rules for who in ("cls", "ben")}
    for b in pf.iter_batches(batch_size=BATCH, columns=BASE_COLS + feats):
        df = b.to_pandas()
        df["native_label"] = df["native_label"].astype("category")
        for c, k in df["native_label"].value_counts().items():
            cls_rows[str(c)] = cls_rows.get(str(c), 0) + int(k)
        reply = (df["OUT_PKTS"].to_numpy() > 0).astype(np.int8)
        pairs.append(pd.DataFrame({"reply": reply, "port": df["L4_DST_PORT"].to_numpy(),
                                   "l7": df["L7_PROTO"].to_numpy()}).value_counts())
        proto = df["PROTOCOL"].to_numpy()
        it, i4 = df["ICMP_TYPE"].to_numpy(), df["ICMP_IPV4_TYPE"].to_numpy()
        grp = np.where(np.isin(proto, ICMP_PROTOS), "ICMP (1, 58)",
                       np.where(proto == 6, "TCP", np.where(proto == 17, "UDP", "other")))
        for g in np.unique(grp):
            s = grp == g
            acc = proto_rows.setdefault(g, [0, 0, 0, 0])
            acc[0] += int(s.sum())
            acc[1] += int((it[s] != 0).sum())
            acc[2] += int((i4[s] != 0).sum())
            acc[3] += int((i4[s] == it[s] // 256).sum())
        s = np.isin(proto, (6, 17)) & (it != 0)
        if s.any():
            nz_parts.append(df.loc[s, ["L4_SRC_PORT", "L4_DST_PORT", "ICMP_TYPE", "native_label"]].assign(
                native_label=lambda d: d["native_label"].astype(str)))
        lab = df["native_label"].astype(str).to_numpy()
        for r in rules:
            ra = side(df[r["A"][0]].to_numpy(), r["A"][1], r["A"][2])
            rc = side(df[r["C"][0]].to_numpy(), r["C"][1], r["C"][2])
            for who, m in (("cls", lab == r["cls"]), ("ben", lab == BENIGN)):
                a = rule_acc[(r["cls"], who)]
                a += [int(m.sum()), int((ra & m).sum()), int((rc & m).sum()), int(((ra == rc) & m).sum())]
    cnt = pd.concat(pairs).groupby(level=["reply", "port", "l7"]).sum()
    n_rows = int(cnt.sum())

    L = [f"== {ds} ({n_rows:,} rows)", "-- Part 1: L7_PROTO and L4_DST_PORT"]
    allc = cnt.groupby(level=["port", "l7"]).sum()
    L += part1_block(allc, "all")
    for flag, label in ((0, "no reply"), (1, "reply")):
        sub = cnt.xs(flag, level="reply") if flag in cnt.index.get_level_values("reply") else allc.iloc[:0]
        L += part1_block(sub, label)
    L.append(off_port_source(ds, allc))
    L += top_tables(allc)

    L.append("-- Part 2: probe rules re-applied to all rows (class side of each rule)")
    if not rules:
        L.append("  no class with a rule on L4_DST_PORT, L7_PROTO or ICMP fields")
    for r in rules:
        assert r["cls"] in cls_rows, f"{ds}: class {r['cls']} not found"
        c, bn = rule_acc[(r["cls"], "cls")], rule_acc[(r["cls"], "ben")]
        fa, oa, ta = r["A"]
        fc, oc, tc = r["C"]
        L.append(f"  {r['cls']}  (probe d1 A {r['d1A']}, d1 C {r['d1C']})")
        L.append(f"    rule A {fa} {oa} {ta:g}: class {100 * c[1] / c[0]:5.1f} %, Benign {100 * bn[1] / bn[0]:5.1f} %")
        L.append(f"    rule C {fc} {oc} {tc:g}: class {100 * c[2] / c[0]:5.1f} %, Benign {100 * bn[2] / bn[0]:5.1f} %")
        L.append(f"    both rules on the same side: class {100 * c[3] / c[0]:5.1f} % of {c[0]:,} rows, "
                 f"Benign {100 * bn[3] / bn[0]:5.1f} % of {bn[0]:,} rows")

    L.append("-- Part 3: ICMP_TYPE and ICMP_IPV4_TYPE by protocol")
    for g in ("ICMP (1, 58)", "TCP", "UDP", "other"):
        if g not in proto_rows:
            continue
        n, a, b4, eq = proto_rows[g]
        L.append(f"  {g:13s} rows {n:>11,}  ICMP_TYPE != 0 {100 * a / n:6.2f} %  ICMP_IPV4_TYPE != 0 "
                 f"{100 * b4 / n:6.2f} %  ICMP_IPV4_TYPE == ICMP_TYPE // 256 {100 * eq / n:6.2f} %")
    nz = pd.concat(nz_parts, ignore_index=True) if nz_parts else pd.DataFrame()
    n_tu = sum(proto_rows.get(g, [0])[0] for g in ("TCP", "UDP"))
    L.append(f"  TCP and UDP flows with ICMP_TYPE != 0: {len(nz):,} ({100 * len(nz) / n_tu:.2f} % of TCP and UDP)")
    if len(nz):
        it = nz["ICMP_TYPE"].to_numpy()
        sp, dp = nz["L4_SRC_PORT"].to_numpy(), nz["L4_DST_PORT"].to_numpy()
        L.append(f"    low byte zero {100 * np.mean(it % 256 == 0):.2f} %; equal to source port "
                 f"{100 * np.mean(it == sp):.2f} %, to destination port {100 * np.mean(it == dp):.2f} %; "
                 f"high byte = source-port high byte {100 * np.mean(it // 256 == sp // 256):.2f} %, "
                 f"= destination-port high byte {100 * np.mean(it // 256 == dp // 256):.2f} %; "
                 f"{nz['ICMP_TYPE'].nunique()} distinct values")
        vc = nz["native_label"].value_counts()
        L.append("    classes (rows with ICMP_TYPE != 0 on TCP/UDP, share of all rows of the class):")
        for c, k in vc.items():
            share = k / cls_rows[c]
            if k >= 1000 or share >= 0.05:
                top = (nz.loc[nz["native_label"] == c, "ICMP_TYPE"] // 256).value_counts()
                tops = ", ".join(f"{int(v)} {100 * x / k:.0f} %" for v, x in top.head(3).items())
                L.append(f"      {c:28s} {k:>10,}  {100 * share:5.1f} % of class  (high byte: {tops})")
    L.append("")
    return L


def main() -> None:
    rules, every = probe_rules()
    L = ["Third reviewer report, major issue 1: L7_PROTO against the destination port; ICMP fields on non-ICMP flows",
         f"Generated: {datetime.now(timezone.utc).isoformat()}",
         "All rows, raw values of tabular.parquet. Reply = OUT_PKTS > 0. port->L7 / L7->port = share of rows equal to",
         "the most frequent value given the other field; baseline = share of the most frequent L7_PROTO value.",
         f"Part 2 rules parsed from {PROBES.relative_to(WORK)} (set A = 40 common features; set C = A without",
         "ports, TTL, TCP window and DNS fields; one grouped split, seed 42); here re-applied to all rows.", ""]
    for ds in DATASETS:
        L += scan(ds, rules.get(ds, []))
        print(f"done {ds}", flush=True)
    L += rule_summary(every)
    OUT.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
