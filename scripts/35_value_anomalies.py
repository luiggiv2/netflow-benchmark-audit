"""35_value_anomalies.py — RQ1: physically impossible or inconsistent values,
NF-v2 vs NF-v3 (PLAN-PAPER-B.md, P3.6).

Row-level consistency checks on the raw values of `data.parquet` (all rows,
no deduplication), counted per dataset and per native class:

  rate_nonfinite      *_SECOND_BYTES is NaN or Inf
  rate_over_100gbps   finite *_SECOND_BYTES above 1.25e10 bytes/s (100 Gbit/s)
  thpt_over_100gbps   *_AVG_THROUGHPUT above 1e11 bit/s
  duration_near_2p32  FLOW_DURATION_MILLISECONDS >= 4,000,000 (2^32 / 1000 = 4,294,967.3).
                      No NF-v3 flow lasts more than 121 s, and no NF-v2 value lies
                      between 1 ms and 4,175,393 ms, so every value above this
                      threshold belongs to the artefact (internal audit B1,
                      data/exploration/f8_duration_check.txt; the first version
                      used 4,294,000 ms and missed most of it in NF-BoT-IoT-v2)
  dir_duration_gt     DURATION_IN or DURATION_OUT > FLOW_DURATION_MILLISECONDS
  bytes_below_ip_hdr  bytes < 20 x packets in a direction (20 = minimum IPv4 header)
  bytes_without_pkts  bytes > 0 with 0 packets in a direction
  pkt_len_order       LONGEST < SHORTEST or MAX_IP_PKT_LEN < MIN_IP_PKT_LEN
  pkt_len_over_65535  LONGEST_FLOW_PKT or MAX_IP_PKT_LEN above 65,535
  size_bins_gt_pkts   sum of NUM_PKTS_* bins > IN_PKTS + OUT_PKTS
  ttl_order           MIN_TTL > MAX_TTL, or MAX_TTL > 255
  tcp_flags_non_tcp   any TCP flag field non-zero with PROTOCOL != 6
  retrans_gt_total    retransmitted bytes or packets > total in that direction
  dns_ttl_over_2p31   DNS_TTL_ANSWER > 2^31 - 1 (RFC 2181: must be treated as 0)
  port_out_of_range   a port < 0 or > 65,535
NF-v3 only:
  ts_order            FLOW_END_MILLISECONDS < FLOW_START_MILLISECONDS
  ts_duration_gap     |(END - START) - FLOW_DURATION_MILLISECONDS| > 1,000 ms
  iat_order           an IAT_MAX < IAT_MIN, or a negative IAT

Also recorded: constant features per dataset, and the destination ports of
service-specific classes (FTP-BruteForce: 21, SSH-Bruteforce: 22).

Usage:
  python scripts/35_value_anomalies.py --dataset NF-ToN-IoT-v2
  python scripts/35_value_anomalies.py --report
Outputs: data/audit/value_anomalies/<dataset>.json and
data/audit/value_anomalies_report.txt
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np
import pyarrow.parquet as pq

from nfv3_config import DATASETS, PROCESSED_ROOT, WORK, feature_cols, is_v2

BATCH = 1_000_000
OUT = WORK / "data/audit"
DURATION_WRAP_MIN = 4_000_000  # ms; see duration_near_2p32 in the docstring
SERVICE_PORT = {"FTP-BruteForce": 21, "SSH-Bruteforce": 22}
BINS = ["NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES", "NUM_PKTS_256_TO_512_BYTES",
        "NUM_PKTS_512_TO_1024_BYTES", "NUM_PKTS_1024_TO_1514_BYTES"]
IAT = [("SRC_TO_DST_IAT_MIN", "SRC_TO_DST_IAT_MAX"), ("DST_TO_SRC_IAT_MIN", "DST_TO_SRC_IAT_MAX")]


def checks(d: dict, v3: bool) -> dict:
    g = lambda c: d[c].astype(np.float64)  # noqa: E731
    r1, r2 = g("SRC_TO_DST_SECOND_BYTES"), g("DST_TO_SRC_SECOND_BYTES")
    inb, outb, inp, outp = g("IN_BYTES"), g("OUT_BYTES"), g("IN_PKTS"), g("OUT_PKTS")
    dur = g("FLOW_DURATION_MILLISECONDS")
    pk = (inp + outp) > 0
    with np.errstate(invalid="ignore"):
        c = {
            "rate_nonfinite": ~np.isfinite(r1) | ~np.isfinite(r2),
            "rate_over_100gbps": (np.isfinite(r1) & (r1 > 1.25e10)) | (np.isfinite(r2) & (r2 > 1.25e10)),
            "thpt_over_100gbps": (g("SRC_TO_DST_AVG_THROUGHPUT") > 1e11) | (g("DST_TO_SRC_AVG_THROUGHPUT") > 1e11),
            "duration_near_2p32": dur >= DURATION_WRAP_MIN,
            "dir_duration_gt": (g("DURATION_IN") > dur) | (g("DURATION_OUT") > dur),
            "bytes_below_ip_hdr": (inb < 20 * inp) | (outb < 20 * outp),
            "bytes_without_pkts": ((inb > 0) & (inp == 0)) | ((outb > 0) & (outp == 0)),
            "pkt_len_order": pk & ((g("LONGEST_FLOW_PKT") < g("SHORTEST_FLOW_PKT"))
                                   | (g("MAX_IP_PKT_LEN") < g("MIN_IP_PKT_LEN"))),
            "pkt_len_over_65535": (g("LONGEST_FLOW_PKT") > 65535) | (g("MAX_IP_PKT_LEN") > 65535),
            "size_bins_gt_pkts": sum(g(b) for b in BINS) > inp + outp,
            "ttl_order": (pk & (g("MIN_TTL") > g("MAX_TTL"))) | (g("MAX_TTL") > 255),
            "tcp_flags_non_tcp": (g("PROTOCOL") != 6) & ((g("TCP_FLAGS") != 0) | (g("CLIENT_TCP_FLAGS") != 0)
                                                          | (g("SERVER_TCP_FLAGS") != 0)),
            "retrans_gt_total": (g("RETRANSMITTED_IN_BYTES") > inb) | (g("RETRANSMITTED_OUT_BYTES") > outb)
                                | (g("RETRANSMITTED_IN_PKTS") > inp) | (g("RETRANSMITTED_OUT_PKTS") > outp),
            "dns_ttl_over_2p31": g("DNS_TTL_ANSWER") > 2**31 - 1,
            "port_out_of_range": (g("L4_SRC_PORT") < 0) | (g("L4_SRC_PORT") > 65535)
                                 | (g("L4_DST_PORT") < 0) | (g("L4_DST_PORT") > 65535),
        }
        if v3:
            start, end = g("FLOW_START_MILLISECONDS"), g("FLOW_END_MILLISECONDS")
            c["ts_order"] = end < start
            c["ts_duration_gap"] = np.abs((end - start) - dur) > 1000
            iat = np.zeros(len(dur), dtype=bool)
            for lo, hi in IAT:
                iat |= (g(hi) < g(lo)) | (g(lo) < 0)
            c["iat_order"] = iat
    return c


def run(dataset: str) -> dict:
    v3 = not is_v2(dataset)
    cols = list(dict.fromkeys(feature_cols(dataset) + (["FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS"] if v3 else [])))
    pf = pq.ParquetFile(PROCESSED_ROOT / dataset / "data.parquet")
    rows = Counter()
    hits: dict = defaultdict(Counter)
    lo, hi = {}, {}
    ports = defaultdict(Counter)
    near = Counter()
    for b in pf.iter_batches(batch_size=BATCH, columns=cols + ["native_label"]):
        df = b.to_pandas()
        lab = df["native_label"].astype(str).to_numpy()
        rows.update(lab)
        d = {c: df[c].to_numpy() for c in cols}
        for name, m in checks(d, v3).items():
            if m.any():
                hits[name].update(lab[m])
        for c in feature_cols(dataset):
            x = d[c].astype(np.float64)
            x = x[np.isfinite(x)]
            if x.size:
                lo[c] = min(lo.get(c, np.inf), float(x.min()))
                hi[c] = max(hi.get(c, -np.inf), float(x.max()))
        for cls in SERVICE_PORT:
            m = lab == cls
            if m.any():
                ports[cls].update(d["L4_DST_PORT"][m].tolist())
        dur = d["FLOW_DURATION_MILLISECONDS"]
        near.update(dur[dur >= DURATION_WRAP_MIN].tolist())
    out = {"dataset": dataset, "generated_utc": datetime.now(timezone.utc).isoformat(),
           "rows": int(sum(rows.values())), "class_rows": dict(rows),
           "checks": {k: {"rows": int(sum(v.values())), "per_class": dict(v)} for k, v in hits.items()},
           "constant_features": sorted(c for c in lo if lo[c] == hi[c]),
           "constant_values": {c: lo[c] for c in lo if lo[c] == hi[c]},
           "duration_near_2p32_values": {str(k): int(v) for k, v in near.most_common(5)},
           "service_ports": {cls: {"rows": int(sum(p.values())),
                                   "expected_port_share": p[SERVICE_PORT[cls]] / max(sum(p.values()), 1),
                                   "top_ports": {str(k): int(v) for k, v in p.most_common(5)}}
                             for cls, p in ports.items()}}
    d = OUT / "value_anomalies"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{dataset}.json").write_text(json.dumps(out, indent=2))
    print(f"{dataset}: {out['rows']:,} rows; checks hit: "
          + ", ".join(f"{k} {v['rows']:,}" for k, v in out["checks"].items()))
    return out


def report():
    res = {json.loads(f.read_text())["dataset"]: json.loads(f.read_text())
           for f in sorted((OUT / "value_anomalies").glob("*.json"))}
    order = ["NF-ToN-IoT-v2", "NF-ToN-IoT-v3", "NF-BoT-IoT-v2", "NF-BoT-IoT-v3",
             "NF-CSE-CIC-IDS2018-v2", "NF-CSE-CIC-IDS2018-v3"]
    order = [o for o in order if o in res]
    names = sorted({k for r in res.values() for k in r["checks"]} | {
        "rate_nonfinite", "rate_over_100gbps", "thpt_over_100gbps", "duration_near_2p32", "dir_duration_gt",
        "bytes_below_ip_hdr", "bytes_without_pkts", "pkt_len_order", "pkt_len_over_65535",
        "size_bins_gt_pkts", "ttl_order", "tcp_flags_non_tcp", "retrans_gt_total",
        "dns_ttl_over_2p31", "port_out_of_range", "ts_order", "ts_duration_gap", "iat_order"})
    short = {o: o.replace("NF-", "").replace("CSE-CIC-IDS2018", "CIC").replace("-IoT", "") for o in order}
    L = ["RQ1: value anomalies, rows failing each consistency check (share of all rows)",
         f"Generated: {datetime.now(timezone.utc).isoformat()}",
         "Definitions in the docstring of scripts/35_value_anomalies.py. ts_* and iat_* exist in NF-v3 only.", "",
         f"  {'check':20s} " + " ".join(f"{short[o]:>18s}" for o in order)]
    for n in names:
        cells = []
        for o in order:
            r = res[o]
            if n.startswith(("ts_", "iat_")) and is_v2(o):
                cells.append(f"{'n/a':>18s}")
                continue
            k = r["checks"].get(n, {"rows": 0})["rows"]
            cells.append(f"{k:>10,} {100 * k / r['rows']:6.2f}%")
        L.append(f"  {n:20s} " + " ".join(cells))
    L.append("")
    L.append("Classes most affected (top 3 per check and dataset, rows):")
    for n in names:
        for o in order:
            pc = res[o]["checks"].get(n, {}).get("per_class", {})
            if pc:
                top = sorted(pc.items(), key=lambda x: -x[1])[:3]
                L.append(f"  {n:20s} {short[o]:12s} " + "; ".join(
                    f"{c} {v:,} ({100 * v / res[o]['class_rows'][c]:.1f}% of class)" for c, v in top))
    L.append("")
    L.append("Constant features (same value in every row):")
    for o in order:
        L.append(f"  {short[o]:12s} " + (", ".join(f"{c}={v:g}" for c, v in res[o]["constant_values"].items()) or "none"))
    L.append("")
    L.append(f"FLOW_DURATION_MILLISECONDS >= {DURATION_WRAP_MIN:,} (most frequent values):")
    for o in order:
        L.append(f"  {short[o]:12s} {res[o]['duration_near_2p32_values'] or 'none'}")
    L.append("")
    L.append("Destination ports of service-specific classes:")
    for o in order:
        for cls, p in res[o]["service_ports"].items():
            L.append(f"  {short[o]:12s} {cls:16s} rows {p['rows']:,}; expected port "
                     f"{SERVICE_PORT[cls]}: {100 * p['expected_port_share']:.2f} %; top ports {p['top_ports']}")
    text = "\n".join(L) + "\n"
    (OUT / "value_anomalies_report.txt").write_text(text)
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.dataset:
        run(a.dataset)
    if a.report:
        report()


if __name__ == "__main__":
    main()
