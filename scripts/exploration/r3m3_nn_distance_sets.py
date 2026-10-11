"""r3m3_nn_distance_sets.py — third reviewer report, major issues 3 and 5,
2026-10-10 (plan task P6.1b, block 3): the model-free check of
nn_distance_bot.py repeated without the testbed fingerprint.

nn_distance_bot.py computed the distances on the model base set, which keeps
the destination port, TTL, TCP window, DNS fields, L7_PROTO and the ICMP
fields, so "the traffic itself differs" mixed traffic with testbed identity.
Here the same queries and reference pools (same seed, same sampling order) are
compared on three feature sets of nfv3_config:
  base     model base set (G0); must reproduce data/exploration/nn_distance_bot.txt
  noid     base without ports, TTL, TCP window and DNS fields (G-NI)
  noidsvc  noid without L7_PROTO, ICMP_TYPE and ICMP_IPV4_TYPE (G-NIS0)
Transform, standardisation, distances and 1-NN vote as in nn_distance_bot.py.

Additions:
  - port-53 breakdown of the NF-ToN-IoT DoS queries (major issue 5): share of
    queries sent to port 53 and, for those and the rest, the median distance
    to the nearest Benign flow of each dataset and the share whose nearest
    Benign flow also goes to port 53 (ports read from the raw rows);
  - a summary per set: ranges of the median nearest distances and of the 1-NN
    votes for the BoT floods, the NF-CSE-CIC-IDS2018 DoS queries (calibration)
    and the NF-ToN-IoT DoS queries, against the other testbeds.

Usage: python scripts/exploration/r3m3_nn_distance_sets.py
Output: data/exploration/r3m3_nn_distance_sets.txt
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
_spec = importlib.util.spec_from_file_location("nnd", ROOT / "scripts/exploration/nn_distance_bot.py")
N = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(N)
C = N.C
from nfv3_config import SERVICE_COLS  # noqa: E402

STORED = ROOT / "data/exploration/nn_distance_bot.txt"
OUT = ROOT / "data/exploration/r3m3_nn_distance_sets.txt"
SETS = ("base", "noid", "noidsvc")
SET_LABEL = {"base": "model base set (G0)", "noid": "no ports, TTL, TCP window, DNS (G-NI)",
             "noidsvc": "G-NI without L7_PROTO and ICMP fields (G-NIS0)"}


def feature_set(d: dict, key: str) -> list[str]:
    if key == "noidsvc":
        return [c for c in d["sets"]["noid"] if c not in SERVICE_COLS]
    return d["sets"][key]


def raw_col(d: dict, rows: np.ndarray, col: str) -> np.ndarray:
    return np.asarray(d["X"][rows][:, d["cols"].index(col)])


def rng_range(v) -> str:
    return f"{min(v):.3f}-{max(v):.3f}" if v else "n/a"


def main() -> None:
    rng = np.random.default_rng(N.SEED)
    L = ["Nearest-neighbour check on three feature sets (third reviewer report, major issues 3 and 5)",
         f"Generated: {datetime.now(timezone.utc).isoformat()}",
         f"Same queries and pools as nn_distance_bot.py (seed {N.SEED}; Q = {N.Q:,}, R = {N.R:,});"
         " sign*log1p, standardised on the pooled rows of each set; Euclidean distance.",
         "1-NN vote: class of the nearest row over the three equal-size pools of one dataset.", ""]
    body = {k: [] for k in SETS}          # per-set lines in the format of nn_distance_bot.txt
    recs = {k: [] for k in SETS}          # (v, query ds, query class, target ds, dist medians, votes)
    port53 = {k: [] for k in SETS}
    for v in ("v2", "v3"):
        names = [f"NF-ToN-IoT-{v}", f"NF-BoT-IoT-{v}", f"NF-CSE-CIC-IDS2018-{v}"]
        data = {n: C.open_cache(n) for n in names}
        bot, ton = names[1], names[0]
        # Sampling exactly as nn_distance_bot.py (same rng sequence, release by release).
        queries, pools = {}, {}
        for n, d in data.items():
            for cls in N.CORE:
                idx = N.rows_of(d, cls)
                if n == bot and cls != "Benign":
                    q = rng.choice(idx, min(N.Q, len(idx)), replace=False)
                    queries[(n, cls)] = q
                    idx = np.setdiff1d(idx, q)
                elif cls == "DoS":
                    q = rng.choice(idx, min(N.Q, len(idx)), replace=False)
                    queries[(n, cls)] = q
                    idx = np.setdiff1d(idx, q)
                pools[(n, cls)] = np.sort(rng.choice(idx, min(N.R, len(idx)), replace=False))
        qsorted = {k: np.sort(i) for k, i in queries.items()}
        udp = {k: 100 * np.mean(raw_col(data[k[0]], i, "PROTOCOL") == 17) for k, i in qsorted.items()}
        ton_q = qsorted[(ton, "DoS")]
        ton_is53 = raw_col(data[ton], ton_q, "L4_DST_PORT") == 53
        benign_port = {n: raw_col(data[n], pools[(n, "Benign")], "L4_DST_PORT") for n in names}
        for key in SETS:
            feat = feature_set(data[names[0]], key)
            assert all(feature_set(data[n], key) == feat for n in names)
            Xp = {k: N.transform(np.asarray(data[k[0]]["X"][i][:, [data[k[0]]["cols"].index(c) for c in feat]]))
                  for k, i in pools.items()}
            Xq = {k: N.transform(np.asarray(data[k[0]]["X"][i][:, [data[k[0]]["cols"].index(c) for c in feat]]))
                  for k, i in qsorted.items()}
            allp = np.vstack(list(Xp.values()))
            mu, sd = allp.mean(0), allp.std(0)
            keepf = sd > 0
            z = lambda A: (A[:, keepf] - mu[keepf]) / sd[keepf]  # noqa: E731
            nn = {k: NearestNeighbors(n_neighbors=1).fit(z(A)) for k, A in Xp.items()}
            B = body[key]
            B.append(f"== {v}: {int(keepf.sum())} features with non-zero SD; pool sizes "
                     + ", ".join(f"{N.short(n)} {c} {len(pools[(n, c)]):,}" for n in names for c in N.CORE))
            for (qn, qc), qi in queries.items():
                B.append(f"  queries {N.short(qn)} {qc} ({len(qi):,}; UDP {udp[(qn, qc)]:.1f} %)")
                zq = z(Xq[(qn, qc)])
                for n in names:
                    res = [nn[(n, c)].kneighbors(zq) for c in N.CORE]
                    dist = np.column_stack([r[0][:, 0] for r in res])
                    vote = dist.argmin(1)
                    med = [float(np.median(dist[:, j])) for j in range(3)]
                    share = [100 * float(np.mean(vote == j)) for j in range(3)]
                    B.append(f"    vs {N.short(n):7s} median NN distance " + ", ".join(
                        f"{c} {med[j]:6.3f}" for j, c in enumerate(N.CORE))
                        + "  | 1-NN vote " + ", ".join(f"{c} {share[j]:5.1f} %" for j, c in enumerate(N.CORE)))
                    recs[key].append((v, qn, qc, n, med, share))
                    if (qn, qc) == (ton, "DoS"):
                        nb_port = benign_port[n][res[0][1][:, 0]]
                        for lab, m in (("port 53", ton_is53), ("other ports", ~ton_is53)):
                            port53[key].append(
                                f"  {N.short(ton)} DoS to {lab:11s} ({int(m.sum()):,} of {len(m):,}) vs {N.short(n):7s}"
                                f" Benign: median NN distance {np.median(dist[m, 0]):6.3f};"
                                f" nearest Benign flow to port 53 {100 * np.mean(nb_port[m] == 53):5.1f} %")
            B.append("")
        del data

    stored = STORED.read_text().splitlines()
    stored_body = stored[stored.index(next(s for s in stored if s.startswith("== v2"))):]
    same = [a == b for a, b in zip(body["base"], stored_body)]
    L.append(f"Reproduction of nn_distance_bot.txt with the base set: {sum(same)} of {len(stored_body)} lines identical"
             + ("" if len(body["base"]) == len(stored_body) else f" (line counts {len(body['base'])} vs {len(stored_body)})"))
    L.append("")
    for key in SETS:
        L.append(f"######## Set {key}: {SET_LABEL[key]}")
        L.extend(body[key])
        L.append("Port-53 breakdown of the NF-ToN-IoT DoS queries")
        L.extend(port53[key])
        L.append("")
        L.append("Summary against the other testbeds (median nearest distances; 1-NN vote over equal pools)")
        for qlab, qds in (("BoT DoS/DDoS", "BoT"), ("CIC DoS (calibration)", "CSE"), ("ToN DoS", "ToN")):
            rows = [r for r in recs[key] if qds in r[1] and r[3] != r[1] and (qds == "BoT" or r[2] == "DoS")]
            own = [r for r in recs[key] if qds in r[1] and r[3] == r[1]]
            att = [m for r in rows for m in r[4][1:]]
            ben = [r[4][0] for r in rows]
            L.append(f"  {qlab}: to foreign DoS/DDoS {rng_range(att)}; to foreign Benign {rng_range(ben)};"
                     f" to own Benign {rng_range([r[4][0] for r in own])}")
            L.append("    nearest flow Benign (% of queries) per combination: " + ", ".join(
                f"{r[0]} {N.short(r[1]).split('-')[0]} {r[2]} vs {N.short(r[3]).split('-')[0]} {r[5][0]:.1f}"
                for r in rows))
        L.append("")
    OUT.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
