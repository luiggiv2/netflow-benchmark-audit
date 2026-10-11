"""32_flood_heuristics.py — reimplementation of the five heuristics of Flood et
al. (2024), applied per native attack class to the NF-v2 and NF-v3 datasets
(PLAN-PAPER-B.md, P3.4). Parameters and deviations from the authors' tool are
documented in research/flood_heuristics_spec.md; no code from their
repository is used.

Per dataset:
  pass 1  full-record hash of every row (all raw columns, label included) to
          drop exact duplicate records, as the authors' loader does; class
          sizes and destination ports of the deduplicated rows (UGT).
  pass 2  seeded samples of at most CAP deduplicated rows per label, with the
          41 features common to NF-v2 and NF-v3.
Per attack class (benign is the background):
  UGT  share of flows to background-service ports (eq. 1)
  WL   Edited Nearest Neighbours, k = 4, >= 2 benign neighbours (eq. 2);
       500-flow reference as in the tool (5 seeds), a 20,000-flow variant, and
       a class-balanced 20,000-flow reference with held-out queries (3 seeds)
  HDF  max macro F1 of a 20-tree random forest on one feature (eq. 3), over
       41 features and over the 39 without ports
  PDD, TC  elbow + KMeans clusters, 500 cosine pairs per cluster (eqs. 5-6),
       3 seeds

Usage:
  python scripts/32_flood_heuristics.py --dataset NF-BoT-IoT-v3 --smoke
  python scripts/32_flood_heuristics.py --dataset NF-BoT-IoT-v3
Output: data/audit/flood_heuristics/<dataset>.json (smoke: <dataset>_smoke.json)
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from nfv3_config import DATASETS, FEATURE_COLS_V2, PROCESSED_ROOT, WORK

FEATS41 = FEATURE_COLS_V2
PORTS = ["L4_SRC_PORT", "L4_DST_PORT"]
FEATS39 = [c for c in FEATS41 if c not in PORTS]
BG_PORTS = [0, 53, 67, 68, 111, 123, 137, 161, 179, 389, 427, 520, 1723, 1900]
DERIVED = {"native_label", "canonical_label", "split"}
BENIGN = "Benign"
SEED = 2026
BATCH = 1_000_000

FULL = {"cap": 500_000, "hdf_total": 500_000, "hdf_min_class": 20_000,
        "wl_ref": 500, "wl_seeds": 5, "wl_queries": 500_000,
        "wl_ref_big": 20_000, "wl_queries_big": 200_000,
        "pdd_rows": 100_000, "pdd_pairs": 500, "pdd_seeds": 3}
SMOKE = {"cap": 50_000, "hdf_total": 50_000, "hdf_min_class": 5_000,
         "wl_ref": 500, "wl_seeds": 2, "wl_queries": 20_000,
         "wl_ref_big": 2_000, "wl_queries_big": 10_000,
         "pdd_rows": 10_000, "pdd_pairs": 500, "pdd_seeds": 1}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def scan(path):
    """Pass 1: dedup mask (first occurrence of each full record), labels, dst ports."""
    pf = pq.ParquetFile(path)
    raw = [c for c in pf.schema_arrow.names if c not in DERIVED]
    hs, labs, ports = [], [], []
    codes: dict[str, int] = {}
    for b in pf.iter_batches(batch_size=BATCH, columns=raw + ["native_label"]):
        df = b.to_pandas()
        hs.append(pd.util.hash_pandas_object(df[raw], index=False).to_numpy())
        lab = df["native_label"].astype(str)
        for v in lab.unique():
            codes.setdefault(v, len(codes))
        labs.append(lab.map(codes).to_numpy(dtype=np.int16))
        ports.append(df["L4_DST_PORT"].to_numpy(dtype=np.int64))
    h = np.concatenate(hs)
    del hs
    keep = np.zeros(len(h), dtype=bool)
    keep[np.unique(h, return_index=True)[1]] = True
    names = {i: v for v, i in codes.items()}
    return keep, np.concatenate(labs), np.concatenate(ports), names


def load_samples(path, keep, labs, names, cap, rng):
    """Pass 2: up to `cap` deduplicated rows per label (41 features)."""
    chosen = np.zeros(len(keep), dtype=bool)
    for lab in np.unique(labs):
        idx = np.flatnonzero(keep & (labs == lab))
        if len(idx) > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        chosen[idx] = True
    parts, start = [], 0
    for b in pq.ParquetFile(path).iter_batches(batch_size=BATCH, columns=FEATS41):
        n = b.num_rows
        m = chosen[start:start + n]
        if m.any():
            df = b.to_pandas().loc[m]
            df["label"] = pd.Series(labs[start:start + n][m]).map(names).to_numpy()
            parts.append(df)
        start += n
    data = pd.concat(parts, ignore_index=True)
    return {lab: g.drop(columns="label").reset_index(drop=True) for lab, g in data.groupby("label")}


def finite(df, cols):
    x = df[cols].to_numpy(dtype=np.float64)
    return x[np.isfinite(x).all(axis=1)]


def minmax(x):
    lo, hi = x.min(axis=0), x.max(axis=0)
    rng_ = hi - lo
    keep = rng_ > 0
    return (x[:, keep] - lo[keep]) / rng_[keep], keep


# --------------------------------------------------------------------------- #
# Heuristics
# --------------------------------------------------------------------------- #


def ugt(ports_c):
    return float(np.isin(ports_c, BG_PORTS).mean())


def hdf(xb, xt, n_b, n_t, p, seed):
    """xb, xt: DataFrames (41 features). Natural proportion with a class floor."""
    rng = np.random.default_rng(seed)
    pt = n_t / (n_t + n_b)
    s_t = max(min(n_t, p["hdf_min_class"]), int(round(p["hdf_total"] * pt)))
    s_t = min(s_t, len(xt))
    s_b = min(len(xb), max(p["hdf_total"] - s_t, 1))
    sb = xb.iloc[rng.choice(len(xb), size=s_b, replace=False)]
    st = xt.iloc[rng.choice(len(xt), size=s_t, replace=False)]
    X = pd.concat([sb, st], ignore_index=True)
    y = np.r_[np.zeros(len(sb), dtype=int), np.ones(len(st), dtype=int)]
    per = {}
    for col in FEATS41:
        x = X[[col]].to_numpy(dtype=np.float64)
        # the forest works in float32; NF-v2 rate values above its range count as non-finite
        ok = (np.isfinite(x) & (np.abs(x) <= np.finfo(np.float32).max)).ravel()
        xtr, xte, ytr, yte = train_test_split(x[ok], y[ok], test_size=0.2, random_state=100)
        rf = RandomForestClassifier(n_estimators=20, random_state=100, n_jobs=-1).fit(xtr, ytr)
        per[col] = float(f1_score(yte, rf.predict(xte), average="macro"))
    best = max(per, key=per.get)
    best39 = max(FEATS39, key=per.get)
    return {"hdf": per[best], "feature": best, "hdf_no_ports": per[best39],
            "feature_no_ports": best39, "rows": {"benign": int(s_b), "class": int(s_t)},
            "per_feature": per}


def wl(xb, xt, n_b, n_t, n_ref, n_query, seeds):
    """ENN with k=4 on min-max scaled benign + class (39 features)."""
    b, t = finite(xb, FEATS39), finite(xt, FEATS39)
    scaled, _ = minmax(np.vstack([b, t]))
    sb, stt = scaled[:len(b)], scaled[len(b):]
    pt = n_t / (n_t + n_b)
    out = []
    for s in seeds:
        rng = np.random.default_rng(s)
        r_t = int(rng.binomial(n_ref, pt))
        r_t, r_b = min(r_t, len(stt)), min(n_ref - r_t, len(sb))
        ref = np.vstack([sb[rng.choice(len(sb), r_b, replace=False)],
                         stt[rng.choice(len(stt), r_t, replace=False)]])
        is_benign = np.r_[np.ones(r_b, bool), np.zeros(r_t, bool)]
        q = stt if len(stt) <= n_query else stt[rng.choice(len(stt), n_query, replace=False)]
        nn = NearestNeighbors(n_neighbors=4).fit(ref)
        _, idx = nn.kneighbors(q)
        out.append(float((is_benign[idx].sum(axis=1) >= 2).mean()))
    return {"mean": float(np.mean(out)), "min": float(np.min(out)), "max": float(np.max(out)),
            "seeds": out, "reference": n_ref, "queries": int(min(len(stt), n_query))}


def wl_balanced(xb, xt, n_ref, n_query, seeds):
    """ENN (k=4) with a class-balanced reference; the queried class flows are
    held out of the reference, so prevalence and self-matches do not drive it."""
    b, t = finite(xb, FEATS39), finite(xt, FEATS39)
    scaled, _ = minmax(np.vstack([b, t]))
    sb, stt = scaled[:len(b)], scaled[len(b):]
    out = []
    for s in seeds:
        rng = np.random.default_rng(s)
        perm = rng.permutation(len(stt))
        r_t = min(n_ref // 2, len(stt) // 2)
        ref_t, q = stt[perm[:r_t]], stt[perm[r_t:]]
        if len(q) > n_query:
            q = q[rng.choice(len(q), n_query, replace=False)]
        ref_b = sb[rng.choice(len(sb), min(r_t, len(sb)), replace=False)]
        ref = np.vstack([ref_b, ref_t])
        is_benign = np.r_[np.ones(len(ref_b), bool), np.zeros(r_t, bool)]
        _, idx = NearestNeighbors(n_neighbors=4).fit(ref).kneighbors(q)
        out.append(float((is_benign[idx].sum(axis=1) >= 2).mean()))
    return {"mean": float(np.mean(out)), "min": float(np.min(out)), "max": float(np.max(out)),
            "seeds": out, "reference_per_side": int(min(n_ref // 2, len(stt) // 2)),
            "queries": int(min(len(stt) - min(n_ref // 2, len(stt) // 2), n_query))}


def knee(inertia):
    k = np.arange(1, len(inertia) + 1, dtype=float)
    if inertia[0] <= 0:
        return 1
    y = (inertia - inertia.min()) / (inertia.max() - inertia.min() + 1e-12)
    x = (k - k.min()) / (k.max() - k.min())
    dist = (1 - x) - y          # distance below the chord from (0,1) to (1,0)
    best = int(np.argmax(dist))
    return int(k[best]) if dist[best] > 1e-9 else 1


def pdd_tc(xt, n_rows, n_pairs, seeds):
    t = finite(xt, FEATS39)
    res = []
    for s in seeds:
        rng = np.random.default_rng(s)
        x = t if len(t) <= n_rows else t[rng.choice(len(t), n_rows, replace=False)]
        if len(x) < 4:
            return {"pdd": None, "tc": None, "note": "too few flows"}
        x, _ = minmax(x)
        if x.shape[1] == 0:
            res.append((1.0, 1.0, 1))
            continue
        inertia = np.array([KMeans(n_clusters=k, n_init="auto", random_state=s).fit(x).inertia_
                            for k in range(1, min(7, len(x)) + 1)])
        n_cl = knee(inertia)
        lab = KMeans(n_clusters=n_cl, n_init="auto", random_state=s).fit_predict(x)
        pdd_num, wsum, tcs = 0.0, 0, []
        for c in range(n_cl):
            xc = x[lab == c]
            if len(xc) < 2:
                continue
            i = rng.integers(0, len(xc), n_pairs)
            j = (i + rng.integers(1, len(xc), n_pairs)) % len(xc)   # j != i
            a, b = xc[i], xc[j]
            na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
            both0 = (na == 0) & (nb == 0)
            cs = np.where(both0, 1.0, (a * b).sum(axis=1) / np.where(na * nb == 0, np.inf, na * nb))
            pdd_num += len(xc) * cs.mean()
            wsum += len(xc)
            tcs.append(float((cs > 0.95).mean()))
        res.append((pdd_num / wsum, max(tcs), n_cl))
    arr = np.array([r[:2] for r in res])
    return {"pdd": float(arr[:, 0].mean()), "tc": float(arr[:, 1].mean()),
            "clusters": [int(r[2]) for r in res], "rows": int(min(len(t), n_rows))}


# --------------------------------------------------------------------------- #


def run(dataset, smoke, classes):
    p = SMOKE if smoke else FULL
    path = PROCESSED_ROOT / dataset / "data.parquet"
    t0 = time.time()
    keep, labs, ports, names = scan(path)
    code = {v: k for k, v in names.items()}
    n_dedup = {names[i]: int((keep & (labs == i)).sum()) for i in names}
    samples = load_samples(path, keep, labs, names, p["cap"], np.random.default_rng(SEED))
    out = {"dataset": dataset, "mode": "smoke" if smoke else "full", "params": p,
           "rows": int(len(labs)), "rows_after_record_dedup": int(keep.sum()),
           "class_rows_after_dedup": n_dedup, "classes": {}}
    print(f"{dataset}: scan + samples in {time.time() - t0:.0f}s; "
          f"{len(labs):,} rows, {int(keep.sum()):,} after record dedup", flush=True)
    targets = [c for c in sorted(n_dedup) if c != BENIGN and (not classes or c in classes)]
    for c in targets:
        t1 = time.time()
        n_t, n_b = n_dedup[c], n_dedup[BENIGN]
        xb, xt = samples[BENIGN], samples[c]
        r = {"rows_dedup": n_t, "ugt": ugt(ports[keep & (labs == code[c])])}
        r["wl_500"] = wl(xb, xt, n_b, n_t, p["wl_ref"], p["wl_queries"],
                         [SEED + i for i in range(p["wl_seeds"])])
        r["wl_big"] = wl(xb, xt, n_b, n_t, p["wl_ref_big"], p["wl_queries_big"], [SEED])
        r["wl_balanced"] = wl_balanced(xb, xt, p["wl_ref_big"], p["wl_queries_big"],
                                       [SEED + i for i in range(3)])
        r["hdf"] = hdf(xb, xt, n_b, n_t, p, SEED)
        r["pdd_tc"] = pdd_tc(xt, p["pdd_rows"], p["pdd_pairs"],
                             [SEED + i for i in range(p["pdd_seeds"])])
        r["elapsed_s"] = round(time.time() - t1, 1)
        out["classes"][c] = r
        print(f"  {c:28s} UGT {r['ugt']:.2f}  WL {r['wl_500']['mean']:.2f} "
              f"(20k ref {r['wl_big']['mean']:.2f}, balanced {r['wl_balanced']['mean']:.2f})  "
              f"HDF {r['hdf']['hdf']:.2f} "
              f"[{r['hdf']['feature']}] no-ports {r['hdf']['hdf_no_ports']:.2f}  "
              f"PDD {r['pdd_tc']['pdd'] if r['pdd_tc']['pdd'] is None else round(r['pdd_tc']['pdd'], 2)}  "
              f"TC {r['pdd_tc']['tc'] if r['pdd_tc']['tc'] is None else round(r['pdd_tc']['tc'], 2)}  "
              f"({r['elapsed_s']}s)", flush=True)
    out["generated_utc"] = datetime.now(timezone.utc).isoformat()
    out["elapsed_s"] = round(time.time() - t0, 1)
    d = WORK / "data/audit/flood_heuristics"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{dataset}{'_smoke' if smoke else ''}.json").write_text(json.dumps(out, indent=2))
    print(f"done in {out['elapsed_s']:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=DATASETS, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--classes", default="", help="comma-separated native labels (default: all)")
    a = ap.parse_args()
    run(a.dataset, a.smoke, [c for c in a.classes.split(",") if c])


if __name__ == "__main__":
    main()
