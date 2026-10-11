"""43_inspect_in_domain.py — P4.2b (research/phase4_protocol.md §12): why the
grouped split is unstable across seeds. No experiment is re-run except one
diagnostic refit (part 3).

Part 1, concentration: for each dataset and native class, under the base key
(all features but L4_SRC_PORT): rows, distinct vectors, share of the class
rows held by its largest 1, 10 and 100 vectors, and the number of vectors
that hold half of the rows.
Part 2, conflicts: vectors of the base key that carry more than one canonical
label (rows outside the taxonomy as "(excluded)"), by label set; and vectors
with both binary labels.
Part 3, NF-BoT-IoT-v2 random forest: in seeds 2, 4 and 5 of G0 BIN about 10 %
of the DDoS test rows were predicted Benign. The largest DDoS vectors are
listed with their split per seed; G0 BIN is refitted for seed 2 (XGBoost and
RF, same sample as 42_in_domain.py) and the predictions for those vectors
are reported with their feature values.

Summary (--summary; added 2026-10-05 for the Phase 4 gate, no training): the
P4.2 figures behind the gate claims, read from data/eval/in_domain/*.jsonl:
seed ranges of G0 and R0; conflict ceilings; G-TS against R0; paired G-ND
and G-NC differences against G0; R0 against G0 and the R0 leakage.

Usage: python scripts/43_inspect_in_domain.py [--part 1 2 3] | --summary
Outputs: data/eval/in_domain_inspection.txt and .json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pyarrow.parquet as pq

from nfv3_config import (DATASETS, PROCESSED_ROOT, WORK, grouped_split_codes, model_feature_sets,
                         tabular_row_hash)

BATCH = 1_000_000
EXCLUDED = "(excluded)"
OUT = WORK / "data/eval"
SEEDS = [1, 2, 3, 4, 5]


def hashes_and_labels(dataset):
    cols = model_feature_sets(dataset)["base"]
    pf = pq.ParquetFile(PROCESSED_ROOT / dataset / "tabular.parquet")
    h, nat, can, lab = [], [], [], []
    for b in pf.iter_batches(batch_size=BATCH, columns=cols + ["Label", "native_label", "canonical_label"]):
        df = b.to_pandas()
        h.append(tabular_row_hash(df, cols))
        nat.append(df["native_label"].astype(str).to_numpy())
        can.append(df["canonical_label"].astype(object).fillna("").astype(str).replace("", EXCLUDED).to_numpy())
        lab.append(df["Label"].to_numpy().astype(np.int8))
    return np.concatenate(h), np.concatenate(nat), np.concatenate(can), np.concatenate(lab)


def part1(dataset, h, nat):
    out = {}
    for c in sorted(np.unique(nat)):
        hc = h[nat == c]
        _, cnt = np.unique(hc, return_counts=True)
        cnt = np.sort(cnt)[::-1]
        cum = np.cumsum(cnt)
        n = int(cum[-1])
        out[c] = {"rows": n, "distinct_vectors": int(len(cnt)),
                  "top1_share": round(float(cum[0] / n), 4),
                  "top10_share": round(float(cum[min(9, len(cum) - 1)] / n), 4),
                  "top100_share": round(float(cum[min(99, len(cum) - 1)] / n), 4),
                  "vectors_for_half": int(np.searchsorted(cum, n / 2) + 1),
                  "largest_vector_rows": int(cnt[0])}
    _, cnt = np.unique(h, return_counts=True)
    cnt = np.sort(cnt)[::-1]
    whole = {"rows": int(cnt.sum()), "distinct_vectors": int(len(cnt)),
             "top10_share": round(float(cnt[:10].sum() / cnt.sum()), 4),
             "top100_share": round(float(cnt[:100].sum() / cnt.sum()), 4),
             "vectors_for_half": int(np.searchsorted(np.cumsum(cnt), cnt.sum() / 2) + 1)}
    return {"classes": out, "dataset": whole}


def label_sets(h, y):
    """Label set of every vector carrying more than one label: {set: (vectors, rows)}."""
    order = np.lexsort((y, h))
    hs, ys = h[order], y[order]
    first = np.ones(len(hs), dtype=bool)
    first[1:] = hs[1:] != hs[:-1]
    gid = np.cumsum(first) - 1
    rows_per = np.bincount(gid)
    pair_new = first.copy()
    pair_new[1:] |= ys[1:] != ys[:-1]
    n_lab = np.bincount(gid, weights=pair_new)
    multi = np.flatnonzero(n_lab > 1)
    res = Counter()
    rows = Counter()
    if len(multi):
        sel = np.isin(gid, multi) & pair_new
        sets = {}
        for g, v in zip(gid[sel], ys[sel]):
            sets.setdefault(g, []).append(v)
        for g, labs in sets.items():
            key = " / ".join(sorted(map(str, labs)))
            res[key] += 1
            rows[key] += int(rows_per[g])
    return res, rows, int(rows_per[multi].sum()) if len(multi) else 0


def part2(h, can, lab):
    can_codes, can_names = np.unique(can, return_inverse=True)[1], np.unique(can)
    v, r, total = label_sets(h, can_codes)
    sets = [{"labels": " / ".join(can_names[int(i)] for i in k.split(" / ")), "vectors": v[k], "rows": r[k]}
            for k in sorted(v, key=lambda k: -r[k])]
    vb, rb, total_b = label_sets(h, lab)
    return {"canonical_conflict_rows": total, "canonical_sets": sets,
            "binary_conflict_rows": total_b, "binary_conflict_vectors": int(sum(vb.values()))}


def part3():
    ds = "NF-BoT-IoT-v2"
    spec = importlib.util.spec_from_file_location("in_domain", WORK / "scripts/42_in_domain.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    d = m.load(ds)
    cols, feat = d["cols"], d["sets"]["base"]
    fidx = [cols.index(c) for c in feat]
    h = d["hashes"]["base"]
    ddos = d["canon_names"].index("DDoS")
    hd = h[d["canon"] == ddos]
    uh, cnt = np.unique(hd, return_counts=True)
    top = uh[np.argsort(cnt)[::-1][:5]]
    top_rows = np.sort(cnt)[::-1][:5]
    names = {0: "train", 1: "val", 2: "test"}
    rows_out = []
    for v, n in zip(top, top_rows):
        sp = {s: names[int(grouped_split_codes(np.array([v], dtype=np.uint64), s)[0])] for s in SEEDS}
        rows_out.append({"hash": int(v), "ddos_rows": int(n), "split_by_seed": sp})
    # refit G0 BIN, seed 2, as in 42_in_domain.py
    seed = 2
    split = m.grouped_split_codes(h, seed)
    keep = np.ones(len(h), dtype=bool)
    take = m.train_sample(split, d["canon"], keep, np.random.default_rng(seed))
    y = d["label"].astype(np.int64)
    Xtr = d["X"][np.ix_(np.flatnonzero(take), fidx)]
    preds = {}
    for name in ("xgb", "rf"):
        model = m.make_model(name, seed)
        model.fit(Xtr, y[take])
        for r in rows_out:
            idx = np.flatnonzero(h == r["hash"])[:1]
            r.setdefault("pred_seed2", {})[name] = int(model.predict(d["X"][np.ix_(idx, fidx)])[0])
            if name == "rf":
                r["rf_benign_vote_share"] = round(float(model.predict_proba(d["X"][np.ix_(idx, fidx)])[0][0]), 3)
        preds[name] = True
    for r in rows_out:
        idx = np.flatnonzero(h == r["hash"])[:1]
        r["features"] = {c: float(d["X"][idx[0], cols.index(c)]) for c in feat}
        r["benign_rows_with_this_vector"] = int(((h == r["hash"]) & (d["label"] == 0)).sum())
    return {"dataset": ds, "seed_refit": seed, "largest_ddos_vectors": rows_out}


def summary():
    import glob
    from collections import defaultdict
    R = defaultdict(dict)
    for f in glob.glob(str(OUT / "in_domain" / "*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            R[(r["dataset"], r["task"], r["model"], r["condition"])][r["seed"]] = r
    keys = sorted({k[:3] for k in R})
    mf = lambda d, k: [r[k] for r in d.values()]  # noqa: E731
    L = ["P4.2 summary for the Phase 4 gate (from data/eval/in_domain/*.jsonl; 5 seeds per cell)",
         f"Generated: {datetime.now(timezone.utc).isoformat()}", ""]
    L.append("1. Seed range (max - min) of macro-F1, G0 vs R0")
    rg = []
    for ds, t, m in keys:
        g, r0 = mf(R[(ds, t, m, "G0")], "macro_f1"), mf(R[(ds, t, m, "R0")], "macro_f1")
        rg.append((max(g) - min(g), max(r0) - min(r0), ds, t, m, min(g), max(g), min(r0), max(r0)))
        L.append(f"   {ds:22s} {t} {m:3s} G0 {min(g):.3f}-{max(g):.3f} (range {max(g) - min(g):.3f})  "
                 f"R0 {min(r0):.3f}-{max(r0):.3f} (range {max(r0) - min(r0):.3f})")
    L.append(f"   cells with G0 range >= 0.05: {sum(x[0] >= 0.05 for x in rg)} of {len(rg)}; "
             f"R0 range >= 0.05: {sum(x[1] >= 0.05 for x in rg)} of {len(rg)}; "
             f"median range G0 {np.median([x[0] for x in rg]):.3f}, R0 {np.median([x[1] for x in rg]):.3f}")
    L.append("")
    L.append("2. Conflict ceiling of G0 (median over seeds; same for both models)")
    for ds in DATASETS:
        for t in ("BIN", "CAN"):
            c = mf(R[(ds, t, "xgb", "G0")], "conflict_ceiling_macro_f1")
            L.append(f"   {ds:22s} {t}: {np.median(c):.3f} [{min(c):.3f}, {max(c):.3f}]")
    L.append("")
    L.append("3. Absolute timestamps (NF-v3): R0 vs G-TS median macro-F1 (both near-random splits)")
    for ds, t, m in keys:
        if (ds, t, m, "G-TS") in R:
            a, b = np.median(mf(R[(ds, t, m, "R0")], "macro_f1")), np.median(mf(R[(ds, t, m, "G-TS")], "macro_f1"))
            L.append(f"   {ds:22s} {t} {m:3s} R0 {a:.3f} -> G-TS {b:.3f} ({100 * (b - a):+.1f} points)")
    L.append("")
    L.append("4. Paired per-seed difference with G0 (points): G-ND (exact duplicate records removed) and G-NC")
    allnd = []
    for ds, t, m in keys:
        for c in ("G-ND", "G-NC"):
            if (ds, t, m, c) not in R:
                continue
            d = [100 * (R[(ds, t, m, c)][s]["macro_f1"] - R[(ds, t, m, "G0")][s]["macro_f1"]) for s in R[(ds, t, m, c)]]
            if c == "G-ND" and "BoT" not in ds:
                allnd += d
            L.append(f"   {ds:22s} {t} {m:3s} {c}: median {np.median(d):+.2f} [{min(d):+.2f}, {max(d):+.2f}]")
    L.append(f"   G-ND over NF-ToN-IoT-v3 and NF-CSE-CIC-IDS2018-v3 (NF-BoT-IoT-v3 has no such records): median "
             f"{np.median(allnd):+.2f} points, IQR [{np.percentile(allnd, 25):+.2f}, {np.percentile(allnd, 75):+.2f}], n={len(allnd)}")
    L.append("")
    L.append("5. Random (R0) vs grouped (G0) split: median macro-F1 and R0 leakage (CAN, seed 1)")
    pos = 0
    for ds, t, m in keys:
        a, b = np.median(mf(R[(ds, t, m, "G0")], "macro_f1")), np.median(mf(R[(ds, t, m, "R0")], "macro_f1"))
        pos += b > a
        L.append(f"   {ds:22s} {t} {m:3s} G0 {a:.3f}  R0 {b:.3f}  ({100 * (b - a):+.1f})")
    L.append(f"   R0 median above G0 median in {pos} of {len(keys)} cells")
    for ds in DATASETS:
        L.append(f"   {ds:22s} R0 test rows with their vector in train: "
                 f"{R[(ds, 'CAN', 'xgb', 'R0')][1]['test_rows_with_vector_in_train']:.3f}")
    text = "\n".join(L) + "\n"
    (OUT / "in_domain_summary.txt").write_text(text)
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--part", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--summary", action="store_true", help="P4.2 figures for the Phase 4 gate")
    a = ap.parse_args()
    if a.summary:
        summary()
        return
    res = {"generated_utc": datetime.now(timezone.utc).isoformat()}
    L = ["P4.2b inspection of the in-domain results (no experiment re-run except part 3)",
         f"Generated: {res['generated_utc']}", "Base key: all features but L4_SRC_PORT.", ""]
    if 1 in a.part or 2 in a.part:
        res["concentration"], res["conflicts"] = {}, {}
        for ds in DATASETS:
            h, nat, can, lab = hashes_and_labels(ds)
            if 1 in a.part:
                c = part1(ds, h, nat)
                res["concentration"][ds] = c
                w = c["dataset"]
                L.append(f"== Part 1, {ds}: {w['rows']:,} rows, {w['distinct_vectors']:,} distinct vectors; "
                         f"largest 10 vectors hold {w['top10_share']:.1%}, largest 100 {w['top100_share']:.1%}; "
                         f"{w['vectors_for_half']:,} vectors hold half of the rows")
                L.append(f"  {'native class':28s} {'rows':>11s} {'vectors':>10s} {'top1':>6s} {'top10':>6s} "
                         f"{'top100':>6s} {'half':>8s} {'largest':>10s}")
                for k, x in c["classes"].items():
                    L.append(f"  {k:28s} {x['rows']:11,d} {x['distinct_vectors']:10,d} {x['top1_share']:6.1%} "
                             f"{x['top10_share']:6.1%} {x['top100_share']:6.1%} {x['vectors_for_half']:8,d} "
                             f"{x['largest_vector_rows']:10,d}")
            if 2 in a.part:
                c = part2(h, can, lab)
                res["conflicts"][ds] = c
                n = len(h)
                L.append(f"-- Part 2, {ds}: rows in vectors with >1 canonical label {c['canonical_conflict_rows']:,} "
                         f"({c['canonical_conflict_rows'] / n:.1%}); with both binary labels "
                         f"{c['binary_conflict_rows']:,} ({c['binary_conflict_rows'] / n:.1%}) in "
                         f"{c['binary_conflict_vectors']:,} vectors")
                for s in c["canonical_sets"][:8]:
                    L.append(f"  {s['labels']:45s} vectors {s['vectors']:>9,d}  rows {s['rows']:>11,d}")
            L.append("")
            del h, nat, can, lab
    if 3 in a.part:
        r3 = part3()
        res["bot_v2_rf"] = r3
        L.append(f"== Part 3, {r3['dataset']}: largest DDoS vectors (base key), split per seed, "
                 f"G0 BIN predictions after refitting seed {r3['seed_refit']} (1 = attack)")
        for r in r3["largest_ddos_vectors"]:
            L.append(f"  DDoS rows {r['ddos_rows']:>9,d}  Benign rows with the same vector "
                     f"{r['benign_rows_with_this_vector']}  split {r['split_by_seed']}  "
                     f"pred xgb {r['pred_seed2']['xgb']} rf {r['pred_seed2']['rf']} "
                     f"(RF Benign vote share {r['rf_benign_vote_share']})")
        nz = {k: v for k, v in r3["largest_ddos_vectors"][0]["features"].items() if v != 0}
        L.append(f"  non-zero features of the largest DDoS vector: {nz}")
    text = "\n".join(L) + "\n"
    (OUT / "in_domain_inspection.txt").write_text(text)
    (OUT / "in_domain_inspection.json").write_text(json.dumps(res, indent=2, default=str))
    print(text)


if __name__ == "__main__":
    main()
