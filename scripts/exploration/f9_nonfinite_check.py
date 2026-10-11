"""f9_nonfinite_check.py — second reviewer report, minor 24 (first report minor
10), 2026-10-08: non-finite values (NaN, inf) in the model matrix of each
dataset (cache of 44_cross_dataset.py, base feature set), so it can be said
whether models trained on one dataset ever saw the missing rates of
NF-BoT-IoT-v3 (F9). Per dataset: rows with any non-finite base feature, the
features involved, and the share per native class where it is not zero.

Usage: python scripts/exploration/f9_nonfinite_check.py
Output: data/exploration/f9_nonfinite_check.txt
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "results/eval/cache"
DATASETS = ["NF-ToN-IoT-v2", "NF-BoT-IoT-v2", "NF-CSE-CIC-IDS2018-v2", "NF-ToN-IoT-v3", "NF-BoT-IoT-v3",
            "NF-CSE-CIC-IDS2018-v3"]
CHUNK = 2_000_000


def main() -> None:
    L = ["Non-finite values in the model matrix (base feature set)", f"Generated: {datetime.now(timezone.utc).isoformat()}", ""]
    for ds in DATASETS:
        meta = json.loads((CACHE / ds / "meta.json").read_text())
        X = np.load(CACHE / ds / "X.npy", mmap_mode="r")
        native = np.load(CACHE / ds / "native.npy")
        idx = [meta["cols"].index(c) for c in meta["sets"]["base"]]
        bad = np.zeros(X.shape[0], dtype=bool)
        per_feat = np.zeros(len(idx), dtype=np.int64)
        for i in range(0, X.shape[0], CHUNK):
            B = ~np.isfinite(np.asarray(X[i:i + CHUNK])[:, idx])
            bad[i:i + CHUNK] = B.any(axis=1)
            per_feat += B.sum(axis=0)
        feats = [f"{meta['sets']['base'][j]} {per_feat[j]:,}" for j in np.flatnonzero(per_feat)]
        L.append(f"  {ds:24s} rows with a non-finite base feature: {int(bad.sum()):,} of {X.shape[0]:,} "
                 f"({100 * bad.mean():.2f} %)" + (f"; features: {', '.join(feats)}" if feats else ""))
        if bad.any():
            names = meta["native_names"]
            for k in np.unique(native[bad]):
                m = native == k
                L.append(f"      {names[k]:24s} {100 * bad[m].mean():6.2f} % of {int(m.sum()):,} rows")
    out = ROOT / "data/exploration/f9_nonfinite_check.txt"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
