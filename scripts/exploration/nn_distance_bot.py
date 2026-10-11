"""nn_distance_bot.py — reviewer point M7, 2026-10-07: a model-independent
check of "different traffic under the same name" (Section 5.3.2). Are the
NF-BoT-IoT DoS and DDoS flows closer to the Benign flows of ToN and CIC than
to their DoS and DDoS flows?

Per release (v2, v3), with the model feature set of the release (base set of
nfv3_config.model_feature_sets, read from the cache of 44_cross_dataset.py):
  - queries: Q random NF-BoT-IoT DoS rows and Q DDoS rows;
  - reference pools: R random rows of each core class (Benign, DoS, DDoS) of
    each dataset of the release; for NF-BoT-IoT the pools exclude the queries
    (within-dataset calibration);
  - transform: sign(x) * log1p(|x|), non-finite -> 0, then each feature
    standardised with the mean and SD of all pooled rows of the release;
  - per query class and dataset: median Euclidean distance to the nearest
    row of each class, and the class of the nearest row over the three equal
    pools (a 1-NN vote with equal class sizes, so it does not depend on the
    class proportions of the datasets).
The source DoS rows are also queried against their own pools (held out) as a
second calibration, and the share of UDP flows per query class is given.

Usage: python scripts/exploration/nn_distance_bot.py
Output: data/exploration/nn_distance_bot.txt
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
_spec = importlib.util.spec_from_file_location("cross", ROOT / "scripts/44_cross_dataset.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

Q, R, SEED = 10_000, 50_000, 2026
CORE = ["Benign", "DoS", "DDoS"]
SHORT = {"ToN": "ToN", "BoT": "BoT", "CSE": "CIC"}


def short(ds: str) -> str:
    return SHORT[ds.split("-")[1]] + "-" + ds[-2:]


def rows_of(d: dict, cls: str) -> np.ndarray:
    return np.flatnonzero(d["canon"] == d["canon_names"].index(cls)) if cls in d["canon_names"] else np.array([], int)


def transform(X: np.ndarray) -> np.ndarray:
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float64)
    return np.sign(X) * np.log1p(np.abs(X))


def main() -> None:
    rng = np.random.default_rng(SEED)
    L = ["Nearest-neighbour check: NF-BoT-IoT DoS/DDoS flows against the core classes of each dataset",
         f"Generated: {datetime.now(timezone.utc).isoformat()}",
         f"Q = {Q:,} queries per class, R = {R:,} reference rows per class (fewer if the class is smaller);",
         "features: model base set of the release, sign*log1p, standardised on the pooled rows; Euclidean distance.",
         "1-NN vote: class of the nearest row over the three equal-size pools of one dataset.", ""]
    for v in ("v2", "v3"):
        names = [f"NF-ToN-IoT-{v}", f"NF-BoT-IoT-{v}", f"NF-CSE-CIC-IDS2018-{v}"]
        data = {n: C.open_cache(n) for n in names}
        feat = data[names[0]]["sets"]["base"]
        bot = names[1]
        queries, pools = {}, {}
        for n, d in data.items():
            fidx = [d["cols"].index(c) for c in feat]
            for cls in CORE:
                idx = rows_of(d, cls)
                if n == bot and cls != "Benign":
                    q = rng.choice(idx, min(Q, len(idx)), replace=False)
                    queries[(n, cls)] = q
                    idx = np.setdiff1d(idx, q)
                elif cls == "DoS":
                    q = rng.choice(idx, min(Q, len(idx)), replace=False)
                    queries[(n, cls)] = q
                    idx = np.setdiff1d(idx, q)
                pools[(n, cls)] = np.sort(rng.choice(idx, min(R, len(idx)), replace=False))
        Xp = {k: transform(np.asarray(data[k[0]]["X"][i][:, [data[k[0]]["cols"].index(c) for c in feat]]))
              for k, i in pools.items()}
        Xq = {k: transform(np.asarray(data[k[0]]["X"][np.sort(i)][:, [data[k[0]]["cols"].index(c) for c in feat]]))
              for k, i in queries.items()}
        allp = np.vstack(list(Xp.values()))
        mu, sd = allp.mean(0), allp.std(0)
        keepf = sd > 0
        z = lambda A: (A[:, keepf] - mu[keepf]) / sd[keepf]
        nn = {k: NearestNeighbors(n_neighbors=1).fit(z(A)) for k, A in Xp.items()}
        proto = feat.index("PROTOCOL")
        L.append(f"== {v}: {int(keepf.sum())} features with non-zero SD; pool sizes "
                 + ", ".join(f"{short(n)} {c} {len(pools[(n, c)]):,}" for n in names for c in CORE))
        for (qn, qc), qi in queries.items():
            raw = np.asarray(data[qn]["X"][np.sort(qi)][:, data[qn]["cols"].index("PROTOCOL")])
            L.append(f"  queries {short(qn)} {qc} ({len(qi):,}; UDP {100 * np.mean(raw == 17):.1f} %)")
            zq = z(Xq[(qn, qc)])
            for n in names:
                dist = np.column_stack([nn[(n, c)].kneighbors(zq)[0][:, 0] for c in CORE])
                vote = dist.argmin(1)
                L.append(f"    vs {short(n):7s} median NN distance " + ", ".join(
                    f"{c} {np.median(dist[:, j]):6.3f}" for j, c in enumerate(CORE))
                    + "  | 1-NN vote " + ", ".join(f"{c} {100 * np.mean(vote == j):5.1f} %"
                                                   for j, c in enumerate(CORE)))
        L.append("")
        del data
    out = ROOT / "data/exploration/nn_distance_bot.txt"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
