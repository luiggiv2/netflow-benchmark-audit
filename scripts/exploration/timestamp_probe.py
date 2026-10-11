"""timestamp_probe.py — quick check before Phase 4 (plan section 0, P4.1 item
8): does the absolute flow start time alone separate each NF-v3 attack class
from benign traffic?

Luay et al. (2026) use FLOW_START_MILLISECONDS and FLOW_END_MILLISECONDS as
model features and find them the most influential, while labels were assigned
by matching flows to the parent ground truth by 5-tuple and time overlap. If
the start time alone separates a class, the timestamp encodes the attack
schedule of the testbed rather than traffic behaviour.

Per NF-v3 dataset and native attack class: decision trees of depth 1 and 3 on
FLOW_START_MILLISECONDS only, class vs benign, balanced samples (at most
TRAIN_CAP rows per label in train and TEST_CAP in test), trained on the stored
grouped train split and scored (macro F1) on the grouped test split, as in
scripts/34_shortcut_probes.py. For reference, the depth-3 macro F1 of set A
(41 features, no timestamps) is read from data/audit/shortcut_probes/.

Common-split variant (--common; reviewer point M2, 2026-10-06): the split is
the grouped split keyed on the 40 features without L4_SRC_PORT, seed 42, as
in `34_shortcut_probes.py --common` (same hash-based assignment), and the
reference is set A of data/audit/shortcut_probes_common/.

Usage: python scripts/exploration/timestamp_probe.py [--common]
Output: data/exploration/timestamp_probe[_common].txt
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nfv3_config import (DATASETS, FEATURE_COLS_V2, PROCESSED_ROOT, WORK, grouped_split_codes,  # noqa: E402
                         is_v2, tabular_row_hash)

COMMON_COLS = [c for c in FEATURE_COLS_V2 if c != "L4_SRC_PORT"]
COMMON_SEED = 42  # REKEY_SEED of 34_shortcut_probes.py

SEED = 2026
TRAIN_CAP, TEST_CAP = 200_000, 100_000
BENIGN = "Benign"
TS = "FLOW_START_MILLISECONDS"


def balanced(d: pd.DataFrame, cls: str, cap: int, rng) -> pd.DataFrame:
    c, b = d[d["c"] == cls], d[d["c"] == BENIGN]
    n = min(len(c), len(b), cap)
    return pd.concat([b.iloc[rng.choice(len(b), n, replace=False)],
                      c.iloc[rng.choice(len(c), n, replace=False)]])


def common_split(ds: str) -> np.ndarray:
    """Split names of the common grouped split (40-feature key, seed 42)."""
    names = np.array(["train", "val", "test"])
    parts = [names[grouped_split_codes(tabular_row_hash(b.to_pandas(), COMMON_COLS), COMMON_SEED)]
             for b in pq.ParquetFile(PROCESSED_ROOT / ds / "tabular.parquet").iter_batches(
                 batch_size=1_000_000, columns=COMMON_COLS)]
    return np.concatenate(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--common", action="store_true")
    common = ap.parse_args().common
    ref_dir = "shortcut_probes_common" if common else "shortcut_probes"
    L = ["FLOW_START_MILLISECONDS alone: attack class vs benign, macro F1 on the grouped test split"
         + (" (common split: 40-feature key, seed 42)" if common else ""),
         f"Generated: {datetime.now(timezone.utc).isoformat()}",
         "Balanced samples; reference = depth-3 macro F1 of set A of script 34, no timestamps"
         + (" (40 features, common split)." if common else " (41 features)."), ""]
    allf = []
    for ds in [d for d in DATASETS if not is_v2(d)]:
        t = pq.read_table(PROCESSED_ROOT / ds / "data.parquet", columns=[TS, "native_label", "split"])
        df = pd.DataFrame({"t": t[TS].to_numpy(), "c": t["native_label"].to_pandas().astype(str).to_numpy(),
                           "s": common_split(ds) if common else t["split"].to_pandas().astype(str).to_numpy()})
        ref = json.loads((WORK / "data/audit" / ref_dir / f"{ds}.json").read_text())["classes"]
        tr_all, te_all = df[df["s"] == "train"], df[df["s"] == "test"]
        L.append(f"== {ds}")
        L.append(f"  {'class':28s} {'test rows':>9s} {'ts d1':>6s} {'ts d3':>6s} {'A d3':>7s}")
        for cls in sorted(set(df["c"]) - {BENIGN}):
            rng = np.random.default_rng(SEED)
            tr, te = balanced(tr_all, cls, TRAIN_CAP, rng), balanced(te_all, cls, TEST_CAP, rng)
            if (te["c"] == cls).sum() < 10 or (tr["c"] == cls).sum() < 10:
                L.append(f"  {cls:28s} too few rows")
                continue
            ytr, yte = (tr["c"] == cls).to_numpy(int), (te["c"] == cls).to_numpy(int)
            f = []
            for depth in (1, 3):
                m = DecisionTreeClassifier(max_depth=depth, random_state=SEED).fit(tr[["t"]].to_numpy(), ytr)
                f.append(f1_score(yte, m.predict(te[["t"]].to_numpy()), average="macro"))
            r = ref.get(cls, {}).get("sets", {}).get("A", {}).get("f1_d3")
            allf.append(f[1])
            L.append(f"  {cls:28s} {int(yte.sum()):>9,} {f[0]:6.3f} {f[1]:6.3f} "
                     + (f"{r:7.3f}" if r is not None else f"{'-':>7s}"))
        L.append("")
    a = np.array(allf)
    L.append(f"NF-v3 classes probed: {len(a)}; timestamp depth-3 macro F1 >= 0.95: {(a >= 0.95).sum()}; "
             f">= 0.90: {(a >= 0.90).sum()}; median {np.median(a):.3f}")
    out = WORK / f"data/exploration/timestamp_probe{'_common' if common else ''}.txt"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
