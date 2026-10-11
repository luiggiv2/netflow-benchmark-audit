"""Check that every numeric column of the NF-v2 CSVs fits its loader dtype
(`nfv3_config.raw_dtypes`). pandas 3.x wraps integer overflow silently and
float32 turns values above 3.4e38 into Inf, so the ranges are measured at full
width (int64 / float64) first."""

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pv

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
from nfv3_config import RAW_CSV, WORK, raw_dtypes  # noqa: E402

V2 = ["NF-ToN-IoT-v2", "NF-BoT-IoT-v2", "NF-CSE-CIC-IDS2018-v2"]

lines = []
for name in V2:
    path = RAW_CSV[name]
    header = path.open().readline().strip().split(",")
    dt = raw_dtypes(name)
    cols = [c for c in header if dt[c].startswith(("int", "float"))]
    types = {c: (pa.int64() if dt[c].startswith("int") else pa.float64()) for c in cols}
    reader = pv.open_csv(path, convert_options=pv.ConvertOptions(column_types=types,
                                                                 include_columns=cols))
    lo = {c: None for c in cols}
    hi = {c: None for c in cols}
    for batch in reader:
        for c in cols:
            mm = pc.min_max(batch.column(c))
            a, b = mm["min"].as_py(), mm["max"].as_py()
            if a is not None:
                lo[c] = a if lo[c] is None else min(lo[c], a)
                hi[c] = b if hi[c] is None else max(hi[c], b)
    lines.append(f"=== {name}: {len(cols)} numeric columns")
    bad = 0
    for c in cols:
        t = dt[c]
        info = np.iinfo(t) if t.startswith("int") else np.finfo(t)
        finite = [v for v in (lo[c], hi[c]) if v is not None and np.isfinite(v)]
        ok = all(info.min <= v <= info.max for v in finite)
        bad += not ok
        lines.append(f"  {'OK  ' if ok else 'FAIL'} {c:30s} {t:8s} min {lo[c]}  max {hi[c]}")
    lines.append(f"  columns outside their dtype: {bad}")
    lines.append("")

out = WORK / "data/exploration/nfv2_dtype_ranges.txt"
out.write_text("\n".join(lines) + "\n")
print("\n".join(l for l in lines if "FAIL" in l or "===" in l or "outside" in l))
