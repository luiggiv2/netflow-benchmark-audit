"""DNS_TTL_ANSWER values above the int32 range in the three NF-v3 datasets,
measured on the current data.parquet files (raw values preserved as int64)."""

from pathlib import Path

import pyarrow.parquet as pq

PROCESSED = Path(__file__).resolve().parents[2] / "data/processed"
INT32_MAX = 2**31 - 1

for ds in ["NF-BoT-IoT-v3", "NF-CSE-CIC-IDS2018-v3", "NF-ToN-IoT-v3"]:
    df = pq.read_table(PROCESSED / ds / "data.parquet",
                       columns=["DNS_TTL_ANSWER", "canonical_label", "split"]).to_pandas()
    big = df[df["DNS_TTL_ANSWER"] > INT32_MAX]
    rest = df.loc[df["DNS_TTL_ANSWER"] <= INT32_MAX, "DNS_TTL_ANSWER"]
    print(f"{ds}: rows above int32 max = {len(big):,}; "
          f"max of remaining rows = {int(rest.max()):,}")
    if len(big):
        print(f"  value range: {int(big['DNS_TTL_ANSWER'].min()):,} .. "
              f"{int(big['DNS_TTL_ANSWER'].max()):,}")
        print("  by canonical class and split:")
        counts = big.groupby(["canonical_label", "split"], observed=True).size()
        for (cls, sp), n in counts.items():
            print(f"    {cls:18s} {sp:5s} {n:>6,}")
