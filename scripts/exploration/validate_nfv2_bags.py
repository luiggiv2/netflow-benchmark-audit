"""Validate the NF-v2 BagIt bags (P3.2): SHA-1 of payload and tag files against
the bag manifests, schema compared with NF-v3, row and label counts compared
with the totals published by Layeghy and Portmann (2023, Table 1 of the
fact sheet in research/literature/)."""

import hashlib
import sys
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.csv as pv

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
from nfv3_config import NFV3_DTYPES, WORK  # noqa: E402

BAGS = {
    "NF-ToN-IoT-v2": ("9bafce9d380588c2_MOHANAD_A4706", "NF-ToN-IoT-v2.csv", 16_940_496),
    "NF-CSE-CIC-IDS2018-v2": ("b3427ed8ad063a09_MOHANAD_A4706", "NF-CSE-CIC-IDS2018-v2.csv", 18_893_708),
    "NF-BoT-IoT-v2": ("befb58edf3428167_MOHANAD_A4706", "NF-BoT-IoT-v2.csv", 37_763_497),
}


def sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def check_manifest(bag: Path, manifest: str) -> list[str]:
    out = []
    for line in (bag / manifest).read_text().splitlines():
        if not line.strip():
            continue
        expected, rel = line.split(maxsplit=1)
        actual = sha1(bag / rel)
        out.append(f"  {'OK  ' if actual == expected else 'FAIL'} {rel}  {actual}")
    return out


lines = []
for name, (bag_dir, csv_name, published_rows) in BAGS.items():
    bag = WORK / bag_dir
    lines += [f"=== {name} ({bag_dir})", "payload (manifest-sha1.txt):"]
    lines += check_manifest(bag, "manifest-sha1.txt")
    lines += ["tag files (tagmanifest-sha1.txt):"]
    lines += check_manifest(bag, "tagmanifest-sha1.txt")
    info = (bag / "FurtherInformation.txt").read_text().strip()
    lines.append(f"FurtherInformation: {info}")

    csv = bag / "data" / csv_name
    header = csv.open().readline().strip().split(",")
    v3 = list(NFV3_DTYPES)
    lines += [f"columns: {len(header)}",
              f"  in NF-v3 but not in this file: {[c for c in v3 if c not in header]}",
              f"  in this file but not in NF-v3: {[c for c in header if c not in v3]}"]

    t = pv.read_csv(csv, convert_options=pv.ConvertOptions(include_columns=["Label", "Attack"]))
    n = t.num_rows
    lines.append(f"rows: {n:,} (published total {published_rows:,}; "
                 f"{'match' if n == published_rows else 'DIFFERENT'})")
    vc = pc.value_counts(t["Attack"]).to_pylist()
    lines.append("Attack counts:")
    for d in sorted(vc, key=lambda x: -x["counts"]):
        lines.append(f"  {d['values']:28s} {d['counts']:>12,}  {100 * d['counts'] / n:6.2f} %")
    benign_attack = pc.equal(t["Attack"], "Benign")
    label0 = pc.equal(t["Label"], 0)
    agree = pc.sum(pc.equal(benign_attack, label0).cast("int64")).as_py()
    lines.append(f"Label == 0 iff Attack == 'Benign': {agree:,} of {n:,} rows agree")
    lines.append("")

out = WORK / "data/exploration/nfv2_bag_validation.txt"
out.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
