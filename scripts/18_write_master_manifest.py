"""18_write_master_manifest.py — consolidate every artefact hash produced by
Phase 1 into a single, human-readable manifest.

The output is written to `data/schemas/MANIFEST.txt`, extending the raw-CSV
section that was seeded in commit 00d73ea (P1.1) with the hashes of every
downstream artefact produced by the loader (P1.5), the tabular builder
(P1.6) and the reproducibility audit (P1.7). It also records the SHA-1 of the two
schema files (`class_mapping.csv` and `NetFlow_v3_Features.csv`) so anyone
reading the manifest can pin the join key used at ingestion time.

The manifest is regenerated from scratch every time the script runs; hashes
come from either the file itself or from the per-dataset JSON manifests
produced by the earlier steps. The script does not fabricate values.

Usage:
  python scripts/18_write_master_manifest.py

The script fails loudly if any expected artefact is missing.

Sections 7-9 (added 2026-10-05, P4.6; inventory in research/artifact_inventory.md):
the public code, the result files of Phases 3-4 and the partition hashes.
PUBLIC_SCRIPTS and EXCLUDED_SCRIPTS must cover every tracked script; the
script stops if one is unclassified.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
MANIFEST_PATH = WORK / "data/schemas/MANIFEST.txt"
CLASS_MAPPING = WORK / "data/schemas/class_mapping.csv"
CLASS_MAPPING_V2 = WORK / "data/schemas/class_mapping_v2.csv"

DATASETS = [
    "NF-ToN-IoT-v3",
    "NF-BoT-IoT-v3",
    "NF-CSE-CIC-IDS2018-v3",
    "NF-ToN-IoT-v2",
    "NF-BoT-IoT-v2",
    "NF-CSE-CIC-IDS2018-v2",
]

BAG_ROOTS = {
    "NF-ToN-IoT-v3": WORK / "02934b58528a226b_NFV3DATA-A11964_A11964",
    "NF-BoT-IoT-v3": WORK / "d509c9db7490cf92_NFV3DATA-A11964_A11964",
    "NF-CSE-CIC-IDS2018-v3": WORK / "f78acbaa2afe1595_NFV3DATA-A11964_A11964",
    "NF-ToN-IoT-v2": WORK / "9bafce9d380588c2_MOHANAD_A4706",
    "NF-BoT-IoT-v2": WORK / "befb58edf3428167_MOHANAD_A4706",
    "NF-CSE-CIC-IDS2018-v2": WORK / "b3427ed8ad063a09_MOHANAD_A4706",
}

RAW_CSV = {
    "NF-ToN-IoT-v3": "data/NF-ToN-IoT-v3.csv",
    "NF-BoT-IoT-v3": "data/NF-BoT-IoT-v3.csv",
    "NF-CSE-CIC-IDS2018-v3": "data/NF-CICIDS2018-v3.csv",
    "NF-ToN-IoT-v2": "data/NF-ToN-IoT-v2.csv",
    "NF-BoT-IoT-v2": "data/NF-BoT-IoT-v2.csv",
    "NF-CSE-CIC-IDS2018-v2": "data/NF-CSE-CIC-IDS2018-v2.csv",
}


# Inventory (P4.6, author decisions of 2026-10-05). Groups: A data pipeline,
# B audit (RQ1-RQ2), C Phase 4 (RQ3-RQ4), P manuscript tables and figures (Phase 5); excluded: D support without claims,
# E superseded, F smoke outputs.
PUBLIC_SCRIPTS = {
    "A": ["nfv3_config.py", "01_load_nfv3.py", "02_build_tabular_nfv3.py", "17_reproducibility_audit.py",
          "18_write_master_manifest.py"],
    "B": ["30_audit_duplicates_conflicts.py", "31_inspect_conflict_groups.py", "32_flood_heuristics.py",
          "33_flood_comparison.py", "34_shortcut_probes.py", "35_value_anomalies.py",
          "36_inspect_undetermined_cases.py",
          "exploration/f8_duration_check.py", "exploration/split_key_sensitivity.py",
          "exploration/timestamp_probe.py", "exploration/rq2_threshold_sensitivity.py",
          "exploration/class_vector_counts.py", "exploration/dns_ttl_anomaly.py",
          "exploration/nfv2_dtype_ranges.py", "exploration/src_port_signal.py",
          "exploration/p14_ip_split_exploration.py", "exploration/p14_temporal_split_exploration.py",
          "exploration/validate_nfv2_bags.py", "exploration/review_audit_negative_controls.py",
          "exploration/compare_shortcut_probes.py",
          # added 2026-10-08 (reviewer responses P5.10 and P5.11)
          "exploration/f2b_window_check.py", "exploration/f2d_v2_label_sets.py", "exploration/f9_nonfinite_check.py",
          "exploration/five_tuple_split_check.py", "exploration/n2_setc_leak_check.py",
          "exploration/n9_minority_composition.py",
          # added 2026-10-10 (third reviewer report, P6.1b)
          "exploration/r3m1_l7proto_icmp_check.py", "exploration/r3m1_probes_no_service.py",
          "exploration/r3m7_probes_seeds.py"],
    "C": ["40_split_verification.py", "42_in_domain.py", "43_inspect_in_domain.py", "44_cross_dataset.py",
          "45_inspect_cross.py", "46_replication.py", "47_shap.py", "48_split_hashes.py",
          # added 2026-10-08 (reviewer responses P5.10 and P5.11)
          "exploration/m8_reply_check.py", "exploration/n1_reply_proxy_check.py", "exploration/n3_split_control.py",
          "exploration/n5_threshold_check.py", "exploration/nn_distance_bot.py", "exploration/replication_gaps.py",
          "exploration/rf_shap_cost.py", "exploration/rq4_trace_checks.py",
          "exploration/timestamp_float32_resolution.py", "exploration/xgb_collapse_diagnosis.py",
          "exploration/xgb_default_gap.py", "exploration/xgb_default_sensitivity.py",
          # added 2026-10-10 (third reviewer report, P6.1b)
          "exploration/r3m1_shap_service.py", "exploration/r3m1_nosvc_transfer.py",
          "exploration/r3m2_reply_tcp_check.py", "exploration/r3m3_nn_distance_sets.py",
          "exploration/r3n5_gnc_removed.py", "exploration/r3n15_ts_unique.py", "59_split_checksums.py"],
    "P": ["figstyle.py", "50_dataset_tables.py", "51_figure_pipeline.py", "52_figures_rq1.py", "53_figure_rq2.py", "54_figures_rq3.py", "55_figures_rq4.py",
          "56_table_s5.py", "57_table5.py", "58_table_s6.py"],
}
EXCLUDED_SCRIPTS = {
    "20_decision_test.py": "E: Phase 2 decision test, conclusion withdrawn (audit C1), redone in P4.2",
    "41_timing_smoke.py": "D: run-time measurement for planning only",
    "exploration/verify_references.py": "D: DOI check for the internal audit",
}
RESULT_DIRS = ["data/audit", "data/exploration", "data/eval", "data/paper"]
EXCLUDED_RESULTS = {
    "data/exploration/reference_check.txt": "D: output of verify_references.py",
    "data/eval/timing_smoke.json": "D: output of 41_timing_smoke.py",
    "data/eval/timing_smoke.txt": "D: output of 41_timing_smoke.py",
}
EXCLUDED_RESULT_MARK = "smoke"  # F: one-seed smoke tests of P4.2-P4.5 (data/decision_test is E and lies outside RESULT_DIRS)


def tracked(prefix: str) -> list[str]:
    out = subprocess.run(["git", "ls-files", prefix], cwd=WORK, capture_output=True, text=True, check=True)
    return sorted(x for x in out.stdout.splitlines() if x)


def code_and_result_lines() -> list[str]:
    lines: list[str] = []
    scripts = [x.removeprefix("scripts/") for x in tracked("scripts") if x.endswith(".py")]
    public = {s for g in PUBLIC_SCRIPTS.values() for s in g}
    unclassified = sorted(set(scripts) - public - set(EXCLUDED_SCRIPTS))
    if unclassified:
        raise RuntimeError(f"scripts not in the inventory: {unclassified}")
    lines.append("# ============================================================")
    lines.append("# 7. Public code (groups A data pipeline, B audit, C Phase 4)")
    lines.append("# ============================================================")
    for g, names in PUBLIC_SCRIPTS.items():
        lines.append(f"# --- group {g} ---")
        for n in names:
            path = require(WORK / "scripts" / n)
            lines.append(f"{sha1_file(path)}  {rel(path)}")
    lines.append("# Excluded from the public archive (kept in the private history):")
    for n, why in EXCLUDED_SCRIPTS.items():
        lines.append(f"#   scripts/{n}: {why}")
    lines.append("")
    lines.append("# ============================================================")
    lines.append("# 8. Result files of Phases 3-4 (data/audit, data/exploration, data/eval)")
    lines.append("# ============================================================")
    excluded = []
    for d in RESULT_DIRS:
        lines.append(f"# --- {d} ---")
        for f in tracked(d):
            if f in EXCLUDED_RESULTS or EXCLUDED_RESULT_MARK in f:
                excluded.append(f)
                continue
            lines.append(f"{sha1_file(WORK / f)}  {f}")
    lines.append(f"# Excluded: {len(excluded)} files (smoke tests; outputs of excluded scripts):")
    for f in excluded:
        lines.append(f"#   {f}")
    lines.append("")
    lines.append("# ============================================================")
    lines.append("# 9. Partition hashes (scripts/48_split_hashes.py)")
    lines.append("# ============================================================")
    for n in ("data/schemas/split_hashes.csv", "data/schemas/split_hashes.txt"):
        lines.append(f"{sha1_file(require(WORK / n))}  {n}")
    lines.append("# Training and test rows of every run configuration (scripts/59_split_checksums.py):")
    lines.append("#   data/paper/split_checksums.json and .txt, hashed in section 8")
    lines.append("")
    return lines


def features_dict(dataset: str) -> str:
    return "data/NetFlow_v2_Features.csv" if dataset.endswith("-v2") else "data/NetFlow_v3_Features.csv"


def sha1_file(path: Path, block: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(block)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def rel(p: Path) -> str:
    return str(p.relative_to(WORK))


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"required artefact missing: {rel(path)}")
    return path


def load_manifest(path: Path) -> dict:
    return json.loads(require(path).read_text())


def bag_lines(dataset: str) -> list[str]:
    """Rehash the files inside the BagIt bag that we recorded in P1.1.

    Kept verbatim across manifest regenerations so the raw-data section
    of the manifest stays authoritative.
    """
    bag = BAG_ROOTS[dataset]
    raw = bag / RAW_CSV[dataset]
    lines: list[str] = []
    lines.append(f"# --- {dataset} ---")
    for candidate in [
        raw,
        bag / features_dict(dataset),
        bag / "manifest-sha1.txt",
        bag / "bag-info.txt",
        bag / "bagit.txt",
    ]:
        require(candidate)
        lines.append(f"{sha1_file(candidate)}  {rel(candidate)}")
    return lines


def processed_lines(dataset: str) -> list[str]:
    """Fetch hashes from the loader, tabular-builder and audit manifests.

    The values are cross-checked against a re-hash of the file on disk to
    catch tampering or accidental regeneration mismatches.
    """
    lines: list[str] = []
    d = WORK / "data/processed" / dataset
    loader_m = load_manifest(d / "manifest.json")
    tab_m = load_manifest(d / "tabular_manifest.json")
    audit_m = load_manifest(d / "audit_manifest.json")

    data_path = WORK / loader_m["outputs"]["parquet"]["path"]
    tab_path = WORK / tab_m["outputs"]["tabular"]["path"]
    qc_paths = [
        d / "qc_report.txt",
        d / "tabular_qc.txt",
        d / "audit_report.txt",
    ]
    manifest_paths = [
        d / "manifest.json",
        d / "tabular_manifest.json",
        d / "audit_manifest.json",
    ]

    def check(path: Path, recorded: str, label: str) -> str:
        actual = sha1_file(path)
        if actual != recorded:
            raise RuntimeError(
                f"{label}: SHA-1 mismatch. recorded={recorded} actual={actual}"
            )
        return actual

    data_sha1 = check(data_path, loader_m["outputs"]["parquet"]["sha1"],
                      "data.parquet")
    tab_sha1 = check(tab_path, tab_m["outputs"]["tabular"]["sha1"],
                     "tabular.parquet")

    lines.append(f"# --- {dataset} ---")
    lines.append(f"{data_sha1}  {rel(data_path)}")
    lines.append(f"{tab_sha1}  {rel(tab_path)}")
    for p in qc_paths:
        require(p)
        lines.append(f"{sha1_file(p)}  {rel(p)}")
    for p in manifest_paths:
        require(p)
        lines.append(f"{sha1_file(p)}  {rel(p)}")

    # Cross-file metadata SHA-1 (batch-boundary invariant hash produced by
    # the audit). Recording it here documents that the four metadata
    # columns are byte-identical in data.parquet and tabular.parquet on the
    # day this manifest was written.
    md = audit_m["measurements"]["metadata_sha1"]
    if md["data.parquet"] != md["tabular.parquet"]:
        raise RuntimeError(
            f"metadata SHA-1 mismatch between data and tabular for {dataset}: {md}"
        )
    lines.append(f"{md['data.parquet']}  {dataset} / cross-file metadata SHA-1")

    # Sanity: audit passed
    if audit_m["failed"] != 0:
        raise RuntimeError(
            f"audit for {dataset} reports {audit_m['failed']} failures; "
            f"refusing to write manifest"
        )
    return lines


def main() -> None:
    now = datetime.now(timezone.utc).isoformat()
    lines: list[str] = []
    lines.append("# MANIFEST — SHA-1 checksums for every artefact of Paper B "
                 "(Phases 1-4)")
    lines.append("#")
    lines.append(f"# Regenerated: {now}")
    lines.append("# Producer:    scripts/18_write_master_manifest.py")
    lines.append("#")
    lines.append("# Format: <sha1>  <path-relative-to-workspace-root>")
    lines.append("# Lines starting with '#' are comments.")
    lines.append("#")
    lines.append("# Sections:")
    lines.append("#   1. Raw datasets (BagIt bags from UQ eSpace)")
    lines.append("#   2. Schemas (join keys and feature dictionaries)")
    lines.append("#   3. Loader outputs (data.parquet + QC + manifest per dataset)")
    lines.append("#   4. Tabular-builder outputs (tabular.parquet + QC + manifest)")
    lines.append("#   5. Audit outputs (audit_report.txt, audit_manifest.json)")
    lines.append("#   6. Cross-file metadata SHA-1 (batch-boundary invariant)")
    lines.append("#   7. Public code")
    lines.append("#   8. Result files of Phases 3-4")
    lines.append("#   9. Partition hashes")
    lines.append("")

    # ------------------------------------------------------------------ #
    # 1. Raw datasets — bag files (verified against the original bag
    #    manifests during P1.1; re-hashed here to lock the current values).
    # ------------------------------------------------------------------ #
    lines.append("# ============================================================")
    lines.append("# 1. Raw datasets (UQ eSpace bags; do not modify)")
    lines.append("# ============================================================")
    for ds in DATASETS:
        lines.extend(bag_lines(ds))
        lines.append("")

    # ------------------------------------------------------------------ #
    # 2. Schemas
    # ------------------------------------------------------------------ #
    lines.append("# ============================================================")
    lines.append("# 2. Schemas (join keys and feature dictionaries)")
    lines.append("# ============================================================")
    for mp in (CLASS_MAPPING, CLASS_MAPPING_V2):
        require(mp)
        lines.append(f"{sha1_file(mp)}  {rel(mp)}")
    # The split key, dtypes and feature list live in this module.
    config = require(WORK / "scripts/nfv3_config.py")
    lines.append(f"{sha1_file(config)}  {rel(config)}")
    lock = WORK / "requirements-lock.txt"
    if lock.exists():
        lines.append(f"{sha1_file(lock)}  {rel(lock)}")
    # Each feature dictionary is byte-identical across the bags of its
    # version and is already listed in section 1; the references here are
    # informational.
    for ds in ("NF-ToN-IoT-v3", "NF-ToN-IoT-v2"):
        ref_feat = BAG_ROOTS[ds] / features_dict(ds)
        lines.append(f"{sha1_file(ref_feat)}  {rel(ref_feat)}  (same file in every bag of this version)")
    lines.append("")

    # ------------------------------------------------------------------ #
    # 3. Loader outputs — data.parquet (P1.5)
    # 4. Tabular-builder outputs — tabular.parquet (P1.6)
    # 5. Audit outputs (P1.7)
    # 6. Cross-file metadata SHA-1
    # (Emitted together per dataset so all lines about the same dataset stay
    # visually grouped.)
    # ------------------------------------------------------------------ #
    lines.append("# ============================================================")
    lines.append("# 3-6. Processed artefacts per dataset")
    lines.append("#      loader (P1.5) + tabular builder (P1.6) + audit (P1.7)")
    lines.append("#      including cross-file metadata SHA-1")
    lines.append("# ============================================================")
    for ds in DATASETS:
        lines.extend(processed_lines(ds))
        lines.append("")

    lines.extend(code_and_result_lines())

    # ------------------------------------------------------------------ #
    # Footer
    # ------------------------------------------------------------------ #
    lines.append("# End of manifest.")
    lines.append(
        "# To verify a single file: "
        "shasum -a 1 <path> | awk '{print $1}'  == <sha1>."
    )
    lines.append(
        "# To rebuild this manifest: python scripts/18_write_master_manifest.py"
    )

    MANIFEST_PATH.write_text("\n".join(lines) + "\n")
    n_hashes = sum(
        1 for ln in lines if ln and not ln.startswith("#") and "  " in ln
    )
    print(f"Wrote {rel(MANIFEST_PATH)} with {n_hashes} hash entries.", file=sys.stderr)


if __name__ == "__main__":
    main()
