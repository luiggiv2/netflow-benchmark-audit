# Auditing the NetFlow versions of IoT and enterprise intrusion detection benchmarks

Code and result files for the article *Auditing the NetFlow versions of IoT and enterprise intrusion detection
benchmarks: data quality, shortcuts and cross-dataset evaluation* (Luiggi Valencia Vélez; submitted to *Computer
Networks*).

The study audits the NF-v2 and NF-v3 releases of NF-ToN-IoT, NF-BoT-IoT and NF-CSE-CIC-IDS2018 at record level and
measures how their problems affect in-domain and cross-dataset evaluation. This repository holds the code that
produces every result of the article, the result files from which every number in it is taken, and checksums of every
data partition used. The datasets themselves are not redistributed (see [Data](#data)).

## Contents

| Path | Content |
|---|---|
| `scripts/` | Loading, audit, evaluation, explanation and table and figure scripts (see [Scripts](#scripts)) |
| `scripts/exploration/` | Focused checks; each writes `data/exploration/<name>.txt` (and sometimes `.json`) |
| `data/processed/<dataset>/` | Quality-control reports, manifests and audit reports of the loading steps (the Parquet files are not included) |
| `data/audit/` | Record-level audit (RQ1) and shortcut probes (RQ2) |
| `data/eval/` | In-domain, cross-dataset, replication and SHAP runs (RQ3, RQ4), one JSON line per run, with reports; the folders ending in `_xgbdefault` repeat the runs with the library-default XGBoost (sensitivity analysis, Supplementary S6) |
| `data/exploration/` | Outputs of the focused checks |
| `data/paper/` | Values plotted in the figures and tabulated in the tables of the article |
| `data/schemas/` | Class mappings, partition hashes and `MANIFEST.txt` (SHA-1 of every artefact) |
| `manuscript/figures/` | Figures 1 to 10 of the article (PDF and PNG), as produced by scripts `51` to `55` |
| `manuscript/tables/T5_findings.csv` | Source of Table 5 (findings by origin) |

## Data

The six NetFlow files are published by the University of Queensland on UQ eSpace under non-commercial re-use with
citation. Download them from their records:

| Dataset | DOI | SHA-1 of the main CSV |
|---|---|---|
| NF-ToN-IoT-v3 | [10.48610/44d7c5e](https://doi.org/10.48610/44d7c5e) | 24854ca3072ab7c2ade9ebfb202345a666a94cfe |
| NF-BoT-IoT-v3 | [10.48610/73c4ebc](https://doi.org/10.48610/73c4ebc) | 357cc7ffe46ef57ba8193a58d7a66eff5cf44db8 |
| NF-CSE-CIC-IDS2018-v3 | [10.48610/ece9b83](https://doi.org/10.48610/ece9b83) | ddcad6d826b73a128a447a0189137efd6c8f70c9 |
| NF-ToN-IoT-v2 | [10.48610/38a2d07](https://doi.org/10.48610/38a2d07) | c0e014512c9a825146c52982d3cf12d3fce1ccc9 |
| NF-BoT-IoT-v2 | [10.48610/ec73920](https://doi.org/10.48610/ec73920) | cddcc616a032f55a6a5ab3e382093636449653d6 |
| NF-CSE-CIC-IDS2018-v2 | [10.48610/e9636b7](https://doi.org/10.48610/e9636b7) | 80ff7fd07e2bc31164197f07fe9d9606ee565f9a |

Unzip each BagIt download at the repository root. The folder names expected by the code are listed in `RAW_CSV` in
`scripts/nfv3_config.py`; edit them there if your folders are named differently. Section 1 of
`data/schemas/MANIFEST.txt` records the SHA-1 of every file of the six bags.

## Environment

Python 3.12 with the packages pinned in `requirements.txt`:

```
pip install -r requirements.txt
```

The results were produced on a laptop CPU (Apple M3, 16 GB of memory); no GPU is needed. Loading NF-BoT-IoT-v2
(37.8 million rows) needs more memory than that machine has, so close other applications and allow the system to
swap.

## Reproducing the results

Run every script from the repository root; the docstring at the top of each script gives its options and outputs.

1. **Load and verify the files** (for each of the six datasets): `01_load_nfv3.py`, `02_build_tabular_nfv3.py` and
   `17_reproducibility_audit.py`. They write `data/processed/<dataset>/` (Parquet files and reports).
2. **Audit (RQ1) and shortcut probes (RQ2):** `30` to `36`; `34_shortcut_probes.py --common` gives the probes
   reported in the article.
3. **Evaluation (RQ3):** `40_split_verification.py`, `42_in_domain.py` (about 5.5 hours on the machine above),
   `43_inspect_in_domain.py`, `44_cross_dataset.py` (about 7 hours; it builds a cache under `results/`),
   `45_inspect_cross.py` and `46_replication.py`.
4. **Explanations (RQ4):** `47_shap.py`.
5. **Partitions:** `48_split_hashes.py` and `59_split_checksums.py`.
6. **Focused checks:** the scripts in `scripts/exploration/`, after the steps they read from.
7. **Tables and figures:** `50` to `58` write to `manuscript/tables/` and `manuscript/figures/`, and the values they
   plot or tabulate to `data/paper/`.

`data/schemas/MANIFEST.txt` lists the SHA-1 of every script and result file in this repository, so a rerun can be
compared with the files used in the article.

## Scripts

Shared definitions (dataset paths, data types, feature sets, split keys and split functions) are in
`scripts/nfv3_config.py`; the figure style is in `scripts/figstyle.py`.

### Loading and verification

| Script | Role |
|---|---|
| `01_load_nfv3.py` | Reads each published CSV with explicit data types, joins the canonical classes and writes `data/processed/<dataset>/data.parquet` with a quality-control report and a manifest |
| `02_build_tabular_nfv3.py` | Builds `tabular.parquet` (normalised features and labels) with its report and manifest |
| `17_reproducibility_audit.py` | Independent re-read of the raw and derived files: ten checks per dataset, including raw-value fidelity and the absence of any feature vector in more than one partition |
| `18_write_master_manifest.py` | Regenerates `data/schemas/MANIFEST.txt` (SHA-1 of every artefact) |

### Audit of the records (RQ1) and shortcut probes (RQ2)

| Script | Role |
|---|---|
| `30_audit_duplicates_conflicts.py` | Repeated feature vectors, records duplicated in every column and label conflicts per class (Figure 2, Table 5) |
| `31_inspect_conflict_groups.py` | Addresses, ports, flags and times of the largest conflict groups |
| `32_flood_heuristics.py`, `33_flood_comparison.py` | Reimplementation of the heuristics of Flood et al. (2024) and comparison with their published values (Figure 4, Table S2) |
| `34_shortcut_probes.py` | Depth-1 to depth-3 decision trees per class against benign traffic; `--common` gives the reported probes (Figure 5) |
| `35_value_anomalies.py` | Row-level consistency checks (durations, rates, packet sizes; Table S1) |
| `36_inspect_undetermined_cases.py` | Inspection of the CIC Infiltration and web-attack conflicts |

### Evaluation (RQ3) and explanations (RQ4)

| Script | Role |
|---|---|
| `40_split_verification.py` | Verifies the grouped splits on every feature set (no vector in two partitions) |
| `42_in_domain.py`, `43_inspect_in_domain.py` | In-domain runs under every condition and their summary (Figure 6, Table S5) |
| `44_cross_dataset.py`, `45_inspect_cross.py` | Cross-dataset runs, including the repetitions on the split of G0, and their inspection and sign tests (Figures 7 and 8, Table S4) |
| `46_replication.py` | Replication of the binary results of Layeghy and Portmann (2023) (Table 6) |
| `47_shap.py` | SHAP values of the XGBoost models and permutation importance of the random forests (Figures 9 and 10) |
| `48_split_hashes.py`, `59_split_checksums.py` | Partition hashes and the training and test rows of every run configuration |

### Tables and figures

| Script | Output |
|---|---|
| `50_dataset_tables.py` | Tables 2 and 3 |
| `51_figure_pipeline.py` | Figure 1 |
| `52_figures_rq1.py` | Figures 2 to 4 |
| `53_figure_rq2.py` | Figure 5 |
| `54_figures_rq3.py` | Figures 6 to 8, Table 6 and Table S4 (`--tables-only` for the tables) |
| `55_figures_rq4.py` | Figures 9 and 10 |
| `56_table_s5.py`, `57_table5.py`, `58_table_s6.py` | Tables S5, 5 and S6 |

Figures and tables are written to `manuscript/figures/` and `manuscript/tables/`; the values they show are written to
`data/paper/`.

### Focused checks (`exploration/`)

Each check writes `data/exploration/<name>.txt` (and sometimes `.json`) and documents its question in its docstring.

- **Data and loading:** `nfv2_dtype_ranges`, `validate_nfv2_bags`, `review_audit_negative_controls`,
  `dns_ttl_anomaly`.
- **Duplicates, conflicts and value anomalies (RQ1):** `class_vector_counts`, `f2b_window_check`,
  `f2d_v2_label_sets`, `f8_duration_check`, `f9_nonfinite_check`, `n9_minority_composition`.
- **Split keys and leakage:** `src_port_signal`, `split_key_sensitivity`, `five_tuple_split_check`,
  `p14_ip_split_exploration`, `p14_temporal_split_exploration` (Supplementary S3), `r3n15_ts_unique`.
- **Shortcut probes (RQ2):** `timestamp_probe`, `compare_shortcut_probes`, `rq2_threshold_sensitivity`,
  `n2_setc_leak_check`, `r3m1_l7proto_icmp_check`, `r3m1_probes_no_service`, `r3m7_probes_seeds`.
- **Evaluation (RQ3):** `n3_split_control`, `r3m1_nosvc_transfer`, `r3n5_gnc_removed`, `nn_distance_bot`,
  `r3m3_nn_distance_sets`, `n5_threshold_check`, `replication_gaps`, `timestamp_float32_resolution`.
- **Explanations (RQ4):** `m8_reply_check`, `n1_reply_proxy_check`, `r3m2_reply_tcp_check`, `r3m1_shap_service`,
  `rq4_trace_checks`, `rf_shap_cost`.
- **XGBoost setting (Supplementary S6):** `xgb_collapse_diagnosis`, `xgb_default_gap`, `xgb_default_sensitivity`.

Some docstrings refer to the internal notes and review rounds in which a check was requested; those notes are not
part of this archive, and every result the article uses is in `data/`.

## Verifying the partitions

Every partition is a deterministic function of the published CSV and the code, so a reproduction can check its
splits without running any model:

- `data/schemas/split_hashes.csv`: SHA-1 of the train, validation and test code of every row, for each split key and
  seed, the random split and the replication samples.
- `data/paper/split_checksums.json`: for every run configuration (in-domain, cross-dataset, replication and shortcut
  probes), the SHA-256 of the sorted row positions (0-based, in the order of the published CSV) of the training and
  test rows.

## Citation

Please cite this archive through its Zenodo record and the article once it is published:

- all versions: [10.5281/zenodo.23291191](https://doi.org/10.5281/zenodo.23291191);
- version v1.0.0, used in the article: [10.5281/zenodo.23291192](https://doi.org/10.5281/zenodo.23291192).

## Licence

The code is released under the MIT licence (`LICENSE`). The NF datasets are not part of this repository and remain
under the terms of their UQ eSpace records.
