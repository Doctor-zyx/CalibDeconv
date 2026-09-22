# CalibDeconv

**Conformal calibration of prediction intervals for PBMC-domain deconvolution from bulk transcriptomes.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

CalibDeconv is a post-hoc reliability calibration layer for cellular deconvolution of PBMC bulk transcriptomes, not a new deconvolution model. It combines:

- **Frozen marker-based NNLS** for cell-type proportion estimation (5 types: T cell, B cell, NK cell, Monocyte, DC)
- **Bootstrap ensemble** (50 iterations) for uncertainty quantification
- **Split conformal calibration** for marginal prediction intervals with finite-sample coverage guarantees

The framework is designed to add calibrated reliability assessment to PBMC deconvolution, not to improve point-estimate accuracy over existing methods.

## Key Results

| Setting | Metric | Value |
|---------|--------|-------|
| Primary PBMC pseudo-bulk | MAE / CCC | 0.081 / 0.848 |
| Raw ensemble coverage (nominal 90%) | Empirical coverage | 0.390 |
| Conformal calibrated (nominal 90%) | Empirical coverage | 0.950 |
| External PBMC 3k (no retraining) | MAE / CCC / Coverage | 0.058 / 0.870 / 0.892 |
| Composition-level simultaneous coverage (nominal 90%) | All 5 types covered | 0.780 (390/500) |
| Bonferroni-adjusted family-wise extension | All 5 types covered | 0.946 (473/500) |

## Installation

```bash
git clone https://github.com/Doctor-zyx/CalibDeconv.git
cd CalibDeconv
pip install -r requirements.txt
```

### Dependencies

- Python 3.10+
- numpy, pandas, scipy, scikit-learn
- scanpy, anndata
- matplotlib, seaborn

### Environment files

Three distinct things, deliberately kept separate:

| | Purpose |
|---|---|
| `requirements.txt` | **Installation.** Broad lower bounds, no upper pins. Use this. |
| `requirements-tested.txt` | **Validated environments.** Exact versions of two independent environments — one on pandas 2.x, one on pandas 3.x — in which the analyses were re-run from a clean checkout. |
| Software versions in the manuscript | **The original analysis environment.** Reported in the manuscript Methods and unchanged. |

Neither validated environment is the one in which the original results were
produced, and neither supersedes the software versions reported in the
manuscript.

As of v1.2.2 the validated scope covers the **stress scripts as well as the
coverage scripts** — `06b_stress_marker5.py` (Figure 4),
`06d_stress_tier2_subset.py` (Figure 5), `15_simultaneous_coverage.py`,
`16_gse107572_real_bulk.py`, `10_publication_figures.py` and
`tests/test_stress_dataframe_compat.py`. In v1.2 only the coverage scripts had
been exercised, which is why the pandas 3 incompatibility fixed in v1.2.2 went
unnoticed.

### Testing

```bash
python tests/test_stress_dataframe_compat.py
```

A dependency-free regression test for the four perturbation functions in
`src/evaluation/stress.py`: it checks that they run without writing through a
read-only `DataFrame.values`, do not mutate the caller's frame, preserve index,
columns, shape and dtype, are deterministic for a fixed seed, and still have
their intended effect. 6/6 pass under both validated environments.

## Project Structure

```
CalibDeconv/
├── src/                    # Core library
│   ├── data/               # Data loading & pseudo-bulk generation
│   ├── deconvolution/      # Signature matrix & NNLS solver
│   ├── uncertainty/        # Ensemble & conformal calibration
│   ├── evaluation/         # Metrics, stress testing
│   └── utils/              # Configuration, I/O
├── scripts/                # Analysis pipeline (numbered sequentially)
├── config/                 # Configuration files
├── results/                # Analysis outputs
├── submission_package/     # Manuscript figures
└── requirements.txt
```

## Pipeline

The analysis pipeline runs sequentially:

```bash
# 1. Data preparation
python scripts/01_download_data.py
python scripts/02_generate_pseudobulk.py

# 2. Deconvolution
python scripts/03_nnls_baseline.py

# 3. Uncertainty estimation
python scripts/04_ensemble_uncertainty.py

# 4. Conformal calibration
python scripts/05_conformal_calibration.py

# 5. Stress testing & validation
python scripts/06b_stress_marker5.py
python scripts/06d_stress_tier2_subset.py

# 6. External validation & out-of-domain analysis
python scripts/11_phase6b_gse107572_pilot.py
python scripts/13_phase7d_neutrophil_ood.py

# 7. Publication figures
python scripts/10_publication_figures.py
python scripts/20_redraw_all_supp_figs.py
```

## Publication figures

`scripts/10_publication_figures.py` is the canonical generator for main
Figures 1–5. There is no second plotting pipeline.

Canonical inputs for the two stress-related figures:

| Figure | Input files |
|---|---|
| Figure 4 (stress tests) | `results/stress_marker_5types/stress_summary_tier1.csv`<br>`results/stress_marker_5types/rejection_curves_tier1.csv` |
| Figure 5 (reference ablation) | `results/stress_marker_5types/stress_summary_tier1.csv`<br>`results/stress_marker_5types/rejection_curves_tier1.csv`<br>`results/stress_marker_5types_tier2_subset/stress_summary_tier2_subset.csv` |

Do **not** use the `*_corrected.csv` files in `results/legacy_pre_v1.1/`. They
are June 2026 snapshots that were not regenerated when Tier 1 was re-run in
v1.1, and despite the name they never corrected any value — the script that
produced them only appended diagnostic columns. No current script reads them.

## Reproducibility limitations

**Tier 1 stress and reliability diagnostics are not bit-reproducible.** Until
v1.2.1, `scripts/06b_stress_marker5.py`, `06c_reliability_diagnostics.py`,
`06d_stress_tier2_subset.py` and `06_stress_test.py` derived their per-scenario
seeds from `abs(hash(scenario_name))`. Python salts the hash of a `str`
per process unless `PYTHONHASHSEED` is set, which this project never set, so
every run drew different seeds. v1.2.1 replaces these with explicit numeric
constants (`SCENARIO_SEEDS`), but the seeds that produced the archived Tier 1
CSVs cannot be recovered.

Consequences:

- The archived files in `results/stress_marker_5types/` remain the frozen
  record behind the manuscript and Figures 4–5. They are **not** overwritten.
- Re-running `06b`/`06c` reproduces the qualitative pattern — dropout is the
  dominant degradation axis, coverage falls below nominal under moderate and
  severe dropout, library-depth reduction has little effect — but not the
  archived numbers to the last decimal.
- The reference-ablation scenario had a second, larger problem: the ablated
  cell type was drawn with `rng.choice()` from the same unstable seed, so DC
  was ablated in only about 21% of runs. v1.2.1 fixes the ablated type to DC
  (`ABLATED_CELL_TYPE`) and pins the scenario seed, which reproduces the
  archived MAE and CCC to within bootstrap-ensemble noise.
- `coverage90_clip` for the reference-ablation scenario cannot match the
  archive even with the original seed: v1.1 replaced the interpolating
  `np.quantile` conformal quantile with an exact order statistic. Point-estimate
  metrics (MAE, CCC) and the uncertainty diagnostics are unaffected by that
  change.
- **The two `low_depth_*` Tier 1 scenarios are numpy-version sensitive.**
  `apply_low_depth` draws from `numpy.random.Generator.binomial`, whose stream
  differs between numpy 2.4.6 and 2.5.3 for some `(n, p)`. Re-running under the
  two validated environments changes those two scenarios by at most 3.6e-04;
  the other seven Tier 1 scenarios and all of Tier 2 are bit-identical. Pin
  numpy if you need `low_depth` reproduced exactly. This is a property of numpy,
  not of CalibDeconv, and it does not affect any value reported in the
  manuscript.

## Composition-level coverage analyses

Two standalone analyses score the **already-frozen** CalibDeconv outputs. They are
**post-hoc evaluation only**: no model is trained, no estimator is refitted, no
signature matrix, marker panel, nonconformity score or conformal radius is
re-selected or re-tuned. Each script reads result CSVs that are already in the
repository and writes new summary CSVs. Both are deterministic — no random
seeds are involved.

```bash
python scripts/15_simultaneous_coverage.py
python scripts/16_gse107572_real_bulk.py
```

### `scripts/15_simultaneous_coverage.py`

How often is the *entire* five-component composition covered, rather than each
cell type separately? Also evaluates a Bonferroni-adjusted family-wise
extension, in which each cell type is calibrated at 1 − α/K with K = 5, so the
radius is the ceil((n+1)(1 − α/K))-th order statistic of the calibration-set
absolute-error nonconformity scores. That construction is
**calibration-set-defined**: the radius is determined from calibration data
alone. Test ground truth is read only in the final scoring step.

| | |
|---|---|
| **Inputs** | `data/processed/true_proportions_test_5type.csv`<br>`results/conformal_marker_5types/intervals_test_clipped.csv`<br>`results/conformal_marker_5types/nonconformity_scores.csv`<br>`results/ensemble_marker_5types/ensemble_summary_test.csv` |
| **Outputs** | `results/simultaneous_coverage/coverage_summary.csv`<br>`results/simultaneous_coverage/coverage_by_cell_type.csv`<br>`results/simultaneous_coverage/bonferroni_radii.csv`<br>`results/simultaneous_coverage/miss_patterns.csv`<br>`results/simultaneous_coverage/miss_correlations.csv` |

Expected values (500 held-out pseudo-bulk mixtures, 500 calibration samples):

| Interval | Nominal | Marginal | Simultaneous | Mean width |
|---|---|---|---|---|
| Raw bootstrap q05–q95 | — | 0.3896 | 0.002 (1/500) | 0.1099 |
| Marginal conformal | 0.90 | 0.9500 | 0.780 (390/500) | 0.288756 |
| Bonferroni family-wise | 0.90 | 0.9892 | 0.946 (473/500) | 0.344433 |

Width cost of the family-wise extension: 0.288756 → 0.344433, an increase of
19.28%. Bonferroni order statistic: 491st of 500, α/K = 0.02.

### `scripts/16_gse107572_real_bulk.py`

Descriptive scoring of the frozen pipeline on the nine GSE107572 PBMC donors
with matched flow cytometry, on the native five-type basis. The conformal radii
are re-used unchanged from the pseudo-bulk calibration set; nothing is
recalibrated on real-bulk data.

| | |
|---|---|
| **Inputs** | `results/phase6b_gse107572/ground_truth_5type.csv`<br>`results/phase6b_gse107572/predicted_proportions.csv`<br>`results/conformal_marker_5types/nonconformity_scores.csv` |
| **Outputs** | `results/gse107572_real_bulk/metrics_overall.csv`<br>`results/gse107572_real_bulk/metrics_per_celltype.csv`<br>`results/gse107572_real_bulk/conformal_coverage.csv` |

Expected values: MAE 0.098474, CCC 0.769255, Pearson 0.770494, RMSE 0.117251
(n = 9). At nominal 0.90, marginal coverage 0.8667 and simultaneous coverage
4/9; under the family-wise extension, 0.9333 and 6/9.

Split conformal guarantees marginal coverage under exchangeability between the
calibration and target samples. Transfer to an independent cohort on a
different platform violates that assumption, so no coverage guarantee carries
over — these figures are descriptive, at n = 9.

## Reference Data

- **Primary reference**: Hao et al. PBMC CITE-seq atlas (2021), accessed via scanpy
- **External validation**: 10x Genomics PBMC 3k dataset
- **Real-bulk feasibility**: GSE107572 (n = 9, flow cytometry)
- **Out-of-domain**: GSE60424 (whole blood)

## Reproducibility

- All random processes use fixed seeds (default: 42)
- Donor-aware splits prevent data leakage
- Frozen marker panel (445 genes) and signature matrix used across all analyses
- Conformal quantiles calibrated once and applied without modification to test and external data

## Citation

Zhou Y.-X., Ding P., Gu B.-Y., Yin H.-Y., and Gu W.-J. CalibDeconv: conformal calibration of prediction intervals for PBMC-domain deconvolution from bulk transcriptomes. (2026). Manuscript submitted.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
