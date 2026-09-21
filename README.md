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
| `requirements-v1.2-tested.txt` | **Clean-room validation snapshot.** The exact versions of one independent environment in which the v1.2 coverage analyses were re-run from a clean checkout and reproduced every reported value byte-for-byte. Recorded as evidence that the results do not depend on a single dependency stack. |
| Software versions in the manuscript | **The original analysis environment.** Reported in the manuscript Methods and unchanged. |

The clean-room snapshot is *not* the environment in which the original results
were produced, and it does not supersede the software versions reported in the
manuscript. It was built from `requirements.txt` at a later date and therefore
resolved to newer releases; the reported values were identical in both.

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
