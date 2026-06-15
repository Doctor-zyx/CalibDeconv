# CalibDeconv

**Conformal calibration of prediction intervals for PBMC-domain deconvolution from bulk transcriptomes.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

CalibDeconv is a reliability-focused framework for cellular deconvolution of PBMC bulk transcriptomes. It combines:

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

Zhang Y. CalibDeconv: conformal calibration of prediction intervals for PBMC-domain deconvolution from bulk transcriptomes. (2025). Manuscript in preparation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
