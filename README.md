# CalibDeconv v1

**Conformal uncertainty calibration for reliable cellular deconvolution of bulk transcriptomes.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)]()

---

## Overview

Traditional deconvolution methods output point estimates of cell-type proportions
from bulk RNA-seq data — but they cannot tell you when those estimates are wrong.
**CalibDeconv** adds a statistical reliability layer to any deconvolution backbone:

- **Ensemble uncertainty estimation** via gene & cell bootstrap perturbations
- **Split conformal prediction** producing calibrated prediction intervals with
  guaranteed marginal coverage
- **Reliability scores** and **failure detection** to flag unreliable samples

**This is Paper 1** — method development and validation on **public data only**
(no clinical samples).  Clinical validation is planned as a follow-up study.

---

## Project structure

```
calibdeconv/
├── config/
│   └── config.yaml              # Global configuration (paths, params, seeds)
├── src/                         # Reusable library code
│   ├── data/                    # Data loading & pseudo-bulk generation
│   ├── deconvolution/           # Signature matrix & NNLS solver
│   ├── uncertainty/             # Ensemble & conformal calibration
│   ├── evaluation/              # Metrics, stress testing, plotting
│   └── utils/                   # Logging, I/O, config
├── scripts/                     # Executable pipeline scripts
│   ├── 01_download_data.py      # Phase 1a: Download scRNA-seq reference
│   ├── 02_generate_pseudobulk.py # Phase 1b: Generate pseudo-bulk samples
│   ├── 03_nnls_baseline.py      # Phase 2:  NNLS baseline deconvolution
│   ├── 04_ensemble_uncertainty.py # Phase 3: Ensemble uncertainty
│   ├── 05_conformal_calibration.py # Phase 4: Conformal calibration
│   ├── 06_stress_test.py        # Phase 5: Stress-test benchmark
│   └── 07_real_benchmark.py     # Phase 6: Real benchmark validation
├── data/
│   ├── raw/                     # Downloaded scRNA-seq data
│   ├── processed/               # Pseudo-bulk CSVs, splits
│   └── external/                # External benchmark datasets
├── results/                     # All outputs
│   ├── nnls/
│   ├── ensemble/
│   ├── conformal/
│   ├── stress/
│   ├── real_benchmark/
│   └── figures/
├── notebooks/                   # Exploratory Jupyter notebooks
├── logs/                        # Timestamped log files
├── paper/                       # Manuscript source
└── README.md
```

---

## Quick start

### 1. Install dependencies

```bash
cd calibdeconv
pip install -r requirements.txt
```

### 2. Download reference scRNA-seq data

```bash
python scripts/01_download_data.py \
    --config config/config.yaml \
    --output-dir data/raw \
    --seed 42
```

### 3. Run the full pipeline

```bash
# Phase 1b: Generate pseudo-bulk
python scripts/02_generate_pseudobulk.py \
    --config config/config.yaml \
    --adata data/raw/pbmc_reference.h5ad \
    --output-dir data/processed \
    --seed 42

# Phase 2: NNLS baseline
python scripts/03_nnls_baseline.py \
    --config config/config.yaml \
    --adata data/raw/pbmc_reference.h5ad \
    --pseudobulk-dir data/processed \
    --seed 42

# Phase 3: Ensemble uncertainty
python scripts/04_ensemble_uncertainty.py \
    --config config/config.yaml \
    --adata data/raw/pbmc_reference.h5ad \
    --pseudobulk-dir data/processed \
    --seed 42

# Phase 4: Conformal calibration
python scripts/05_conformal_calibration.py \
    --config config/config.yaml \
    --ensemble-dir results/ensemble \
    --pseudobulk-dir data/processed \
    --seed 42

# Phase 5: Stress testing
python scripts/06_stress_test.py \
    --config config/config.yaml \
    --adata data/raw/pbmc_reference.h5ad \
    --pseudobulk-dir data/processed \
    --seed 42

# Phase 6: Real benchmark (requires external data)
python scripts/07_real_benchmark.py \
    --config config/config.yaml \
    --adata data/raw/pbmc_reference.h5ad \
    --benchmark newman2015 \
    --seed 42
```

---

## Reproducibility

- All random processes use a fixed `seed` (default: 42).
- Seed values are recorded in output files and logs.
- Train / calibration / test splits are saved for full traceability.
- Intermediate outputs are saved at every phase.

---

## Requirements for real benchmark data

The real benchmark validation (Phase 6) requires external datasets that must
be downloaded manually:

| Benchmark | Source | Notes |
|-----------|--------|-------|
| Newman 2015 | [CIBERSORTx](https://cibersortx.stanford.edu/) | 20 PBMC, FACS gold standard |
| Racle 2017 | `immunedeconv::dataset_racle` (R) or [EPIC](https://github.com/GfellerLab/EPIC) | 4 melanoma, FACS |

Place the data in `data/external/<benchmark_name>/` following the structure
documented in `scripts/07_real_benchmark.py`.

---

## Evaluation metrics

| Category | Metrics |
|----------|---------|
| **Point accuracy** | MAE, RMSE, Pearson *r*, CCC |
| **Uncertainty quality** | Coverage, interval width, calibration curve |
| **Failure detection** | AUROC, AUPRC, rejection curves |

---

## Minimum success criteria (Paper 1)

- [x] Point estimates at least match NNLS baseline accuracy
- [x] 90% calibrated interval coverage ≈ 90% (no over-confidence)
- [x] Uncertainty significantly correlates with true error under stress
- [x] Failure detection AUROC clearly above random
- [x] Rejecting high-uncertainty samples reduces mean error
- [x] At least one real benchmark with explanatory power
- [x] Fully reproducible code, parameters, and data pipeline

---

## License

TBD

---

## Citation

TBD — this work is in progress.
