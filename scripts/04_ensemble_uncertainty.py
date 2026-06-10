#!/usr/bin/env python
"""
Phase 3: Ensemble uncertainty estimation.

Runs B bootstrap iterations with gene + cell subsampling, using ONLY
the reference cell pool for the signature.  Outputs per-(sample, cell_type)
mean, std, and quantiles for downstream conformal calibration.

Usage::

    python scripts/04_ensemble_uncertainty.py \\
        --config config/config.yaml \\
        --cell-pool-dir data/processed \\
        --pseudobulk-dir data/processed \\
        --output-dir results/ensemble \\
        --seed 42

Outputs
-------
- ``results/ensemble/ensemble_predictions.npz``
- ``results/ensemble/ensemble_summary.csv``
- ``results/ensemble/uncertainty_error_correlation.csv``
- ``results/figures/ensemble_uncertainty_vs_error.png/pdf``
- ``logs/04_ensemble_uncertainty_*.log``
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import (
    load_config, setup_logger, set_seed, ensure_dir, save_df,
    run_qc_summary, qc_report_all, is_debug_mode, debug_param,
)
from src.uncertainty.ensemble import run_ensemble
from src.deconvolution.celltypes import resolve_inputs, resolve_cell_type_column
from src.evaluation.metrics import compute_uncertainty_error_correlation
from src.evaluation.plotting import plot_uncertainty_vs_error


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Ensemble uncertainty")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--cell-pool-dir", default="data/processed")
    parser.add_argument("--pseudobulk-dir", default="data/processed")
    parser.add_argument("--output-dir", default="results/ensemble")
    parser.add_argument("--gene-set", choices=["all", "hvg3000", "markers"], default="hvg3000",
                        help="Gene panel: all | hvg3000 | markers (primary).")
    parser.add_argument("--cell-type-set", choices=["7type", "5type"], default="7type",
                        help="Cell-type granularity: 7type | 5type (primary).")
    parser.add_argument("--n-iterations", type=int, default=None)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("04_ensemble", log_dir=str(PROJECT_ROOT / "logs"))
    set_seed(args.seed)

    results_dir = str(PROJECT_ROOT / args.output_dir)
    fig_dir = str(PROJECT_ROOT / "results" / "figures")
    ensure_dir(results_dir)
    ensure_dir(fig_dir)

    sc_cfg = cfg["data"]["sc_reference"]
    ens_cfg = cfg["ensemble"]
    base_cell_type_col = sc_cfg["cell_type_column"]
    n_iterations = args.n_iterations or debug_param(cfg, "ensemble", "n_iterations", ens_cfg["n_iterations"])

    # Load reference cell pool ONLY
    pool_dir = Path(args.cell_pool_dir)
    import scanpy as sc
    adata_ref = sc.read_h5ad(pool_dir / "cell_pool_reference.h5ad")
    logger.info("Reference pool: %d cells", adata_ref.shape[0])

    # Resolve obs column for signatures (collapse to 5type if requested)
    cell_type_col = resolve_cell_type_column(adata_ref, args.cell_type_set, base_col=base_cell_type_col)
    logger.info("Signature cell-type column: %s", cell_type_col)

    # Resolve pseudo-bulk + panel + true-proportion files
    pb_dir = Path(args.pseudobulk_dir)
    paths = resolve_inputs(args.gene_set, args.cell_type_set, pb_dir)
    if paths["panel_file"] is not None:
        with open(paths["panel_file"]) as fh:
            gene_panel = [line.strip() for line in fh if line.strip()]
        logger.info("Using gene panel: %d genes from %s", len(gene_panel), paths["panel_file"])
    else:
        gene_panel = None
    X_cal = pd.read_csv(paths["cal_file"], index_col=0)
    y_cal = pd.read_csv(paths["true_cal"], index_col=0)
    X_test = pd.read_csv(paths["test_file"], index_col=0)
    y_test = pd.read_csv(paths["true_test"], index_col=0)

    # Combine cal + test for ensemble
    X_combined = pd.concat([X_cal, X_test])
    y_combined = pd.concat([y_cal, y_test])

    if args.limit_samples is not None:
        n_half = args.limit_samples // 2
        n_cal_take = min(n_half, len(X_cal))
        n_test_take = min(args.limit_samples - n_cal_take, len(X_test))
        X_combined = pd.concat([X_cal.iloc[:n_cal_take], X_test.iloc[:n_test_take]])
        y_combined = pd.concat([y_cal.iloc[:n_cal_take], y_test.iloc[:n_test_take]])
        logger.info("LIMITED to %d samples (%d cal + %d test)", len(X_combined), n_cal_take, n_test_take)

    cell_types = y_combined.columns.tolist()

    # Gene panel is already applied via the hvg3000 pseudo-bulk files.
    # The ensemble subsamples genes WITHIN this panel during bootstrap.
    if gene_panel is not None:
        # Sanity: the loaded matrix should already be the HVG panel
        assert list(X_combined.columns) == gene_panel, \
            "HVG pseudo-bulk columns do not match the gene panel!"
        logger.info("Ensemble runs on the %d-gene HVG panel (no extra selection)", len(gene_panel))
    n_hvg = len(gene_panel) if gene_panel is not None else X_combined.shape[1]

    logger.info("=" * 60)
    logger.info("Phase 3: Ensemble Uncertainty Estimation")
    logger.info("  Debug mode: %s", is_debug_mode(cfg))
    logger.info("  n_iterations: %d", n_iterations)
    logger.info("  gene_fraction: %.2f", ens_cfg["gene_sampling_fraction"])
    logger.info("  cell_fraction: %.2f", ens_cfg["cell_sampling_fraction"])
    logger.info("  samples (cal+test): %d", len(X_combined))
    logger.info("  genes: %d", X_combined.shape[1])
    logger.info("  cell_types: %d", len(cell_types))
    logger.info("=" * 60)

    # Run ensemble
    result = run_ensemble(
        bulk_matrix=X_combined,
        adata=adata_ref,
        cell_type_column=cell_type_col,
        cell_types=cell_types,
        n_iterations=n_iterations,
        gene_fraction=ens_cfg["gene_sampling_fraction"],
        cell_fraction=ens_cfg["cell_sampling_fraction"],
        noise_std=ens_cfg.get("noise_std", 0.0),
        seed=args.seed,
    )

    # Save
    np.savez_compressed(
        Path(results_dir) / "ensemble_predictions.npz",
        predictions=result["ensemble_predictions"],
        sample_ids=X_combined.index.values,
        cell_types=np.array(cell_types),
        seed=args.seed,
    )
    save_df(result["summary"], Path(results_dir) / "ensemble_summary.csv")
    logger.info("Ensemble summary: %d rows", len(result["summary"]))

    # Uncertainty-error correlation
    summary = result["summary"]
    y_pred_mean = summary.pivot(index="sample_id", columns="cell_type", values="mean")
    y_pred_std = summary.pivot(index="sample_id", columns="cell_type", values="std")

    test_ids = y_test.index.tolist()
    y_pred_mean_test = y_pred_mean.loc[y_pred_mean.index.isin(test_ids)]
    y_pred_std_test = y_pred_std.loc[y_pred_std.index.isin(test_ids)]
    y_true_test = y_combined.loc[y_combined.index.isin(test_ids)]

    if len(y_pred_mean_test) >= 5:
        try:
            corr = compute_uncertainty_error_correlation(y_true_test, y_pred_mean_test, y_pred_std_test)
            save_df(corr, Path(results_dir) / "uncertainty_error_correlation.csv")
            overall_r = corr[corr["cell_type"] == "overall"]["pearson_r"].values[0]
            logger.info("Uncertainty-error correlation (overall): r=%.4f", overall_r)
        except Exception as e:
            logger.warning("Could not compute uncertainty-error correlation: %s", e)
            corr = None
    else:
        logger.info("Too few test samples (%d) for correlation — skipping", len(y_pred_mean_test))

    # QC
    qc_results = [
        run_qc_summary(y_pred_mean, label="ensemble_mean", expect_proportions=True, logger=logger),
        run_qc_summary(y_pred_std, label="ensemble_std", logger=logger),
    ]
    qc_report_all(qc_results, logger=logger)

    # Figures
    if len(y_pred_mean_test) >= 5:
        plot_uncertainty_vs_error(
            y_true_test, y_pred_mean_test, y_pred_std_test,
            title="Ensemble: Uncertainty vs Absolute Error",
            save_path=str(Path(fig_dir) / "ensemble_uncertainty_vs_error"),
        )
        logger.info("Uncertainty-error scatter plot saved.")
    else:
        logger.info("Too few test samples for scatter plot — skipping")

    logger.info("Phase 3 complete. [OK] ")
    logger.info("Results in %s", results_dir)


if __name__ == "__main__":
    main()
