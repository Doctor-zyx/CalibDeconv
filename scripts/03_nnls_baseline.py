#!/usr/bin/env python
"""
Phase 2: NNLS baseline deconvolution.

Builds a signature matrix from the REFERENCE cell pool only, then runs
NNLS deconvolution on the calibration and test pseudo-bulks.

**Expression-scale consistency:** Both the signature matrix and the
pseudo-bulk matrix use the SAME normalisation (CPM), as specified in
``config.yaml`` under ``nnls.expression_scale``.

Usage::

    python scripts/03_nnls_baseline.py \\
        --config config/config.yaml \\
        --cell-pool-dir data/processed \\
        --pseudobulk-dir data/processed \\
        --output-dir results/nnls \\
        --seed 42

Outputs
-------
- ``results/nnls/signature_matrix.csv``
- ``results/nnls/predicted_proportions_cal.csv``
- ``results/nnls/predicted_proportions_test.csv``
- ``results/nnls/accuracy_metrics.csv`` (overall)
- ``results/nnls/accuracy_per_celltype.csv``
- ``results/figures/nnls_true_vs_predicted.png/pdf``
- ``results/figures/nnls_error_boxplot.png/pdf``
- ``logs/03_nnls_baseline_*.log``
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
    run_qc_summary, qc_report_all, is_debug_mode,
)
from src.deconvolution.signature import (
    build_signature_matrix, subset_signature_to_bulk_genes,
)
from src.deconvolution.celltypes import resolve_inputs, resolve_cell_type_column
from src.deconvolution.nnls import deconvolve_batch
from src.evaluation.metrics import compute_accuracy_metrics
from src.evaluation.plotting import plot_true_vs_predicted, plot_error_boxplot


def main():
    parser = argparse.ArgumentParser(description="Phase 2: NNLS baseline deconvolution")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--cell-pool-dir", default="data/processed", help="Dir with cell_pool_*.h5ad")
    parser.add_argument("--pseudobulk-dir", default="data/processed", help="Dir with pseudo-bulk CSVs")
    parser.add_argument("--output-dir", default=None,
                        help="Defaults to results/<auto> based on gene/cell-type set "
                             "(e.g. results/nnls_marker_5types for the primary line).")
    parser.add_argument("--gene-set", choices=["all", "hvg3000", "markers"], default="hvg3000",
                        help="Gene panel: all | hvg3000 | markers (primary).")
    parser.add_argument("--cell-type-set", choices=["7type", "5type"], default="7type",
                        help="Cell-type granularity: 7type | 5type (primary, T subsets merged).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("03_nnls", log_dir=str(PROJECT_ROOT / "logs"))
    set_seed(args.seed)

    sc_cfg = cfg["data"]["sc_reference"]
    nnls_cfg = cfg["nnls"]
    base_cell_type_col = sc_cfg["cell_type_column"]
    expr_scale = nnls_cfg["expression_scale"]
    gene_set = args.gene_set
    cell_type_set = args.cell_type_set

    pb_dir = Path(args.pseudobulk_dir)
    paths = resolve_inputs(gene_set, cell_type_set, pb_dir)

    # Output dir: explicit override, else auto by gene/cell-type set
    out_subdir = args.output_dir or f"results/{paths['default_out_subdir']}"
    results_dir = str(PROJECT_ROOT / out_subdir)
    fig_dir = str(PROJECT_ROOT / "results" / "figures")
    ensure_dir(results_dir)
    ensure_dir(fig_dir)

    logger.info("=" * 60)
    logger.info("Phase 2: NNLS Baseline Deconvolution")
    logger.info("  Debug mode: %s", is_debug_mode(cfg))
    logger.info("  gene_set: %s | cell_type_set: %s", gene_set, cell_type_set)
    logger.info("  expression_scale: %s", expr_scale)
    logger.info("  output_dir: %s", results_dir)
    logger.info("=" * 60)

    # Load reference cell pool (NEVER calibration or test pool)
    pool_dir = Path(args.cell_pool_dir)
    import scanpy as sc
    adata_ref = sc.read_h5ad(pool_dir / "cell_pool_reference.h5ad")
    logger.info("Reference pool: %d cells × %d genes", *adata_ref.shape)

    # Resolve the obs column for signatures (collapses to 5type if requested)
    cell_type_col = resolve_cell_type_column(adata_ref, cell_type_set, base_col=base_cell_type_col)
    logger.info("Signature cell-type column: %s", cell_type_col)

    # ── Determine gene panel from resolver ──
    if paths["panel_file"] is not None:
        panel_path = paths["panel_file"]
        if not panel_path.exists():
            raise FileNotFoundError(
                f"Gene panel not found: {panel_path}. Generate it first "
                f"(02b_select_hvg.py / 03d_finalize_primary.py), or use --gene-set all."
            )
        with open(panel_path) as fh:
            gene_panel = [line.strip() for line in fh if line.strip()]
        logger.info("Loaded gene panel: %d genes from %s", len(gene_panel), panel_path)
    else:
        gene_panel = None
    cal_file = paths["cal_file"]
    test_file = paths["test_file"]
    sig_out_name = paths["sig_out_name"]

    # Load pseudo-bulk data (CPM-normalised) and true proportions
    X_cal = pd.read_csv(cal_file, index_col=0)
    y_cal = pd.read_csv(paths["true_cal"], index_col=0)
    X_test = pd.read_csv(test_file, index_col=0)
    y_test = pd.read_csv(paths["true_test"], index_col=0)

    logger.info("Calibration: %d samples x %d genes, Test: %d samples x %d genes",
                X_cal.shape[0], X_cal.shape[1], X_test.shape[0], X_test.shape[1])
    logger.info("Cell types: %s", y_test.columns.tolist())

    # Build signature from REFERENCE pool ONLY
    cell_types = y_test.columns.tolist()
    signature, _ = build_signature_matrix(
        adata=adata_ref,
        cell_type_column=cell_type_col,
        cell_types=cell_types,
        min_cells=sc_cfg["min_cells_per_type"],
    )

    # Normalise signature to match pseudo-bulk scale
    from src.data.pseudobulk import normalize_signature
    signature_norm = normalize_signature(signature, method=expr_scale)

    # ── Restrict signature to the chosen gene panel ──
    if gene_panel is not None:
        # Keep only panel genes that exist in the signature, in panel order
        missing_in_sig = [g for g in gene_panel if g not in signature_norm.index]
        if missing_in_sig:
            logger.warning("%d panel genes missing from signature: %s ...",
                           len(missing_in_sig), missing_in_sig[:5])
        panel_in_sig = [g for g in gene_panel if g in signature_norm.index]
        sig_subset = signature_norm.loc[panel_in_sig]
        # Align pseudo-bulk to exactly the same genes/order
        X_cal = X_cal[panel_in_sig]
        X_test = X_test[panel_in_sig]
    else:
        sig_subset = subset_signature_to_bulk_genes(signature_norm, X_test.columns.tolist())
        # Align pseudo-bulk to signature genes/order
        X_cal = X_cal[sig_subset.index.tolist()]
        X_test = X_test[sig_subset.index.tolist()]

    # ── MANDATORY consistency checks (gene set + order + scale) ──
    logger.info("-" * 40)
    logger.info("Gene consistency checks (gene_set=%s):", gene_set)
    sig_genes = list(sig_subset.index)
    cal_genes = list(X_cal.columns)
    test_genes = list(X_test.columns)

    check_sig_cal_set = set(sig_genes) == set(cal_genes)
    check_sig_test_set = set(sig_genes) == set(test_genes)
    check_sig_cal_order = sig_genes == cal_genes
    check_sig_test_order = sig_genes == test_genes
    logger.info("  signature == cal  (set):   %s", check_sig_cal_set)
    logger.info("  signature == test (set):   %s", check_sig_test_set)
    logger.info("  signature == cal  (order): %s", check_sig_cal_order)
    logger.info("  signature == test (order): %s", check_sig_test_order)
    logger.info("  n genes: signature=%d cal=%d test=%d", len(sig_genes), len(cal_genes), len(test_genes))

    if not (check_sig_cal_set and check_sig_test_set and check_sig_cal_order and check_sig_test_order):
        raise RuntimeError(
            "Gene consistency check FAILED. Signature and pseudo-bulk genes "
            "must match exactly in set AND order before NNLS. Aborting."
        )
    logger.info("  [OK] Signature and pseudo-bulk genes are identical in set and order")
    logger.info("  Expression scale: %s (CPM) — same for signature and pseudo-bulk", expr_scale)

    sig_path = Path(results_dir) / sig_out_name
    save_df(sig_subset, sig_path)
    logger.info("Signature: %d genes × %d cell types, saved to %s", *sig_subset.shape, sig_path)

    # Deconvolve
    logger.info("Deconvolving calibration set...")
    pred_cal = deconvolve_batch(X_cal, sig_subset)
    save_df(pred_cal, Path(results_dir) / "predicted_proportions_cal.csv")

    logger.info("Deconvolving test set...")
    pred_test = deconvolve_batch(X_test, sig_subset)
    save_df(pred_test, Path(results_dir) / "predicted_proportions_test.csv")

    # Evaluate
    logger.info("-" * 40)
    logger.info("Accuracy metrics (test set):")
    acc = compute_accuracy_metrics(y_test, pred_test)
    save_df(acc["overall"], Path(results_dir) / "accuracy_metrics.csv")
    save_df(acc["per_cell_type"], Path(results_dir) / "accuracy_per_celltype.csv")

    for k, v in acc["overall"].iloc[0].items():
        logger.info("  %s: %.4f", k, v)

    # QC
    qc_results = [
        run_qc_summary(pred_test, label="pred_test", expect_proportions=True, logger=logger),
        run_qc_summary(y_test, label="true_test", expect_proportions=True, logger=logger),
        run_qc_summary(sig_subset, label="signature", logger=logger),
    ]
    for r in qc_results:
        r["file_path"] = str(Path(results_dir) / "predicted_proportions_test.csv")
    qc_report_all(qc_results, logger=logger)

    # Expression-scale check: compare mean expression in signature vs pseudo-bulk
    sig_mean = sig_subset.values.mean()
    pb_mean = X_test.values.mean()
    logger.info("Scale check — signature mean: %.4f, pseudo-bulk mean: %.4f", sig_mean, pb_mean)
    if abs(np.log10(max(sig_mean, 1e-8) / max(pb_mean, 1e-8))) > 1:
        logger.warning("[WARN]  Signature and pseudo-bulk may be on different scales!")
    else:
        logger.info("[OK]  Signature and pseudo-bulk scales consistent")

    # Figures
    plot_true_vs_predicted(
        y_test, pred_test,
        title="NNLS Deconvolution: True vs Predicted",
        save_path=str(Path(fig_dir) / "nnls_true_vs_predicted"),
    )
    plot_error_boxplot(
        y_test, pred_test,
        title="NNLS: Absolute Error by Cell Type",
        save_path=str(Path(fig_dir) / "nnls_error_boxplot"),
    )

    logger.info("Phase 2 complete. [OK] ")
    logger.info("Results in %s", results_dir)


if __name__ == "__main__":
    main()
