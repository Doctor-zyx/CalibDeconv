#!/usr/bin/env python
"""
Phase 6: Public real benchmark validation.

[WARN]  **PENDING MANUAL REVIEW — DO NOT RUN YET** [WARN] 

Before executing this script, you MUST manually confirm:
1. Sample-to-sample correspondence between bulk expression and ground-truth
   cell proportions (are they from the SAME sample?)
2. Cell-type name mapping between reference scRNA-seq and FACS labels
3. Gene identifier format (gene symbols? Ensembl IDs?) — and that they
   match the reference single-cell data
4. That the ground truth is a genuine gold standard, not a weak reference

This script will only run once Phases 1–5 have passed review and the
above checks are confirmed.

Expected datasets:
- data/external/newman2015/  : 20 PBMC, microarray + FACS
- data/external/racle2017/   :  4 melanoma, RNA-seq + FACS
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
    run_qc_summary, qc_report_all,
)
from src.deconvolution.signature import build_signature_matrix, subset_signature_to_bulk_genes
from src.deconvolution.nnls import deconvolve_batch
from src.evaluation.metrics import compute_accuracy_metrics
from src.evaluation.plotting import plot_true_vs_predicted


def main():
    parser = argparse.ArgumentParser(description="Phase 6: Real benchmark validation")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--cell-pool-dir", default="data/processed")
    parser.add_argument("--benchmark", default="newman2015")
    parser.add_argument("--output-dir", default="results/real_benchmark")
    parser.add_argument("--external-dir", default="data/external")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("07_real_benchmark", log_dir=str(PROJECT_ROOT / "logs"))
    set_seed(args.seed)

    logger.info("=" * 60)
    logger.info("[WARN]  PHASE 6 — PENDING MANUAL REVIEW")
    logger.info("=" * 60)
    logger.info("This script requires manual confirmation before execution.")
    logger.info("Please verify:")
    logger.info("  1. Bulk-vs-FACS sample correspondence")
    logger.info("  2. Cell-type name mapping")
    logger.info("  3. Gene identifier format matches reference")
    logger.info("  4. Ground truth is genuine gold standard")
    logger.info("")
    logger.info("After confirmation, add a file named 'REVIEW_APPROVED.txt'")
    logger.info("to data/external/ and re-run this script.")
    logger.info("=" * 60)

    external_dir = Path(PROJECT_ROOT / args.external_dir)
    approval_file = external_dir / "REVIEW_APPROVED.txt"

    if not approval_file.exists():
        logger.warning(
            "[STOP]  REVIEW NOT APPROVED. To proceed, create: %s",
            approval_file,
        )
        logger.info("  echo 'reviewed by [your name] on [date]' > %s", approval_file)
        logger.info("Phase 6 aborted — manual review required.")
        return

    # Only runs after manual approval
    results_dir = str(PROJECT_ROOT / args.output_dir)
    fig_dir = str(PROJECT_ROOT / "results" / "figures")
    ensure_dir(results_dir)
    ensure_dir(fig_dir)

    # Load reference pool
    pool_dir = Path(args.cell_pool_dir)
    import scanpy as sc
    adata_ref = sc.read_h5ad(pool_dir / "cell_pool_reference.h5ad")

    sc_cfg = cfg["data"]["sc_reference"]
    cell_type_col = sc_cfg["cell_type_column"]

    # Load benchmark data (placeholder — paths subject to manual review)
    data_path = external_dir / args.benchmark
    logger.info("Loading benchmark data from %s", data_path)

    try:
        bulk_expr = pd.read_csv(data_path / "bulk_expression.csv", index_col=0)
        facs_true = pd.read_csv(data_path / "facs_proportions.csv", index_col=0)
        logger.info("Bulk: %s, FACS: %s", bulk_expr.shape, facs_true.shape)
    except FileNotFoundError:
        logger.error("Benchmark data not found. Expected files in %s/", data_path)
        return

    # Build signature
    common_types = [ct for ct in facs_true.columns if ct in adata_ref.obs[cell_type_col].unique()]
    if len(common_types) < 2:
        logger.warning("Few common cell types! Manual mapping needed.")
        common_types = facs_true.columns.tolist()

    signature, _ = build_signature_matrix(adata_ref, cell_type_col, cell_types=common_types)
    from src.data.pseudobulk import normalize_signature
    sig_norm = normalize_signature(signature)
    sig_subset = subset_signature_to_bulk_genes(sig_norm, bulk_expr.columns.tolist() if bulk_expr.shape[0] < bulk_expr.shape[1] else bulk_expr.index.tolist())

    # Deconvolve
    bulk_for_deconv = bulk_expr if bulk_expr.shape[0] < bulk_expr.shape[1] else bulk_expr.T
    predictions = deconvolve_batch(bulk_for_deconv, sig_subset)

    # Evaluate
    common_eval = [ct for ct in facs_true.columns if ct in predictions.columns]
    acc = compute_accuracy_metrics(facs_true[common_eval], predictions[common_eval])

    save_df(predictions, Path(results_dir) / "predictions.csv")
    save_df(acc["overall"], Path(results_dir) / "metrics.csv")
    save_df(acc["per_cell_type"], Path(results_dir) / "metrics_per_celltype.csv")

    logger.info("Real benchmark results:")
    for k, v in acc["overall"].iloc[0].items():
        logger.info("  %s: %.4f", k, v)

    # QC
    qc_results = [
        run_qc_summary(predictions, label="real_predictions", expect_proportions=True, logger=logger),
        run_qc_summary(facs_true, label="real_facs", expect_proportions=True, logger=logger),
    ]
    qc_report_all(qc_results, logger=logger)

    plot_true_vs_predicted(
        facs_true[common_eval], predictions[common_eval],
        title=f"Real Benchmark ({args.benchmark}): True vs Predicted",
        save_path=str(Path(fig_dir) / f"real_benchmark_{args.benchmark}"),
    )

    logger.info("Phase 6 complete. [OK] ")


if __name__ == "__main__":
    main()
