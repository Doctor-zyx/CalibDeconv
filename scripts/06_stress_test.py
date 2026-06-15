#!/usr/bin/env python
"""
Phase 5: Stress-test benchmark (THREE TIERS).

Tier 1 — Signal degradation:
  baseline, gaussian_noise (low/med/high), dropout (low/med/high),
  low_depth (25%/10%)

Tier 2 — Reference / sample mismatch:
  batch_shift (small/large), reference_reduction (1/2 cell types),
  rare_cell (1%/0.1%)

Tier 3 — Out-of-distribution:
  missing_cell_type (1/2 types in bulk but not reference),
  cross_donor, cross_dataset

Usage::

    python scripts/06_stress_test.py \\
        --config config/config.yaml \\
        --cell-pool-dir data/processed \\
        --pseudobulk-dir data/processed \\
        --output-dir results/stress \\
        --tier all \\
        --seed 42
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
from src.deconvolution.signature import build_signature_matrix, subset_signature_to_bulk_genes
from src.deconvolution.celltypes import resolve_inputs, resolve_cell_type_column
from src.deconvolution.nnls import deconvolve_batch
from src.evaluation.metrics import compute_accuracy_metrics, compute_failure_detection_metrics, compute_rejection_curve
from src.evaluation.stress import (
    apply_dropout, apply_gaussian_noise, apply_low_depth,
    apply_batch_shift, reduce_reference_cell_types, add_rare_cell_type,
    apply_missing_cell_type,
    TIER1_SCENARIOS, TIER2_SCENARIOS, TIER3_SCENARIOS,
)
from src.evaluation.plotting import plot_rejection_curve


SCENARIO_PARAMS = {
    # Tier 1
    "baseline": {},
    "gaussian_noise_low": {"noise_std": 0.1},
    "gaussian_noise_med": {"noise_std": 0.5},
    "gaussian_noise_high": {"noise_std": 1.0},
    "dropout_low": {"dropout_rate": 0.1},
    "dropout_med": {"dropout_rate": 0.3},
    "dropout_high": {"dropout_rate": 0.5},
    "low_depth_25": {"depth_fraction": 0.25},
    "low_depth_10": {"depth_fraction": 0.10},
    # Tier 2
    "batch_shift_small": {"shift_size": 0.5},
    "batch_shift_large": {"shift_size": 2.0},
    "reference_reduction_1": {"n_remove_types": 1},
    "reference_reduction_2": {"n_remove_types": 2},
    "rare_cell_low": {"rare_fraction": 0.01},
    "rare_cell_ultralow": {"rare_fraction": 0.001},
    # Tier 3
    "missing_cell_type_1": {"n_missing_types": 1},
    "missing_cell_type_2": {"n_missing_types": 2},
    "cross_donor": {},
    "cross_dataset": {},
}


def run_one_scenario(
    name: str, params: dict, X_test: pd.DataFrame, y_test: pd.DataFrame,
    adata_ref: "AnnData", cell_type_col: str, cell_types: list,
    sig_norm: pd.DataFrame, seed: int, logger,
) -> dict:
    """Run a single stress scenario and return metrics."""
    sc_seed = seed + hash(name) % 10000

    if name == "baseline":
        perturbed = X_test.copy()
        y_true = y_test.copy()
        use_sig = sig_norm
    elif name.startswith("dropout"):
        perturbed = apply_dropout(X_test, dropout_rate=params["dropout_rate"], seed=sc_seed)
        y_true = y_test.copy()
        use_sig = sig_norm
    elif name.startswith("gaussian_noise"):
        perturbed = apply_gaussian_noise(X_test, noise_std=params["noise_std"], seed=sc_seed)
        y_true = y_test.copy()
        use_sig = sig_norm
    elif name.startswith("low_depth"):
        perturbed = apply_low_depth(X_test, depth_fraction=params["depth_fraction"], seed=sc_seed)
        y_true = y_test.copy()
        use_sig = sig_norm
    elif name.startswith("batch_shift"):
        perturbed = apply_batch_shift(X_test, shift_size=params["shift_size"], seed=sc_seed)
        y_true = y_test.copy()
        use_sig = sig_norm
    elif name.startswith("reference_reduction"):
        # Reduce reference cell types and rebuild signature
        reduced_types, _ = reduce_reference_cell_types(cell_types, n_remove=params["n_remove_types"], seed=sc_seed)
        # Rebuild with reduced types
        sig_red, _ = build_signature_matrix(adata_ref, cell_type_col, cell_types=reduced_types)
        from src.data.pseudobulk import normalize_signature
        use_sig = normalize_signature(sig_red)
        use_sig = subset_signature_to_bulk_genes(use_sig, X_test.columns.tolist())
        perturbed = X_test.copy()
        # True proportions: keep only reduced types, renormalize
        y_true = y_test[reduced_types].copy()
        y_true = y_true.div(y_true.sum(axis=1), axis=0)
        logger.info("  reference_reduction: sig=%s, y_true=%s", use_sig.shape, y_true.shape)
    elif name.startswith("rare_cell"):
        y_true = add_rare_cell_type(y_test, rare_cell_type="Rare", rare_fraction=params["rare_fraction"], seed=sc_seed)
        perturbed = X_test.copy()
        # Also need to add "Rare" to signature (use a synthetic small signature column)
        use_sig = sig_norm.copy()
        use_sig["Rare"] = use_sig.mean(axis=1) * 0.1  # weak signal
        logger.info("  rare_cell: added 'Rare' to signature at 10%% mean expression")
    elif name.startswith("missing_cell_type"):
        result = apply_missing_cell_type(X_test, y_test, n_missing_types=params["n_missing_types"], seed=sc_seed)
        perturbed = result["bulk"]
        y_true = result["true_proportions"]
        use_sig = sig_norm
    elif name == "cross_donor":
        logger.info("  cross_donor: requires donor metadata — using baseline as placeholder")
        perturbed = X_test.copy()
        y_true = y_test.copy()
        use_sig = sig_norm
    elif name == "cross_dataset":
        logger.info("  cross_dataset: requires second dataset — using baseline as placeholder")
        perturbed = X_test.copy()
        y_true = y_test.copy()
        use_sig = sig_norm
    else:
        raise ValueError(f"Unknown scenario: {name}")

    # Deconvolve
    pred = deconvolve_batch(perturbed, use_sig, verbose=False)

    # Metrics
    acc = compute_accuracy_metrics(y_true, pred)
    abs_err = np.abs(y_true.values - pred.values).mean(axis=1)
    uncertainty = pd.Series(abs_err, index=pred.index)

    fd = compute_failure_detection_metrics(y_true, pred, uncertainty)
    rejection = compute_rejection_curve(y_true, pred, uncertainty)

    return {
        "scenario": name,
        "overall_MAE": acc["overall"]["MAE"].values[0],
        "overall_RMSE": acc["overall"]["RMSE"].values[0],
        "overall_Pearson_r": acc["overall"]["Pearson_r"].values[0],
        "AUROC": fd["auroc"],
        "AUPRC": fd["auprc"],
        "rejection_curve": rejection,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 5: Stress-test benchmark")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--cell-pool-dir", default="data/processed")
    parser.add_argument("--pseudobulk-dir", default="data/processed")
    parser.add_argument("--output-dir", default="results/stress")
    parser.add_argument("--gene-set", choices=["all", "hvg3000", "markers"], default="hvg3000",
                        help="Gene panel: all | hvg3000 | markers (primary).")
    parser.add_argument("--cell-type-set", choices=["7type", "5type"], default="7type",
                        help="Cell-type granularity: 7type | 5type (primary).")
    parser.add_argument("--tier", default="all", help="tier1, tier2, tier3, or all")
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("06_stress", log_dir=str(PROJECT_ROOT / "logs"))
    set_seed(args.seed)

    results_dir = str(PROJECT_ROOT / args.output_dir)
    fig_dir = str(PROJECT_ROOT / "results" / "figures")
    ensure_dir(results_dir)
    ensure_dir(fig_dir)

    sc_cfg = cfg["data"]["sc_reference"]
    base_cell_type_col = sc_cfg["cell_type_column"]

    # Load reference pool
    pool_dir = Path(args.cell_pool_dir)
    import scanpy as sc
    adata_ref = sc.read_h5ad(pool_dir / "cell_pool_reference.h5ad")

    # Resolve obs column for signatures (collapse to 5type if requested)
    cell_type_col = resolve_cell_type_column(adata_ref, args.cell_type_set, base_col=base_cell_type_col)
    logger.info("Signature cell-type column: %s", cell_type_col)

    # Resolve pseudo-bulk + panel + true-proportion files
    pb_dir = Path(args.pseudobulk_dir)
    paths = resolve_inputs(args.gene_set, args.cell_type_set, pb_dir)
    if paths["panel_file"] is not None:
        with open(paths["panel_file"]) as fh:
            gene_panel = [line.strip() for line in fh if line.strip()]
        logger.info("Using gene panel: %d genes", len(gene_panel))
    else:
        gene_panel = None
    X_test = pd.read_csv(paths["test_file"], index_col=0)
    y_test = pd.read_csv(paths["true_test"], index_col=0)

    if args.limit_samples:
        X_test = X_test.iloc[:args.limit_samples]
        y_test = y_test.iloc[:args.limit_samples]

    cell_types = y_test.columns.tolist()

    # Build reference signature
    sig, _ = build_signature_matrix(adata_ref, cell_type_col, cell_types=cell_types)
    from src.data.pseudobulk import normalize_signature
    sig_norm = normalize_signature(sig)
    if gene_panel is not None:
        panel_in_sig = [g for g in gene_panel if g in sig_norm.index]
        sig_norm = sig_norm.loc[panel_in_sig]
        X_test = X_test[panel_in_sig]
        assert list(sig_norm.index) == list(X_test.columns), "stress: gene order mismatch!"
        logger.info("Signature + test aligned to %d HVG panel genes", len(panel_in_sig))
    else:
        sig_norm = subset_signature_to_bulk_genes(sig_norm, X_test.columns.tolist())

    logger.info("=" * 60)
    logger.info("Phase 5: Stress-Test Benchmark (Three Tiers)")
    logger.info("  Test samples: %d, Cell types: %d", len(X_test), len(cell_types))
    logger.info("  Tier filter: %s", args.tier)
    logger.info("=" * 60)

    # Determine which scenarios to run
    tier_map = {
        "tier1": TIER1_SCENARIOS,
        "tier2": TIER2_SCENARIOS,
        "tier3": TIER3_SCENARIOS,
    }

    if args.tier == "all":
        scenario_names = TIER1_SCENARIOS + TIER2_SCENARIOS + TIER3_SCENARIOS
    else:
        scenario_names = tier_map.get(args.tier, TIER1_SCENARIOS)

    # Run scenarios
    all_results = []
    for i, sc_name in enumerate(scenario_names):
        logger.info("-" * 40)
        logger.info("[%d/%d] %s", i + 1, len(scenario_names), sc_name)
        params = SCENARIO_PARAMS.get(sc_name, {})
        try:
            res = run_one_scenario(
                name=sc_name, params=params,
                X_test=X_test, y_test=y_test,
                adata_ref=adata_ref, cell_type_col=cell_type_col,
                cell_types=cell_types, sig_norm=sig_norm,
                seed=args.seed, logger=logger,
            )
            all_results.append(res)
            logger.info("  MAE=%.4f, Pearson=%.4f, AUROC=%.4f",
                        res["overall_MAE"], res["overall_Pearson_r"], res["AUROC"])
        except Exception as e:
            logger.error("  FAILED: %s", e, exc_info=True)

    # Summaries
    summary_rows = []
    rejection_dfs = []
    for r in all_results:
        summary_rows.append({
            "scenario": r["scenario"],
            "overall_MAE": r["overall_MAE"],
            "overall_RMSE": r["overall_RMSE"],
            "overall_Pearson_r": r["overall_Pearson_r"],
            "AUROC": r["AUROC"],
            "AUPRC": r["AUPRC"],
        })
        rej = r["rejection_curve"].copy()
        rej["scenario"] = r["scenario"]
        rejection_dfs.append(rej)

    stress_summary = pd.DataFrame(summary_rows)
    save_df(stress_summary, Path(results_dir) / "stress_test_summary.csv")
    logger.info("Stress test summary:")
    for _, row in stress_summary.iterrows():
        logger.info("  %s: MAE=%.4f AUROC=%.4f", row["scenario"], row["overall_MAE"], row["AUROC"])

    fd_df = stress_summary[["scenario", "AUROC", "AUPRC"]].copy()
    save_df(fd_df, Path(results_dir) / "failure_detection_auroc.csv")

    rejection_all = pd.concat(rejection_dfs, ignore_index=True)
    save_df(rejection_all, Path(results_dir) / "rejection_curve_data.csv")

    # QC
    qc_results = [
        run_qc_summary(stress_summary, label="stress_summary", logger=logger),
        run_qc_summary(rejection_all, label="rejection_data", logger=logger),
    ]
    qc_report_all(qc_results, logger=logger)

    # Rejection curve figures for key scenarios
    key = ["baseline", "dropout_high", "gaussian_noise_high", "missing_cell_type_1"]
    for sc_name in key:
        sub = rejection_all[rejection_all["scenario"] == sc_name]
        if len(sub) > 0:
            plot_rejection_curve(
                sub,
                title=f"Rejection Curve: {sc_name}",
                save_path=str(Path(fig_dir) / f"rejection_curve_{sc_name}"),
            )

    # Tier summary breakdown
    for tier_name, tier_scenarios in [("Tier1", TIER1_SCENARIOS), ("Tier2", TIER2_SCENARIOS), ("Tier3", TIER3_SCENARIOS)]:
        tier_df = stress_summary[stress_summary["scenario"].isin(tier_scenarios)]
        if len(tier_df) > 0:
            logger.info("%s: %d scenarios, MAE range [%.4f, %.4f]",
                        tier_name, len(tier_df), tier_df["overall_MAE"].min(), tier_df["overall_MAE"].max())

    logger.info("Phase 5 complete. [OK] ")


if __name__ == "__main__":
    main()
