#!/usr/bin/env python
"""
Phase 4: Split conformal calibration.

Uses the calibration set to compute conformal quantiles, then applies
them to the test set.  Outputs BOTH raw and boundary-clipped intervals
with separate coverage and width metrics.

Usage::

    python scripts/05_conformal_calibration.py \\
        --config config/config.yaml \\
        --ensemble-dir results/ensemble \\
        --pseudobulk-dir data/processed \\
        --output-dir results/conformal \\
        --seed 42

Outputs
-------
- ``results/conformal/intervals_conformal.csv`` (raw + clipped in one table)
- ``results/conformal/coverage_by_nominal.csv``
- ``results/conformal/calibration_quantiles.csv``
- ``results/figures/calibration_curve.png/pdf``
- ``results/figures/conformal_interval_widths.png/pdf``
- ``logs/05_conformal_calibration_*.log``
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
from src.uncertainty.conformal import (
    calibrate, predict_intervals, evaluate_calibration,
)
from src.evaluation.plotting import plot_calibration_curve, plot_interval_widths


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Conformal calibration")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--ensemble-dir", default="results/ensemble")
    parser.add_argument("--pseudobulk-dir", default="data/processed")
    parser.add_argument("--output-dir", default="results/conformal")
    parser.add_argument("--gene-set", choices=["all", "hvg3000", "markers"], default="hvg3000",
                        help="Informational: which gene panel the ensemble used (primary: markers).")
    parser.add_argument("--cell-type-set", choices=["7type", "5type"], default="7type",
                        help="Selects the true-proportion files (primary: 5type).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("05_conformal", log_dir=str(PROJECT_ROOT / "logs"))
    set_seed(args.seed)

    results_dir = str(PROJECT_ROOT / args.output_dir)
    fig_dir = str(PROJECT_ROOT / "results" / "figures")
    ensure_dir(results_dir)
    ensure_dir(fig_dir)

    conf_cfg = cfg["conformal"]
    nominal_coverages = conf_cfg["nominal_coverages"]
    score_fn = conf_cfg["nonconformity_score"]
    per_ct = conf_cfg["per_cell_type"]
    do_clip = conf_cfg["clip_intervals"]

    # Load ensemble summary
    ens_dir = Path(args.ensemble_dir)
    if not ens_dir.is_absolute():
        ens_dir = PROJECT_ROOT / ens_dir
    summary = pd.read_csv(ens_dir / "ensemble_summary.csv")
    y_pred_mean = summary.pivot(index="sample_id", columns="cell_type", values="mean")
    y_pred_std = summary.pivot(index="sample_id", columns="cell_type", values="std")

    # Load true proportions (cell-type-set selects 7type vs 5type files)
    from src.deconvolution.celltypes import resolve_inputs
    pb_dir = Path(args.pseudobulk_dir)
    if not pb_dir.is_absolute():
        pb_dir = PROJECT_ROOT / pb_dir
    paths = resolve_inputs(args.gene_set, args.cell_type_set, pb_dir)
    logger.info("Cell-type set: %s | true props: %s", args.cell_type_set, paths["true_cal"].name)
    y_cal_true = pd.read_csv(paths["true_cal"], index_col=0)
    y_test_true = pd.read_csv(paths["true_test"], index_col=0)

    # Align
    common_cal = y_cal_true.index.intersection(y_pred_mean.index)
    common_test = y_test_true.index.intersection(y_pred_mean.index)
    y_cal_true = y_cal_true.loc[common_cal]
    y_pred_mean_cal = y_pred_mean.loc[common_cal]
    y_pred_std_cal = y_pred_std.loc[common_cal]
    y_test_true = y_test_true.loc[common_test]
    y_pred_mean_test = y_pred_mean.loc[common_test]
    y_pred_std_test = y_pred_std.loc[common_test]

    logger.info("=" * 60)
    logger.info("Phase 4: Split Conformal Calibration")
    logger.info("  Calibration set: %d samples", len(y_cal_true))
    logger.info("  Test set: %d samples", len(y_test_true))
    logger.info("  Nominal coverages: %s", nominal_coverages)
    logger.info("  Score function: %s", score_fn)
    logger.info("  Per cell type: %s", per_ct)
    logger.info("  Clip intervals: %s", do_clip)
    logger.info("=" * 60)

    # Calibrate
    cal_result = calibrate(
        y_true_cal=y_cal_true,
        y_pred_cal=y_pred_mean_cal,
        y_std_cal=y_pred_std_cal if score_fn == "normalized_residual" else None,
        nominal_coverages=nominal_coverages,
        score_function=score_fn,
        per_cell_type=per_ct,
    )
    save_df(cal_result["quantiles"], Path(results_dir) / "calibration_quantiles.csv")
    save_df(cal_result["scores"], Path(results_dir) / "calibration_scores.csv")

    # Predict intervals (BOTH raw and clipped)
    intervals = predict_intervals(
        y_pred_test=y_pred_mean_test,
        calibration_quantiles=cal_result["quantiles"],
        y_std_test=y_pred_std_test if score_fn == "normalized_residual" else None,
        score_function=score_fn,
        clip=do_clip,
    )
    save_df(intervals, Path(results_dir) / "intervals_conformal.csv")

    # Evaluate (raw + clip)
    coverage_metrics = evaluate_calibration(intervals, y_test_true)
    save_df(coverage_metrics, Path(results_dir) / "coverage_by_nominal.csv")

    # Report both raw and clip
    logger.info("Coverage results (overall):")
    overall = coverage_metrics[coverage_metrics["cell_type"] == "overall"]
    for _, row in overall.iterrows():
        logger.info(
            "  nominal=%.0f%% | raw coverage=%.4f width=%.4f | clip coverage=%.4f width=%.4f",
            row["nominal_coverage"] * 100,
            row["empirical_coverage_raw"], row["mean_interval_width_raw"],
            row["empirical_coverage_clip"], row["mean_interval_width_clip"],
        )

    # QC
    qc_results = [
        run_qc_summary(intervals, label="intervals", logger=logger),
        run_qc_summary(coverage_metrics, label="coverage_metrics", logger=logger),
    ]
    qc_report_all(qc_results, logger=logger)

    # Check: is 90% coverage close to 90%?
    target_row = coverage_metrics[
        (coverage_metrics["nominal_coverage"] == 0.90) & (coverage_metrics["cell_type"] == "overall")
    ]
    if len(target_row) > 0:
        emp_cov = target_row["empirical_coverage_clip"].values[0]
        if abs(emp_cov - 0.90) > 0.05:
            logger.warning("[WARN]  90%% nominal coverage but empirical=%.4f — check calibration!", emp_cov)
        else:
            logger.info("[OK]  90%% nominal coverage: empirical=%.4f", emp_cov)

    # Figures
    plot_calibration_curve(
        coverage_metrics,
        title="Conformal Calibration: Nominal vs Empirical Coverage",
        save_path=str(Path(fig_dir) / "calibration_curve"),
    )
    plot_interval_widths(
        intervals,
        title="Conformal: Interval Width by Cell Type (90% coverage)",
        save_path=str(Path(fig_dir) / "conformal_interval_widths"),
    )

    logger.info("Phase 4 complete. [OK] ")
    logger.info("Results in %s", results_dir)


if __name__ == "__main__":
    main()
