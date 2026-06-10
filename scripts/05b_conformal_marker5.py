#!/usr/bin/env python
"""
Phase 4 (PRIMARY line): split conformal calibration for marker_5type.

Calibrates prediction intervals around the Phase 3 ensemble point estimate
using split conformal prediction on the calibration set, then evaluates
marginal coverage on the test set at nominal levels 80/90/95%.

Reuses src/uncertainty/conformal.py (calibrate / predict_intervals /
evaluate_calibration). Strict alignment: every true/pred/interval table is
reindexed to the true-proportion sample_id order and canonical column order;
metrics are computed only after index/column asserts pass.

PRIMARY line:
  gene_set = markers | cell_type_set = 5type
  cell types = Monocyte, NK, B, DC, T_cell
  point estimate = Phase 3 ensemble mean (recorded in calibration_summary.csv)
  uncertainty source = Phase 3 ensemble summary (std / q0.05 / q0.95 / width)

Does NOT run stress test (Phase 5).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.io import setup_logger, set_seed, ensure_dir
from src.uncertainty.conformal import calibrate, predict_intervals, evaluate_calibration

CANON = ["Monocyte", "NK", "B", "DC", "T_cell"]


# ── Helpers (strict sample_id + canonical-column alignment) ──────────────────

def load_true_proportions(proc_dir, split):
    """Load true proportions for a split, columns forced to CANON order."""
    df = pd.read_csv(Path(proc_dir) / f"true_proportions_{split}_5type.csv", index_col=0)
    assert set(df.columns) == set(CANON), f"{split} true cols {list(df.columns)} != CANON"
    df = df[CANON]
    df.index.name = "sample_id"
    return df


def load_ensemble_prediction(ens_dir, split):
    """Load Phase 3 ensemble-mean prediction (has sample_id index), CANON cols."""
    df = pd.read_csv(Path(ens_dir) / f"predicted_proportions_{split}_ensemble.csv", index_col=0)
    assert set(df.columns) == set(CANON), f"{split} pred cols {list(df.columns)} != CANON"
    df = df[CANON]
    df.index.name = "sample_id"
    return df


def pivot_ensemble_summary(summary, stat, true_index, true_columns):
    """Long ensemble summary -> wide (sample_id x cell_type) for one stat.

    Reindexed to (true_index, true_columns) via .loc — NEVER positional
    relabeling. Asserts exact index/column match (guards lexicographic scramble).
    """
    wide = summary.pivot_table(index="sample_id", columns="cell_type",
                               values=stat, aggfunc="first")
    wide = wide.loc[true_index, true_columns]
    assert list(wide.index) == list(true_index), f"{stat}: index != true index"
    assert list(wide.columns) == list(true_columns), f"{stat}: columns != true columns"
    return wide


def align_and_assert(y_true, y_pred, name):
    """Return y_true aligned to y_pred; assert index AND columns identical."""
    y_al = y_true.loc[y_pred.index, y_pred.columns]
    assert list(y_al.index) == list(y_pred.index), f"{name}: sample index mismatch"
    assert list(y_al.columns) == list(y_pred.columns), f"{name}: cell type column mismatch"
    return y_al


def save_with_sample_id(df, path):
    """Save a wide table preserving sample_id index."""
    out = df.copy()
    out.index.name = "sample_id"
    out.to_csv(path)


def parse_args():
    p = argparse.ArgumentParser(description="Phase 4 conformal calibration (marker_5type)")
    p.add_argument("--gene-set", choices=["markers"], default="markers")
    p.add_argument("--cell-type-set", choices=["5type"], default="5type")
    p.add_argument("--proc-dir", default="data/processed")
    p.add_argument("--ensemble-dir", default="results/ensemble_marker_5types")
    p.add_argument("--output-dir", default="results/conformal_marker_5types")
    p.add_argument("--nominal-levels", type=float, nargs="+", default=[0.8, 0.9, 0.95])
    p.add_argument("--point-estimate-source", choices=["ensemble_mean", "phase2_baseline"],
                   default="ensemble_mean")
    p.add_argument("--score-function", choices=["absolute_error", "normalized_residual"],
                   default="absolute_error")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("05b_conformal_m5", log_dir=str(PROJECT_ROOT / "logs"))
    set_seed(args.seed)

    proc_dir = PROJECT_ROOT / args.proc_dir
    ens_dir = PROJECT_ROOT / args.ensemble_dir
    out_dir = ensure_dir(str(PROJECT_ROOT / args.output_dir))
    fig_dir = ensure_dir(str(PROJECT_ROOT / "results" / "figures"))

    logger.info("=" * 60)
    logger.info("Phase 4 conformal: gene_set=%s cell_type_set=%s", args.gene_set, args.cell_type_set)
    logger.info("  nominal levels: %s", args.nominal_levels)
    logger.info("  point estimate source: %s", args.point_estimate_source)
    logger.info("  score function: %s", args.score_function)
    logger.info("=" * 60)

    # ── Step 3: load data ──
    y_cal = load_true_proportions(proc_dir, "cal")
    y_test = load_true_proportions(proc_dir, "test")
    logger.info("True props: cal %s, test %s, cols=%s", y_cal.shape, y_test.shape, list(y_cal.columns))

    # Point estimate = Phase 3 ensemble mean (recorded for provenance).
    pred_cal = load_ensemble_prediction(ens_dir, "cal").loc[y_cal.index, CANON]
    pred_test = load_ensemble_prediction(ens_dir, "test").loc[y_test.index, CANON]
    assert list(pred_cal.index) == list(y_cal.index) and list(pred_cal.columns) == CANON
    assert list(pred_test.index) == list(y_test.index) and list(pred_test.columns) == CANON

    # Ensemble summaries (long) -> wide std/q05/q95/width, reindexed to true.
    sum_cal = pd.read_csv(ens_dir / "ensemble_summary_cal.csv")
    sum_test = pd.read_csv(ens_dir / "ensemble_summary_test.csv")
    std_cal = pivot_ensemble_summary(sum_cal, "std", y_cal.index, CANON)
    std_test = pivot_ensemble_summary(sum_test, "std", y_test.index, CANON)
    q05_test = pivot_ensemble_summary(sum_test, "q0.05", y_test.index, CANON)
    q95_test = pivot_ensemble_summary(sum_test, "q0.95", y_test.index, CANON)
    width_test = pivot_ensemble_summary(sum_test, "interval_width", y_test.index, CANON)
    logger.info("Ensemble summary wide tables built and aligned (std/q05/q95/width).")

    # ── Step 4: conformal calibration ──
    std_cal_arg = std_cal if args.score_function == "normalized_residual" else None
    std_test_arg = std_test if args.score_function == "normalized_residual" else None

    cal_res = calibrate(
        y_true_cal=y_cal, y_pred_cal=pred_cal, y_std_cal=std_cal_arg,
        nominal_coverages=args.nominal_levels, score_function=args.score_function,
        per_cell_type=True,
    )
    quantiles = cal_res["quantiles"]
    scores = cal_res["scores"]  # wide (sample_id x cell_type) nonconformity scores
    save_with_sample_id(scores, Path(out_dir) / "nonconformity_scores.csv")
    quantiles.to_csv(Path(out_dir) / "conformal_quantiles.csv", index=False)
    logger.info("Calibrated %d quantile records over %d levels", len(quantiles), len(args.nominal_levels))

    # Intervals on test (long format: raw + clip together)
    intervals = predict_intervals(
        y_pred_test=pred_test, calibration_quantiles=quantiles,
        y_std_test=std_test_arg, score_function=args.score_function, clip=True,
    )
    # Split raw vs clipped wide files (sample_id x [cell_type x bound] per level)
    raw_cols = ["sample_id", "cell_type", "nominal_coverage", "mean",
                "lower_raw", "upper_raw", "interval_width_raw"]
    clip_cols = ["sample_id", "cell_type", "nominal_coverage", "mean",
                 "lower_clip", "upper_clip", "interval_width_clip"]
    intervals[raw_cols].to_csv(Path(out_dir) / "intervals_test_raw.csv", index=False)
    intervals[clip_cols].to_csv(Path(out_dir) / "intervals_test_clipped.csv", index=False)

    # Coverage evaluation (raw + clip, per cell type + overall)
    cov = evaluate_calibration(intervals, y_test)
    cov_overall = cov[cov["cell_type"] == "overall"].copy()
    cov_per_ct = cov[cov["cell_type"] != "overall"].copy()
    cov_overall.to_csv(Path(out_dir) / "coverage_by_nominal.csv", index=False)
    cov_per_ct.to_csv(Path(out_dir) / "coverage_by_cell_type.csv", index=False)

    # Interval width summary (raw + clip, per cell type + overall, per level)
    width_rows = []
    for lvl, g in intervals.groupby("nominal_coverage"):
        for ct in CANON:
            gct = g[g["cell_type"] == ct]
            width_rows.append({
                "nominal_coverage": lvl, "cell_type": ct,
                "mean_width_raw": gct["interval_width_raw"].mean(),
                "mean_width_clip": gct["interval_width_clip"].mean(),
            })
        width_rows.append({
            "nominal_coverage": lvl, "cell_type": "overall",
            "mean_width_raw": g["interval_width_raw"].mean(),
            "mean_width_clip": g["interval_width_clip"].mean(),
        })
    pd.DataFrame(width_rows).to_csv(Path(out_dir) / "interval_width_summary.csv", index=False)

    # Calibration summary (provenance + headline coverage)
    summ_rows = []
    for _, r in cov_overall.iterrows():
        summ_rows.append({
            "nominal_coverage": r["nominal_coverage"],
            "empirical_coverage_raw": r["empirical_coverage_raw"],
            "empirical_coverage_clip": r["empirical_coverage_clip"],
            "mean_interval_width_raw": r["mean_interval_width_raw"],
            "mean_interval_width_clip": r["mean_interval_width_clip"],
            "point_estimate_source": args.point_estimate_source,
            "uncertainty_source": "phase3_ensemble_summary",
            "score_function": args.score_function,
            "n_cal": len(y_cal), "n_test": len(y_test),
            "gene_set": args.gene_set, "cell_type_set": args.cell_type_set,
        })
    pd.DataFrame(summ_rows).to_csv(Path(out_dir) / "calibration_summary.csv", index=False)
    logger.info("Step 4 outputs written to %s", out_dir)

    # ── Step 5: figures ──
    levels = sorted(args.nominal_levels)

    # 1. coverage vs nominal (raw + clip), with y=x reference
    plt.figure(figsize=(6, 5))
    co = cov_overall.sort_values("nominal_coverage")
    plt.plot(co["nominal_coverage"], co["empirical_coverage_raw"], "o-", label="raw")
    plt.plot(co["nominal_coverage"], co["empirical_coverage_clip"], "s-", label="clipped")
    plt.plot([0.75, 1.0], [0.75, 1.0], "k--", lw=1, label="ideal")
    plt.xlabel("Nominal coverage")
    plt.ylabel("Empirical coverage (test)")
    plt.title("Conformal coverage vs nominal (marker_5type)")
    plt.legend()
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(Path(fig_dir) / f"conf5_coverage_vs_nominal.{ext}", dpi=150)
    plt.close()

    # 2. interval width by cell type (clipped) at 90%
    lvl90 = 0.90 if 0.90 in levels else levels[len(levels) // 2]
    g90 = intervals[intervals["nominal_coverage"] == lvl90]
    plt.figure(figsize=(6, 5))
    data = [g90[g90["cell_type"] == ct]["interval_width_clip"].values for ct in CANON]
    plt.boxplot(data, labels=CANON)
    plt.ylabel(f"Clipped interval width @ {int(lvl90*100)}%")
    plt.title("Conformal interval width by cell type (marker_5type)")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(Path(fig_dir) / f"conf5_interval_width_by_celltype.{ext}", dpi=150)
    plt.close()

    # 3. true vs interval examples (first 12 test samples, T_cell, @90%)
    ex_ct = "T_cell"
    ex = g90[g90["cell_type"] == ex_ct].head(12).reset_index(drop=True)
    ex_true = y_test.loc[ex["sample_id"].values, ex_ct].values
    plt.figure(figsize=(7, 5))
    xpos = np.arange(len(ex))
    plt.errorbar(xpos, ex["mean"].values,
                 yerr=[ex["mean"].values - ex["lower_clip"].values,
                       ex["upper_clip"].values - ex["mean"].values],
                 fmt="o", capsize=3, label="pred ± conformal")
    plt.scatter(xpos, ex_true, color="red", marker="x", s=60, label="true", zorder=5)
    plt.xticks(xpos, ex["sample_id"].values, rotation=45, ha="right", fontsize=7)
    plt.ylabel(f"{ex_ct} proportion")
    plt.title(f"True vs conformal interval @ {int(lvl90*100)}% ({ex_ct}, first 12)")
    plt.legend()
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(Path(fig_dir) / f"conf5_true_vs_interval_examples.{ext}", dpi=150)
    plt.close()

    # 4. raw ensemble width vs conformal width by cell type (@90%, clipped)
    plt.figure(figsize=(6, 5))
    raw_w = [width_test[ct].mean() for ct in CANON]  # phase3 ensemble q95-q05 width
    conf_w = [g90[g90["cell_type"] == ct]["interval_width_clip"].mean() for ct in CANON]
    xpos = np.arange(len(CANON))
    plt.bar(xpos - 0.2, raw_w, width=0.4, label="raw ensemble (q95-q05)")
    plt.bar(xpos + 0.2, conf_w, width=0.4, label=f"conformal clipped @{int(lvl90*100)}%")
    plt.xticks(xpos, CANON)
    plt.ylabel("Mean interval width")
    plt.title("Raw ensemble vs conformal interval width (marker_5type)")
    plt.legend()
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(Path(fig_dir) / f"conf5_raw_vs_conformal_width.{ext}", dpi=150)
    plt.close()

    logger.info("Step 5 figures written to %s", fig_dir)

    # ── Step 6: QC summary ──
    # Boundary check on raw intervals
    raw_lower_min = intervals["lower_raw"].min()
    raw_upper_max = intervals["upper_raw"].max()
    clip_lower_min = intervals["lower_clip"].min()
    clip_upper_max = intervals["upper_clip"].max()
    any_below0_raw = bool(raw_lower_min < 0)
    any_above1_raw = bool(raw_upper_max > 1)
    clip_ok = bool(clip_lower_min >= 0 and clip_upper_max <= 1)

    print("\n" + "=" * 70)
    print("PHASE 4 CONFORMAL (marker_5type) — QC SUMMARY")
    print("=" * 70)
    print(f"\n1. Nominal levels: {levels}")
    print(f"   Point estimate source: {args.point_estimate_source} | score: {args.score_function}")
    print(f"   n_cal={len(y_cal)} n_test={len(y_test)}")

    print(f"\n2. Overall coverage (raw / clipped):")
    for _, r in co.iterrows():
        print(f"   nominal={r['nominal_coverage']:.0%}: raw={r['empirical_coverage_raw']:.4f} "
              f"clip={r['empirical_coverage_clip']:.4f}")

    print(f"\n3. Per-cell-type coverage (raw / clipped):")
    for lvl in levels:
        print(f"   @ {lvl:.0%}:")
        sub = cov_per_ct[cov_per_ct["nominal_coverage"] == lvl]
        for ct in CANON:
            row = sub[sub["cell_type"] == ct]
            if len(row):
                print(f"     {ct:<10} raw={row['empirical_coverage_raw'].values[0]:.4f} "
                      f"clip={row['empirical_coverage_clip'].values[0]:.4f}")

    print(f"\n4. Mean interval width (raw / clipped), overall:")
    for _, r in co.iterrows():
        print(f"   nominal={r['nominal_coverage']:.0%}: raw={r['mean_interval_width_raw']:.4f} "
              f"clip={r['mean_interval_width_clip']:.4f}")

    print(f"\n5. Boundary check:")
    print(f"   raw  lower_min={raw_lower_min:.4f} upper_max={raw_upper_max:.4f} "
          f"(lower<0: {any_below0_raw}, upper>1: {any_above1_raw})")
    print(f"   clip lower_min={clip_lower_min:.4f} upper_max={clip_upper_max:.4f} "
          f"(in [0,1]: {clip_ok})")

    print(f"\n6. Coverage near nominal? (clipped, |emp-nominal|<=0.05):")
    for _, r in co.iterrows():
        dev = abs(r["empirical_coverage_clip"] - r["nominal_coverage"])
        print(f"   nominal={r['nominal_coverage']:.0%}: emp_clip={r['empirical_coverage_clip']:.4f} "
              f"dev={dev:.4f} {'OK' if dev <= 0.05 else 'OFF'}")

    print(f"\n7. Alignment checks: all index/column asserts passed (no mismatch).")

    print(f"\n8. Output files:")
    for f in ["nonconformity_scores.csv", "conformal_quantiles.csv",
              "intervals_test_raw.csv", "intervals_test_clipped.csv",
              "coverage_by_nominal.csv", "coverage_by_cell_type.csv",
              "interval_width_summary.csv", "calibration_summary.csv"]:
        p = Path(out_dir) / f
        print(f"   {'[OK]' if p.exists() else '[MISSING]'} {p}")

    print(f"\n9. Figures:")
    for f in ["conf5_coverage_vs_nominal", "conf5_interval_width_by_celltype",
              "conf5_true_vs_interval_examples", "conf5_raw_vs_conformal_width"]:
        p = Path(fig_dir) / f"{f}.png"
        print(f"   {'[OK]' if p.exists() else '[MISSING]'} {p}")

    print("\n" + "=" * 70)
    logger.info("Phase 4 conformal complete.")


if __name__ == "__main__":
    main()
