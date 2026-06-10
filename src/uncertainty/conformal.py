"""
Split conformal prediction for calibrated prediction intervals.

Given ensemble summary statistics and a held-out calibration set with
known true proportions, this module:

1. Computes nonconformity scores on the calibration set.
2. Derives calibration quantiles (one per cell type or globally).
3. Applies those quantiles to the test set to produce calibrated
   prediction intervals with guaranteed marginal coverage.

**Boundary handling (proportions in [0, 1])**

We output TWO sets of intervals per (sample, cell_type, coverage):

- **raw**  : unclipped  [ŷ - q̂,  ŷ + q̂]
- **clip** : clipped to [0, 1]  [max(0, ŷ - q̂),  min(1, ŷ + q̂)]

Coverage and interval width are reported separately for each version.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Nonconformity score functions ──────────────────────────────────────────

def absolute_error_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """|y_true − y_pred|"""
    return np.abs(y_true - y_pred)


def normalized_residual_score(
    y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    """|y_true − y_pred| / (y_std + ε)"""
    return np.abs(y_true - y_pred) / (y_std + eps)


# ── Main calibration API ───────────────────────────────────────────────────

def calibrate(
    y_true_cal: pd.DataFrame,
    y_pred_cal: pd.DataFrame,
    y_std_cal: Optional[pd.DataFrame] = None,
    nominal_coverages: List[float] = None,
    score_function: str = "absolute_error",
    per_cell_type: bool = True,
) -> Dict:
    """Compute conformal calibration quantiles from a calibration set.

    Returns a dict with ``"quantiles"``, ``"scores"``, ``"metadata"``.
    """
    if nominal_coverages is None:
        nominal_coverages = [0.80, 0.90, 0.95]
    cell_types = y_true_cal.columns.tolist()
    n_cal = len(y_true_cal)

    if score_function == "absolute_error":
        scores = absolute_error_score(y_true_cal.values, y_pred_cal.values)
    elif score_function == "normalized_residual":
        if y_std_cal is None:
            raise ValueError("y_std_cal required for normalized_residual score")
        scores = normalized_residual_score(y_true_cal.values, y_pred_cal.values, y_std_cal.values)
    else:
        raise ValueError(f"Unknown score function: {score_function}")

    scores_df = pd.DataFrame(scores, index=y_true_cal.index, columns=cell_types)

    quantile_records = []
    for coverage in nominal_coverages:
        alpha = 1 - coverage
        if per_cell_type:
            for ct in cell_types:
                ct_scores = scores_df[ct].values
                q_hat = _conformal_quantile(ct_scores, alpha, n_cal)
                quantile_records.append({
                    "nominal_coverage": coverage, "cell_type": ct,
                    "quantile": q_hat, "n_cal": n_cal,
                })
        else:
            all_scores = scores_df.values.flatten()
            q_hat = _conformal_quantile(all_scores, alpha, n_cal * len(cell_types))
            quantile_records.append({
                "nominal_coverage": coverage, "cell_type": "global",
                "quantile": q_hat, "n_cal": n_cal * len(cell_types),
            })

    quantiles_df = pd.DataFrame(quantile_records)
    logger.info("Calibration: %d samples × %d types, %d coverage levels",
                n_cal, len(cell_types), len(nominal_coverages))
    return {"quantiles": quantiles_df, "scores": scores_df,
            "metadata": {"score_function": score_function, "per_cell_type": per_cell_type}}


def predict_intervals(
    y_pred_test: pd.DataFrame,
    calibration_quantiles: pd.DataFrame,
    y_std_test: Optional[pd.DataFrame] = None,
    score_function: str = "absolute_error",
    clip: bool = True,
) -> pd.DataFrame:
    """Apply calibrated quantiles to produce prediction intervals.

    Returns a long-format DataFrame with both raw and clipped intervals.
    Columns: sample_id, cell_type, nominal_coverage,
             mean, lower_raw, upper_raw, interval_width_raw,
             lower_clip, upper_clip, interval_width_clip.
    """
    cell_types = y_pred_test.columns.tolist()
    sample_ids = y_pred_test.index.tolist()
    rows = []

    for coverage, group in calibration_quantiles.groupby("nominal_coverage"):
        q_map = dict(zip(group["cell_type"], group["quantile"]))

        for ct in cell_types:
            q = q_map.get(ct, q_map.get("global", 0))
            means = y_pred_test[ct].values

            if score_function == "normalized_residual" and y_std_test is not None:
                half_widths = q * (y_std_test[ct].values + 1e-8)
            else:
                half_widths = np.full_like(means, q)

            for i, sid in enumerate(sample_ids):
                hw = half_widths[i] if hasattr(half_widths, '__iter__') else half_widths
                mean_val = means[i]
                lower_raw = mean_val - hw
                upper_raw = mean_val + hw
                width_raw = 2 * hw

                lower_clip = max(0.0, lower_raw)
                upper_clip = min(1.0, upper_raw)
                width_clip = upper_clip - lower_clip

                rows.append({
                    "sample_id": sid,
                    "cell_type": ct,
                    "nominal_coverage": coverage,
                    "mean": mean_val,
                    "lower_raw": lower_raw,
                    "upper_raw": upper_raw,
                    "interval_width_raw": width_raw,
                    "lower_clip": lower_clip,
                    "upper_clip": upper_clip,
                    "interval_width_clip": width_clip,
                })

    return pd.DataFrame(rows)


def evaluate_calibration(
    intervals: pd.DataFrame,
    y_true_test: pd.DataFrame,
) -> pd.DataFrame:
    """Compute empirical coverage and width for BOTH raw and clipped intervals.

    Returns a DataFrame with columns for raw_ and clip_ coverage/width.
    """
    # Build true_proportion long table
    if "sample_id" in y_true_test.reset_index().columns:
        true_long = y_true_test.reset_index().melt(
            id_vars="sample_id", var_name="cell_type", value_name="true_proportion"
        )
    else:
        true_long = y_true_test.reset_index().melt(
            id_vars=y_true_test.index.name or "index",
            var_name="cell_type", value_name="true_proportion",
        )
        true_long.rename(columns={y_true_test.index.name or "index": "sample_id"}, inplace=True)

    merged = intervals.merge(true_long, on=["sample_id", "cell_type"])
    merged["covered_raw"] = (merged["true_proportion"] >= merged["lower_raw"]) & (
        merged["true_proportion"] <= merged["upper_raw"]
    )
    merged["covered_clip"] = (merged["true_proportion"] >= merged["lower_clip"]) & (
        merged["true_proportion"] <= merged["upper_clip"]
    )

    # Per (coverage, cell_type)
    metrics = merged.groupby(["nominal_coverage", "cell_type"]).agg(
        empirical_coverage_raw=("covered_raw", "mean"),
        mean_interval_width_raw=("interval_width_raw", "mean"),
        empirical_coverage_clip=("covered_clip", "mean"),
        mean_interval_width_clip=("interval_width_clip", "mean"),
    ).reset_index()

    # Overall
    overall = merged.groupby("nominal_coverage").agg(
        empirical_coverage_raw=("covered_raw", "mean"),
        mean_interval_width_raw=("interval_width_raw", "mean"),
        empirical_coverage_clip=("covered_clip", "mean"),
        mean_interval_width_clip=("interval_width_clip", "mean"),
    ).reset_index()
    overall["cell_type"] = "overall"

    result = pd.concat([metrics, overall], ignore_index=True)
    return result


# ── Internal helpers ───────────────────────────────────────────────────────

def _conformal_quantile(scores: np.ndarray, alpha: float, n_cal: int) -> float:
    """Compute the conformal quantile q̂.

    q̂ = the ⌈(n_cal + 1)(1 − α)⌉-th smallest score.
    """
    level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
    level = min(max(level, 0.0), 1.0)
    return float(np.quantile(scores, level))
