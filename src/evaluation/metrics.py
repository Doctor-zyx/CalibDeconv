"""
Evaluation metrics for deconvolution accuracy and uncertainty.

Accuracy metrics:
- MAE, RMSE (overall and per-cell-type)
- Pearson correlation
- CCC (Lin's Concordance Correlation Coefficient)

Uncertainty metrics:
- Empirical coverage (fraction of true values inside prediction intervals)
- Interval width
- Uncertainty-error correlation (Pearson r between predicted std and abs error)
- Calibration curve data

Failure detection metrics:
- AUROC (can uncertainty detect large errors?)
- Rejection curve (error vs fraction of samples retained)
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score

logger = logging.getLogger(__name__)


# ── Accuracy ───────────────────────────────────────────────────────────────

def compute_accuracy_metrics(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """Compute all accuracy metrics.

    Returns
    -------
    dict
        ``"overall"``: scalar metrics across all (sample, cell_type) pairs.
        ``"per_cell_type"``: metrics per cell type.
    """
    cell_types = y_true.columns.tolist()
    true_vals = y_true.values
    pred_vals = y_pred.values

    # Overall
    overall = {
        "MAE": float(np.mean(np.abs(true_vals - pred_vals))),
        "RMSE": float(np.sqrt(np.mean((true_vals - pred_vals) ** 2))),
        "Pearson_r": float(stats.pearsonr(true_vals.flatten(), pred_vals.flatten())[0]),
        "CCC": _ccc(true_vals.flatten(), pred_vals.flatten()),
    }

    # Per cell type
    per_ct = []
    for j, ct in enumerate(cell_types):
        tv = true_vals[:, j]
        pv = pred_vals[:, j]
        per_ct.append({
            "cell_type": ct,
            "MAE": float(np.mean(np.abs(tv - pv))),
            "RMSE": float(np.sqrt(np.mean((tv - pv) ** 2))),
            "Pearson_r": float(stats.pearsonr(tv, pv)[0]) if tv.std() > 0 and pv.std() > 0 else np.nan,
            "CCC": _ccc(tv, pv),
        })

    return {
        "overall": pd.DataFrame([overall]),
        "per_cell_type": pd.DataFrame(per_ct),
    }


# ── Uncertainty ────────────────────────────────────────────────────────────

def compute_uncertainty_error_correlation(
    y_true: pd.DataFrame,
    y_pred_mean: pd.DataFrame,
    y_pred_std: pd.DataFrame,
) -> pd.DataFrame:
    """Correlation between prediction uncertainty (std) and absolute error.

    Returns per-cell-type and overall Pearson r.
    """
    abs_error = np.abs(y_true.values - y_pred_mean.values)
    cell_types = y_true.columns.tolist()

    rows = []
    for j, ct in enumerate(cell_types):
        ae = abs_error[:, j]
        std = y_pred_std.values[:, j]
        if std.std() > 0:
            r, p = stats.pearsonr(std, ae)
        else:
            r, p = np.nan, np.nan
        rows.append({
            "cell_type": ct,
            "pearson_r": r,
            "p_value": p,
        })

    # Overall
    r, p = stats.pearsonr(y_pred_std.values.flatten(), abs_error.flatten())
    rows.append({
        "cell_type": "overall",
        "pearson_r": r,
        "p_value": p,
    })

    return pd.DataFrame(rows)


# ── Failure detection ──────────────────────────────────────────────────────

def compute_failure_detection_metrics(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    uncertainty: pd.Series,
    error_threshold_quantile: float = 0.75,
) -> Dict:
    """Evaluate whether uncertainty can detect large prediction errors.

    Parameters
    ----------
    y_true, y_pred : pd.DataFrame
        True and predicted proportions.
    uncertainty : pd.Series
        Per-sample uncertainty scores (higher = more uncertain).
    error_threshold_quantile : float
        Quantile of absolute error above which a prediction is a "failure".

    Returns
    -------
    dict
        ``"auroc"``, ``"auprc"``, ``"threshold"``
    """
    abs_error = np.abs(y_true.values - y_pred.values).mean(axis=1)  # per-sample MAE
    threshold = np.quantile(abs_error, error_threshold_quantile)
    labels = (abs_error > threshold).astype(int)

    # Need at least one positive and one negative
    if labels.sum() == 0 or labels.sum() == len(labels):
        logger.warning("All samples same class — AUROC not computable")
        return {"auroc": np.nan, "auprc": np.nan, "threshold": threshold, "labels": labels}

    auroc = roc_auc_score(labels, uncertainty)
    auprc = average_precision_score(labels, uncertainty)

    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "threshold": float(threshold),
    }


def compute_rejection_curve(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    uncertainty: pd.Series,
    n_points: int = 20,
) -> pd.DataFrame:
    """Compute the error-rejection curve.

    Samples are sorted by uncertainty (descending).  We progressively
    remove the most uncertain samples and compute the mean error of
    the retained set.

    Returns
    -------
    pd.DataFrame
        Columns: fraction_retained, mean_absolute_error.
    """
    abs_error = np.abs(y_true.values - y_pred.values).mean(axis=1)
    order = np.argsort(-uncertainty.values)  # descending uncertainty
    n = len(order)

    rows = []
    for i in range(n_points + 1):
        frac = 1.0 - i / n_points
        n_keep = max(1, int(n * frac))
        retained = order[:n_keep]
        mae = abs_error[retained].mean()
        rows.append({
            "fraction_retained": frac,
            "n_retained": n_keep,
            "mean_absolute_error": float(mae),
        })

    return pd.DataFrame(rows)


# ── Helpers ────────────────────────────────────────────────────────────────

def _ccc(x: np.ndarray, y: np.ndarray) -> float:
    """Lin's Concordance Correlation Coefficient."""
    mean_x, mean_y = np.mean(x), np.mean(y)
    var_x, var_y = np.var(x, ddof=1), np.var(y, ddof=1)
    cov_xy = np.cov(x, y, ddof=1)[0, 1]
    # Handle edge case of zero variance
    denom = var_x + var_y + (mean_x - mean_y) ** 2
    if denom == 0:
        return 1.0 if np.allclose(x, y) else 0.0
    return float(2 * cov_xy / denom)


def compute_all_metrics(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    y_std: Optional[pd.DataFrame] = None,
    uncertainty_scores: Optional[pd.Series] = None,
) -> Dict:
    """Convenience function: compute all metrics in one call.

    Returns a dict with all metric DataFrames.
    """
    results = {}
    results["accuracy"] = compute_accuracy_metrics(y_true, y_pred)

    if y_std is not None:
        results["uncertainty_error_corr"] = compute_uncertainty_error_correlation(
            y_true, y_pred, y_std
        )

    if uncertainty_scores is not None:
        fd = compute_failure_detection_metrics(y_true, y_pred, uncertainty_scores)
        results["failure_detection"] = fd
        results["rejection_curve"] = compute_rejection_curve(
            y_true, y_pred, uncertainty_scores
        )

    return results
