"""
Publication-quality visualisation for CalibDeconv.

All figures use a consistent style and are saved in both PNG and PDF
formats.  Every function returns the figure and axes for further
customisation.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

logger = logging.getLogger(__name__)

# ── Global style ───────────────────────────────────────────────────────────

_STYLE_CONFIG = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
}


def set_style() -> None:
    """Apply the CalibDeconv house style."""
    sns.set_style("ticks")
    matplotlib.rcParams.update(_STYLE_CONFIG)
    # Use a clean colour palette
    sns.set_palette("colorblind")


set_style()  # apply on import


# ── Figure 1: True vs Predicted scatter ────────────────────────────────────

def plot_true_vs_predicted(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    title: str = "True vs Predicted Proportions",
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Scatter plot of true vs predicted cell-type proportions."""
    true_vals = y_true.values.flatten()
    pred_vals = y_pred.values.flatten()

    # Remove zero-true zero-pred pairs (mostly artifacts)
    mask = ~((true_vals == 0) & (pred_vals == 0))
    true_vals = true_vals[mask]
    pred_vals = pred_vals[mask]

    r, p = stats.pearsonr(true_vals, pred_vals)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(true_vals, pred_vals, alpha=0.3, s=4, edgecolors="none")
    ax.plot([0, 1], [0, 1], "r--", linewidth=0.8, label="y = x")
    ax.set_xlabel("True proportion")
    ax.set_ylabel("Predicted proportion")
    ax.set_title(f"{title}\nPearson r = {r:.4f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_aspect("equal")

    if save_path:
        _save_figure(fig, save_path)
    return fig, ax


# ── Figure 2: Per-cell-type error boxplot ──────────────────────────────────

def plot_error_boxplot(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    title: str = "Error Distribution by Cell Type",
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Boxplot of absolute errors per cell type."""
    errors = np.abs(y_true.values - y_pred.values)
    df = pd.DataFrame(errors, columns=y_true.columns)
    df_long = df.melt(var_name="Cell type", value_name="Absolute error")

    fig, ax = plt.subplots(figsize=(max(6, len(df.columns) * 0.8), 4))
    sns.boxplot(data=df_long, x="Cell type", y="Absolute error", ax=ax)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylabel("Absolute error")

    if save_path:
        _save_figure(fig, save_path)
    return fig, ax


# ── Figure 3: Calibration curve ────────────────────────────────────────────

def plot_calibration_curve(
    coverage_by_nominal: pd.DataFrame,
    title: str = "Calibration Curve",
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Nominal vs empirical coverage (calibration curve)."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0.8, 0.95], [0.8, 0.95], "k--", linewidth=0.8, label="Perfect calibration")

    for ct in coverage_by_nominal["cell_type"].unique():
        subset = coverage_by_nominal[coverage_by_nominal["cell_type"] == ct]
        ax.plot(
            subset["nominal_coverage"],
            subset["empirical_coverage_clip"],
            "o-",
            markersize=4,
            label=ct if ct != "overall" else None,
        )

    # Highlight overall
    overall = coverage_by_nominal[coverage_by_nominal["cell_type"] == "overall"]
    if len(overall) > 0:
        ax.plot(
            overall["nominal_coverage"],
            overall["empirical_coverage_clip"],
            "s-", color="black", linewidth=2, markersize=6, label="Overall",
        )

    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(title)
    ax.set_xlim(0.78, 0.97)
    ax.set_ylim(0.5, 1.02)
    ax.legend(fontsize=7)
    ax.set_aspect("equal")

    if save_path:
        _save_figure(fig, save_path)
    return fig, ax


# ── Figure 4: Interval width by cell type ──────────────────────────────────

def plot_interval_widths(
    intervals: pd.DataFrame,
    title: str = "Prediction Interval Width by Cell Type",
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Boxplot of interval widths per cell type at the default coverage level."""
    # Use 90% coverage by default
    subset = intervals[intervals["nominal_coverage"] == 0.90]

    fig, ax = plt.subplots(figsize=(max(6, len(subset["cell_type"].unique()) * 0.8), 4))
    sns.boxplot(data=subset, x="cell_type", y="interval_width_clip", ax=ax)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylabel("Interval width (90% coverage)")

    if save_path:
        _save_figure(fig, save_path)
    return fig, ax


# ── Figure 5: Uncertainty vs Error ─────────────────────────────────────────

def plot_uncertainty_vs_error(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    y_std: pd.DataFrame,
    cell_type: Optional[str] = None,
    title: str = "Uncertainty vs Absolute Error",
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Scatter of predicted std vs absolute error."""
    if cell_type is not None:
        abs_err = np.abs(y_true[cell_type].values - y_pred[cell_type].values)
        stds = y_std[cell_type].values
    else:
        abs_err = np.abs(y_true.values - y_pred.values).flatten()
        stds = y_std.values.flatten()

    r, p = stats.pearsonr(stds, abs_err)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(stds, abs_err, alpha=0.3, s=4, edgecolors="none")
    # Trend line
    sns.regplot(x=stds, y=abs_err, scatter=False, ax=ax, color="red", line_kws={"linewidth": 1})
    ax.set_xlabel("Predicted uncertainty (std)")
    ax.set_ylabel("Absolute error")
    ax.set_title(f"{title}\nr = {r:.4f} (p = {p:.2e})")

    if save_path:
        _save_figure(fig, save_path)
    return fig, ax


# ── Figure 6: Rejection curve ──────────────────────────────────────────────

def plot_rejection_curve(
    rejection_df: pd.DataFrame,
    title: str = "Error-Rejection Curve",
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot mean error as a function of fraction of (least uncertain) samples retained."""
    baseline_mae = rejection_df["mean_absolute_error"].iloc[0]  # 100% retained

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(
        rejection_df["fraction_retained"],
        rejection_df["mean_absolute_error"],
        "o-", markersize=4, linewidth=1.5,
    )
    ax.axhline(baseline_mae, color="grey", linestyle="--", linewidth=0.8, label=f"All samples (MAE={baseline_mae:.4f})")
    ax.set_xlabel("Fraction of samples retained")
    ax.set_ylabel("Mean absolute error")
    ax.set_title(title)
    ax.invert_xaxis()
    ax.legend(fontsize=8)

    if save_path:
        _save_figure(fig, save_path)
    return fig, ax


# ── Figure 7: Stress coverage heatmap ──────────────────────────────────────

def plot_stress_heatmap(
    stress_df: pd.DataFrame,
    metric: str = "overall_MAE",
    title: str = "Stress Test Summary",
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Heatmap of a performance metric across stress scenarios and severity levels."""
    pivot = stress_df.pivot_table(
        index="scenario", columns="severity", values=metric, aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(max(6, pivot.shape[1] * 1.2), max(4, pivot.shape[0] * 0.6)))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlOrRd", ax=ax, cbar_kws={"label": metric})
    ax.set_title(title)
    ax.set_ylabel("Scenario")
    ax.set_xlabel("Severity")

    if save_path:
        _save_figure(fig, save_path)
    return fig, ax


# ── Helpers ────────────────────────────────────────────────────────────────

def _save_figure(fig: plt.Figure, path: str) -> None:
    """Save figure in PNG and PDF formats."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    base = p.with_suffix("")
    fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    logger.info("Saved figure: %s.{png,pdf}", base)
