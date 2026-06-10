"""
Ensemble uncertainty estimation via bootstrap perturbations.

For each pseudo-bulk sample we run B iterations of deconvolution with:
- Random gene subsampling (default 80% of genes)
- Random cell subsampling (default 80% of reference cells per cell type)
- Optional Gaussian noise injection on bulk expression

This produces an empirical distribution of predicted proportions for each
(sample, cell_type).  Summary statistics (mean, std, quantiles, interval
width) are computed from this distribution.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..deconvolution.nnls import deconvolve_with_cell_subsampling

logger = logging.getLogger(__name__)


def run_ensemble(
    bulk_matrix: pd.DataFrame,
    adata: "AnnData",
    cell_type_column: str,
    cell_types: List[str],
    n_iterations: int = 100,
    gene_fraction: float = 0.8,
    cell_fraction: float = 0.8,
    noise_std: float = 0.0,
    quantiles: Tuple[float, ...] = (0.05, 0.25, 0.75, 0.95),
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Run ensemble uncertainty estimation for all samples.

    Parameters
    ----------
    bulk_matrix : pd.DataFrame
        Pseudo-bulk expression matrix, (n_samples, n_genes).
    adata : AnnData
        Reference single-cell data.
    cell_type_column : str
        Column in ``adata.obs`` with cell-type labels.
    cell_types : list of str
        Cell types to include.
    n_iterations : int
        Number of bootstrap iterations (B).
    gene_fraction : float
        Fraction of genes sampled per iteration.
    cell_fraction : float
        Fraction of reference cells sampled per iteration, per cell type.
    noise_std : float
        Std of additive Gaussian noise on bulk expression (0 = off).
    quantiles : tuple of float
        Quantiles to compute from the empirical distribution.
    seed : int
        Random seed.

    Returns
    -------
    dict
        ``"ensemble_predictions"``: 3-D array (n_iterations, n_samples, n_cell_types)
        ``"summary"``: DataFrame with mean, std, quantiles, interval_width
        ``"raw_intervals"``: 5%/95% intervals before calibration
        ``"coverage_raw"``: empirical coverage of raw intervals
    """
    rng = np.random.default_rng(seed)
    gene_list = bulk_matrix.columns.tolist()
    bulk_values = bulk_matrix.values
    n_samples, _ = bulk_values.shape
    n_types = len(cell_types)

    # 3-D storage: (iterations, samples, cell_types)
    all_preds = np.zeros((n_iterations, n_samples, n_types), dtype=np.float32)

    t0 = time.time()
    for it in range(n_iterations):
        iter_rng = np.random.default_rng(seed + it * 10000 + 1)
        for s in range(n_samples):
            bulk_vec = bulk_values[s].copy()

            # Optional noise
            if noise_std > 0:
                bulk_vec += iter_rng.normal(0, noise_std * bulk_vec.std(), size=len(bulk_vec))
                bulk_vec = np.clip(bulk_vec, 0, None)

            all_preds[it, s] = deconvolve_with_cell_subsampling(
                bulk_vector=bulk_vec,
                adata=adata,
                cell_type_column=cell_type_column,
                cell_types=cell_types,
                gene_list=gene_list,
                cell_fraction=cell_fraction,
                gene_fraction=gene_fraction,
                rng=iter_rng,
            )

        elapsed = time.time() - t0
        if (it + 1) % 10 == 0 or it == 0:
            logger.info(
                "  Iteration %d/%d (%.1fs elapsed, ~%.0fs remaining)",
                it + 1, n_iterations, elapsed,
                elapsed / (it + 1) * (n_iterations - it - 1),
            )

    # Compute summary statistics
    summary = _compute_summary(all_preds, cell_types, bulk_matrix.index, quantiles)
    raw_intervals = _make_raw_intervals(summary)
    coverage_raw = _compute_raw_coverage(summary)

    logger.info("Ensemble complete: %d iterations × %d samples", n_iterations, n_samples)
    return {
        "ensemble_predictions": all_preds,
        "summary": summary,
        "raw_intervals": raw_intervals,
        "coverage_raw": coverage_raw,
    }


def _compute_summary(
    all_preds: np.ndarray,
    cell_types: List[str],
    sample_ids: pd.Index,
    quantiles: Tuple[float, ...],
) -> pd.DataFrame:
    """Compute per-(sample, cell_type) summary statistics."""
    n_iter, n_samples, n_types = all_preds.shape
    rows = []

    for s in range(n_samples):
        for t in range(n_types):
            dist = all_preds[:, s, t]
            row = {
                "sample_id": sample_ids[s],
                "cell_type": cell_types[t],
                "mean": np.mean(dist),
                "std": np.std(dist),
                "interval_width": np.percentile(dist, 95) - np.percentile(dist, 5),
            }
            for q in quantiles:
                row[f"q{q:.2f}"] = np.quantile(dist, q)
            rows.append(row)

    return pd.DataFrame(rows)


def _make_raw_intervals(summary: pd.DataFrame) -> pd.DataFrame:
    """Extract 5%–95% raw prediction intervals."""
    piv = summary.pivot_table(
        index="sample_id", columns="cell_type",
        values=["q0.05", "q0.95"], aggfunc="first",
    )
    return piv


def _compute_raw_coverage(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Compute how often the mean falls inside the raw interval.

    This is a placeholder — actual coverage is computed against true
    proportions in the evaluation module.
    """
    return summary.groupby("cell_type").agg(
        mean_width=("interval_width", "mean"),
        mean_std=("std", "mean"),
    ).reset_index()
