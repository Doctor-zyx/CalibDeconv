"""
Non-negative least squares (NNLS) deconvolution.

Given a bulk expression vector **b** (genes × 1) and a signature matrix
**S** (genes × cell_types), NNLS solves:

    min ||S·x - b||₂   subject to   x ≥ 0

The solution **x** is then optionally normalised so that Σx = 1,
giving estimated cell-type proportions.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import nnls

logger = logging.getLogger(__name__)


def deconvolve_nnls(
    bulk: np.ndarray,
    signature: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """Solve a single NNLS deconvolution problem.

    Parameters
    ----------
    bulk : np.ndarray
        1-D array of bulk gene expression (n_genes,).
    signature : np.ndarray
        2-D signature matrix (n_genes, n_cell_types).
    normalize : bool
        If True, normalise proportions to sum to 1.

    Returns
    -------
    np.ndarray
        Estimated cell-type proportions (n_cell_types,).
    """
    x, residual = nnls(signature, bulk, maxiter=10000)
    if normalize and x.sum() > 0:
        x = x / x.sum()
    return x


def deconvolve_batch(
    bulk_matrix: pd.DataFrame,
    signature: pd.DataFrame,
    normalize: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run NNLS deconvolution on every sample in a bulk matrix.

    Parameters
    ----------
    bulk_matrix : pd.DataFrame
        Bulk expression, shape (n_samples, n_genes).
    signature : pd.DataFrame
        Signature matrix, shape (n_genes, n_cell_types).
    normalize : bool
        Normalise each sample's proportions to sum to 1.
    verbose : bool
        Log progress every 100 samples.

    Returns
    -------
    pd.DataFrame
        Predicted proportions, shape (n_samples, n_cell_types).
    """
    # Align genes
    common_genes = bulk_matrix.columns.intersection(signature.index)
    if len(common_genes) == 0:
        raise ValueError(
            "No common genes between bulk and signature. "
            "Check gene naming conventions."
        )
    logger.info("Deconvolving with %d common genes", len(common_genes))

    bulk_aligned = bulk_matrix[common_genes].values
    sig_aligned = signature.loc[common_genes].values
    cell_types = signature.columns.tolist()

    n_samples = bulk_aligned.shape[0]
    n_types = len(cell_types)
    predictions = np.zeros((n_samples, n_types), dtype=np.float64)

    for i in range(n_samples):
        predictions[i] = deconvolve_nnls(
            bulk_aligned[i], sig_aligned, normalize=normalize
        )
        if verbose and (i + 1) % 500 == 0:
            logger.info("  Deconvolved %d / %d samples", i + 1, n_samples)

    result = pd.DataFrame(predictions, index=bulk_matrix.index, columns=cell_types)
    result.index.name = "sample_id"
    result.columns.name = "cell_type"
    return result


def deconvolve_with_cell_subsampling(
    bulk_vector: np.ndarray,
    adata: "AnnData",
    cell_type_column: str,
    cell_types: list,
    gene_list: list,
    cell_fraction: float = 0.8,
    gene_fraction: float = 0.8,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Single-iteration ensemble deconvolution with cell + gene subsampling.

    This is the workhorse used inside the ensemble loop (Phase 3).

    1. Randomly sample ``cell_fraction`` of reference cells per cell type.
    2. Build a signature from the sampled cells.
    3. Randomly sample ``gene_fraction`` of genes from signature and bulk.
    4. Run NNLS and return proportions.

    Parameters
    ----------
    bulk_vector : np.ndarray
        1-D bulk expression (n_genes,).
    adata : AnnData
        Reference single-cell data.
    cell_type_column : str
        Column in ``adata.obs`` with cell-type labels.
    cell_types : list of str
        Cell types to include.
    gene_list : list of str
        Gene names corresponding to *bulk_vector*.
    cell_fraction : float
        Fraction of reference cells to keep per cell type.
    gene_fraction : float
        Fraction of genes to keep.
    rng : np.random.Generator or None

    Returns
    -------
    np.ndarray
        Estimated proportions (len(cell_types),), normalised to sum to 1.
    """
    if rng is None:
        rng = np.random.default_rng()

    # 1. Gene subsampling
    n_genes = len(gene_list)
    n_gene_keep = max(10, int(n_genes * gene_fraction))
    gene_idx = rng.choice(n_genes, size=n_gene_keep, replace=False)

    # 2. Per-cell-type cell subsampling + build mini-signature
    n_types = len(cell_types)
    mini_sig = np.zeros((n_gene_keep, n_types), dtype=np.float64)

    for j, ct in enumerate(cell_types):
        mask = adata.obs[cell_type_column] == ct
        expr = adata[mask].X
        if sparse.issparse(expr):
            expr = expr.toarray()
        n_cells = expr.shape[0]
        n_cell_keep = max(1, int(n_cells * cell_fraction))
        chosen = rng.choice(n_cells, size=n_cell_keep, replace=False)
        mini_sig[:, j] = expr[chosen][:, gene_idx].mean(axis=0)

    # 3. Subsample bulk
    bulk_sub = bulk_vector[gene_idx]

    # 4. NNLS
    return deconvolve_nnls(bulk_sub, mini_sig, normalize=True)


# Import for type hints
from scipy import sparse
