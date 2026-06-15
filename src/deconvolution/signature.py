"""
Construction of signature matrices from single-cell reference data.

A signature matrix S (genes × cell_types) is built by computing the
mean expression per cell type across all reference cells.  This matrix
is the input to the NNLS deconvolution step.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

logger = logging.getLogger(__name__)


def build_signature_matrix(
    adata: "AnnData",
    cell_type_column: str,
    cell_types: Optional[List[str]] = None,
    min_cells: int = 100,
    min_expression: float = 0.0,
    gene_list: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Build a signature matrix (genes × cell_types) from scRNA-seq reference.

    Parameters
    ----------
    adata : AnnData
        Reference single-cell data.
    cell_type_column : str
        Column in ``adata.obs`` with cell-type labels.
    cell_types : list of str or None
        Cell types to include. If None, uses all types with ≥ min_cells.
    min_cells : int
        Minimum number of cells required per cell type.
    min_expression : float
        Genes with mean expression below this threshold across ALL cell
        types are dropped.
    gene_list : list of str or None
        If given, restrict to these genes (for matching with bulk data).

    Returns
    -------
    signature_df : pd.DataFrame
        Signature matrix, shape (genes, cell_types).
    kept_types : list of str
        Cell types included.
    """
    if cell_types is None:
        counts = adata.obs[cell_type_column].value_counts()
        cell_types = counts[counts >= min_cells].index.tolist()
        logger.info("Auto-selected %d cell types", len(cell_types))
    else:
        # Validate
        available = set(adata.obs[cell_type_column].unique())
        missing = set(cell_types) - available
        if missing:
            raise ValueError(f"Cell types not found in data: {missing}")

    # Restrict to requested genes
    if gene_list is not None:
        adata = adata[:, adata.var_names.isin(gene_list)].copy()
        logger.info("Restricted to %d genes from gene_list", len(gene_list))

    n_genes = adata.shape[1]
    gene_names = adata.var_names.tolist()
    n_types = len(cell_types)
    signature = np.zeros((n_genes, n_types), dtype=np.float64)

    for j, ct in enumerate(cell_types):
        mask = adata.obs[cell_type_column] == ct
        expr = adata[mask].X
        if sparse.issparse(expr):
            expr = expr.toarray()
        signature[:, j] = expr.mean(axis=0)

    # Filter low-expression genes
    if min_expression > 0:
        keep = signature.mean(axis=1) >= min_expression
        signature = signature[keep, :]
        gene_names = [g for g, k in zip(gene_names, keep) if k]
        logger.info(
            "Filtered genes: %d kept, %d removed (min_expression=%.2f)",
            keep.sum(), (~keep).sum(), min_expression,
        )

    signature_df = pd.DataFrame(signature, index=gene_names, columns=cell_types)
    signature_df.index.name = "gene"
    signature_df.columns.name = "cell_type"

    logger.info(
        "Signature matrix built: %d genes × %d cell types", n_genes, n_types
    )
    return signature_df, cell_types


def subset_signature_to_bulk_genes(
    signature: pd.DataFrame,
    bulk_genes: List[str],
) -> pd.DataFrame:
    """Intersect signature genes with the genes present in bulk data.

    Parameters
    ----------
    signature : pd.DataFrame
        Signature matrix (genes × cell_types).
    bulk_genes : list of str
        Genes present in the bulk expression matrix.

    Returns
    -------
    pd.DataFrame
        Subsetted signature matrix.
    """
    common = signature.index.intersection(bulk_genes)
    n_total = len(signature)
    n_common = len(common)
    logger.info(
        "Gene intersection: %d / %d (%.1f%%)",
        n_common, n_total, 100 * n_common / max(n_total, 1),
    )
    if n_common < 10:
        logger.warning("Very few overlapping genes — results may be unreliable!")
    return signature.loc[common]


def subsample_signature(
    signature: pd.DataFrame,
    cell_fraction: float = 0.8,
    rng: np.random.Generator = None,
) -> pd.DataFrame:
    """Randomly subsample genes from a signature matrix (for ensemble).

    Parameters
    ----------
    signature : pd.DataFrame
        Signature matrix (genes × cell_types).
    cell_fraction : float
        Fraction of genes to retain (e.g. 0.8).
    rng : np.random.Generator
        Random state.

    Returns
    -------
    pd.DataFrame
        Subsampled signature matrix (fewer rows).
    """
    if rng is None:
        rng = np.random.default_rng()
    n_genes = signature.shape[0]
    n_keep = max(1, int(n_genes * cell_fraction))
    chosen = rng.choice(signature.index, size=n_keep, replace=False)
    return signature.loc[chosen]
