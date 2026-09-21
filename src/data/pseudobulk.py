"""
Pseudo-bulk generation from single-cell RNA-seq data.

**Cell-pool separation (critical for valid inference)**

Single cells are split into three *disjoint* pools BEFORE any pseudo-bulk
generation or signature construction:

1. **Reference pool**  – used ONLY for building the signature matrix.
2. **Calibration pool** – used ONLY for generating calibration pseudo-bulks.
3. **Test pool**        – used ONLY for generating test pseudo-bulks.

This guarantees that no cell appears in both the reference signature and
any pseudo-bulk sample, preventing information leakage.

**Expression scale consistency**

Pseudo-bulks are aggregated from **raw counts** (``use_raw_counts=True``)
via ``sum``, then normalised jointly (CPM or TPM).  The signature matrix
is built from the reference pool using the SAME normalisation pipeline.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

logger = logging.getLogger(__name__)


# ── Public API ─────────────────────────────────────────────────────────────

def split_cell_pools(
    adata: "AnnData",
    cell_type_column: str,
    ratios: Dict[str, float],
    donor_column: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, "AnnData"]:
    """Split single cells into reference / calibration / test pools.

    Cells are split **independently within each cell type** to maintain
    balanced cell-type composition across pools.  When *donor_column* is
    provided, donor-aware splitting is attempted (avoids splitting cells
    of the same donor across pools, but this requires enough donors).

    Parameters
    ----------
    adata : AnnData
        Full single-cell data with cell-type labels.
    cell_type_column : str
        Column in ``adata.obs`` with cell-type annotations.
    ratios : dict
        e.g. ``{"reference": 0.5, "calibration": 0.25, "test": 0.25}``.
        Must sum to 1.0.
    donor_column : str or None
        Column in ``adata.obs`` with donor/sample IDs (if available).
    seed : int
        Random seed.

    Returns
    -------
    dict
        ``{"reference": AnnData, "calibration": AnnData, "test": AnnData}``
    """
    assert abs(sum(ratios.values()) - 1.0) < 1e-9, f"Ratios must sum to 1, got {ratios}"
    rng = np.random.default_rng(seed)
    cell_types = adata.obs[cell_type_column].unique()
    logger.info("Splitting %d cell types across 3 pools: %s", len(cell_types), ratios)

    pool_indices: Dict[str, list] = {k: [] for k in ratios}

    # ── Donor-aware split (GLOBAL): assign whole donors to one pool each ──
    # This is the strictest leakage control: every cell of a given donor
    # lands in exactly one pool, so donors never overlap across pools.
    if donor_column is not None and donor_column in adata.obs.columns:
        all_donors = adata.obs[donor_column].astype(str)
        unique_donors = np.array(sorted(all_donors.unique()))
        n_donors = len(unique_donors)

        if n_donors >= 3:
            donor_order = rng.permutation(unique_donors)
            n_ref = max(1, int(round(n_donors * ratios["reference"])))
            n_cal = max(1, int(round(n_donors * ratios["calibration"])))
            # Ensure at least 1 donor remains for test
            n_cal = min(n_cal, n_donors - n_ref - 1)
            n_cal = max(1, n_cal)
            ref_donors = set(donor_order[:n_ref])
            cal_donors = set(donor_order[n_ref:n_ref + n_cal])
            test_donors = set(donor_order[n_ref + n_cal:])

            logger.info("Donor-aware GLOBAL split (%d donors):", n_donors)
            logger.info("  reference donors: %s", sorted(ref_donors))
            logger.info("  calibration donors: %s", sorted(cal_donors))
            logger.info("  test donors: %s", sorted(test_donors))

            donor_to_pool = {}
            for d in ref_donors:
                donor_to_pool[d] = "reference"
            for d in cal_donors:
                donor_to_pool[d] = "calibration"
            for d in test_donors:
                donor_to_pool[d] = "test"

            donor_arr = all_donors.values
            for idx in range(adata.shape[0]):
                pool_indices[donor_to_pool[donor_arr[idx]]].append(idx)

            return _build_pool_anndata(adata, pool_indices, cell_type_column)
        else:
            logger.warning(
                "Only %d donors — too few for donor-aware split. "
                "Falling back to cell-level split.", n_donors,
            )

    # ── Cell-level split (per cell type, balanced) ──
    for ct in cell_types:
        ct_mask = adata.obs[cell_type_column] == ct
        ct_indices = np.where(ct_mask)[0]
        n_ct = len(ct_indices)

        perm = rng.permutation(ct_indices)
        n_ref = max(1, int(n_ct * ratios["reference"]))
        n_cal = max(1, int(n_ct * ratios["calibration"]))
        pool_indices["reference"].extend(perm[:n_ref].tolist())
        pool_indices["calibration"].extend(perm[n_ref:n_ref + n_cal].tolist())
        pool_indices["test"].extend(perm[n_ref + n_cal:].tolist())

    return _build_pool_anndata(adata, pool_indices, cell_type_column)


def _build_pool_anndata(adata, pool_indices, cell_type_column):
    """Materialise pool index lists into AnnData subsets."""
    result = {}
    for pool_name, indices in pool_indices.items():
        result[pool_name] = adata[indices].copy()
        logger.info("Pool '%s': %d cells across %d types",
                    pool_name, len(indices), adata[indices].obs[cell_type_column].nunique())
    return result


def generate_pseudobulk_from_pool(
    adata_pool: "AnnData",
    cell_type_column: str,
    n_samples: int,
    total_cells_per_sample: int = 500,
    aggregation: str = "sum",
    dirichlet_alpha: float = 1.0,
    min_cells_per_type: int = 100,
    use_raw_counts: bool = True,
    seed: int = 42,
    output_dir: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Generate pseudo-bulk samples from a single-cell pool.

    This should be called separately for the calibration and test pools.
    **Never** call this on the reference pool — the reference pool is
    used only for the signature matrix.

    Parameters
    ----------
    adata_pool : AnnData
        Single-cell pool (calibration or test).
    cell_type_column : str
        Column in ``adata.obs`` with cell-type labels.
    n_samples : int
        Number of pseudo-bulk samples.
    total_cells_per_sample : int
        Total cells per sample (N).
    aggregation : str
        ``"sum"`` (raw counts) or ``"mean"``.
    dirichlet_alpha : float
        Dirichlet concentration parameter.
    min_cells_per_type : int
        Cell types with fewer cells are excluded.
    use_raw_counts : bool
        If True, use ``adata.raw.X`` for expression values.
    seed : int
        Random seed.
    output_dir : str or None
        If given, write CSV files.

    Returns
    -------
    dict
        ``"pseudobulk_matrix"``, ``"true_proportions"``, ``"cell_type_summary"``
    """
    rng = np.random.default_rng(seed)
    logger.info("Generating %d pseudo-bulk samples from pool (seed=%d)", n_samples, seed)

    # 1. Filter cell types
    adata_pool, kept_types = _filter_cell_types(adata_pool, cell_type_column, min_cells_per_type)
    logger.info("Kept %d cell types after filtering (<%d cells removed)", len(kept_types), min_cells_per_type)

    # 2. Get expression data
    if use_raw_counts and adata_pool.raw is not None:
        logger.info("Using raw counts from adata.raw")
        expr_adata = adata_pool.raw.to_adata()
    else:
        expr_adata = adata_pool

    # 3. Build cell pools (cell_type → array of expression vectors)
    cell_pools = _build_cell_pools(expr_adata, cell_type_column, kept_types)

    # 4. Generate samples
    matrices, proportions = _generate_samples(
        cell_pools=cell_pools,
        cell_types=kept_types,
        n_samples=n_samples,
        total_cells=total_cells_per_sample,
        rng=rng,
        aggregation=aggregation,
        dirichlet_alpha=dirichlet_alpha,
    )

    # 5. Build DataFrames
    gene_names = expr_adata.var_names.tolist()
    pb_df = pd.DataFrame(matrices, columns=gene_names)
    pb_df.index.name = "sample_id"

    prop_df = pd.DataFrame(proportions, columns=list(kept_types))
    prop_df.index.name = "sample_id"

    summary_df = _build_summary(adata_pool, cell_type_column, kept_types)

    # 6. Optional write
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        pb_df.to_csv(out / "pseudobulk_matrix.csv")
        prop_df.to_csv(out / "true_proportions.csv")
        summary_df.to_csv(out / "cell_type_summary.csv")
        logger.info("Pseudo-bulk outputs written to %s", output_dir)

    return {
        "pseudobulk_matrix": pb_df,
        "true_proportions": prop_df,
        "cell_type_summary": summary_df,
    }


def normalize_pseudobulk(
    pseudobulk: pd.DataFrame,
    method: str = "cpm",
) -> pd.DataFrame:
    """Normalise pseudo-bulk expression matrix.

    Parameters
    ----------
    pseudobulk : pd.DataFrame
        Raw aggregated counts (samples × genes).
    method : str
        ``"cpm"`` (counts per million), ``"tpm"``, or ``"none"``.

    Returns
    -------
    pd.DataFrame
        Normalised expression.
    """
    if method == "none":
        return pseudobulk.copy()

    lib_sizes = pseudobulk.values.sum(axis=1)
    if method == "cpm":
        normed = pseudobulk.values / lib_sizes[:, None] * 1e6
    elif method == "tpm":
        # TPM: first divide by gene length, then scale
        # Without gene lengths, TPM ≈ CPM
        logger.warning("TPM requested but no gene lengths available — falling back to CPM")
        normed = pseudobulk.values / lib_sizes[:, None] * 1e6
    else:
        raise ValueError(f"Unknown normalisation method: {method}")

    result = pd.DataFrame(normed, index=pseudobulk.index, columns=pseudobulk.columns)
    logger.info("Normalised pseudo-bulk with %s (mean lib size: %.1f)", method, lib_sizes.mean())
    return result


def normalize_signature(
    signature: pd.DataFrame,
    method: str = "cpm",
) -> pd.DataFrame:
    """Apply the same normalisation to a signature matrix.

    For a signature matrix (gene-mean expression per cell type), we do a
    per-column (per-cell-type) CPM normalisation so that the scale
    matches the pseudo-bulk data.
    """
    if method == "none":
        return signature.copy()
    if method in ("cpm", "tpm"):
        # Per cell-type normalisation
        col_sums = signature.values.sum(axis=0)
        normed = signature.values / col_sums[None, :] * 1e6
        result = pd.DataFrame(normed, index=signature.index, columns=signature.columns)
        logger.info("Normalised signature matrix with %s", method)
        return result
    raise ValueError(f"Unknown normalisation method: {method}")


# ── Internal helpers ───────────────────────────────────────────────────────

def _filter_cell_types(
    adata: "AnnData",
    cell_type_column: str,
    min_cells: int,
) -> Tuple["AnnData", List[str]]:
    """Remove cell types with fewer than *min_cells* cells."""
    counts = adata.obs[cell_type_column].value_counts()
    kept = counts[counts >= min_cells].index.tolist()
    removed = counts[counts < min_cells].index.tolist()
    if removed:
        logger.warning("Removing %d cell types (<%d cells): %s", len(removed), min_cells, removed)
    mask = adata.obs[cell_type_column].isin(kept)
    return adata[mask].copy(), kept


def _build_cell_pools(
    adata: "AnnData",
    cell_type_column: str,
    cell_types: List[str],
) -> Dict[str, np.ndarray]:
    """Group expression data by cell type."""
    pools = {}
    for ct in cell_types:
        mask = adata.obs[cell_type_column] == ct
        expr = adata[mask].X
        if sparse.issparse(expr):
            expr = expr.toarray()
        pools[ct] = expr.astype(np.float64)
        logger.debug("Cell type '%s': %d cells", ct, expr.shape[0])
    return pools


def _generate_samples(
    cell_pools: Dict[str, np.ndarray],
    cell_types: List[str],
    n_samples: int,
    total_cells: int,
    rng: np.random.Generator,
    aggregation: str = "sum",
    dirichlet_alpha: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate pseudo-bulk samples and their true proportions."""
    n_genes = next(iter(cell_pools.values())).shape[1]
    n_types = len(cell_types)

    pb_matrix = np.zeros((n_samples, n_genes), dtype=np.float64)
    prop_matrix = np.zeros((n_samples, n_types), dtype=np.float64)

    for i in range(n_samples):
        props = rng.dirichlet([dirichlet_alpha] * n_types)
        prop_matrix[i] = props
        cell_counts = rng.multinomial(total_cells, props)

        sample_expr = np.zeros(n_genes, dtype=np.float64)
        for j, ct in enumerate(cell_types):
            if cell_counts[j] == 0:
                continue
            pool = cell_pools[ct]
            chosen_idx = rng.choice(pool.shape[0], size=cell_counts[j], replace=True)
            sample_expr += pool[chosen_idx].sum(axis=0)

        if aggregation == "mean":
            sample_expr /= total_cells

        pb_matrix[i] = sample_expr
        if (i + 1) % 500 == 0:
            logger.info("  Generated %d / %d samples", i + 1, n_samples)

    return pb_matrix, prop_matrix


def _build_summary(adata: "AnnData", cell_type_column: str, cell_types: List[str]) -> pd.DataFrame:
    """Build a per-cell-type summary table."""
    rows = []
    for ct in cell_types:
        mask = adata.obs[cell_type_column] == ct
        n_cells = mask.sum()
        expr_sub = adata[mask].X
        if sparse.issparse(expr_sub):
            expr_sub = expr_sub.toarray()
        mean_umi = expr_sub.sum(axis=1).mean()
        rows.append({"cell_type": ct, "n_cells": n_cells, "mean_umi": round(float(mean_umi), 1)})
    return pd.DataFrame(rows)
