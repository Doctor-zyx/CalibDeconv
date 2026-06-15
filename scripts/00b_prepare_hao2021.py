#!/usr/bin/env python
"""
Prepare the Hao et al. 2021 PBMC reference (from CELLxGENE) for CalibDeconv.

The CELLxGENE h5ad has specific conventions that need normalisation:
- ``adata.X`` holds normalised/log data; raw counts are in ``adata.raw.X``
  or a ``adata.layers`` entry.
- ``adata.var_names`` are Ensembl gene IDs; human-readable symbols are
  in ``adata.var['feature_name']``.
- Cell types are in ``adata.obs['cell_type']`` (fine-grained, ~30 types).
- Donor IDs in ``adata.obs['donor_id']``.

This script:
1. Loads the full reference.
2. Maps fine-grained cell types -> a configurable set of MAJOR cell types.
3. Sets gene symbols as var_names (for matching with bulk data later).
4. Ensures raw counts are accessible in ``adata.raw``.
5. Writes a cleaned ``pbmc_reference.h5ad`` plus diagnostic summaries.

Usage::

    python scripts/00b_prepare_hao2021.py \\
        --input data/raw/hao2021_pbmc_full.h5ad \\
        --output data/raw/pbmc_reference.h5ad \\
        --seed 42
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import setup_logger, set_seed, ensure_dir, save_df


# Map fine-grained CELLxGENE cell_type labels -> major PBMC lineages.
# Based on Hao 2021 Level-1 / Azimuth L1 conventions.
MAJOR_TYPE_MAP = {
    # ---- CD4 T ----
    "CD4-positive, alpha-beta T cell": "CD4_T",
    "CD4-positive, alpha-beta cytotoxic T cell": "CD4_T",
    "central memory CD4-positive, alpha-beta T cell": "CD4_T",
    "effector memory CD4-positive, alpha-beta T cell": "CD4_T",
    "naive thymus-derived CD4-positive, alpha-beta T cell": "CD4_T",
    "memory regulatory T cell": "CD4_T",
    "naive regulatory T cell": "CD4_T",
    # ---- CD8 T ----
    "CD8-positive, alpha-beta T cell": "CD8_T",
    "central memory CD8-positive, alpha-beta T cell": "CD8_T",
    "effector memory CD8-positive, alpha-beta T cell": "CD8_T",
    "naive thymus-derived CD8-positive, alpha-beta T cell": "CD8_T",
    "mucosal invariant T cell": "CD8_T",
    # ---- Other T ----
    "gamma-delta T cell": "other_T",
    "double negative thymocyte": "other_T",
    # ---- B ----
    "B cell": "B",
    "memory B cell": "B",
    "naive B cell": "B",
    "plasmablast": "B",
    # ---- NK ----
    "natural killer cell": "NK",
    "CD16-negative, CD56-bright natural killer cell, human": "NK",
    # ---- Monocytes ----
    "CD14-positive monocyte": "Monocyte",
    "CD14-low, CD16-positive monocyte": "Monocyte",
    # ---- Dendritic cells ----
    "conventional dendritic cell": "DC",
    "myeloid dendritic cell, human": "DC",
    "plasmacytoid dendritic cell": "DC",
    # ---- Other (dropped by default) ----
    "innate lymphoid cell": "other",
    "hematopoietic stem cell": "other",
    "erythrocyte": "other",
    "platelet": "other",
    "unknown": "other",
}


def main():
    parser = argparse.ArgumentParser(description="Prepare Hao 2021 PBMC reference")
    parser.add_argument("--input", default="data/raw/hao2021_pbmc_full.h5ad")
    parser.add_argument("--output", default="data/raw/pbmc_reference.h5ad")
    parser.add_argument("--drop-other", action="store_true", default=True,
                        help="Drop the 'other' major type (rare/ambiguous cells)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger = setup_logger("00b_prepare_hao2021", log_dir=str(PROJECT_ROOT / "logs"))
    set_seed(args.seed)

    import scanpy as sc
    import anndata
    from scipy import sparse

    input_path = PROJECT_ROOT / args.input if not Path(args.input).is_absolute() else Path(args.input)
    output_path = PROJECT_ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)

    logger.info("=" * 60)
    logger.info("Preparing Hao et al. 2021 PBMC reference")
    logger.info("  Input: %s", input_path)
    logger.info("=" * 60)

    logger.info("Loading (this may take a minute for 2.6GB)...")
    adata = sc.read_h5ad(input_path)
    logger.info("Loaded: %d cells x %d genes", *adata.shape)
    logger.info("obs columns: %s", list(adata.obs.columns))
    logger.info("var columns: %s", list(adata.var.columns))

    # ---- 1. Identify raw counts ----
    # CELLxGENE convention: raw counts in adata.raw.X, normalised in adata.X
    has_raw = adata.raw is not None
    logger.info("adata.raw present: %s", has_raw)
    if "feature_name" in adata.var.columns:
        logger.info("Gene symbol column 'feature_name' found")

    # ---- 2. Map cell types to major lineages ----
    if "cell_type" not in adata.obs.columns:
        raise ValueError("No 'cell_type' column found in adata.obs!")

    fine_types = adata.obs["cell_type"].astype(str)
    unmapped = set(fine_types.unique()) - set(MAJOR_TYPE_MAP.keys())
    if unmapped:
        logger.warning("Unmapped cell types (-> 'other'): %s", unmapped)

    adata.obs["major_cell_type"] = fine_types.map(
        lambda x: MAJOR_TYPE_MAP.get(x, "other")
    )

    logger.info("Major cell type distribution:")
    for ct, n in adata.obs["major_cell_type"].value_counts().items():
        logger.info("  %-12s %6d cells", ct, n)

    # ---- 3. Drop 'other' if requested ----
    if args.drop_other:
        before = adata.shape[0]
        adata = adata[adata.obs["major_cell_type"] != "other"].copy()
        logger.info("Dropped 'other': %d -> %d cells", before, adata.shape[0])

    # Use major_cell_type as the working cell_type
    adata.obs["cell_type"] = adata.obs["major_cell_type"].astype("category")

    # ---- 4. Set gene symbols as var_names ----
    if "feature_name" in adata.var.columns:
        adata.var["ensembl_id"] = adata.var_names
        new_names = adata.var["feature_name"].astype(str).values
        # Drop the feature_name column to avoid index/column name clash on write
        adata.var = adata.var.drop(columns=["feature_name"])
        adata.var_names = new_names
        adata.var.index.name = "gene_symbol"
        adata.var_names_make_unique()
        logger.info("Set gene symbols as var_names")

        # Reconstruct adata.raw with aligned var_names (raw also carries
        # a 'feature_name' column that clashes with the index on write)
        if adata.raw is not None:
            import anndata as _ad
            raw_adata = _ad.AnnData(
                X=adata.raw.X,
                var=pd.DataFrame(index=adata.var_names),
                obs=adata.obs[[]].copy(),
            )
            adata.raw = raw_adata
            logger.info("Rebuilt adata.raw with aligned gene symbols")

    # ---- 5. Ensure raw counts in adata.raw ----
    # Verify raw is integer counts
    if has_raw:
        raw_X = adata.raw.X
        sample = raw_X[:100].toarray() if sparse.issparse(raw_X) else raw_X[:100]
        is_integer = np.allclose(sample, np.round(sample))
        logger.info("Raw counts appear to be integers: %s", is_integer)
        # Align raw var_names with main var_names
        if adata.raw.shape[1] == adata.shape[1]:
            logger.info("Raw and main have same gene count")
    else:
        logger.warning("No adata.raw — using adata.X as counts (may be normalised!)")

    # ---- 6. Donor / batch info ----
    donor_col = "donor_id" if "donor_id" in adata.obs.columns else None
    if donor_col:
        logger.info("Donor distribution:")
        for d, n in adata.obs[donor_col].value_counts().items():
            logger.info("  %-6s %6d cells", str(d), n)

    # ---- 7. Save ----
    ensure_dir(str(output_path.parent))
    adata.write(output_path)
    logger.info("Saved cleaned reference to %s (%d cells x %d genes, %d cell types)",
                output_path, adata.shape[0], adata.shape[1],
                adata.obs["cell_type"].nunique())

    # ---- 8. Diagnostic summaries ----
    diag_dir = ensure_dir(str(PROJECT_ROOT / "results" / "data_prep"))

    ct_summary = adata.obs["cell_type"].value_counts().reset_index()
    ct_summary.columns = ["cell_type", "n_cells"]
    save_df(ct_summary, Path(diag_dir) / "cell_type_counts.csv")

    if donor_col:
        donor_summary = adata.obs.groupby([donor_col, "cell_type"], observed=True).size().reset_index(name="n_cells")
        save_df(donor_summary, Path(diag_dir) / "donor_celltype_distribution.csv")
        donor_totals = adata.obs[donor_col].value_counts().reset_index()
        donor_totals.columns = ["donor_id", "n_cells"]
        save_df(donor_totals, Path(diag_dir) / "donor_counts.csv")

    logger.info("Diagnostics saved to %s", diag_dir)
    logger.info("Done. [OK]")


if __name__ == "__main__":
    main()
