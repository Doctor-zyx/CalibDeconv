#!/usr/bin/env python
"""
Phase 1a: Download and prepare the scRNA-seq reference dataset.

**IMPORTANT — Data source note**

10x Genomics PBMC 10k does NOT include cell-type labels.  This script
includes a provisional marker-based auto-annotation step for quick
testing.  For formal analysis, replace this with a pre-annotated
dataset (Monaco 2019, Azimuth PBMC, etc.).

Usage::

    python scripts/01_download_data.py \\
        --config config/config.yaml \\
        --output-dir data/raw \\
        --seed 42

Outputs
-------
- ``data/raw/pbmc_reference.h5ad``
- ``data/raw/data_manifest.csv``
- ``logs/01_download_data_*.log``
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
    run_qc_summary, is_debug_mode,
)


def download_pbmc_10x(output_dir: str, logger) -> str:
    """Download 10x Genomics PBMC 10k v3 filtered feature-barcode matrix."""
    import urllib.request

    url = (
        "https://cf.10xgenomics.com/samples/cell-exp/3.0.0/"
        "pbmc_10k_v3/pbmc_10k_v3_filtered_feature_bc_matrix.h5"
    )
    dest = Path(output_dir) / "pbmc_10k_v3_filtered_feature_bc_matrix.h5"
    if dest.exists():
        logger.info("File already exists: %s", dest)
        return str(dest)
    logger.info("Downloading 10x PBMC 10k v3 dataset from 10x Genomics...")
    urllib.request.urlretrieve(url, dest)
    logger.info("Downloaded to %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    return str(dest)


def build_anndata_from_10x(h5_path: str, logger) -> "AnnData":
    """Read 10x .h5 and build an AnnData with basic QC and provisional annotation."""
    import scanpy as sc

    adata = sc.read_10x_h5(h5_path)
    adata.var_names_make_unique()

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    # Store raw counts before normalisation
    adata.raw = adata.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    logger.info("QC complete: %d cells × %d genes", *adata.shape)

    # Provisional marker-based annotation
    logger.info("Running provisional marker-based cell-type annotation...")
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
    sc.tl.pca(adata, n_comps=30)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=0.8)

    _annotate_pbmc_clusters(adata, logger)
    for ct, count in adata.obs["cell_type"].value_counts().items():
        logger.info("  %s: %d cells", ct, count)
    return adata


def _annotate_pbmc_clusters(adata, logger):
    """Map Leiden clusters to PBMC cell types using canonical markers."""
    markers = {
        "CD4+_T_cells": ["CD3D", "CD4"],
        "CD8+_T_cells": ["CD3D", "CD8A", "CD8B"],
        "B_cells": ["MS4A1", "CD79A"],
        "NK_cells": ["NKG7", "GNLY"],
        "Monocytes": ["CD14", "FCGR3A"],
        "Dendritic_cells": ["FCER1A", "CST3"],
    }
    from scipy import sparse

    for ct_name, ct_markers in markers.items():
        available = [m for m in ct_markers if m in adata.var_names]
        if not available:
            continue
        expr = adata[:, available].X
        if sparse.issparse(expr):
            expr = expr.toarray()
        for cluster in adata.obs["leiden"].unique():
            mask = adata.obs["leiden"] == cluster
            adata.obs.loc[adata.obs["leiden"] == cluster, f"score_{ct_name}"] = expr[mask].mean()

    # Assign each cluster to the cell type with highest score
    score_cols = [c for c in adata.obs.columns if c.startswith("score_")]
    best_type = adata.obs[score_cols].idxmax(axis=1).str.replace("score_", "")
    adata.obs["cell_type"] = best_type
    logger.info("Cluster->cell_type mapping complete")


def main():
    parser = argparse.ArgumentParser(description="Phase 1a: Download scRNA-seq data")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--source", default="10x_pbmc")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("01_download", log_dir=str(PROJECT_ROOT / "logs"))
    set_seed(args.seed)
    debug = is_debug_mode(cfg)
    logger.info("Debug mode: %s", debug)

    output_dir = str(PROJECT_ROOT / args.output_dir)
    ensure_dir(output_dir)

    sc_cfg = cfg["data"]["sc_reference"]
    logger.info("=" * 60)
    logger.info("Phase 1a: Download & prepare scRNA-seq reference")
    logger.info("  Source: %s", sc_cfg["source"])
    logger.info("  [WARN]  10x PBMC does NOT include cell-type labels!")
    logger.info("  Provisional auto-annotation will be applied.")
    logger.info("  For formal runs, use Monaco 2019 or Azimuth PBMC.")
    logger.info("=" * 60)

    # Download
    if args.source == "10x_pbmc":
        h5_path = download_pbmc_10x(output_dir, logger)
    else:
        raise ValueError(f"Unknown source: {args.source}")

    # Build AnnData
    adata = build_anndata_from_10x(h5_path, logger)

    # Save
    h5ad_path = Path(output_dir) / "pbmc_reference.h5ad"
    adata.write(h5ad_path)
    logger.info("Saved AnnData to %s", h5ad_path)

    # QC on expression and cell-type distribution
    from scipy import sparse
    expr = adata.raw.X if adata.raw is not None else adata.X
    if sparse.issparse(expr):
        expr = expr.toarray()
    expr_df = pd.DataFrame(
        expr[:10, :100],  # first 10 cells, 100 genes as sample
    )
    qc = run_qc_summary(
        pd.DataFrame(adata.obs["cell_type"].value_counts()).reset_index(),
        label="cell_type_counts",
        logger=logger,
    )
    qc["n_cells"] = adata.shape[0]
    qc["n_genes"] = adata.shape[1]
    qc["n_cell_types"] = adata.obs["cell_type"].nunique()
    qc["file_path"] = str(h5ad_path.resolve())
    logger.info("QC: %d cells, %d genes, %d cell types", qc["n_cells"], qc["n_genes"], qc["n_cell_types"])

    # Manifest
    manifest = pd.DataFrame([{
        "source": args.source,
        "n_cells": adata.shape[0],
        "n_genes": adata.shape[1],
        "cell_types": ",".join(sorted(adata.obs["cell_type"].unique())),
        "h5ad_path": str(h5ad_path),
        "seed": args.seed,
        "debug": debug,
    }])
    save_df(manifest, Path(output_dir) / "data_manifest.csv")
    logger.info("Phase 1a complete. [OK] ")


if __name__ == "__main__":
    main()
