#!/usr/bin/env python
"""
Generate synthetic PBMC-like scRNA-seq data for pipeline testing.

Creates an AnnData with realistic cell-type expression patterns so the
full CalibDeconv pipeline can be tested end-to-end without downloading
external data.  This approach also serves as a controlled in-silico
benchmark.

The synthetic data has:
- 6 cell types (CD4+ T, CD8+ T, B, NK, Monocytes, DC)
- ~300 cells per type (total ~1800 cells)
- 2000 genes with type-specific marker expression
- Raw counts + normalised layers (mimics adata.raw)
- "cell_type" annotation in adata.obs

Usage::

    python scripts/00_generate_synthetic_data.py --output-dir data/raw --seed 42
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import ensure_dir, set_seed


def generate_synthetic_pbmc(
    n_genes: int = 2000,
    n_cells_per_type: int = 300,
    seed: int = 42,
) -> "AnnData":
    """Generate a synthetic PBMC AnnData with realistic expression patterns.

    Each cell type has a set of highly-expressed marker genes, plus a
    shared background expression.  Negative binomial noise is added to
    simulate count data.

    Returns
    -------
    AnnData
        With ``.raw`` (raw counts) and ``.obs["cell_type"]``.
    """
    import anndata
    rng = np.random.default_rng(seed)

    # Cell type definitions
    cell_types = {
        "CD4+_T_cells": ["CD3D", "CD4", "IL7R", "CCR7"],
        "CD8+_T_cells": ["CD3D", "CD8A", "CD8B", "GZMK"],
        "B_cells": ["MS4A1", "CD79A", "CD79B", "PAX5"],
        "NK_cells": ["NKG7", "GNLY", "KLRD1", "PRF1"],
        "Monocytes": ["CD14", "FCGR3A", "LYZ", "S100A9"],
        "Dendritic_cells": ["FCER1A", "CST3", "CLEC10A", "CD1C"],
    }

    type_names = list(cell_types.keys())
    n_types = len(type_names)
    n_cells_total = n_types * n_cells_per_type

    # Gene names: real markers first, then generic genes
    all_markers = []
    for markers in cell_types.values():
        all_markers.extend(markers)
    # Ensure uniqueness while preserving order
    seen = set()
    unique_markers = []
    for m in all_markers:
        if m not in seen:
            unique_markers.append(m)
            seen.add(m)

    # Fill remaining with generic gene names
    n_fill = n_genes - len(unique_markers)
    generic_genes = [f"GENE_{i:04d}" for i in range(n_fill)]
    gene_names = unique_markers + generic_genes

    # Expression matrix: each cell type has a base mean + marker peaks
    # log mean expression ~ N(3, 1) for baseline genes
    log_base_mean = rng.normal(3.0, 1.0, size=n_genes)

    # Build per-cell-type log mean expression
    # Markers get a boost of +3 to +5
    marker_boost = {}
    for ct, markers in cell_types.items():
        boost = {}
        for m in markers:
            if m in gene_names:
                idx = gene_names.index(m)
                boost[idx] = rng.uniform(3.0, 5.0)
        marker_boost[ct] = boost

    # Generate cells
    X = np.zeros((n_cells_total, n_genes), dtype=np.float32)
    obs_cell_types = []

    for i, ct in enumerate(type_names):
        start = i * n_cells_per_type
        end = start + n_cells_per_type

        # Per-cell-type log mean with marker boost
        ct_log_mean = log_base_mean.copy()
        for gene_idx, boost_val in marker_boost[ct].items():
            ct_log_mean[gene_idx] += boost_val

        # Each cell has slight variation around the type mean
        for j in range(start, end):
            cell_log_mean = ct_log_mean + rng.normal(0, 0.3, size=n_genes)
            # Negative binomial: mu = exp(log_mean), size (dispersion) varies
            mu = np.exp(cell_log_mean)
            # dispersion parameter: smaller = more overdispersion
            size_param = 2.0
            # NB: Gamma-Poisson mixture
            p = size_param / (size_param + mu)
            counts = rng.negative_binomial(size_param, 1 - p[:size_param]) if False else np.array([
                rng.poisson(m) for m in mu
            ])
            X[j] = counts

        obs_cell_types.extend([ct] * n_cells_per_type)

    # Build AnnData
    adata = anndata.AnnData(
        X=X,
        obs=pd.DataFrame({"cell_type": obs_cell_types}),
        var=pd.DataFrame(index=gene_names),
    )
    adata.var_names_make_unique()

    # Store raw counts
    adata.raw = adata.copy()

    # Add normalised layer
    import scanpy as sc
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    return adata


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic PBMC data")
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--n-genes", type=int, default=2000)
    parser.add_argument("--n-cells-per-type", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    output_dir = Path(PROJECT_ROOT) / args.output_dir
    ensure_dir(str(output_dir))

    print(f"Generating synthetic PBMC data:")
    print(f"  {len(['CD4+_T_cells','CD8+_T_cells','B_cells','NK_cells','Monocytes','Dendritic_cells'])} cell types")
    print(f"  {args.n_cells_per_type} cells per type = ~{6 * args.n_cells_per_type} total")
    print(f"  {args.n_genes} genes")

    adata = generate_synthetic_pbmc(
        n_genes=args.n_genes,
        n_cells_per_type=args.n_cells_per_type,
        seed=args.seed,
    )

    out_path = output_dir / "pbmc_reference.h5ad"
    adata.write(out_path)
    print(f"Saved: {out_path}")
    print(f"  {adata.shape[0]} cells × {adata.shape[1]} genes")
    print(f"  Cell types: {adata.obs['cell_type'].value_counts().to_dict()}")
    print(f"  Raw counts in adata.raw: {'Yes' if adata.raw is not None else 'No'}")
    print("Done.")


if __name__ == "__main__":
    main()
