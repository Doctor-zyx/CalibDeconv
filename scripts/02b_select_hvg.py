#!/usr/bin/env python
"""
Phase 1c: Select a 3000 highly-variable-gene (HVG) panel for formal analysis.

CRITICAL — leakage control:
  HVG selection uses ONLY the reference cell pool (cell_pool_reference.h5ad).
  Calibration and test pseudo-bulks are NOT consulted, so no information
  from the evaluation sets leaks into the gene panel.

This script:
1. Loads the reference cell pool.
2. Selects the top-N highly variable genes (scanpy seurat flavor) on
   log-normalised reference expression.
3. Restricts the panel to genes that also exist in the pseudo-bulk
   matrices (so the panel is directly usable downstream).
4. Saves the gene list and HVG-subset CPM pseudo-bulk matrices.
5. Updates run_metadata.json with the gene-panel provenance.

Usage::

    python scripts/02b_select_hvg.py \\
        --cell-pool-dir data/processed \\
        --pseudobulk-dir data/processed \\
        --n-hvg 3000 \\
        --seed 42
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import load_config, setup_logger, set_seed, ensure_dir, save_df


def main():
    parser = argparse.ArgumentParser(description="Phase 1c: HVG panel selection")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--cell-pool-dir", default="data/processed")
    parser.add_argument("--pseudobulk-dir", default="data/processed")
    parser.add_argument("--n-hvg", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("02b_select_hvg", log_dir=str(PROJECT_ROOT / "logs"))
    set_seed(args.seed)

    pool_dir = Path(PROJECT_ROOT / args.cell_pool_dir)
    pb_dir = Path(PROJECT_ROOT / args.pseudobulk_dir)

    logger.info("=" * 60)
    logger.info("Phase 1c: HVG panel selection (reference pool ONLY)")
    logger.info("  n_hvg target: %d", args.n_hvg)
    logger.info("=" * 60)

    # ── 1. Load reference pool ONLY ──
    import scanpy as sc
    ref_path = pool_dir / "cell_pool_reference.h5ad"
    logger.info("Loading reference pool from %s", ref_path)
    adata_ref = sc.read_h5ad(ref_path)
    logger.info("Reference pool: %d cells x %d genes", *adata_ref.shape)
    logger.info("LEAKAGE CHECK: calibration/test pools are NOT loaded. [OK]")

    # ── 2. HVG selection on reference (log-normalised) ──
    # Work on a fresh copy from raw counts to control normalisation.
    if adata_ref.raw is not None:
        adata_hvg = adata_ref.raw.to_adata()
        logger.info("Using raw counts from adata.raw for HVG selection")
    else:
        adata_hvg = adata_ref.copy()
        logger.warning("No adata.raw — using adata.X (may already be normalised)")

    # Normalise + log1p, then HVG (seurat flavor works on log data)
    sc.pp.normalize_total(adata_hvg, target_sum=1e4)
    sc.pp.log1p(adata_hvg)
    sc.pp.highly_variable_genes(adata_hvg, n_top_genes=args.n_hvg, flavor="seurat")
    hvg_genes = adata_hvg.var_names[adata_hvg.var["highly_variable"]].tolist()
    logger.info("Selected %d HVGs from reference pool", len(hvg_genes))

    # ── 3. Restrict to genes present in pseudo-bulk matrices ──
    # Read only the header of pseudobulk to get the gene list (fast)
    pb_cal_path = pb_dir / "pseudobulk_matrix_cal_cpm.csv"
    pb_test_path = pb_dir / "pseudobulk_matrix_test_cpm.csv"
    pb_cal_header = pd.read_csv(pb_cal_path, index_col=0, nrows=0)
    pb_genes = set(pb_cal_header.columns)

    hvg_in_pb = [g for g in hvg_genes if g in pb_genes]
    n_dropped = len(hvg_genes) - len(hvg_in_pb)
    if n_dropped > 0:
        logger.warning("%d HVGs not in pseudo-bulk matrix — dropped", n_dropped)
    logger.info("Final HVG panel size (present in pseudo-bulk): %d", len(hvg_in_pb))

    # Preserve a stable order: by HVG ranking but only kept genes
    hvg_panel = hvg_in_pb

    # ── 4. Save gene list ──
    gene_panel_path = pb_dir / "selected_genes_hvg3000.txt"
    with open(gene_panel_path, "w") as fh:
        for g in hvg_panel:
            fh.write(g + "\n")
    logger.info("Gene panel saved to %s", gene_panel_path)

    # ── 5. Subset pseudo-bulk matrices to HVG panel ──
    logger.info("Subsetting cal/test CPM matrices to HVG panel...")
    pb_cal = pd.read_csv(pb_cal_path, index_col=0)
    pb_test = pd.read_csv(pb_test_path, index_col=0)

    # Same gene order for both
    pb_cal_hvg = pb_cal[hvg_panel]
    pb_test_hvg = pb_test[hvg_panel]

    cal_out = pb_dir / "pseudobulk_matrix_cal_hvg3000_cpm.csv"
    test_out = pb_dir / "pseudobulk_matrix_test_hvg3000_cpm.csv"
    pb_cal_hvg.to_csv(cal_out)
    pb_test_hvg.to_csv(test_out)
    logger.info("Saved HVG cal matrix: %d x %d -> %s", *pb_cal_hvg.shape, cal_out)
    logger.info("Saved HVG test matrix: %d x %d -> %s", *pb_test_hvg.shape, test_out)

    # ── 6. Update run_metadata.json ──
    meta_path = pb_dir / "run_metadata.json"
    if meta_path.exists():
        with open(meta_path) as fh:
            meta = json.load(fh)
    else:
        meta = {}
    meta.update({
        "n_hvg": len(hvg_panel),
        "gene_selection_source": "reference_pool_only",
        "gene_panel_file": str(gene_panel_path),
        "analysis_gene_set": "hvg3000",
        "hvg_flavor": "seurat",
        "hvg_seed": args.seed,
    })
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    logger.info("Updated run_metadata.json")

    # ── 7. Consistency checks ──
    logger.info("-" * 40)
    logger.info("Consistency checks:")
    same_order = list(pb_cal_hvg.columns) == list(pb_test_hvg.columns)
    same_as_panel = list(pb_cal_hvg.columns) == hvg_panel
    logger.info("  cal/test HVG gene order identical: %s", same_order)
    logger.info("  matrices match gene panel order:   %s", same_as_panel)
    logger.info("  cal HVG matrix: %s", pb_cal_hvg.shape)
    logger.info("  test HVG matrix: %s", pb_test_hvg.shape)
    logger.info("  NA in cal: %s, NA in test: %s",
                bool(pb_cal_hvg.isna().any().any()), bool(pb_test_hvg.isna().any().any()))

    logger.info("Phase 1c complete. [OK]")
    logger.info("First 10 genes: %s", hvg_panel[:10])


if __name__ == "__main__":
    main()
