#!/usr/bin/env python
"""
Phase 1b: Generate pseudo-bulk samples from scRNA-seq reference data.

**Cell-pool separation enforced**

1. Split single cells into three DISJOINT pools: reference / calibration / test.
2. Reference pool -> signature matrix only.  Calibration pool -> calibration
   pseudo-bulk.  Test pool -> test pseudo-bulk.
3. Pseudo-bulk aggregated from raw counts, then jointly normalised (CPM).

Usage::

    python scripts/02_generate_pseudobulk.py \\
        --config config/config.yaml \\
        --adata data/raw/pbmc_reference.h5ad \\
        --output-dir data/processed \\
        --seed 42

Outputs
-------
- ``data/processed/cell_pools.h5ad``         : reference / cal / test AnnData pools
- ``data/processed/pseudobulk_matrix_cal.csv`` : calibration pseudo-bulk (raw)
- ``data/processed/true_proportions_cal.csv``   : calibration ground-truth
- ``data/processed/pseudobulk_matrix_test.csv`` : test pseudo-bulk (raw)
- ``data/processed/true_proportions_test.csv``  : test ground-truth
- ``data/processed/pseudobulk_*_cpm.csv``       : CPM-normalised versions
- ``logs/02_generate_pseudobulk_*.log``
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import (
    load_config, setup_logger, set_seed, ensure_dir, save_df,
    run_qc_summary, run_qc_on_path, qc_report_all, is_debug_mode, debug_param,
)
from src.data.pseudobulk import (
    split_cell_pools,
    generate_pseudobulk_from_pool,
    normalize_pseudobulk,
)


def main():
    parser = argparse.ArgumentParser(description="Phase 1b: Pseudo-bulk generation")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--adata", required=True, help="Path to Anndata .h5ad file")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Total pseudo-bulk samples (cal + test). Overrides config.")
    parser.add_argument("--donor-aware", dest="donor_aware", action="store_true", default=None,
                        help="Force donor-aware split (whole donors per pool).")
    parser.add_argument("--no-donor-aware", dest="donor_aware", action="store_false",
                        help="Force random cell-level split.")
    parser.add_argument("--cell-type-col", default=None,
                        help="Column in adata.obs with cell-type labels. Overrides config.")
    parser.add_argument("--donor-col", default=None,
                        help="Column in adata.obs with donor IDs. Overrides config.")
    parser.add_argument("--n-highly-variable-genes", "--n-hvg", dest="n_hvg", type=int, default=None,
                        help="Number of highly-variable genes to keep. Overrides config.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("02_pseudobulk", log_dir=str(PROJECT_ROOT / "logs"))
    set_seed(args.seed)

    output_dir = str(PROJECT_ROOT / args.output_dir)
    ensure_dir(output_dir)

    # Load data
    import scanpy as sc
    adata_path = PROJECT_ROOT / args.adata if not Path(args.adata).is_absolute() else Path(args.adata)
    logger.info("Loading AnnData from %s", adata_path)
    adata = sc.read_h5ad(adata_path)

    sc_cfg = cfg["data"]["sc_reference"]
    pb_cfg = cfg["pseudobulk"]

    # ---- CLI overrides config ----
    cell_type_col = args.cell_type_col or sc_cfg["cell_type_column"]
    donor_col = args.donor_col if args.donor_col is not None else sc_cfg.get("donor_column")
    use_raw = sc_cfg["use_raw_counts"]
    n_hvg = args.n_hvg if args.n_hvg is not None else sc_cfg.get("n_highly_variable_genes")

    # n_samples: CLI > debug-mode override > config default
    if args.n_samples is not None:
        n_total = args.n_samples
    else:
        n_total = debug_param(cfg, "pseudobulk", "n_samples", pb_cfg["n_samples"])
    n_samples_cal = n_total // 2
    n_samples_test = n_total - n_samples_cal
    normalization = pb_cfg["normalization"]

    # donor_aware: CLI > config
    if args.donor_aware is not None:
        donor_aware = args.donor_aware
    else:
        donor_aware = cfg["cell_pool_split"].get("donor_aware", False)

    logger.info("=" * 60)
    logger.info("Phase 1b: Pseudo-bulk generation (cell-pool separation)")
    logger.info("  Debug mode: %s", is_debug_mode(cfg))
    logger.info("  cell_type_column: %s", cell_type_col)
    logger.info("  donor_column: %s", donor_col)
    logger.info("  donor_aware split: %s", donor_aware)
    logger.info("  n_highly_variable_genes: %s", n_hvg)
    logger.info("  use_raw_counts: %s", use_raw)
    logger.info("  n_samples: %d cal + %d test = %d", n_samples_cal, n_samples_test, n_samples_cal + n_samples_test)
    logger.info("  total_cells/sample: %d", pb_cfg["total_cells_per_sample"])
    logger.info("  aggregation: %s", pb_cfg["aggregation"])
    logger.info("  normalisation: %s", normalization)
    logger.info("=" * 60)

    # ── Step 0: Pre-split donor × cell_type distribution ──
    logger.info("-" * 40)
    logger.info("Overall donor x cell_type distribution:")
    if donor_col and donor_col in adata.obs.columns:
        overall_dist = adata.obs.groupby([donor_col, cell_type_col], observed=True).size().unstack(fill_value=0)
        logger.info("\n%s", overall_dist.to_string())
        save_df(overall_dist.reset_index(), Path(output_dir) / "donor_celltype_distribution_all.csv")
    else:
        logger.warning("No donor column — cannot show donor x cell_type table")

    # ── Step 1: Split cell pools ──
    pool_ratios = {
        "reference": cfg["cell_pool_split"]["reference"],
        "calibration": cfg["cell_pool_split"]["calibration"],
        "test": cfg["cell_pool_split"]["test"],
    }
    logger.info("Splitting cells into: ref=%.0f%% / cal=%.0f%% / test=%.0f%% (donor_aware=%s)",
                pool_ratios["reference"] * 100, pool_ratios["calibration"] * 100,
                pool_ratios["test"] * 100, donor_aware)

    pools = split_cell_pools(
        adata=adata,
        cell_type_column=cell_type_col,
        ratios=pool_ratios,
        donor_column=donor_col if donor_aware else None,
        seed=args.seed,
    )

    # Save pools for downstream use
    for name, pool_adata in pools.items():
        pool_adata.write(Path(output_dir) / f"cell_pool_{name}.h5ad")
    logger.info("Cell pools saved to %s/cell_pool_*.h5ad", output_dir)

    # Verify pools are disjoint (barcode intersection MUST be 0)
    ref_cells = set(pools["reference"].obs_names)
    cal_cells = set(pools["calibration"].obs_names)
    test_cells = set(pools["test"].obs_names)
    inter_rc = len(ref_cells & cal_cells)
    inter_rt = len(ref_cells & test_cells)
    inter_ct = len(cal_cells & test_cells)
    logger.info("Barcode intersections (must all be 0): ref∩cal=%d ref∩test=%d cal∩test=%d",
                inter_rc, inter_rt, inter_ct)
    assert inter_rc == 0, "Reference and calibration pools overlap!"
    assert inter_rt == 0, "Reference and test pools overlap!"
    assert inter_ct == 0, "Calibration and test pools overlap!"
    logger.info("[OK]  Cell pools verified disjoint: ref=%d cal=%d test=%d cells",
                len(ref_cells), len(cal_cells), len(test_cells))

    # ── Per-pool donor list + cell-type distribution ──
    logger.info("-" * 40)
    logger.info("Per-pool composition:")
    pool_diag_rows = []
    for name, pool_adata in pools.items():
        if donor_col and donor_col in pool_adata.obs.columns:
            donors_here = sorted(pool_adata.obs[donor_col].unique().tolist())
        else:
            donors_here = ["(cell-level split)"]
        ct_counts = pool_adata.obs[cell_type_col].value_counts().to_dict()
        logger.info("  [%s] %d cells | donors=%s", name, pool_adata.shape[0], donors_here)
        for ct, n in sorted(ct_counts.items()):
            logger.info("       %-12s %6d", ct, n)
            pool_diag_rows.append({"pool": name, "cell_type": ct, "n_cells": n})
        # Warn if any cell type is too small in this pool
        for ct, n in ct_counts.items():
            if n < sc_cfg["min_cells_per_type"]:
                logger.warning("    [WARN] %s in %s pool has only %d cells (<%d)",
                               ct, name, n, sc_cfg["min_cells_per_type"])

    pool_diag = pd.DataFrame(pool_diag_rows)
    save_df(pool_diag, Path(output_dir) / "pool_celltype_distribution.csv")

    # Donor overlap check across pools
    if donor_col and donor_col in adata.obs.columns and donor_aware:
        ref_donors = set(pools["reference"].obs[donor_col].unique())
        cal_donors = set(pools["calibration"].obs[donor_col].unique())
        test_donors = set(pools["test"].obs[donor_col].unique())
        logger.info("Donor sets — ref=%s cal=%s test=%s",
                    sorted(ref_donors), sorted(cal_donors), sorted(test_donors))
        d_overlap = (ref_donors & cal_donors) | (ref_donors & test_donors) | (cal_donors & test_donors)
        if d_overlap:
            logger.warning("[WARN] Donors appear in multiple pools: %s", d_overlap)
        else:
            logger.info("[OK]  Donors do NOT overlap across pools")

    # ── Step 2: Generate pseudo-bulk from calibration pool ──
    logger.info("-" * 40)
    logger.info("Generating CALIBRATION pseudo-bulk samples...")
    cal_result = generate_pseudobulk_from_pool(
        adata_pool=pools["calibration"],
        cell_type_column=cell_type_col,
        n_samples=n_samples_cal,
        total_cells_per_sample=pb_cfg["total_cells_per_sample"],
        aggregation=pb_cfg["aggregation"],
        dirichlet_alpha=pb_cfg["dirichlet_alpha"],
        min_cells_per_type=sc_cfg["min_cells_per_type"],
        use_raw_counts=use_raw,
        seed=args.seed,
    )

    # ── Step 3: Generate pseudo-bulk from test pool ──
    logger.info("-" * 40)
    logger.info("Generating TEST pseudo-bulk samples...")
    test_result = generate_pseudobulk_from_pool(
        adata_pool=pools["test"],
        cell_type_column=cell_type_col,
        n_samples=n_samples_test,
        total_cells_per_sample=pb_cfg["total_cells_per_sample"],
        aggregation=pb_cfg["aggregation"],
        dirichlet_alpha=pb_cfg["dirichlet_alpha"],
        min_cells_per_type=sc_cfg["min_cells_per_type"],
        use_raw_counts=use_raw,
        seed=args.seed + 1,
    )

    # ── Step 4: Assign unique sample IDs across cal/test ──
    for tag, res in [("cal", cal_result), ("test", test_result)]:
        n = len(res["pseudobulk_matrix"])
        res["pseudobulk_matrix"].index = [f"{tag}_{i}" for i in range(n)]
        res["true_proportions"].index = [f"{tag}_{i}" for i in range(n)]
        res["pseudobulk_matrix"].to_csv(Path(output_dir) / f"pseudobulk_matrix_{tag}.csv")
        res["true_proportions"].to_csv(Path(output_dir) / f"true_proportions_{tag}.csv")
        logger.info("Saved %s: %d samples × %d genes", tag, *res["pseudobulk_matrix"].shape)

    # ── Step 5: Joint normalisation ──
    logger.info("-" * 40)
    logger.info("Applying joint normalisation (%s)...", normalization)
    pb_raw_all = pd.concat([cal_result["pseudobulk_matrix"], test_result["pseudobulk_matrix"]])
    pb_norm_all = normalize_pseudobulk(pb_raw_all, method=normalization)

    n_cal = len(cal_result["pseudobulk_matrix"])
    pb_norm_cal = pb_norm_all.iloc[:n_cal]
    pb_norm_test = pb_norm_all.iloc[n_cal:]

    for tag, pb_norm in [("cal", pb_norm_cal), ("test", pb_norm_test)]:
        pb_norm.to_csv(Path(output_dir) / f"pseudobulk_matrix_{tag}_cpm.csv")
    logger.info("Normalised matrices saved.")

    # ── Step 6: QC Summary ──
    logger.info("-" * 40)
    logger.info("Running QC checks...")
    qc_results = []
    for tag, res in [("cal", cal_result), ("test", test_result)]:
        qc_pb = run_qc_summary(res["pseudobulk_matrix"], label=f"pseudobulk_raw_{tag}", logger=logger)
        qc_pb["file_path"] = str(Path(output_dir) / f"pseudobulk_matrix_{tag}.csv")
        qc_results.append(qc_pb)

        qc_prop = run_qc_summary(
            res["true_proportions"], label=f"true_proportions_{tag}",
            expect_proportions=True, logger=logger,
        )
        qc_prop["file_path"] = str(Path(output_dir) / f"true_proportions_{tag}.csv")
        qc_results.append(qc_prop)

    qc_report_all(qc_results, logger=logger)

    # Cell type alignment check
    cal_types = set(cal_result["true_proportions"].columns)
    test_types = set(test_result["true_proportions"].columns)
    if cal_types != test_types:
        logger.warning("[WARN]  Cell type mismatch between cal and test!")
        logger.warning("  Cal only: %s", cal_types - test_types)
        logger.warning("  Test only: %s", test_types - cal_types)
    else:
        logger.info("[OK]  Cell types consistent across cal/test: %d types", len(cal_types))

    # ── Step 7: Per-cell-type proportion distribution ──
    logger.info("-" * 40)
    logger.info("Per-cell-type proportion distribution (test set):")
    prop_stats_rows = []
    for tag, res in [("cal", cal_result), ("test", test_result)]:
        props = res["true_proportions"]
        for ct in props.columns:
            col = props[ct]
            prop_stats_rows.append({
                "set": tag, "cell_type": ct,
                "mean": round(float(col.mean()), 4),
                "std": round(float(col.std()), 4),
                "min": round(float(col.min()), 4),
                "max": round(float(col.max()), 4),
            })
            if tag == "test":
                logger.info("  %-12s mean=%.4f std=%.4f range=[%.4f, %.4f]",
                            ct, col.mean(), col.std(), col.min(), col.max())
    prop_stats = pd.DataFrame(prop_stats_rows)
    save_df(prop_stats, Path(output_dir) / "proportion_distribution.csv")

    # Proportion-sum check (each sample must sum to ~1)
    for tag, res in [("cal", cal_result), ("test", test_result)]:
        sums = res["true_proportions"].sum(axis=1)
        logger.info("  %s proportion sums: mean=%.6f min=%.6f max=%.6f",
                    tag, sums.mean(), sums.min(), sums.max())

    # ── Step 8: Save run metadata for downstream scripts ──
    run_meta = {
        "n_samples_total": n_total,
        "n_samples_cal": n_samples_cal,
        "n_samples_test": n_samples_test,
        "n_highly_variable_genes": n_hvg,
        "donor_aware": donor_aware,
        "cell_type_column": cell_type_col,
        "donor_column": donor_col,
        "normalization": normalization,
        "n_genes": int(cal_result["pseudobulk_matrix"].shape[1]),
        "n_cell_types": len(cal_types),
        "cell_types": sorted(cal_types),
        "seed": args.seed,
    }
    with open(Path(output_dir) / "run_metadata.json", "w") as fh:
        json.dump(run_meta, fh, indent=2, default=str)
    logger.info("Run metadata saved to run_metadata.json")

    logger.info("Phase 1b complete. [OK] ")
    logger.info("Output files in %s", output_dir)
    logger.info("Key outputs:")
    logger.info("  true_proportions_cal.csv  : %s", Path(output_dir) / "true_proportions_cal.csv")
    logger.info("  true_proportions_test.csv : %s", Path(output_dir) / "true_proportions_test.csv")
    logger.info("  pseudobulk_matrix_cal_cpm.csv  : %s", Path(output_dir) / "pseudobulk_matrix_cal_cpm.csv")
    logger.info("  pseudobulk_matrix_test_cpm.csv : %s", Path(output_dir) / "pseudobulk_matrix_test_cpm.csv")


if __name__ == "__main__":
    main()
