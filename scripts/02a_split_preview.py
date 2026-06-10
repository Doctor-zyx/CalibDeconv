#!/usr/bin/env python
"""
Split preview: run the donor-aware cell-pool split and report diagnostics
WITHOUT generating any pseudo-bulk samples.

This is a safety gate: it confirms that every pool contains all major
cell types with enough cells, donors do not overlap, and barcodes are
disjoint — before committing to the (slower) pseudo-bulk generation.

Usage::

    python scripts/02a_split_preview.py \\
        --adata data/raw/pbmc_reference.h5ad \\
        --cell-type-col cell_type \\
        --donor-col donor_id \\
        --min-cells 100 \\
        --seed 42
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import load_config, setup_logger, set_seed, ensure_dir, save_df
from src.data.pseudobulk import split_cell_pools


def main():
    parser = argparse.ArgumentParser(description="Split preview (no pseudo-bulk)")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--adata", required=True)
    parser.add_argument("--cell-type-col", default="cell_type")
    parser.add_argument("--donor-col", default="donor_id")
    parser.add_argument("--donor-aware", dest="donor_aware", action="store_true", default=True)
    parser.add_argument("--no-donor-aware", dest="donor_aware", action="store_false")
    parser.add_argument("--min-cells", type=int, default=100)
    parser.add_argument("--output-dir", default="results/data_prep")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("02a_split_preview", log_dir=str(PROJECT_ROOT / "logs"))
    set_seed(args.seed)

    out_dir = ensure_dir(str(PROJECT_ROOT / args.output_dir))

    import scanpy as sc
    adata_path = PROJECT_ROOT / args.adata if not Path(args.adata).is_absolute() else Path(args.adata)
    logger.info("Loading AnnData from %s", adata_path)
    adata = sc.read_h5ad(adata_path)
    logger.info("Loaded: %d cells x %d genes", *adata.shape)

    ct_col = args.cell_type_col
    donor_col = args.donor_col
    min_cells = args.min_cells

    # ── 1. Full data donor × cell_type table ──
    logger.info("=" * 60)
    logger.info("STEP 1: Full data donor x cell_type distribution")
    logger.info("=" * 60)
    full_table = adata.obs.groupby([donor_col, ct_col], observed=True).size().unstack(fill_value=0)
    logger.info("\n%s", full_table.to_string())
    save_df(full_table.reset_index(), Path(out_dir) / "split_preview_donor_celltype_all.csv")

    n_major = adata.obs[ct_col].nunique()
    logger.info("Total cell types: %d", n_major)
    logger.info("Total donors: %d (%s)", adata.obs[donor_col].nunique(),
                sorted(adata.obs[donor_col].astype(str).unique()))

    # ── 2. Run the split ──
    logger.info("=" * 60)
    logger.info("STEP 2: Cell-pool split (donor_aware=%s)", args.donor_aware)
    logger.info("=" * 60)
    pool_ratios = {
        "reference": cfg["cell_pool_split"]["reference"],
        "calibration": cfg["cell_pool_split"]["calibration"],
        "test": cfg["cell_pool_split"]["test"],
    }
    pools = split_cell_pools(
        adata=adata,
        cell_type_column=ct_col,
        ratios=pool_ratios,
        donor_column=donor_col if args.donor_aware else None,
        seed=args.seed,
    )

    # ── 3. Barcode intersection (must be 0) ──
    logger.info("=" * 60)
    logger.info("STEP 3: Barcode intersection check (must all be 0)")
    logger.info("=" * 60)
    ref_b = set(pools["reference"].obs_names)
    cal_b = set(pools["calibration"].obs_names)
    test_b = set(pools["test"].obs_names)
    i_rc, i_rt, i_ct = len(ref_b & cal_b), len(ref_b & test_b), len(cal_b & test_b)
    logger.info("  ref ∩ cal  = %d", i_rc)
    logger.info("  ref ∩ test = %d", i_rt)
    logger.info("  cal ∩ test = %d", i_ct)
    barcodes_ok = (i_rc == 0 and i_rt == 0 and i_ct == 0)

    # ── 4. Donor lists per pool + overlap ──
    logger.info("=" * 60)
    logger.info("STEP 4: Donor lists per pool")
    logger.info("=" * 60)
    donor_sets = {}
    for name in ["reference", "calibration", "test"]:
        ds = sorted(pools[name].obs[donor_col].astype(str).unique())
        donor_sets[name] = set(ds)
        logger.info("  %-12s donors=%s (%d cells)", name, ds, pools[name].shape[0])
    donor_overlap = (donor_sets["reference"] & donor_sets["calibration"]) | \
                    (donor_sets["reference"] & donor_sets["test"]) | \
                    (donor_sets["calibration"] & donor_sets["test"])
    donors_ok = (len(donor_overlap) == 0)
    if donors_ok:
        logger.info("  [OK] Donors do NOT overlap across pools")
    else:
        logger.warning("  [WARN] Donor overlap: %s", donor_overlap)

    # ── 5. Per-pool cell-type distribution + min-cell check ──
    logger.info("=" * 60)
    logger.info("STEP 5: Per-pool cell-type counts (threshold=%d)", min_cells)
    logger.info("=" * 60)
    all_types = sorted(adata.obs[ct_col].unique().tolist())
    rows = []
    all_types_present = True
    all_above_min = True
    for name in ["reference", "calibration", "test"]:
        counts = pools[name].obs[ct_col].value_counts().to_dict()
        logger.info("  --- %s pool ---", name)
        for ct in all_types:
            n = counts.get(ct, 0)
            flag = ""
            if n == 0:
                flag = " <-- MISSING!"
                all_types_present = False
                all_above_min = False
            elif n < min_cells:
                flag = f" <-- BELOW {min_cells}!"
                all_above_min = False
            logger.info("      %-12s %6d%s", ct, n, flag)
            rows.append({"pool": name, "cell_type": ct, "n_cells": n,
                         "below_min": n < min_cells, "missing": n == 0})
    pool_ct_df = pd.DataFrame(rows)
    save_df(pool_ct_df, Path(out_dir) / "split_preview_pool_celltype.csv")

    # Save donor assignment
    donor_assign = []
    for name, ds in donor_sets.items():
        for d in sorted(ds):
            donor_assign.append({"pool": name, "donor_id": d})
    save_df(pd.DataFrame(donor_assign), Path(out_dir) / "split_preview_donor_assignment.csv")

    # ── 6. Verdict ──
    logger.info("=" * 60)
    logger.info("VERDICT")
    logger.info("=" * 60)
    logger.info("  Barcodes disjoint (all intersections 0): %s", barcodes_ok)
    logger.info("  Donors non-overlapping across pools:      %s", donors_ok)
    logger.info("  All %d major types present in every pool:  %s", len(all_types), all_types_present)
    logger.info("  Every cell type >= %d cells in every pool: %s", min_cells, all_above_min)

    passed = barcodes_ok and donors_ok and all_types_present and all_above_min
    if passed:
        logger.info("  >>> SPLIT PREVIEW PASSED. Safe to generate pseudo-bulk. <<<")
    else:
        logger.warning("  >>> SPLIT PREVIEW FAILED. Do NOT generate pseudo-bulk yet. <<<")
        if not all_above_min:
            logger.warning("  Suggestion: a cell type is too small in one pool.")
            logger.warning("    - Try a different seed to reshuffle donor assignment, OR")
            logger.warning("    - Adjust split ratios (e.g. give test/cal more donors), OR")
            logger.warning("    - Lower min_cells if the type is genuinely rare, OR")
            logger.warning("    - Merge that rare type into a broader lineage.")

    logger.info("Diagnostics saved to %s", out_dir)
    return passed


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
