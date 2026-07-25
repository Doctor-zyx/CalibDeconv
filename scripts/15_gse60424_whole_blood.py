#!/usr/bin/env python
"""
GSE60424 whole-blood domain-shift evaluation (end-to-end).

Reproduces the GSE60424 analysis from downloaded GEO files:
1. Parse series_matrix metadata → extract FACS-sort cell counts
2. Select 14 donors with complete 4-class ground truth (exclude 6 lacking NK)
3. Map Ensembl IDs → HGNC symbols via annotation table
4. Normalize whole-blood expression to CPM
5. Run frozen NNLS point estimate + bootstrap ensemble (B=50)
6. Compute MAE, CCC, Pearson r, per-cell-type metrics
7. Save predictions, ground truth, metrics, and Figure S6 source data

Data requirements (all under data/real_bulk/gse60424/):
  - counts.txt.gz : GEO supplementary file (expression matrix)
  - series_matrix.txt.gz : GEO series matrix (sample metadata)
  - Annotation_genes.txt : Ensembl ID → Gene Symbol mapping

Frozen reference (under data/processed/ and results/):
  - cell_pool_reference.h5ad : single-cell reference atlas
  - selected_genes_markers.txt : 445-gene frozen marker panel
  - config/config.yaml : ensemble hyperparameters
"""

import argparse
import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import load_config, setup_logger, set_seed, ensure_dir
from src.deconvolution.nnls import deconvolve_nnls

# Reuse the validated fast-ensemble core from Phase 3.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "ens04b", str(PROJECT_ROOT / "scripts" / "04b_ensemble_marker5.py"))
_ens = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_ens)
build_panel_celltype_arrays = _ens.build_panel_celltype_arrays
fast_ensemble = _ens.fast_ensemble
summarize = _ens.summarize

# 4-class cell types for whole-blood evaluation
CLASSES_4 = ["Monocyte", "NK", "B", "T_cell"]
# 5-class cell types in the frozen reference
CANON_5 = ["Monocyte", "NK", "B", "DC", "T_cell"]

SEED = 42


# ── Step 1: Parse GEO metadata ──────────────────────────────────────────────

def parse_series_matrix(path):
    """Extract per-sample metadata from GEO series_matrix.txt.gz."""
    fields = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("!Sample_title"):
                fields["title"] = [v.strip().strip('"') for v in line.strip().split("\t")[1:]]
            elif line.startswith("!Sample_characteristics"):
                vals = [v.strip().strip('"') for v in line.strip().split("\t")[1:]]
                key = vals[0].split(":")[0].strip()
                fields[key] = [v.split(":", 1)[1].strip() if ":" in v else v for v in vals]
    return pd.DataFrame(fields)


def compute_ground_truth(meta):
    """Derive 4-class reference fractions from FACS-sort cell counts.

    For each donor: cellcount values are extracted for sorted immune-cell subsets.
    CD4 and CD8 T-cell counts are summed into T_cell. Fractions are computed
    over Monocytes, NK, B-cells, and T_cell, renormalized to sum to one.

    Returns (gt_df, donor_list, excluded_donors):
      gt_df: DataFrame (14 donors × 4 cell types)
      donor_list: list of donor IDs retained
      excluded_donors: list of donor IDs excluded (no NK)
    """
    # Map GEO celltype labels to our 4-class names
    type_map = {
        "Monocytes": "Monocyte",
        "NK": "NK",
        "B-cells": "B",
        "CD4": "T_cell",
        "CD8": "T_cell",
    }

    # Filter to sorted (non-WB, non-Neutrophil) samples with valid cellcount
    sorted_mask = (~meta["celltype"].isin(["Whole Blood", "Neutrophils"])) & \
                  (meta["cellcount"] != "--")
    sorted_meta = meta[sorted_mask].copy()
    sorted_meta["mapped_type"] = sorted_meta["celltype"].map(type_map)
    sorted_meta["cellcount_num"] = sorted_meta["cellcount"].astype(float)

    # Aggregate per donor: sum CD4+CD8 into T_cell
    agg = sorted_meta.groupby(["donorid", "mapped_type"])["cellcount_num"].sum().reset_index()

    # Find donors with all 4 classes
    donor_types = agg.groupby("donorid")["mapped_type"].apply(set)
    required = set(CLASSES_4)
    complete_donors = [d for d, ts in donor_types.items() if ts >= required]
    excluded_donors = [d for d, ts in donor_types.items() if ts < required]

    # Build ground truth table
    rows = []
    for donor in sorted(complete_donors, key=int):
        d_agg = agg[agg["donorid"] == donor].set_index("mapped_type")["cellcount_num"]
        total = sum(d_agg.get(ct, 0) for ct in CLASSES_4)
        fracs = {ct: d_agg.get(ct, 0) / total for ct in CLASSES_4}
        rows.append({"donor": int(donor), **fracs})

    gt_df = pd.DataFrame(rows).set_index("donor")
    return gt_df, [int(d) for d in complete_donors], [int(d) for d in excluded_donors]


# ── Step 2: Gene mapping ────────────────────────────────────────────────────

def build_ensembl_to_symbol(adata_ref):
    """Use Hao et al. reference atlas ensembl_id → gene_symbol mapping.

    This matches the original analysis and gives 438/445 panel gene overlap
    (the GEO-provided Annotation_genes.txt maps far fewer genes).
    """
    ens2sym = dict(zip(adata_ref.var["ensembl_id"], adata_ref.var_names))
    return ens2sym


# ── Step 3: Expression processing ───────────────────────────────────────────

def load_expression(counts_path, meta, donor_list, ens2sym, gene_panel, logger):
    """Load expression for whole-blood samples of selected donors.

    Returns expression DataFrame (n_donors × n_panel_genes) in CPM.
    """
    # Identify whole-blood library IDs for selected donors
    wb_mask = (meta["celltype"] == "Whole Blood") & \
              (meta["donorid"].astype(int).isin(donor_list))
    wb_meta = meta[wb_mask].copy()
    lib_to_donor = dict(zip(wb_meta["title"], wb_meta["donorid"].astype(int)))
    lib_ids = list(lib_to_donor.keys())
    logger.info("  WB libraries to load: %d for %d donors", len(lib_ids), len(donor_list))

    # Load expression matrix (Ensembl IDs × libraries)
    counts = pd.read_csv(counts_path, sep="\t", index_col=0, compression="gzip")
    counts = counts[lib_ids]  # subset to WB libraries

    # Map Ensembl → HGNC symbols
    counts.index = counts.index.map(lambda x: ens2sym.get(x, None))
    counts = counts[counts.index.notna()]
    # Drop duplicate gene symbols (keep first = highest expression on average)
    counts = counts[~counts.index.duplicated(keep="first")]

    # Overlap with frozen panel
    panel_in_data = [g for g in gene_panel if g in counts.index]
    logger.info("  Gene mapping: %d Ensembl → %d unique HGNC symbols",
                len(ens2sym), len(counts))
    logger.info("  Panel overlap: %d / %d marker genes present", len(panel_in_data), len(gene_panel))

    # Normalize to CPM (using full gene set for denominator)
    col_sums = counts.sum(axis=0)
    logger.info("  Library sizes: min=%.0f, max=%.0f, median=%.0f",
                col_sums.min(), col_sums.max(), col_sums.median())
    cpm = counts / col_sums * 1e6

    # Subset to panel genes, transpose to (samples × genes)
    expr = cpm.loc[panel_in_data].T
    # Rename columns to donors
    expr.index = [lib_to_donor[lib] for lib in expr.index]
    expr.index.name = "donor"

    # Fill missing panel genes with 0
    for g in gene_panel:
        if g not in expr.columns:
            expr[g] = 0.0
    expr = expr[gene_panel]  # reorder to panel order

    return expr, len(panel_in_data)


# ── Step 4: Deconvolution ────────────────────────────────────────────────────

def ccc(y_true, y_pred):
    """Lin's concordance correlation coefficient."""
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    mt, mp = yt.mean(), yp.mean()
    vt, vp = yt.var(), yp.var()
    cov = ((yt - mt) * (yp - mp)).mean()
    d = vt + vp + (mt - mp) ** 2
    return float(2 * cov / d) if d > 0 else float("nan")


def run_deconvolution(expr_cpm, gene_panel, adata_ref, cell_type_col, cell_types,
                      n_iterations, gene_frac, cell_frac, logger):
    """Run NNLS point estimate and bootstrap ensemble on whole-blood CPM.

    Returns: (pred_single, pred_ensemble, std_ensemble, sample_diagnostics)
    All DataFrames indexed by donor.
    """
    # Build per-cell-type arrays for ensemble
    per_type_arrays, per_type_totals = build_panel_celltype_arrays(
        adata_ref, cell_type_col, cell_types, gene_panel, logger)

    # Build frozen signature (mean CPM per cell type)
    sig = np.zeros((len(gene_panel), len(cell_types)), dtype=np.float64)
    for j, (arr, totals) in enumerate(zip(per_type_arrays, per_type_totals)):
        col = arr.mean(axis=0)
        denom = totals.mean()
        sig[:, j] = col / denom * 1e6 if denom > 0 else col

    donors = list(expr_cpm.index)
    X = expr_cpm.values.astype(np.float64)

    # Single NNLS
    pred_single = np.zeros((len(donors), len(cell_types)))
    for i in range(len(donors)):
        pred_single[i] = deconvolve_nnls(X[i], sig, normalize=True)

    # Bootstrap ensemble
    all_preds = fast_ensemble(
        X_panel=X, per_type_arrays=per_type_arrays,
        per_type_totals=per_type_totals, cell_types=cell_types,
        n_panel_genes=len(gene_panel), n_iterations=n_iterations,
        gene_frac=gene_frac, cell_frac=cell_frac,
        noise_std=0.0, seed=SEED, logger=logger, tag="gse60424")
    summ = summarize(all_preds, cell_types, donors)

    # Extract ensemble mean and std (wide format, 4 classes only)
    def wide(stat, cols):
        w = summ.pivot_table(index="sample_id", columns="cell_type",
                             values=stat, aggfunc="first")
        w = w[[c for c in cols if c in w.columns]]
        return w

    ens_mean = wide("mean", CLASSES_4)
    ens_std = wide("std", CLASSES_4)

    # Build DataFrames
    pred_single_df = pd.DataFrame(pred_single, index=donors, columns=cell_types)
    pred_single_df.index.name = "donor"
    pred_4class = pred_single_df[CLASSES_4]

    # Sample diagnostics
    diag_rows = []
    for donor in donors:
        ens_vals = ens_mean.loc[donor].values
        std_vals = ens_std.loc[donor].values
        diag_rows.append({
            "donor": donor,
            "mean_std": float(std_vals.mean()),
        })
    diag_df = pd.DataFrame(diag_rows).set_index("donor")

    return pred_4class, ens_mean, ens_std, diag_df


def parse_args():
    p = argparse.ArgumentParser(description="GSE60424 whole-blood domain-shift evaluation")
    p.add_argument("--data-dir", default="data/real_bulk/gse60424")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--cell-pool-dir", default="data/processed")
    p.add_argument("--output-dir", default="results/phase6b_gse60424")
    p.add_argument("--n-iterations", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    global SEED
    SEED = args.seed
    logger = setup_logger("15_gse60424", log_dir=str(PROJECT_ROOT / "logs"))
    set_seed(args.seed)

    data_dir = PROJECT_ROOT / args.data_dir
    out_dir = ensure_dir(str(PROJECT_ROOT / args.output_dir))

    logger.info("=" * 60)
    logger.info("GSE60424 whole-blood domain-shift evaluation")
    logger.info("=" * 60)

    # ── 1. Parse metadata ──
    logger.info("[1] Parsing GEO series_matrix metadata...")
    meta = parse_series_matrix(data_dir / "series_matrix.txt.gz")
    logger.info("  Total samples: %d", len(meta))
    logger.info("  Cell types: %s", sorted(meta["celltype"].unique()))

    # ── 2. Compute ground truth ──
    logger.info("[2] Computing sort-derived reference fractions...")
    gt, donor_list, excluded = compute_ground_truth(meta)
    logger.info("  Complete donors (n=%d): %s", len(donor_list), donor_list)
    logger.info("  Excluded donors (n=%d, no NK): %s", len(excluded), excluded)
    gt.to_csv(Path(out_dir) / "ground_truth_4class.csv")

    # ── 3. Gene mapping ──
    logger.info("[3] Building Ensembl → HGNC symbol mapping from reference atlas...")
    import scanpy as sc
    adata_ref = sc.read_h5ad(PROJECT_ROOT / args.cell_pool_dir / "cell_pool_reference.h5ad")
    ens2sym = build_ensembl_to_symbol(adata_ref)
    logger.info("  Ensembl→symbol mappings from reference: %d", len(ens2sym))

    # ── 4. Load panel ──
    panel_file = PROJECT_ROOT / "data" / "processed" / "selected_genes_markers.txt"
    with open(panel_file) as fh:
        gene_panel = [l.strip() for l in fh if l.strip()]
    logger.info("[4] Frozen marker panel: %d genes", len(gene_panel))

    # ── 5. Load and process expression ──
    logger.info("[5] Loading and normalizing whole-blood expression...")
    expr_cpm, n_overlap = load_expression(
        data_dir / "counts.txt.gz", meta, donor_list, ens2sym, gene_panel, logger)
    logger.info("  Expression matrix: %s", expr_cpm.shape)

    # ── 6. Load reference config ──
    logger.info("[6] Loading reference config...")
    cfg = load_config(args.config)
    ens_cfg = cfg["ensemble"]
    gene_frac = ens_cfg["gene_sampling_fraction"]
    cell_frac = ens_cfg["cell_sampling_fraction"]
    base_col = cfg["data"]["sc_reference"]["cell_type_column"]

    from src.deconvolution.celltypes import resolve_cell_type_column
    cell_type_col = resolve_cell_type_column(adata_ref, "5type", base_col=base_col)
    cell_types = sorted(adata_ref.obs[cell_type_col].unique().tolist())
    logger.info("  Reference: %d cells, %d types (%s)",
                adata_ref.shape[0], len(cell_types), cell_types)

    # ── 7. Deconvolution ──
    logger.info("[7] Running NNLS + ensemble (B=%d)...", args.n_iterations)
    pred_single, pred_ens, std_ens, diag = run_deconvolution(
        expr_cpm, gene_panel, adata_ref, cell_type_col, cell_types,
        args.n_iterations, gene_frac, cell_frac, logger)

    # ── 8. Metrics ──
    logger.info("[8] Computing metrics...")
    gt_aligned = gt.loc[pred_ens.index, CLASSES_4]

    ae = np.abs(gt_aligned.values - pred_ens.values)
    mae = float(ae.mean())
    pr = np.corrcoef(gt_aligned.values.ravel(), pred_ens.values.ravel())[0, 1]
    from scipy.stats import spearmanr
    sr, _ = spearmanr(gt_aligned.values.ravel(), pred_ens.values.ravel())
    ccc_val = ccc(gt_aligned.values, pred_ens.values)

    metrics = pd.DataFrame([{
        "MAE": mae, "Pearson": float(pr), "Spearman": float(sr),
        "CCC": ccc_val, "n": len(donor_list), "classes": 4,
        "marker_overlap": n_overlap,
    }])

    # Per-sample diagnostics
    for donor in gt_aligned.index:
        diag.loc[donor, "sample_MAE"] = float(
            np.abs(gt_aligned.loc[donor].values - pred_ens.loc[donor].values).mean())
    diag = diag[["sample_MAE", "mean_std"]]

    # ── 9. Save ──
    logger.info("[9] Saving results...")
    pred_single.to_csv(Path(out_dir) / "predicted_proportions_4class.csv")
    pred_ens.to_csv(Path(out_dir) / "predicted_proportions_ensemble_4class.csv")
    std_ens.to_csv(Path(out_dir) / "ensemble_std_4class.csv")
    metrics.to_csv(Path(out_dir) / "metrics_overall.csv", index=False)
    diag.to_csv(Path(out_dir) / "sample_diagnostics.csv")

    # ── 10. Print summary ──
    print("\n" + "=" * 60)
    print("GSE60424 WHOLE-BLOOD DOMAIN-SHIFT EVALUATION")
    print("=" * 60)
    print(f"\nDonors: {len(donor_list)} (excluded {len(excluded)} without NK)")
    print(f"Marker panel: {len(gene_panel)} genes ({n_overlap} present in data)")
    print(f"\nOverall metrics (ensemble mean, 4-class):")
    print(f"  MAE    = {mae:.6f}")
    print(f"  CCC    = {ccc_val:.6f}")
    print(f"  Pearson= {float(pr):.6f}")
    print(f"  Spearman={float(sr):.6f}")

    print(f"\nPer-cell-type MAE:")
    for ct in CLASSES_4:
        ct_mae = float(np.abs(gt_aligned[ct].values - pred_ens[ct].values).mean())
        ct_ccc = ccc(gt_aligned[ct].values, pred_ens[ct].values)
        print(f"  {ct:<10} MAE={ct_mae:.4f}  CCC={ct_ccc:.4f}")

    print(f"\nGround truth (first 3 donors):")
    print(gt.head(3).to_string())
    print(f"\nEnsemble predictions (first 3 donors):")
    print(pred_ens.head(3).to_string())
    print(f"\nEnsemble std (first 3 donors):")
    print(std_ens.head(3).to_string())
    print("=" * 60)

    logger.info("GSE60424 analysis complete. Results in %s", out_dir)


if __name__ == "__main__":
    main()
