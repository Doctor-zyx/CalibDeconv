#!/usr/bin/env python
"""
Marker-gene experiment: compare two deconvolution configurations to fix the
poor HVG-based NNLS baseline (T-subtype collinearity collapse).

  Config A: marker-gene signature, 7 cell types (CD4_T/CD8_T/other_T kept)
  Config B: marker-gene signature, 5 cell types (T subsets merged -> T_cell)

LEAKAGE CONTROL (identical to HVG path):
  - Marker genes are selected from the REFERENCE pool ONLY
    (cell_pool_reference.h5ad). Calibration/test pseudo-bulks and their
    true proportions are NEVER consulted for gene selection.
  - Signature is built from the reference pool ONLY.

SCALE CONSISTENCY (identical to HVG path):
  - Signature: CPM normalised over ALL genes, THEN subset to markers.
  - Pseudo-bulk: full-gene CPM matrices, THEN subset to markers.

Outputs (per config) under results/nnls_markers/<tag>/:
  - selected_genes_markers.txt
  - signature_matrix_markers.csv
  - predicted_proportions_cal.csv / predicted_proportions_test.csv
  - accuracy_per_celltype.csv
A combined comparison table is printed at the end.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import setup_logger, set_seed, ensure_dir, save_df
from src.deconvolution.signature import build_signature_matrix
from src.deconvolution.nnls import deconvolve_batch
from src.data.pseudobulk import normalize_signature

# CD4_T / CD8_T / other_T are the closely-related T subsets that collapse
# under NNLS. The 5-type config merges them into a single T_cell label.
T_SUBSETS = ["CD4_T", "CD8_T", "other_T"]


def ccc(y_true, y_pred):
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    mt, mp = yt.mean(), yp.mean()
    vt, vp = yt.var(), yp.var()
    cov = ((yt - mt) * (yp - mp)).mean()
    denom = vt + vp + (mt - mp) ** 2
    return 2 * cov / denom if denom > 0 else float("nan")


def pearson(y_true, y_pred):
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    if yt.std() == 0 or yp.std() == 0:
        return float("nan")
    return np.corrcoef(yt, yp)[0, 1]


def select_markers(adata_ref, cell_type_col, pb_genes, n_markers, logger):
    """Select union of top-N up-regulated markers per cell type.

    Uses reference pool ONLY, on log-normalised counts (Wilcoxon).
    Restricts to genes present in the pseudo-bulk matrices.
    """
    import scanpy as sc

    if adata_ref.raw is not None:
        adata = adata_ref.raw.to_adata()
        adata.obs = adata_ref.obs.copy()
    else:
        adata = adata_ref.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.tl.rank_genes_groups(adata, groupby=cell_type_col, method="wilcoxon")
    names = adata.uns["rank_genes_groups"]["names"]
    groups = names.dtype.names

    marker_set = []
    seen = set()
    for grp in groups:
        top = list(names[grp][:n_markers])
        kept = 0
        for g in top:
            if g in pb_genes and g not in seen:
                marker_set.append(g)
                seen.add(g)
                kept += 1
        logger.info("  %-10s: %d/%d top markers present in pseudo-bulk", grp, kept, n_markers)
    logger.info("  Union marker panel size: %d", len(marker_set))
    return marker_set


def collapse_labels(series):
    """Map the 7-type labels to 5 types (T subsets -> T_cell)."""
    return series.map(lambda x: "T_cell" if x in T_SUBSETS else x)


def collapse_true_props(y_df):
    """Collapse true-proportion columns: sum T subsets into T_cell."""
    out = y_df.copy()
    out["T_cell"] = out[T_SUBSETS].sum(axis=1)
    out = out.drop(columns=T_SUBSETS)
    return out


def run_config(tag, cell_type_col, adata_ref, X_cal_full, X_test_full,
               y_cal, y_test, pb_genes, n_markers, out_root, seed, logger):
    logger.info("=" * 60)
    logger.info("CONFIG '%s' (cell_type column: %s)", tag, cell_type_col)
    logger.info("=" * 60)

    cell_types = sorted(adata_ref.obs[cell_type_col].unique().tolist())
    logger.info("Cell types (%d): %s", len(cell_types), cell_types)

    # ── 1. Marker selection (reference pool ONLY) ──
    logger.info("Selecting markers (reference pool only, top-%d per type)...", n_markers)
    markers = select_markers(adata_ref, cell_type_col, pb_genes, n_markers, logger)

    # ── 2. Signature: CPM over ALL genes, THEN subset to markers ──
    sig_full, _ = build_signature_matrix(adata_ref, cell_type_col, cell_types=cell_types)
    sig_norm = normalize_signature(sig_full, method="cpm")
    markers_in_sig = [g for g in markers if g in sig_norm.index]
    sig = sig_norm.loc[markers_in_sig]

    # ── 3. Pseudo-bulk: full-gene CPM, THEN subset to markers (same order) ──
    X_cal = X_cal_full[markers_in_sig]
    X_test = X_test_full[markers_in_sig]

    # ── 4. Consistency check (set + order) ──
    assert list(sig.index) == list(X_cal.columns) == list(X_test.columns), \
        f"{tag}: gene set/order mismatch between signature and pseudo-bulk!"
    logger.info("[OK] signature == cal == test genes (set+order): %d genes", len(sig))

    # ── 5. NNLS ──
    pred_cal = deconvolve_batch(X_cal, sig, verbose=False)
    pred_test = deconvolve_batch(X_test, sig, verbose=False)

    # Align true proportions to predicted columns (by name)
    cols = list(pred_test.columns)
    yt_cal = y_cal.reset_index(drop=True)[cols]
    yp_cal = pred_cal.reset_index(drop=True)[cols]
    yt_test = y_test.reset_index(drop=True)[cols]
    yp_test = pred_test.reset_index(drop=True)[cols]

    # ── 6. Metrics ──
    def overall(yt, yp):
        diff = yt.values - yp.values
        return {
            "MAE": np.abs(diff).mean(),
            "RMSE": np.sqrt((diff ** 2).mean()),
            "Pearson": pearson(yt.values, yp.values),
            "CCC": ccc(yt.values, yp.values),
        }

    m_cal = overall(yt_cal, yp_cal)
    m_test = overall(yt_test, yp_test)
    logger.info("CAL : MAE=%.4f RMSE=%.4f Pearson=%.4f CCC=%.4f",
                m_cal["MAE"], m_cal["RMSE"], m_cal["Pearson"], m_cal["CCC"])
    logger.info("TEST: MAE=%.4f RMSE=%.4f Pearson=%.4f CCC=%.4f",
                m_test["MAE"], m_test["RMSE"], m_test["Pearson"], m_test["CCC"])

    per_rows = []
    logger.info("Per-cell-type (test):")
    logger.info("  %-10s %8s %8s %10s %10s", "type", "MAE", "CCC", "true_mean", "pred_mean")
    for ct in cols:
        mae_ct = np.abs(yt_test[ct].values - yp_test[ct].values).mean()
        ccc_ct = ccc(yt_test[ct].values, yp_test[ct].values)
        logger.info("  %-10s %8.4f %8.4f %10.4f %10.4f",
                    ct, mae_ct, ccc_ct, yt_test[ct].mean(), yp_test[ct].mean())
        per_rows.append({"cell_type": ct, "MAE": mae_ct, "CCC": ccc_ct,
                         "true_mean": yt_test[ct].mean(), "pred_mean": yp_test[ct].mean()})

    # ── 7. Save ──
    out_dir = ensure_dir(str(Path(out_root) / tag))
    with open(Path(out_dir) / "selected_genes_markers.txt", "w") as fh:
        for g in markers_in_sig:
            fh.write(g + "\n")
    save_df(sig, Path(out_dir) / "signature_matrix_markers.csv")
    save_df(pred_cal, Path(out_dir) / "predicted_proportions_cal.csv")
    save_df(pred_test, Path(out_dir) / "predicted_proportions_test.csv")
    save_df(pd.DataFrame(per_rows), Path(out_dir) / "accuracy_per_celltype.csv")
    logger.info("Saved config '%s' outputs to %s", tag, out_dir)

    return {
        "tag": tag, "n_types": len(cols), "n_markers": len(markers_in_sig),
        "cal": m_cal, "test": m_test, "per_type": per_rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Marker-gene 7-type vs 5-type experiment")
    parser.add_argument("--cell-pool-dir", default="data/processed")
    parser.add_argument("--pseudobulk-dir", default="data/processed")
    parser.add_argument("--output-dir", default="results/nnls_markers")
    parser.add_argument("--cell-type-col", default="cell_type")
    parser.add_argument("--n-markers", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger = setup_logger("03c_marker_exp", log_dir=str(PROJECT_ROOT / "logs"))
    set_seed(args.seed)

    pool_dir = Path(PROJECT_ROOT / args.cell_pool_dir)
    pb_dir = Path(PROJECT_ROOT / args.pseudobulk_dir)
    out_root = ensure_dir(str(PROJECT_ROOT / args.output_dir))

    import scanpy as sc
    adata_ref = sc.read_h5ad(pool_dir / "cell_pool_reference.h5ad")
    logger.info("Reference pool: %d cells x %d genes", *adata_ref.shape)

    # Full-gene CPM pseudo-bulk (NOT the HVG subset)
    X_cal_full = pd.read_csv(pb_dir / "pseudobulk_matrix_cal_cpm.csv", index_col=0)
    X_test_full = pd.read_csv(pb_dir / "pseudobulk_matrix_test_cpm.csv", index_col=0)
    pb_genes = set(X_cal_full.columns)

    y_cal = pd.read_csv(pb_dir / "true_proportions_cal.csv", index_col=0)
    y_test = pd.read_csv(pb_dir / "true_proportions_test.csv", index_col=0)

    results = []

    # ── Config A: 7 types ──
    results.append(run_config(
        "marker_7type", args.cell_type_col, adata_ref,
        X_cal_full, X_test_full, y_cal, y_test,
        pb_genes, args.n_markers, out_root, args.seed, logger,
    ))

    # ── Config B: 5 types (collapse T subsets) ──
    adata_ref5 = adata_ref.copy()
    adata_ref5.obs["cell_type5"] = collapse_labels(adata_ref5.obs[args.cell_type_col]).astype("category")
    y_cal5 = collapse_true_props(y_cal)
    y_test5 = collapse_true_props(y_test)
    results.append(run_config(
        "marker_5type", "cell_type5", adata_ref5,
        X_cal_full, X_test_full, y_cal5, y_test5,
        pb_genes, args.n_markers, out_root, args.seed, logger,
    ))

    # ── Comparison summary ──
    logger.info("=" * 60)
    logger.info("COMPARISON SUMMARY (test set)")
    logger.info("=" * 60)
    logger.info("  %-14s %6s %8s %8s %8s %8s",
                "config", "types", "MAE", "RMSE", "Pearson", "CCC")
    for r in results:
        t = r["test"]
        logger.info("  %-14s %6d %8.4f %8.4f %8.4f %8.4f",
                    r["tag"], r["n_types"], t["MAE"], t["RMSE"], t["Pearson"], t["CCC"])

    # Highlight degenerate (CCC==0) types per config
    for r in results:
        degen = [p["cell_type"] for p in r["per_type"] if abs(p["CCC"]) < 1e-9]
        logger.info("  %s degenerate (CCC~0) types: %s", r["tag"], degen if degen else "none")

    save_df(pd.DataFrame([
        {"config": r["tag"], "n_types": r["n_types"], "n_markers": r["n_markers"],
         **{f"test_{k}": v for k, v in r["test"].items()},
         **{f"cal_{k}": v for k, v in r["cal"].items()}}
        for r in results
    ]), Path(out_root) / "comparison_summary.csv")
    logger.info("Comparison summary saved to %s", Path(out_root) / "comparison_summary.csv")


if __name__ == "__main__":
    main()
