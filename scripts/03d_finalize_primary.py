#!/usr/bin/env python
"""
Finalize the PRIMARY analysis line: marker-gene signature, 5 collapsed cell types.

This consolidates the experiment outputs into canonical pipeline files and
records the three-way baseline comparison. It does NOT delete the 7-type
diagnostic results — they are copied into clearly named directories.

PRIMARY line:
  cell_type_set = 5type_collapsed   (T_cell = CD4_T+CD8_T+other_T; B; NK; Monocyte; DC)
  gene_set      = markers           (reference-pool DE markers, 445 genes)
  scale         = CPM
  gene_selection_source = reference_pool_only

Reproducibility:
  The 5-type marker panel is RECOMPUTED here (n_markers=100, seed=42) and
  asserted byte-identical to the experiment's saved gene list, so the
  canonical files are guaranteed consistent.
"""

import argparse
import json
import shutil
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

T_SUBSETS = ["CD4_T", "CD8_T", "other_T"]
PRIMARY_TYPES = ["T_cell", "B", "NK", "Monocyte", "DC"]


# ── metric helpers ──────────────────────────────────────────────────────────
def ccc(y_true, y_pred):
    yt = np.asarray(y_true).ravel(); yp = np.asarray(y_pred).ravel()
    mt, mp = yt.mean(), yp.mean(); vt, vp = yt.var(), yp.var()
    cov = ((yt - mt) * (yp - mp)).mean()
    denom = vt + vp + (mt - mp) ** 2
    return 2 * cov / denom if denom > 0 else float("nan")


def pearson(y_true, y_pred):
    yt = np.asarray(y_true).ravel(); yp = np.asarray(y_pred).ravel()
    if yt.std() == 0 or yp.std() == 0:
        return float("nan")
    return np.corrcoef(yt, yp)[0, 1]


def overall_metrics(yt, yp):
    diff = yt.values - yp.values
    return {
        "MAE": float(np.abs(diff).mean()),
        "RMSE": float(np.sqrt((diff ** 2).mean())),
        "Pearson": float(pearson(yt.values, yp.values)),
        "CCC": float(ccc(yt.values, yp.values)),
    }


def read_pred(path, cols):
    """Read a predictions CSV (saved without index) and order columns."""
    df = pd.read_csv(path)
    if df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=df.columns[0])
    return df.reset_index(drop=True)[cols]


def select_markers(adata_ref, cell_type_col, pb_genes, n_markers, logger):
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
    marker_set, seen = [], set()
    for grp in names.dtype.names:
        for g in list(names[grp][:n_markers]):
            if g in pb_genes and g not in seen:
                marker_set.append(g); seen.add(g)
    logger.info("Recomputed marker panel: %d genes", len(marker_set))
    return marker_set


def collapse_labels(series):
    return series.map(lambda x: "T_cell" if x in T_SUBSETS else x)


def collapse_true_props(y_df):
    out = y_df.copy()
    out["T_cell"] = out[T_SUBSETS].sum(axis=1)
    return out.drop(columns=T_SUBSETS)


def main():
    parser = argparse.ArgumentParser(description="Finalize primary marker_5type line")
    parser.add_argument("--cell-pool-dir", default="data/processed")
    parser.add_argument("--pseudobulk-dir", default="data/processed")
    parser.add_argument("--n-markers", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger = setup_logger("03d_finalize", log_dir=str(PROJECT_ROOT / "logs"))
    set_seed(args.seed)

    proc = Path(PROJECT_ROOT / "data" / "processed")
    pool_dir = Path(PROJECT_ROOT / args.cell_pool_dir)
    pb_dir = Path(PROJECT_ROOT / args.pseudobulk_dir)
    res = Path(PROJECT_ROOT / "results")

    out_primary = ensure_dir(str(res / "nnls_marker_5types"))
    out_cmp = ensure_dir(str(res / "nnls_comparison"))

    import scanpy as sc
    adata_ref = sc.read_h5ad(pool_dir / "cell_pool_reference.h5ad")
    logger.info("Reference pool: %d cells x %d genes", *adata_ref.shape)

    X_cal_full = pd.read_csv(pb_dir / "pseudobulk_matrix_cal_cpm.csv", index_col=0)
    X_test_full = pd.read_csv(pb_dir / "pseudobulk_matrix_test_cpm.csv", index_col=0)
    pb_genes = set(X_cal_full.columns)
    y_cal7 = pd.read_csv(pb_dir / "true_proportions_cal.csv", index_col=0)
    y_test7 = pd.read_csv(pb_dir / "true_proportions_test.csv", index_col=0)

    # ── 1. PRIMARY: marker_5type pipeline (recompute) ──
    logger.info("=" * 60)
    logger.info("PRIMARY: marker_5type")
    logger.info("=" * 60)
    adata5 = adata_ref.copy()
    adata5.obs["cell_type5"] = collapse_labels(adata5.obs["cell_type"]).astype("category")
    types5 = sorted(adata5.obs["cell_type5"].unique().tolist())
    logger.info("5 collapsed cell types: %s", types5)

    markers = select_markers(adata5, "cell_type5", pb_genes, args.n_markers, logger)

    # Reproducibility assert vs experiment output
    exp_panel_path = res / "nnls_markers" / "marker_5type" / "selected_genes_markers.txt"
    if exp_panel_path.exists():
        with open(exp_panel_path) as fh:
            exp_panel = [l.strip() for l in fh if l.strip()]
        assert markers == exp_panel, "Recomputed marker panel differs from experiment output!"
        logger.info("[OK] Recomputed panel matches experiment (%d genes)", len(markers))

    sig_full, _ = build_signature_matrix(adata5, "cell_type5", cell_types=types5)
    sig_norm = normalize_signature(sig_full, method="cpm")
    markers_in_sig = [g for g in markers if g in sig_norm.index]
    sig = sig_norm.loc[markers_in_sig]

    X_cal = X_cal_full[markers_in_sig]
    X_test = X_test_full[markers_in_sig]
    assert list(sig.index) == list(X_cal.columns) == list(X_test.columns), "gene order mismatch!"

    y_cal5 = collapse_true_props(y_cal7)
    y_test5 = collapse_true_props(y_test7)

    pred_cal = deconvolve_batch(X_cal, sig, verbose=False)
    pred_test = deconvolve_batch(X_test, sig, verbose=False)

    cols5 = list(pred_test.columns)
    yt_cal = y_cal5.reset_index(drop=True)[cols5]
    yt_test = y_test5.reset_index(drop=True)[cols5]
    yp_cal = pred_cal.reset_index(drop=True)[cols5]
    yp_test = pred_test.reset_index(drop=True)[cols5]

    m_cal = overall_metrics(yt_cal, yp_cal)
    m_test = overall_metrics(yt_test, yp_test)

    per_rows = []
    for ct in cols5:
        per_rows.append({
            "cell_type": ct,
            "MAE": float(np.abs(yt_test[ct].values - yp_test[ct].values).mean()),
            "CCC": float(ccc(yt_test[ct].values, yp_test[ct].values)),
            "true_mean": float(yt_test[ct].mean()),
            "pred_mean": float(yp_test[ct].mean()),
        })
    per_df = pd.DataFrame(per_rows)

    # ── 2. Save canonical PRIMARY files ──
    # data/processed canonical inputs
    with open(proc / "selected_genes_markers.txt", "w") as fh:
        for g in markers_in_sig:
            fh.write(g + "\n")
    save_df(y_cal5, proc / "true_proportions_cal_5type.csv", index=True)
    save_df(y_test5, proc / "true_proportions_test_5type.csv", index=True)
    X_cal.to_csv(proc / "pseudobulk_matrix_cal_markers_cpm.csv")
    X_test.to_csv(proc / "pseudobulk_matrix_test_markers_cpm.csv")

    # results/nnls_marker_5types outputs
    save_df(sig, Path(out_primary) / "signature_matrix_markers.csv", index=True)
    save_df(pred_cal, Path(out_primary) / "predicted_proportions_cal.csv")
    save_df(pred_test, Path(out_primary) / "predicted_proportions_test.csv")
    save_df(pd.DataFrame([m_cal]), Path(out_primary) / "metrics_cal.csv")
    save_df(pd.DataFrame([m_test]), Path(out_primary) / "metrics_test.csv")
    save_df(per_df, Path(out_primary) / "marker_5type_summary.csv")
    logger.info("Saved primary files. TEST: MAE=%.4f RMSE=%.4f Pearson=%.4f CCC=%.4f",
                m_test["MAE"], m_test["RMSE"], m_test["Pearson"], m_test["CCC"])

    # ── 3. Diagnostic baselines: recompute metrics from saved predictions ──
    logger.info("=" * 60)
    logger.info("DIAGNOSTIC baselines (metrics from saved predictions)")
    logger.info("=" * 60)
    cmp_rows = []

    # HVG3000_7types  (results/nnls/)
    cols7 = list(y_test7.columns)
    hvg_dir = res / "nnls"
    hvg_pred_test = read_pred(hvg_dir / "predicted_proportions_test.csv", cols7)
    hvg_pred_cal = read_pred(hvg_dir / "predicted_proportions_cal.csv", cols7)
    h_test = overall_metrics(y_test7.reset_index(drop=True)[cols7], hvg_pred_test)
    h_cal = overall_metrics(y_cal7.reset_index(drop=True)[cols7], hvg_pred_cal)
    cmp_rows.append({"baseline": "HVG3000_7types", "n_types": 7, "role": "diagnostic",
                     **{f"test_{k}": v for k, v in h_test.items()},
                     **{f"cal_{k}": v for k, v in h_cal.items()}})

    # marker_7types  (results/nnls_markers/marker_7type/)
    m7_dir = res / "nnls_markers" / "marker_7type"
    m7_pred_test = read_pred(m7_dir / "predicted_proportions_test.csv", cols7)
    m7_pred_cal = read_pred(m7_dir / "predicted_proportions_cal.csv", cols7)
    m7_test = overall_metrics(y_test7.reset_index(drop=True)[cols7], m7_pred_test)
    m7_cal = overall_metrics(y_cal7.reset_index(drop=True)[cols7], m7_pred_cal)
    cmp_rows.append({"baseline": "marker_7types", "n_types": 7, "role": "diagnostic",
                     **{f"test_{k}": v for k, v in m7_test.items()},
                     **{f"cal_{k}": v for k, v in m7_cal.items()}})

    # marker_5types (PRIMARY)
    cmp_rows.append({"baseline": "marker_5types", "n_types": 5, "role": "primary",
                     **{f"test_{k}": v for k, v in m_test.items()},
                     **{f"cal_{k}": v for k, v in m_cal.items()}})

    cmp_df = pd.DataFrame(cmp_rows)
    save_df(cmp_df, Path(out_cmp) / "nnls_baseline_comparison.csv")
    logger.info("Comparison:\n%s", cmp_df.to_string(index=False))

    # ── 4. Copy diagnostic dirs to clearly-named locations (keep originals) ──
    shutil.copytree(hvg_dir, res / "nnls_hvg3000_7types", dirs_exist_ok=True)
    shutil.copytree(m7_dir, res / "nnls_marker_7types", dirs_exist_ok=True)
    logger.info("Diagnostic dirs copied to nnls_hvg3000_7types/ and nnls_marker_7types/")

    # ── 5. Update run_metadata.json ──
    meta_path = proc / "run_metadata.json"
    meta = json.load(open(meta_path)) if meta_path.exists() else {}
    meta.update({
        "primary_analysis_gene_set": "markers",
        "primary_analysis_n_genes": len(markers_in_sig),
        "primary_analysis_cell_types": PRIMARY_TYPES,
        "primary_analysis_n_types": 5,
        "collapsed_cell_types": {"T_cell": T_SUBSETS},
        "gene_selection_source": "reference_pool_only",
        "marker_selection_method": "reference_pool_differential_expression",
        "primary_baseline": "marker_5type",
        "diagnostic_baselines": ["HVG3000_7types", "marker_7types"],
        "marker_n_per_type": args.n_markers,
    })
    json.dump(meta, open(meta_path, "w"), indent=2, default=str)
    logger.info("Updated run_metadata.json")

    # ── 6. QC SUMMARY ──
    print("\n" + "=" * 70)
    print("PRIMARY LINE FINALIZATION — QC SUMMARY")
    print("=" * 70)

    print(f"\n1. selected_genes_markers.txt: {len(markers_in_sig)} genes")
    print(f"   First 10: {markers_in_sig[:10]}")

    print(f"\n2. 5type true proportions:")
    print(f"   cal : {y_cal5.shape}, test: {y_test5.shape}")
    print(f"   columns: {list(y_cal5.columns)}")
    print("   cal head:")
    print(y_cal5.head().round(4).to_string())
    print("   test head:")
    print(y_test5.head().round(4).to_string())

    print(f"\n3. 5type proportion row-sum:")
    for nm, d in [("cal", y_cal5), ("test", y_test5)]:
        rs = d.sum(axis=1)
        print(f"   {nm}: min={rs.min():.4f} mean={rs.mean():.4f} max={rs.max():.4f}")

    print(f"\n4. marker CPM matrices: cal {X_cal.shape}, test {X_test.shape}")
    print(f"\n5. signature_matrix_markers.csv: {sig.shape} (genes x types)")

    print(f"\n6. marker_5type TEST overall:")
    print(f"   MAE={m_test['MAE']:.4f} RMSE={m_test['RMSE']:.4f} "
          f"Pearson={m_test['Pearson']:.4f} CCC={m_test['CCC']:.4f}")

    print(f"\n7. Per-cell-type (test):")
    print(f"   {'type':<10}{'MAE':>9}{'CCC':>9}{'true_mean':>11}{'pred_mean':>11}")
    near_zero = []
    for p in per_rows:
        print(f"   {p['cell_type']:<10}{p['MAE']:>9.4f}{p['CCC']:>9.4f}"
              f"{p['true_mean']:>11.4f}{p['pred_mean']:>11.4f}")
        if p["pred_mean"] < 0.01 or abs(p["CCC"]) < 0.05:
            near_zero.append(p["cell_type"])

    print(f"\n8. Degenerate (pred_mean<0.01 or |CCC|<0.05) types: "
          f"{near_zero if near_zero else 'NONE — all 5 types healthy'}")

    print(f"\n9. Primary-line file paths:")
    for f in [proc / "selected_genes_markers.txt",
              proc / "true_proportions_cal_5type.csv",
              proc / "true_proportions_test_5type.csv",
              proc / "pseudobulk_matrix_cal_markers_cpm.csv",
              proc / "pseudobulk_matrix_test_markers_cpm.csv",
              Path(out_primary) / "signature_matrix_markers.csv",
              Path(out_primary) / "predicted_proportions_cal.csv",
              Path(out_primary) / "predicted_proportions_test.csv",
              Path(out_primary) / "metrics_cal.csv",
              Path(out_primary) / "metrics_test.csv",
              Path(out_primary) / "marker_5type_summary.csv",
              meta_path]:
        print(f"   {'[OK]' if Path(f).exists() else '[MISSING]'} {f}")

    print(f"\n10. Comparison CSV:")
    print(f"   {Path(out_cmp) / 'nnls_baseline_comparison.csv'}")
    print("   Diagnostic dirs: results/nnls_hvg3000_7types/, results/nnls_marker_7types/")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
