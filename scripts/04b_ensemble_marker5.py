#!/usr/bin/env python
"""
Phase 3 (PRIMARY line): ensemble uncertainty for marker_5type.

Runs B bootstrap perturbations of NNLS deconvolution, restricted to the
marker gene panel and the 5 collapsed cell types. Produces per-(sample,
cell_type) uncertainty summaries, uncertainty-error correlations, ensemble
point-estimate accuracy, and diagnostic figures.

Hard constraints (asserted at runtime):
  - Perturbations stay WITHIN the marker gene panel (445 genes); never
    fall back to all-genes or HVG3000.
  - cell_type_set = 5type (T_cell, B, NK, Monocyte, DC).
  - expression scale = CPM.
  - Reference pool ONLY for signatures/cell bootstrap (leakage control).

Outputs (results/ensemble_marker_5types/), does NOT overwrite results/ensemble/:
  ensemble_predictions_cal.npy / ensemble_predictions_test.npy
  ensemble_summary_cal.csv / ensemble_summary_test.csv  (mean/std/q05/q95/width/disagreement)
  perturbation_log.csv
  uncertainty_error_correlation.csv
  ensemble_accuracy.csv
  predicted_proportions_cal_ensemble.csv / _test_ensemble.csv  (ensemble mean point est.)
  sanity_check.csv
Figures (results/figures/):
  ens5_uncertainty_vs_error.png
  ens5_interval_width_by_celltype.png
  ens5_true_vs_predicted.png
  ens5_per_celltype_unc_err_corr.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.io import load_config, setup_logger, set_seed, ensure_dir, save_df
from src.uncertainty.ensemble import run_ensemble
from src.deconvolution.celltypes import resolve_inputs, resolve_cell_type_column
from src.deconvolution.nnls import deconvolve_nnls
from src.evaluation.metrics import (
    compute_accuracy_metrics, compute_uncertainty_error_correlation,
)
from scipy import sparse


def build_panel_celltype_arrays(adata_ref, cell_type_col, cell_types, gene_panel, logger):
    """Pre-extract dense per-cell-type expression aligned to the marker panel.

    Returns
    -------
    per_type : list of np.ndarray
        Parallel to cell_types; each (n_cells_of_type, n_panel_genes), columns
        in EXACT gene_panel order. Values are on adata.X scale (log-normalized,
        exactly what the Phase-2 baseline signature used).
    per_type_totals : list of np.ndarray
        Parallel to cell_types; each (n_cells_of_type,) = per-cell SUM over ALL
        genes (the CPM denominator). Lets the ensemble replicate the baseline's
        "CPM over all genes, then subset to panel" normalization exactly:
        col_sum_over_all_genes(type mean) == mean over cells of per-cell total.

    Done ONCE so the ensemble loop never re-masks adata or re-densifies. Aligning
    to the panel here also fixes the gene-index aliasing bug in the generic
    subsampling helper (which assumed adata was already panel-restricted).
    """
    var_set = set(adata_ref.var_names)
    missing = [g for g in gene_panel if g not in var_set]
    if missing:
        raise ValueError(f"{len(missing)} panel genes absent from reference: {missing[:10]}")

    # Per-cell total over ALL genes (CPM denominator), on adata.X scale.
    Xall = adata_ref.X
    cell_totals = np.asarray(Xall.sum(axis=1)).ravel().astype(np.float64)

    # Panel-restricted dense matrix, columns reindexed to panel order.
    sub = adata_ref[:, gene_panel]
    assert list(sub.var_names) == gene_panel, "panel reorder failed!"
    Xp = sub.X
    if sparse.issparse(Xp):
        Xp = Xp.toarray()
    Xp = np.asarray(Xp, dtype=np.float64)

    labels = adata_ref.obs[cell_type_col].values
    per_type, per_type_totals = [], []
    for ct in cell_types:
        rows = np.where(labels == ct)[0]
        per_type.append(Xp[rows])
        per_type_totals.append(cell_totals[rows])
        logger.info("  [panel] %s: %d cells x %d genes (mean cell-total=%.1f)",
                    ct, len(rows), Xp.shape[1], cell_totals[rows].mean())
    return per_type, per_type_totals

# __FASTCORE__


def fast_ensemble(X_panel, per_type_arrays, per_type_totals, cell_types, n_panel_genes,
                  n_iterations, gene_frac, cell_frac, noise_std, seed, logger, tag):
    """Fast, panel-aligned ensemble.

    For each iteration: build ONE mini-signature by subsampling cells (per type)
    and genes, then deconvolve ALL samples against it. This is the correct and
    efficient design — one signature rebuild per iteration, not per (sample,iter).

    The mini-signature is CPM-normalized PER CELL TYPE using the per-cell total
    over ALL genes (per_type_totals), reproducing the Phase-2 baseline's
    "CPM over all genes, then subset to panel" scale exactly. Without this the
    log-normalized adata.X scale (~0.08) mismatches the CPM bulk (~1e3), which
    collapses NNLS accuracy.

    Parameters
    ----------
    X_panel : np.ndarray
        Bulk matrix (n_samples, n_panel_genes), columns in panel order, CPM.
    per_type_arrays : list of np.ndarray
        Per-cell-type panel expression (cells, n_panel_genes), adata.X scale.
    per_type_totals : list of np.ndarray
        Per-cell total over ALL genes (cells,) — the CPM denominator.

    Returns
    -------
    all_preds : np.ndarray  (n_iterations, n_samples, n_types)
    """
    n_samples = X_panel.shape[0]
    n_types = len(cell_types)
    all_preds = np.zeros((n_iterations, n_samples, n_types), dtype=np.float32)
    n_gene_keep = max(10, int(n_panel_genes * gene_frac))

    import time
    t0 = time.time()
    for it in range(n_iterations):
        # Per-iteration RNG matches the perturbation_log seed formula.
        iter_rng = np.random.default_rng(seed + it * 10000 + 1)
        gene_idx = iter_rng.choice(n_panel_genes, size=n_gene_keep, replace=False)

        # Build mini-signature: subsample cells per type, mean over panel genes,
        # then per-column CPM using the all-gene per-cell totals.
        mini_sig = np.zeros((n_gene_keep, n_types), dtype=np.float64)
        for j, (arr, totals) in enumerate(zip(per_type_arrays, per_type_totals)):
            n_cells = arr.shape[0]
            n_keep = max(1, int(n_cells * cell_frac))
            chosen = iter_rng.choice(n_cells, size=n_keep, replace=False)
            col = arr[chosen][:, gene_idx].mean(axis=0)
            denom = totals[chosen].mean()  # sum over ALL genes of the type mean
            mini_sig[:, j] = col / denom * 1e6 if denom > 0 else col

        # Bulk for this iteration (optionally noised), restricted to sampled genes.
        Xb = X_panel[:, gene_idx]
        if noise_std > 0:
            Xb = Xb + iter_rng.normal(0, noise_std * Xb.std(), size=Xb.shape)
            Xb = np.clip(Xb, 0, None)

        for s in range(n_samples):
            all_preds[it, s] = deconvolve_nnls(Xb[s], mini_sig, normalize=True)

        if (it + 1) % 10 == 0 or it == 0:
            el = time.time() - t0
            logger.info("  [%s] iter %d/%d (%.1fs, ~%.0fs left)",
                        tag, it + 1, n_iterations, el, el / (it + 1) * (n_iterations - it - 1))
    return all_preds


def summarize(all_preds, cell_types, sample_ids):
    """Per-(sample,cell_type) mean/std/quantiles/width/disagreement."""
    n_iter, n_samples, n_types = all_preds.shape
    rows = []
    eps = 1e-8
    for s in range(n_samples):
        for t in range(n_types):
            dist = all_preds[:, s, t]
            mean = float(np.mean(dist))
            std = float(np.std(dist))
            q05 = float(np.quantile(dist, 0.05))
            q95 = float(np.quantile(dist, 0.95))
            rows.append({
                "sample_id": sample_ids[s],
                "cell_type": cell_types[t],
                "mean": mean, "std": std,
                "q0.05": q05, "q0.25": float(np.quantile(dist, 0.25)),
                "q0.75": float(np.quantile(dist, 0.75)), "q0.95": q95,
                "interval_width": q95 - q05,
                "disagreement": std / (abs(mean) + eps),
            })
    return pd.DataFrame(rows)

# __FASTCORE_END__




def pivot_stat(summary, value):
    """sample_id x cell_type table for one summary statistic.

    NOTE: pivot_table sorts the index lexicographically (test_0, test_1,
    test_10, ...). Callers MUST realign to the canonical sample order via
    align_true_to_pred / explicit .loc — never assume row order matches the
    original generation order.
    """
    return summary.pivot_table(index="sample_id", columns="cell_type",
                               values=value, aggfunc="first")


def align_true_to_pred(y_true, y_pred, name):
    """Align true proportions to a prediction table by index AND columns.

    Both must use real sample IDs as index. Raises on any mismatch — this is
    the guard that prevents the lexicographic-pivot row-scramble bug.
    """
    y_aligned = y_true.loc[y_pred.index, y_pred.columns]
    assert list(y_aligned.index) == list(y_pred.index), f"{name}: sample index mismatch"
    assert list(y_aligned.columns) == list(y_pred.columns), f"{name}: cell type column mismatch"
    return y_aligned


def add_disagreement(summary):
    """Add an ensemble-disagreement score: coefficient of variation std/mean.

    Captures relative spread; robust to the proportion magnitude. Falls back
    to std where mean ~ 0.
    """
    eps = 1e-8
    summary = summary.copy()
    summary["disagreement"] = summary["std"] / (summary["mean"].abs() + eps)
    return summary


def run_one_set(tag, X, per_type_arrays, per_type_totals, cell_types, gene_panel,
                n_iterations, gene_frac, cell_frac, noise_std, seed, logger):
    """Run the fast panel-aligned ensemble on one sample set (cal or test)."""
    # Restrict X to the marker panel, in panel order. HARD assertion.
    X = X[gene_panel]
    assert list(X.columns) == gene_panel, f"{tag}: X columns != marker panel!"
    logger.info("[%s] ensemble on %d samples x %d marker genes, B=%d",
                tag, X.shape[0], X.shape[1], n_iterations)

    all_preds = fast_ensemble(
        X_panel=X.values.astype(np.float64),
        per_type_arrays=per_type_arrays,
        per_type_totals=per_type_totals,
        cell_types=cell_types,
        n_panel_genes=len(gene_panel),
        n_iterations=n_iterations,
        gene_frac=gene_frac,
        cell_frac=cell_frac,
        noise_std=noise_std,
        seed=seed,
        logger=logger,
        tag=tag,
    )
    summary = summarize(all_preds, cell_types, list(X.index))
    return all_preds, summary


def normalize_rows(df):
    """Renormalize each row to sum to 1 (simplex projection of the mean)."""
    s = df.sum(axis=1)
    return df.div(s.where(s > 0, 1.0), axis=0)


def main():
    parser = argparse.ArgumentParser(description="Phase 3 ensemble (marker_5type primary)")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--cell-pool-dir", default="data/processed")
    parser.add_argument("--pseudobulk-dir", default="data/processed")
    parser.add_argument("--output-dir", default="results/ensemble_marker_5types")
    parser.add_argument("--gene-set", choices=["markers"], default="markers")
    parser.add_argument("--cell-type-set", choices=["5type"], default="5type")
    parser.add_argument("--n-iterations", type=int, default=50)
    parser.add_argument("--gene-fraction", type=float, default=None,
                        help="Override config ensemble.gene_sampling_fraction.")
    parser.add_argument("--cell-fraction", type=float, default=None,
                        help="Override config ensemble.cell_sampling_fraction.")
    parser.add_argument("--noise-std", type=float, default=None,
                        help="Override config ensemble.noise_std.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("04b_ensemble_m5", log_dir=str(PROJECT_ROOT / "logs"))
    set_seed(args.seed)

    out_dir = ensure_dir(str(PROJECT_ROOT / args.output_dir))
    fig_dir = ensure_dir(str(PROJECT_ROOT / "results" / "figures"))

    ens_cfg = cfg["ensemble"]
    gene_frac = args.gene_fraction if args.gene_fraction is not None else ens_cfg["gene_sampling_fraction"]
    cell_frac = args.cell_fraction if args.cell_fraction is not None else ens_cfg["cell_sampling_fraction"]
    noise_std = args.noise_std if args.noise_std is not None else ens_cfg["noise_std"]
    base_col = cfg["data"]["sc_reference"]["cell_type_column"]

    # ── Resolve primary-line inputs ──
    pb_dir = Path(PROJECT_ROOT / args.pseudobulk_dir)
    paths = resolve_inputs(args.gene_set, args.cell_type_set, pb_dir)
    with open(paths["panel_file"]) as fh:
        gene_panel = [l.strip() for l in fh if l.strip()]
    logger.info("Marker panel: %d genes", len(gene_panel))

    import scanpy as sc
    adata_ref = sc.read_h5ad(Path(PROJECT_ROOT / args.cell_pool_dir) / "cell_pool_reference.h5ad")
    cell_type_col = resolve_cell_type_column(adata_ref, args.cell_type_set, base_col=base_col)
    cell_types = sorted(adata_ref.obs[cell_type_col].unique().tolist())
    logger.info("Reference pool: %d cells | cell_type col: %s | types: %s",
                adata_ref.shape[0], cell_type_col, cell_types)

    X_cal = pd.read_csv(paths["cal_file"], index_col=0)
    X_test = pd.read_csv(paths["test_file"], index_col=0)
    y_cal = pd.read_csv(paths["true_cal"], index_col=0)
    y_test = pd.read_csv(paths["true_test"], index_col=0)

    logger.info("=" * 60)
    logger.info("Phase 3 ensemble: gene_set=%s cell_type_set=%s B=%d scale=CPM",
                args.gene_set, args.cell_type_set, args.n_iterations)
    logger.info("  gene_fraction=%.2f cell_fraction=%.2f noise_std=%.3f",
                gene_frac, cell_frac, noise_std)
    logger.info("=" * 60)

    # __MAIN_CONTINUE__

    # ── Pre-extract panel-aligned per-cell-type arrays (ONCE, leakage-safe) ──
    logger.info("Pre-extracting panel-aligned reference arrays...")
    per_type_arrays, per_type_totals = build_panel_celltype_arrays(
        adata_ref, cell_type_col, cell_types, gene_panel, logger)

    # ── Run ensemble on cal and test ──
    preds_cal, sum_cal = run_one_set("cal", X_cal, per_type_arrays, per_type_totals,
                                     cell_types, gene_panel, args.n_iterations,
                                     gene_frac, cell_frac, noise_std, args.seed, logger)
    preds_test, sum_test = run_one_set("test", X_test, per_type_arrays, per_type_totals,
                                       cell_types, gene_panel, args.n_iterations,
                                       gene_frac, cell_frac, noise_std, args.seed, logger)

    np.save(Path(out_dir) / "ensemble_predictions_cal.npy", preds_cal)
    np.save(Path(out_dir) / "ensemble_predictions_test.npy", preds_test)
    save_df(sum_cal, Path(out_dir) / "ensemble_summary_cal.csv")
    save_df(sum_test, Path(out_dir) / "ensemble_summary_test.csv")

    # ── Perturbation log (per iteration: seed + config) ──
    # run_ensemble derives each iteration's RNG as seed + it*10000 + 1.
    pert_rows = []
    for it in range(args.n_iterations):
        pert_rows.append({
            "iteration": it,
            "iter_seed": args.seed + it * 10000 + 1,
            "gene_fraction": gene_frac,
            "cell_fraction": cell_frac,
            "noise_std": noise_std,
            "gene_pool": "marker_panel",
            "n_genes_pool": len(gene_panel),
            "n_genes_sampled": max(1, int(len(gene_panel) * gene_frac)),
        })
    save_df(pd.DataFrame(pert_rows), Path(out_dir) / "perturbation_log.csv")
    logger.info("Perturbation log: %d iterations recorded", args.n_iterations)

    # __MAIN_CONTINUE2__

    # ── Build mean/std tables and align to true proportions ──
    # CANONICAL order: rows follow the true-proportion file's sample_id order
    # (generation order: test_0, test_1, test_2, ...), columns follow its
    # column order. pivot_stat sorts lexicographically (test_0, test_1, test_10,
    # ...), so we MUST reindex via .loc[true_order, canonical_cols] — never
    # positional relabeling, which scrambled rows.
    canonical_cell_types = list(y_test.columns)
    logger.info("Canonical cell-type order: %s", canonical_cell_types)

    def build_tables(summary, y_true):
        true_order = list(y_true.index)
        cols = list(y_true.columns)
        mean = pivot_stat(summary, "mean").loc[true_order, cols]
        std = pivot_stat(summary, "std").loc[true_order, cols]
        q05 = pivot_stat(summary, "q0.05").loc[true_order, cols]
        q95 = pivot_stat(summary, "q0.95").loc[true_order, cols]
        width = pivot_stat(summary, "interval_width").loc[true_order, cols]
        disagree = pivot_stat(summary, "disagreement").loc[true_order, cols]
        # Hard guards: row order + column order must match the true file exactly.
        assert list(mean.index) == true_order, "mean row order != true order"
        assert list(mean.columns) == cols, "mean col order != true cols"
        assert list(mean.index[:3]) == true_order[:3], "row order check failed"
        yt = y_true.loc[mean.index, mean.columns]
        assert list(yt.index) == list(mean.index) and list(yt.columns) == list(mean.columns)
        return mean, std, q05, q95, width, disagree, yt

    mean_cal, std_cal, q05_cal, q95_cal, width_cal, disagree_cal, yt_cal = build_tables(sum_cal, y_cal)
    mean_test, std_test, q05_test, q95_test, width_test, disagree_test, yt_test = build_tables(sum_test, y_test)
    # Verify generation order (test_0, test_1, test_2 — NOT test_0, test_1, test_10)
    logger.info("test sample_id[:10] = %s", list(mean_test.index[:10]))

    # ── Uncertainty-error correlation: cal, test, per-cell-type + overall ──
    corr_cal = compute_uncertainty_error_correlation(yt_cal, mean_cal, std_cal)
    corr_cal["split"] = "calibration"
    corr_test = compute_uncertainty_error_correlation(yt_test, mean_test, std_test)
    corr_test["split"] = "test"
    # Combined (cal+test) overall+per-type
    yt_all = pd.concat([yt_cal, yt_test], ignore_index=True)
    mean_all = pd.concat([mean_cal, mean_test], ignore_index=True)
    std_all = pd.concat([std_cal, std_test], ignore_index=True)
    corr_all = compute_uncertainty_error_correlation(yt_all, mean_all, std_all)
    corr_all["split"] = "overall"
    corr_df = pd.concat([corr_cal, corr_test, corr_all], ignore_index=True)
    save_df(corr_df, Path(out_dir) / "uncertainty_error_correlation.csv")

    # ── Ensemble point-estimate accuracy (cal + test) ──
    acc_rows = []
    for split, yt, mean in [("calibration", yt_cal, mean_cal), ("test", yt_test, mean_test)]:
        m = compute_accuracy_metrics(yt, mean)
        o = m["overall"].iloc[0].to_dict()
        o["split"] = split
        o["level"] = "overall"
        o["cell_type"] = "ALL"
        acc_rows.append(o)
        for _, r in m["per_cell_type"].iterrows():
            d = r.to_dict()
            d["split"] = split
            d["level"] = "per_cell_type"
            acc_rows.append(d)
    acc_df = pd.DataFrame(acc_rows)
    save_df(acc_df, Path(out_dir) / "ensemble_accuracy.csv")

    # ── Ensemble mean point estimates (saved WITH sample_id index) ──
    for split, mean, y_true in [("cal", mean_cal, y_cal), ("test", mean_test, y_test)]:
        assert list(mean.index) == list(y_true.index), f"{split}: pred index != true index"
        assert list(mean.columns) == list(y_true.columns), f"{split}: pred cols != true cols"
        out = mean.copy()
        out.index.name = "sample_id"
        out.to_csv(Path(out_dir) / f"predicted_proportions_{split}_ensemble.csv")
        logger.info("[%s] saved pred; sample_id[:10]=%s", split, list(out.index[:10]))

    # __MAIN_CONTINUE3__

    # ── Sanity checks on ensemble-mean proportions ──
    sanity_rows = []
    for split, mean in [("calibration", mean_cal), ("test", mean_test)]:
        rs = mean.sum(axis=1)
        vals = mean.values
        sanity_rows.append({
            "split": split,
            "row_sum_min": float(rs.min()),
            "row_sum_mean": float(rs.mean()),
            "row_sum_max": float(rs.max()),
            "has_negative": bool((vals < 0).any()),
            "has_na": bool(np.isnan(vals).any()),
            "has_inf": bool(np.isinf(vals).any()),
            # Ensemble mean of NNLS proportions need not sum to exactly 1;
            # flag whether renormalization to sum=1 is advisable.
            "needs_renormalization": bool(np.abs(rs - 1.0).max() > 1e-6),
            "max_abs_sum_dev": float(np.abs(rs - 1.0).max()),
        })
    sanity_df = pd.DataFrame(sanity_rows)
    save_df(sanity_df, Path(out_dir) / "sanity_check.csv")
    all_ok = (not sanity_df["has_negative"].any()
              and not sanity_df["has_na"].any()
              and not sanity_df["has_inf"].any())

    # ── Figures ──
    abs_err_test = np.abs(yt_test.values - mean_test.values).flatten()
    std_flat_test = std_test.values.flatten()

    # 1. uncertainty vs absolute error scatter (test)
    plt.figure(figsize=(6, 5))
    plt.scatter(std_flat_test, abs_err_test, s=8, alpha=0.3)
    plt.xlabel("Ensemble std (uncertainty)")
    plt.ylabel("Absolute error")
    plt.title("Uncertainty vs absolute error (test, marker_5type)")
    plt.tight_layout()
    plt.savefig(Path(fig_dir) / "ens5_uncertainty_vs_error.png", dpi=150)
    plt.close()

    # 2. interval width by cell type (test)
    plt.figure(figsize=(6, 5))
    width_test[canonical_cell_types].boxplot()
    plt.ylabel("Interval width (q95 - q05)")
    plt.title("Interval width by cell type (test, marker_5type)")
    plt.tight_layout()
    plt.savefig(Path(fig_dir) / "ens5_interval_width_by_celltype.png", dpi=150)
    plt.close()

    # 3. ensemble mean true vs predicted (test, colored by cell type)
    plt.figure(figsize=(6, 6))
    for ct in cell_types:
        plt.scatter(yt_test[ct].values, mean_test[ct].values, s=10, alpha=0.4, label=ct)
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("True proportion")
    plt.ylabel("Ensemble mean predicted")
    plt.title("True vs predicted (test, marker_5type)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(Path(fig_dir) / "ens5_true_vs_predicted.png", dpi=150)
    plt.close()

    # 4. per-cell-type uncertainty-error correlation (test)
    pc = corr_test[corr_test["cell_type"] != "overall"]
    plt.figure(figsize=(6, 5))
    plt.bar(pc["cell_type"], pc["pearson_r"])
    plt.ylabel("Pearson r (std vs abs error)")
    plt.title("Per-cell-type uncertainty-error correlation (test)")
    plt.axhline(0, color="k", lw=0.8)
    plt.tight_layout()
    plt.savefig(Path(fig_dir) / "ens5_per_celltype_unc_err_corr.png", dpi=150)
    plt.close()

    # __MAIN_CONTINUE4__

    # ── QC SUMMARY ──
    o_cal = acc_df[(acc_df["split"] == "calibration") & (acc_df["level"] == "overall")].iloc[0]
    o_test = acc_df[(acc_df["split"] == "test") & (acc_df["level"] == "overall")].iloc[0]
    corr_overall_test = corr_test[corr_test["cell_type"] == "overall"]["pearson_r"].values[0]
    corr_overall_all = corr_all[corr_all["cell_type"] == "overall"]["pearson_r"].values[0]

    print("\n" + "=" * 70)
    print("PHASE 3 ENSEMBLE (marker_5type) — QC SUMMARY")
    print("=" * 70)
    print(f"\n1. Ensemble iterations (B): {args.n_iterations}")
    print(f"2. Genes used (marker panel): {len(gene_panel)}")
    print(f"3. Cell types: {len(cell_types)} -> {cell_types}")
    print(f"4. Samples: cal={X_cal.shape[0]} test={X_test.shape[0]}")

    print(f"\n5. Ensemble point-estimate accuracy:")
    print(f"   CAL : MAE={o_cal['MAE']:.4f} RMSE={o_cal['RMSE']:.4f} "
          f"Pearson={o_cal['Pearson_r']:.4f} CCC={o_cal['CCC']:.4f}")
    print(f"   TEST: MAE={o_test['MAE']:.4f} RMSE={o_test['RMSE']:.4f} "
          f"Pearson={o_test['Pearson_r']:.4f} CCC={o_test['CCC']:.4f}")

    print(f"\n6. Uncertainty-error correlation (Pearson r, std vs abs error):")
    print(f"   test overall   = {corr_overall_test:.4f}")
    print(f"   cal+test overall = {corr_overall_all:.4f}")
    print(f"   per cell type (test):")
    for _, r in corr_test[corr_test["cell_type"] != "overall"].iterrows():
        print(f"     {r['cell_type']:<10} r={r['pearson_r']:.4f} (p={r['p_value']:.2e})")

    print(f"\n7. Proportion sanity check (ensemble mean):")
    for _, r in sanity_df.iterrows():
        print(f"   {r['split']:<12} row_sum[min/mean/max]="
              f"{r['row_sum_min']:.4f}/{r['row_sum_mean']:.4f}/{r['row_sum_max']:.4f} "
              f"neg={r['has_negative']} NA={r['has_na']} inf={r['has_inf']} "
              f"renorm_advised={r['needs_renormalization']} (max_dev={r['max_abs_sum_dev']:.4f})")

    print(f"\n8. Output files:")
    for f in ["ensemble_summary_cal.csv", "ensemble_summary_test.csv",
              "ensemble_predictions_cal.npy", "ensemble_predictions_test.npy",
              "perturbation_log.csv", "uncertainty_error_correlation.csv",
              "ensemble_accuracy.csv", "predicted_proportions_cal_ensemble.csv",
              "predicted_proportions_test_ensemble.csv", "sanity_check.csv"]:
        p = Path(out_dir) / f
        print(f"   {'[OK]' if p.exists() else '[MISSING]'} {p}")

    print(f"\n9. Figures:")
    for f in ["ens5_uncertainty_vs_error.png", "ens5_interval_width_by_celltype.png",
              "ens5_true_vs_predicted.png", "ens5_per_celltype_unc_err_corr.png"]:
        p = Path(fig_dir) / f
        print(f"   {'[OK]' if p.exists() else '[MISSING]'} {p}")

    print(f"\n10. Sanity check: {'ALL PASS' if all_ok else 'CHECK FAILED'}")
    print("=" * 70)
    logger.info("Phase 3 ensemble complete. Sanity: %s", "PASS" if all_ok else "FAIL")


if __name__ == "__main__":
    main()





