#!/usr/bin/env python
"""
Phase 5 Tier 1 (PRIMARY line): stress test for marker_5type.

Goal: verify that CalibDeconv's uncertainty (Phase 3 ensemble std / interval
width) and Phase 4 conformal intervals can flag unreliable predictions under
signal degradation.

For each scenario the TEST bulk is perturbed, then the SAME ensemble procedure
(marker panel, 5 types, B iterations, gene/cell subsampling) is re-run on the
perturbed bulk to obtain a genuine, label-free reliability score. Phase 4
conformal quantiles (calibrated ONCE on clean cal data) are applied to measure
coverage degradation.

Reliability score (no true labels used):
  reliability = mean ensemble std across cell types  (primary)
  also reported: mean conformal interval width across cell types

Failure definition (sample-level mean abs error):
  primary  : MAE > 0.10
  sensitivity: MAE > 0.15

Tier 1 only. Does NOT run Tier 2/3, public benchmark, or script 07.
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

from src.utils.io import load_config, setup_logger, set_seed, ensure_dir
from src.deconvolution.celltypes import resolve_inputs, resolve_cell_type_column
from src.evaluation.stress import apply_gaussian_noise, apply_dropout, apply_low_depth
from src.uncertainty.conformal import predict_intervals, evaluate_calibration
from sklearn.metrics import roc_auc_score

# Reuse the validated fast-ensemble core from Phase 3 (04b) without duplicating.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "ens04b", str(PROJECT_ROOT / "scripts" / "04b_ensemble_marker5.py"))
_ens = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_ens)
build_panel_celltype_arrays = _ens.build_panel_celltype_arrays
fast_ensemble = _ens.fast_ensemble
summarize = _ens.summarize

CANON = ["Monocyte", "NK", "B", "DC", "T_cell"]

# Tier 1 scenarios with a numeric "severity" for trend plots (0 = baseline).
TIER1 = [
    ("baseline",             "none",     {},                       0.0),
    ("gaussian_noise_low",   "gaussian", {"noise_std": 0.1},       0.1),
    ("gaussian_noise_medium","gaussian", {"noise_std": 0.5},       0.5),
    ("gaussian_noise_high",  "gaussian", {"noise_std": 1.0},       1.0),
    ("dropout_low",          "dropout",  {"dropout_rate": 0.1},    0.1),
    ("dropout_medium",       "dropout",  {"dropout_rate": 0.3},    0.3),
    ("dropout_high",         "dropout",  {"dropout_rate": 0.5},    0.5),
    ("low_depth_medium",     "low_depth",{"depth_fraction": 0.25}, 0.75),
    ("low_depth_high",       "low_depth",{"depth_fraction": 0.10}, 0.90),
]


# ── metric helpers ──────────────────────────────────────────────────────────

def ccc(yt, yp):
    yt = np.asarray(yt).ravel(); yp = np.asarray(yp).ravel()
    mt, mp = yt.mean(), yp.mean(); vt, vp = yt.var(), yp.var()
    cov = ((yt - mt) * (yp - mp)).mean(); d = vt + vp + (mt - mp) ** 2
    return float(2 * cov / d) if d > 0 else float("nan")


def overall_metrics(yt, yp):
    A, B = yt.values, yp.values
    diff = A - B
    pr = np.corrcoef(A.ravel(), B.ravel())[0, 1] if A.std() > 0 and B.std() > 0 else np.nan
    return {"MAE": float(np.abs(diff).mean()),
            "RMSE": float(np.sqrt((diff ** 2).mean())),
            "Pearson": float(pr), "CCC": ccc(A, B)}


def auroc_at(abs_err, score, thr):
    """AUROC for detecting failures (abs_err > thr) from a reliability score."""
    labels = (abs_err > thr).astype(int)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return np.nan, int(labels.sum())
    return float(roc_auc_score(labels, score)), int(labels.sum())


def rejection_curve(abs_err, score, n_points=20):
    """Error-rejection curve, BOTH directions.

    reject_high_mae: reject most-uncertain first, KEEP the lowest-score
        samples. This is the PRIMARY oracle-free curve — if the score is a
        valid reliability signal, retained error should DROP as we retain
        fewer (more-confident) samples.
    reject_low_mae: reject least-uncertain first, KEEP the highest-score
        samples (sanity mirror; should rise).
    """
    asc = np.argsort(score)         # low -> high score
    desc = np.argsort(-score)       # high -> low score
    n = len(score)
    rows = []
    for i in range(n_points + 1):
        frac = 1.0 - i / n_points
        n_keep = max(1, int(n * frac))
        keep_low = asc[:n_keep]     # retain lowest-score (reject high)
        keep_high = desc[:n_keep]   # retain highest-score (reject low)
        rows.append({
            "fraction_retained": frac, "n_retained": n_keep,
            "reject_high_mae": float(abs_err[keep_low].mean()),
            "reject_low_mae": float(abs_err[keep_high].mean()),
            "mean_absolute_error": float(abs_err[keep_low].mean()),  # primary = reject_high
        })
    return pd.DataFrame(rows)


def perturb(kind, X, params, seed):
    if kind == "none":
        return X.copy()
    if kind == "gaussian":
        return apply_gaussian_noise(X, noise_std=params["noise_std"], seed=seed)
    if kind == "dropout":
        return apply_dropout(X, dropout_rate=params["dropout_rate"], seed=seed)
    if kind == "low_depth":
        return apply_low_depth(X, depth_fraction=params["depth_fraction"], seed=seed)
    raise ValueError(f"unknown perturbation kind: {kind}")


def parse_args():
    p = argparse.ArgumentParser(description="Phase 5 Tier 1 stress test (marker_5type)")
    p.add_argument("--gene-set", choices=["markers"], default="markers")
    p.add_argument("--cell-type-set", choices=["5type"], default="5type")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--cell-pool-dir", default="data/processed")
    p.add_argument("--proc-dir", default="data/processed")
    p.add_argument("--ensemble-dir", default="results/ensemble_marker_5types")
    p.add_argument("--conformal-dir", default="results/conformal_marker_5types")
    p.add_argument("--output-dir", default="results/stress_marker_5types")
    p.add_argument("--tier", choices=["tier1", "tier2", "tier3", "all"], default="tier1")
    p.add_argument("--n-iterations", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    if args.tier != "tier1":
        raise SystemExit("This script runs Tier 1 only (per request). Use --tier tier1.")
    logger = setup_logger("06b_stress_m5", log_dir=str(PROJECT_ROOT / "logs"))
    set_seed(args.seed)

    cfg = load_config(args.config)
    ens_cfg = cfg["ensemble"]
    gene_frac = ens_cfg["gene_sampling_fraction"]
    cell_frac = ens_cfg["cell_sampling_fraction"]
    base_col = cfg["data"]["sc_reference"]["cell_type_column"]

    proc_dir = PROJECT_ROOT / args.proc_dir
    conf_dir = PROJECT_ROOT / args.conformal_dir
    out_dir = ensure_dir(str(PROJECT_ROOT / args.output_dir))
    fig_dir = ensure_dir(str(PROJECT_ROOT / "results" / "figures"))

    # ── Inputs: panel, reference pool, clean test, conformal quantiles ──
    paths = resolve_inputs(args.gene_set, args.cell_type_set, proc_dir)
    with open(paths["panel_file"]) as fh:
        gene_panel = [l.strip() for l in fh if l.strip()]

    import scanpy as sc
    adata_ref = sc.read_h5ad(Path(PROJECT_ROOT / args.cell_pool_dir) / "cell_pool_reference.h5ad")
    cell_type_col = resolve_cell_type_column(adata_ref, args.cell_type_set, base_col=base_col)
    cell_types = sorted(adata_ref.obs[cell_type_col].unique().tolist())

    X_test_clean = pd.read_csv(paths["test_file"], index_col=0)[gene_panel]
    y_test = pd.read_csv(paths["true_test"], index_col=0)[CANON]
    y_test.index.name = "sample_id"

    # Conformal quantiles (calibrated ONCE on clean cal data in Phase 4)
    quantiles = pd.read_csv(conf_dir / "conformal_quantiles.csv")

    # Pre-extract panel-aligned per-cell-type arrays (leakage-safe, once)
    per_type_arrays, per_type_totals = build_panel_celltype_arrays(
        adata_ref, cell_type_col, cell_types, gene_panel, logger)

    logger.info("=" * 60)
    logger.info("Phase 5 Tier 1: %d scenarios, B=%d, panel=%d genes, types=%s",
                len(TIER1), args.n_iterations, len(gene_panel), cell_types)
    logger.info("=" * 60)

    # ── Step 4: scenario loop ──
    true_order = list(y_test.index)
    summary_rows, per_ct_rows, fd_rows, rej_dfs, qc_rows = [], [], [], [], []

    def wide(summary, stat):
        w = summary.pivot_table(index="sample_id", columns="cell_type",
                                values=stat, aggfunc="first").loc[true_order, CANON]
        assert list(w.index) == true_order and list(w.columns) == CANON
        return w

    for name, kind, params, severity in TIER1:
        sc_seed = args.seed + abs(hash(name)) % 10000
        Xp = perturb(kind, X_test_clean, params, sc_seed)[gene_panel]
        assert list(Xp.index) == true_order and list(Xp.columns) == gene_panel

        preds = fast_ensemble(
            X_panel=Xp.values.astype(np.float64),
            per_type_arrays=per_type_arrays, per_type_totals=per_type_totals,
            cell_types=cell_types, n_panel_genes=len(gene_panel),
            n_iterations=args.n_iterations, gene_frac=gene_frac, cell_frac=cell_frac,
            noise_std=0.0, seed=sc_seed, logger=logger, tag=name)
        summ = summarize(preds, cell_types, list(Xp.index))

        mean = wide(summ, "mean")
        std = wide(summ, "std")
        width = wide(summ, "interval_width")
        yt = y_test.loc[mean.index, mean.columns]
        assert list(yt.index) == list(mean.index) and list(yt.columns) == list(mean.columns)

        # Per-sample error + label-free reliability scores
        abs_err = np.abs(yt.values - mean.values).mean(axis=1)
        rel_std = std.mean(axis=1).values          # primary reliability score
        rel_width = width.mean(axis=1).values      # secondary

        m = overall_metrics(yt, mean)

        # Conformal intervals @ all levels on this scenario's mean, coverage @90%
        intervals = predict_intervals(mean, quantiles, y_std_test=None,
                                      score_function="absolute_error", clip=True)
        cov = evaluate_calibration(intervals, yt)
        cov90 = cov[(cov["cell_type"] == "overall") & (cov["nominal_coverage"] == 0.90)]
        cov90_clip = float(cov90["empirical_coverage_clip"].values[0]) if len(cov90) else np.nan
        w90 = intervals[intervals["nominal_coverage"] == 0.90]["interval_width_clip"].mean()

        # uncertainty-error correlation (rel_std vs abs_err)
        ue_corr = (np.corrcoef(rel_std, abs_err)[0, 1]
                   if np.std(rel_std) > 0 and np.std(abs_err) > 0 else np.nan)

        # Failure detection AUROC at fixed thresholds
        auroc10, n_fail10 = auroc_at(abs_err, rel_std, 0.10)
        auroc15, n_fail15 = auroc_at(abs_err, rel_std, 0.15)

        rej = rejection_curve(abs_err, rel_std)
        rej.insert(0, "scenario", name)
        rej_dfs.append(rej)

        summary_rows.append({
            "scenario": name, "perturbation": kind, "severity": severity,
            "params": str(params), "n_samples": len(yt),
            "MAE": m["MAE"], "RMSE": m["RMSE"], "Pearson": m["Pearson"], "CCC": m["CCC"],
            "mean_uncertainty_std": float(rel_std.mean()),
            "mean_interval_width": float(rel_width.mean()),
            "coverage90_clip": cov90_clip, "conformal_width90_clip": float(w90),
            "uncertainty_error_corr": float(ue_corr) if ue_corr == ue_corr else np.nan,
            "AUROC_fail_gt0.10": auroc10, "n_fail_gt0.10": n_fail10,
            "AUROC_fail_gt0.15": auroc15, "n_fail_gt0.15": n_fail15,
        })

        # per-cell-type metrics
        for ct in CANON:
            ae_ct = np.abs(yt[ct].values - mean[ct].values)
            per_ct_rows.append({
                "scenario": name, "cell_type": ct,
                "MAE": float(ae_ct.mean()), "CCC": ccc(yt[ct].values, mean[ct].values),
                "mean_std": float(std[ct].mean()), "mean_width": float(width[ct].mean()),
            })

        # failure detection detail
        fd_rows.append({
            "scenario": name,
            "AUROC_gt0.10": auroc10, "n_fail_gt0.10": n_fail10,
            "AUROC_gt0.15": auroc15, "n_fail_gt0.15": n_fail15,
            "frac_fail_gt0.10": float((abs_err > 0.10).mean()),
            "frac_fail_gt0.15": float((abs_err > 0.15).mean()),
        })

        # sanity checks
        vals = mean.values
        qc_rows.append({
            "scenario": name,
            "has_na": bool(np.isnan(vals).any()),
            "has_inf": bool(np.isinf(vals).any()),
            "row_sum_min": float(mean.sum(axis=1).min()),
            "row_sum_max": float(mean.sum(axis=1).max()),
            "index_aligned": list(mean.index) == true_order,
            "columns_aligned": list(mean.columns) == CANON,
        })
        logger.info("[%s] MAE=%.4f CCC=%.4f cov90=%.3f AUROC(>0.10)=%s",
                    name, m["MAE"], m["CCC"], cov90_clip,
                    f"{auroc10:.3f}" if auroc10 == auroc10 else "NA")

    summary_df = pd.DataFrame(summary_rows)
    per_ct_df = pd.DataFrame(per_ct_rows)
    fd_df = pd.DataFrame(fd_rows)
    rej_df = pd.concat(rej_dfs, ignore_index=True)
    qc_df = pd.DataFrame(qc_rows)

    summary_df.to_csv(Path(out_dir) / "stress_summary_tier1.csv", index=False)
    per_ct_df.to_csv(Path(out_dir) / "stress_per_celltype_tier1.csv", index=False)
    fd_df.to_csv(Path(out_dir) / "failure_detection_tier1.csv", index=False)
    rej_df.to_csv(Path(out_dir) / "rejection_curves_tier1.csv", index=False)
    qc_df.to_csv(Path(out_dir) / "stress_qc_tier1.csv", index=False)
    logger.info("Tier 1 CSVs written to %s", out_dir)

    # ── Step 5: figures ──
    def savefig(stem):
        for ext in ("png", "pdf"):
            plt.savefig(Path(fig_dir) / f"{stem}.{ext}", dpi=150)
        plt.close()

    sdf = summary_df.copy()

    # 1. error vs severity (per perturbation family)
    plt.figure(figsize=(6, 5))
    for fam in ["gaussian", "dropout", "low_depth"]:
        d = sdf[sdf["perturbation"] == fam].sort_values("severity")
        base = sdf[sdf["perturbation"] == "none"]
        d = pd.concat([base, d])
        plt.plot(d["severity"], d["MAE"], "o-", label=fam)
    plt.xlabel("Severity"); plt.ylabel("Test MAE")
    plt.title("Error vs severity (Tier 1, marker_5type)"); plt.legend()
    plt.tight_layout(); savefig("stress_error_vs_severity")

    # 2. uncertainty vs severity
    plt.figure(figsize=(6, 5))
    for fam in ["gaussian", "dropout", "low_depth"]:
        d = pd.concat([sdf[sdf["perturbation"] == "none"],
                       sdf[sdf["perturbation"] == fam].sort_values("severity")])
        plt.plot(d["severity"], d["mean_uncertainty_std"], "o-", label=fam)
    plt.xlabel("Severity"); plt.ylabel("Mean ensemble std")
    plt.title("Uncertainty vs severity (Tier 1)"); plt.legend()
    plt.tight_layout(); savefig("stress_uncertainty_vs_severity")

    # 3. coverage vs severity (90% clipped)
    plt.figure(figsize=(6, 5))
    for fam in ["gaussian", "dropout", "low_depth"]:
        d = pd.concat([sdf[sdf["perturbation"] == "none"],
                       sdf[sdf["perturbation"] == fam].sort_values("severity")])
        plt.plot(d["severity"], d["coverage90_clip"], "o-", label=fam)
    plt.axhline(0.90, color="k", ls="--", lw=1, label="nominal 90%")
    plt.xlabel("Severity"); plt.ylabel("Empirical coverage @90% (clipped)")
    plt.title("Conformal coverage vs severity (Tier 1)"); plt.legend()
    plt.tight_layout(); savefig("stress_coverage_vs_severity")

    # 4. failure-detection AUROC by scenario (threshold 0.10)
    plt.figure(figsize=(7, 5))
    order = sdf.sort_values("severity")
    plt.bar(order["scenario"], order["AUROC_fail_gt0.10"])
    plt.axhline(0.5, color="k", ls="--", lw=1, label="random")
    plt.ylabel("AUROC (failure MAE>0.10)")
    plt.title("Failure detection AUROC (Tier 1)")
    plt.xticks(rotation=45, ha="right", fontsize=7); plt.legend()
    plt.tight_layout(); savefig("stress_failure_auroc")

    # 5. rejection curves (all scenarios)
    plt.figure(figsize=(7, 5))
    for name in sdf["scenario"]:
        d = rej_df[rej_df["scenario"] == name]
        plt.plot(d["fraction_retained"], d["mean_absolute_error"], label=name, lw=1)
    plt.xlabel("Fraction retained (reject most-uncertain first)")
    plt.ylabel("Mean abs error (retained)")
    plt.title("Rejection curves (Tier 1)")
    plt.legend(fontsize=6, ncol=2); plt.tight_layout()
    savefig("stress_rejection_curves")
    logger.info("Tier 1 figures written to %s", fig_dir)

    # ── Step 6: QC summary ──
    base_mae = sdf[sdf["scenario"] == "baseline"]["MAE"].values[0]
    print("\n" + "=" * 78)
    print("PHASE 5 TIER 1 STRESS TEST (marker_5type) — QC SUMMARY")
    print("=" * 78)

    print("\n1-5. Per-scenario metrics:")
    print(f"{'scenario':<22}{'MAE':>7}{'RMSE':>7}{'Pear':>7}{'CCC':>7}"
          f"{'unc_std':>9}{'width':>8}{'cov90':>7}{'ue_r':>7}{'AUROC.10':>9}")
    for _, r in sdf.iterrows():
        au = f"{r['AUROC_fail_gt0.10']:.3f}" if r['AUROC_fail_gt0.10'] == r['AUROC_fail_gt0.10'] else "NA"
        print(f"{r['scenario']:<22}{r['MAE']:>7.3f}{r['RMSE']:>7.3f}{r['Pearson']:>7.3f}"
              f"{r['CCC']:>7.3f}{r['mean_uncertainty_std']:>9.4f}{r['mean_interval_width']:>8.3f}"
              f"{r['coverage90_clip']:>7.3f}{r['uncertainty_error_corr']:>7.3f}{au:>9}")

    print("\n6. Rejection check (reject_high = reject most-uncertain, keep confident):")
    for name in sdf["scenario"]:
        d = rej_df[rej_df["scenario"] == name].sort_values("fraction_retained")
        full = d[d["fraction_retained"] >= 0.999]["reject_high_mae"]
        row50 = d.iloc[(d["fraction_retained"] - 0.5).abs().argmin()]
        full_v = full.values[0] if len(full) else d["reject_high_mae"].iloc[-1]
        half = row50["reject_high_mae"]
        drop = full_v - half  # positive = error dropped after rejecting uncertain
        print(f"   {name:<22} full={full_v:.4f} -> retain50%(reject_high)={half:.4f} "
              f"(drop={drop:+.4f}) {'OK' if drop >= 0 else 'NO-DROP'}")

    print("\n7. Alignment / sanity (no mismatch expected):")
    bad = qc_df[(qc_df["has_na"]) | (qc_df["has_inf"]) | (~qc_df["index_aligned"])
                | (~qc_df["columns_aligned"])]
    print(f"   scenarios with NA/inf/misalignment: {len(bad)} "
          f"({'NONE' if len(bad) == 0 else list(bad['scenario'])})")
    print(f"   row_sum range across scenarios: [{qc_df['row_sum_min'].min():.4f}, "
          f"{qc_df['row_sum_max'].max():.4f}]")

    print(f"\n8. Output files:")
    for f in ["stress_summary_tier1.csv", "stress_per_celltype_tier1.csv",
              "failure_detection_tier1.csv", "rejection_curves_tier1.csv",
              "stress_qc_tier1.csv"]:
        p = Path(out_dir) / f
        print(f"   {'[OK]' if p.exists() else '[MISSING]'} {p}")
    print(f"   Figures:")
    for f in ["stress_error_vs_severity", "stress_uncertainty_vs_severity",
              "stress_coverage_vs_severity", "stress_failure_auroc",
              "stress_rejection_curves"]:
        p = Path(fig_dir) / f"{f}.png"
        print(f"   {'[OK]' if p.exists() else '[MISSING]'} {p}")
    print("=" * 78)
    logger.info("Phase 5 Tier 1 complete.")


if __name__ == "__main__":
    main()
