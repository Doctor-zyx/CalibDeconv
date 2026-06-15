#!/usr/bin/env python
"""
Phase 5 Tier 2 SUBSET (validation): batch_shift_small/large, reference_reduction_1.

Reuses the validated Phase 3 fast-ensemble core (04b) and the conformal module.
Strict sample_id + canonical-column alignment. Reliability score = mean_std.
Primary rejection = reject most-uncertain (reject_high), keep confident.

Scenario handling:
  batch_shift_*       : perturb TEST bulk only; signature/true/quantiles unchanged
                        -> reuse Phase 4 5-type conformal quantiles.
  reference_reduction_1: remove 1 cell type from the REFERENCE (signature = 4 types);
                        the removed cell stays in the bulk. True props restricted to
                        kept types and renormalized. Conformal is RE-CALIBRATED on the
                        reduced cal set (5-type Phase 4 quantiles no longer apply).

Does NOT run rare_cell, missing_cell_type, cross_donor, cross_dataset, Tier 3.
Outputs to results/stress_marker_5types_tier2_subset/ (does not touch Tier 1).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import load_config, setup_logger, set_seed, ensure_dir
from src.deconvolution.celltypes import resolve_inputs, resolve_cell_type_column
from src.evaluation.stress import apply_batch_shift, reduce_reference_cell_types
from src.uncertainty.conformal import calibrate, predict_intervals, evaluate_calibration
from sklearn.metrics import roc_auc_score

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("ens04b", str(PROJECT_ROOT / "scripts" / "04b_ensemble_marker5.py"))
_ens = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_ens)
build_panel_celltype_arrays = _ens.build_panel_celltype_arrays
fast_ensemble = _ens.fast_ensemble
summarize = _ens.summarize

CANON = ["Monocyte", "NK", "B", "DC", "T_cell"]


def ccc(yt, yp):
    yt = np.asarray(yt).ravel(); yp = np.asarray(yp).ravel()
    mt, mp = yt.mean(), yp.mean(); vt, vp = yt.var(), yp.var()
    cov = ((yt - mt) * (yp - mp)).mean(); d = vt + vp + (mt - mp) ** 2
    return float(2 * cov / d) if d > 0 else float("nan")


def overall_metrics(yt, yp):
    A, B = yt.values, yp.values; diff = A - B
    pr = np.corrcoef(A.ravel(), B.ravel())[0, 1] if A.std() > 0 and B.std() > 0 else np.nan
    return {"MAE": float(np.abs(diff).mean()), "RMSE": float(np.sqrt((diff ** 2).mean())),
            "Pearson": float(pr), "CCC": ccc(A, B)}


def auroc_at(abs_err, score, thr):
    labels = (abs_err > thr).astype(int)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return np.nan, int(labels.sum())
    return float(roc_auc_score(labels, score)), int(labels.sum())


def reject_high_at_half(abs_err, score):
    """Mean retained error after rejecting most-uncertain down to 50% kept."""
    order = np.argsort(-score)  # descending uncertainty
    n = len(order); n_keep = max(1, n // 2)
    kept = order[n_keep:]  # drop the most-uncertain half, keep confident half
    return float(abs_err[kept].mean()), float(abs_err.mean())


def run_ensemble_on(Xp, per_type_arrays, per_type_totals, cell_types, gene_panel,
                    n_iter, gf, cf, seed, logger, tag, order, col_order):
    """Run ensemble on a (perturbed) bulk; return aligned mean/std wide tables."""
    preds = fast_ensemble(
        X_panel=Xp.values.astype(np.float64),
        per_type_arrays=per_type_arrays, per_type_totals=per_type_totals,
        cell_types=cell_types, n_panel_genes=len(gene_panel),
        n_iterations=n_iter, gene_frac=gf, cell_frac=cf,
        noise_std=0.0, seed=seed, logger=logger, tag=tag)
    summ = summarize(preds, cell_types, list(Xp.index))

    def wide(stat):
        w = summ.pivot_table(index="sample_id", columns="cell_type",
                             values=stat, aggfunc="first").loc[order, col_order]
        assert list(w.index) == list(order) and list(w.columns) == list(col_order)
        return w
    return wide("mean"), wide("std"), wide("interval_width")


def parse_args():
    p = argparse.ArgumentParser(description="Phase 5 Tier 2 subset (batch_shift, reference_reduction_1)")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--cell-pool-dir", default="data/processed")
    p.add_argument("--proc-dir", default="data/processed")
    p.add_argument("--ensemble-dir", default="results/ensemble_marker_5types")
    p.add_argument("--conformal-dir", default="results/conformal_marker_5types")
    p.add_argument("--output-dir", default="results/stress_marker_5types_tier2_subset")
    p.add_argument("--n-iterations", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


SCENARIOS = [
    ("batch_shift_small",     "batch_shift",   {"shift_size": 0.5}),
    ("batch_shift_large",     "batch_shift",   {"shift_size": 2.0}),
    ("reference_reduction_1", "ref_reduction", {"n_remove": 1}),
]


def main():
    args = parse_args()
    logger = setup_logger("06d_tier2_subset", log_dir=str(PROJECT_ROOT / "logs"))
    set_seed(args.seed)

    cfg = load_config(args.config)
    ens_cfg = cfg["ensemble"]
    gf = ens_cfg["gene_sampling_fraction"]; cf = ens_cfg["cell_sampling_fraction"]
    base_col = cfg["data"]["sc_reference"]["cell_type_column"]

    proc_dir = PROJECT_ROOT / args.proc_dir
    conf_dir = PROJECT_ROOT / args.conformal_dir
    out_dir = ensure_dir(str(PROJECT_ROOT / args.output_dir))
    fig_dir = ensure_dir(str(PROJECT_ROOT / "results" / "figures"))

    paths = resolve_inputs("markers", "5type", proc_dir)
    with open(paths["panel_file"]) as fh:
        gene_panel = [l.strip() for l in fh if l.strip()]

    import scanpy as sc
    adata_ref = sc.read_h5ad(Path(PROJECT_ROOT / args.cell_pool_dir) / "cell_pool_reference.h5ad")
    cell_type_col = resolve_cell_type_column(adata_ref, "5type", base_col=base_col)
    cell_types = sorted(adata_ref.obs[cell_type_col].unique().tolist())

    X_test_clean = pd.read_csv(paths["test_file"], index_col=0)[gene_panel]
    X_cal_clean = pd.read_csv(paths["cal_file"], index_col=0)[gene_panel]
    y_test = pd.read_csv(paths["true_test"], index_col=0)[CANON]; y_test.index.name = "sample_id"
    y_cal = pd.read_csv(paths["true_cal"], index_col=0)[CANON]; y_cal.index.name = "sample_id"
    test_order = list(y_test.index)

    quantiles5 = pd.read_csv(conf_dir / "conformal_quantiles.csv")
    per_type_arrays, per_type_totals = build_panel_celltype_arrays(
        adata_ref, cell_type_col, cell_types, gene_panel, logger)

    logger.info("=" * 60)
    logger.info("Tier 2 SUBSET: %s | B=%d | panel=%d", [s[0] for s in SCENARIOS],
                args.n_iterations, len(gene_panel))
    logger.info("=" * 60)

    rows = []
    for name, kind, params in SCENARIOS:
        sc_seed = args.seed + abs(hash(name)) % 10000
        logger.info("-" * 40); logger.info("scenario: %s (%s)", name, kind)

        if kind == "batch_shift":
            # Perturb test bulk only; signature, true props, quantiles unchanged.
            Xp = apply_batch_shift(X_test_clean, shift_size=params["shift_size"],
                                   seed=sc_seed)[gene_panel]
            assert list(Xp.index) == test_order and list(Xp.columns) == gene_panel
            mean, std, width = run_ensemble_on(
                Xp, per_type_arrays, per_type_totals, cell_types, gene_panel,
                args.n_iterations, gf, cf, sc_seed, logger, name, test_order, CANON)
            yt = y_test.loc[mean.index, mean.columns]
            quantiles = quantiles5
            n_types_used = 5

        elif kind == "ref_reduction":
            # Remove 1 cell type from the REFERENCE; it stays in the bulk.
            kept, removed = reduce_reference_cell_types(cell_types, n_remove=params["n_remove"], seed=sc_seed)
            kept_canon = [c for c in CANON if c in kept]
            logger.info("  removed=%s kept=%s", removed, kept_canon)
            arrs = [per_type_arrays[cell_types.index(c)] for c in kept]
            tots = [per_type_totals[cell_types.index(c)] for c in kept]

            # Clean cal/test bulk unchanged; re-run ensemble with REDUCED reference.
            mean_t, std_t, width_t = run_ensemble_on(
                X_test_clean[gene_panel], arrs, tots, kept, gene_panel,
                args.n_iterations, gf, cf, sc_seed, logger, name + "_test",
                test_order, kept_canon)
            mean_c, std_c, _ = run_ensemble_on(
                X_cal_clean[gene_panel], arrs, tots, kept, gene_panel,
                args.n_iterations, gf, cf, sc_seed, logger, name + "_cal",
                list(y_cal.index), kept_canon)

            # True props restricted to kept types, renormalized (sum to 1 over kept).
            yt = y_test[kept_canon].copy(); yt = yt.div(yt.sum(axis=1), axis=0)
            yc = y_cal[kept_canon].copy(); yc = yc.div(yc.sum(axis=1), axis=0)
            mean, std, width = mean_t, std_t, width_t

            # RE-CALIBRATE conformal on the reduced cal set (5-type quantiles invalid).
            cal_res = calibrate(y_true_cal=yc, y_pred_cal=mean_c[kept_canon],
                                nominal_coverages=[0.8, 0.9, 0.95],
                                score_function="absolute_error", per_cell_type=True)
            quantiles = cal_res["quantiles"]
            n_types_used = len(kept_canon)
        else:
            raise ValueError(kind)

        assert list(yt.index) == list(mean.index) and list(yt.columns) == list(mean.columns)
        abs_err = np.abs(yt.values - mean.values).mean(axis=1)
        rel_std = std.mean(axis=1).values
        m = overall_metrics(yt, mean)

        intervals = predict_intervals(mean, quantiles, y_std_test=None,
                                      score_function="absolute_error", clip=True)
        cov = evaluate_calibration(intervals, yt)
        cov90 = cov[(cov["cell_type"] == "overall") & (cov["nominal_coverage"] == 0.90)]
        cov90_clip = float(cov90["empirical_coverage_clip"].values[0]) if len(cov90) else np.nan
        w90 = intervals[intervals["nominal_coverage"] == 0.90]["interval_width_clip"].mean()

        ue = (np.corrcoef(rel_std, abs_err)[0, 1]
              if np.std(rel_std) > 0 and np.std(abs_err) > 0 else np.nan)
        auroc10, nf10 = auroc_at(abs_err, rel_std, 0.10)
        auroc15, nf15 = auroc_at(abs_err, rel_std, 0.15)
        rj_half, rj_full = reject_high_at_half(abs_err, rel_std)

        rows.append({
            "scenario": name, "kind": kind, "params": str(params),
            "n_types_used": n_types_used, "n_samples": len(yt),
            "MAE": m["MAE"], "RMSE": m["RMSE"], "Pearson": m["Pearson"], "CCC": m["CCC"],
            "mean_std": float(rel_std.mean()), "mean_interval_width": float(width.mean(axis=1).mean()),
            "coverage90_clip": cov90_clip, "conformal_width90_clip": float(w90),
            "uncertainty_error_corr": float(ue) if ue == ue else np.nan,
            "AUROC_fail_gt0.10": auroc10, "n_fail_gt0.10": nf10,
            "AUROC_fail_gt0.15": auroc15, "n_fail_gt0.15": nf15,
            "reject_high_mae_retain50": rj_half, "full_mae": rj_full,
            "reject_high_delta_vs_all": rj_half - rj_full,
        })
        logger.info("  MAE=%.4f CCC=%.4f cov90=%.3f std=%.4f AUROC10=%s rejhalfΔ=%+.4f",
                    m["MAE"], m["CCC"], cov90_clip, rel_std.mean(),
                    f"{auroc10:.3f}" if auroc10 == auroc10 else "NA", rj_half - rj_full)

    df = pd.DataFrame(rows)
    df.to_csv(Path(out_dir) / "stress_summary_tier2_subset.csv", index=False)
    logger.info("Saved %s", Path(out_dir) / "stress_summary_tier2_subset.csv")

    # ── QC summary ──
    base = pd.read_csv(PROJECT_ROOT / "results" / "stress_marker_5types" /
                       "stress_summary_tier1_corrected.csv") \
        if (PROJECT_ROOT / "results" / "stress_marker_5types" /
            "stress_summary_tier1_corrected.csv").exists() else None
    base_mae = None
    if base is not None and "baseline" in set(base["scenario"]):
        base_mae = float(base[base["scenario"] == "baseline"]["MAE"].values[0])

    print("\n" + "=" * 80)
    print("PHASE 5 TIER 2 SUBSET — QC SUMMARY (reliability score = mean_std)")
    print("=" * 80)
    print(f"\nTier 1 baseline MAE (reference): {base_mae if base_mae is not None else 'NA'}")
    print(f"\n{'scenario':<22}{'types':>6}{'MAE':>8}{'RMSE':>8}{'Pear':>7}{'CCC':>7}"
          f"{'std':>8}{'iwidth':>8}{'cov90':>7}{'ue_r':>7}{'AUROC10':>8}{'rejΔ50':>8}")
    for _, r in df.iterrows():
        au = f"{r['AUROC_fail_gt0.10']:.3f}" if r['AUROC_fail_gt0.10'] == r['AUROC_fail_gt0.10'] else "NA"
        print(f"{r['scenario']:<22}{int(r['n_types_used']):>6}{r['MAE']:>8.4f}{r['RMSE']:>8.4f}"
              f"{r['Pearson']:>7.3f}{r['CCC']:>7.3f}{r['mean_std']:>8.4f}"
              f"{r['mean_interval_width']:>8.3f}{r['coverage90_clip']:>7.3f}"
              f"{r['uncertainty_error_corr']:>7.3f}{au:>8}{r['reject_high_delta_vs_all']:>+8.4f}")

    print("\nInterpretation:")
    for _, r in df.iterrows():
        cov_ok = "≥0.90" if r["coverage90_clip"] >= 0.90 else "<0.90 (degraded)"
        rej_ok = "drops" if r["reject_high_delta_vs_all"] < 0 else "no drop"
        print(f"  {r['scenario']:<22} coverage90 {r['coverage90_clip']:.3f} ({cov_ok}); "
              f"reject_high error {rej_ok}; AUROC10="
              f"{r['AUROC_fail_gt0.10']:.3f}" if r['AUROC_fail_gt0.10'] == r['AUROC_fail_gt0.10']
              else f"  {r['scenario']}: AUROC NA")

    print(f"\nOutput file: {Path(out_dir) / 'stress_summary_tier2_subset.csv'}")
    print("=" * 80)
    logger.info("Tier 2 subset complete.")


if __name__ == "__main__":
    main()
