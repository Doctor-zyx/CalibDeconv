#!/usr/bin/env python
"""
READ-ONLY Phase 5 Tier 1 reliability-score diagnostics.

Re-runs the SAME perturbed ensemble per scenario (identical seeds to 06b) to
get sample-level scores, then evaluates multiple reliability scores for
failure detection and rejection. Writes ONLY two new diagnostic CSVs; does
NOT touch any existing Tier 1 result, figure, or summary.

Outputs:
  results/stress_marker_5types/reliability_score_diagnostics_tier1.csv
  results/stress_marker_5types/rejection_direction_diagnostics_tier1.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import load_config, setup_logger, set_seed
from src.deconvolution.celltypes import resolve_inputs, resolve_cell_type_column
from src.evaluation.stress import apply_gaussian_noise, apply_dropout, apply_low_depth

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("ens04b", str(PROJECT_ROOT / "scripts" / "04b_ensemble_marker5.py"))
_ens = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_ens)
build_panel_celltype_arrays = _ens.build_panel_celltype_arrays
fast_ensemble = _ens.fast_ensemble
summarize = _ens.summarize

CANON = ["Monocyte", "NK", "B", "DC", "T_cell"]
TIER1 = [
    ("baseline", "none", {}, 0.0),
    ("gaussian_noise_low", "gaussian", {"noise_std": 0.1}, 0.1),
    ("gaussian_noise_medium", "gaussian", {"noise_std": 0.5}, 0.5),
    ("gaussian_noise_high", "gaussian", {"noise_std": 1.0}, 1.0),
    ("dropout_low", "dropout", {"dropout_rate": 0.1}, 0.1),
    ("dropout_medium", "dropout", {"dropout_rate": 0.3}, 0.3),
    ("dropout_high", "dropout", {"dropout_rate": 0.5}, 0.5),
    ("low_depth_medium", "low_depth", {"depth_fraction": 0.25}, 0.75),
    ("low_depth_high", "low_depth", {"depth_fraction": 0.10}, 0.90),
]


def perturb(kind, X, params, seed):
    if kind == "none": return X.copy()
    if kind == "gaussian": return apply_gaussian_noise(X, noise_std=params["noise_std"], seed=seed)
    if kind == "dropout": return apply_dropout(X, dropout_rate=params["dropout_rate"], seed=seed)
    if kind == "low_depth": return apply_low_depth(X, depth_fraction=params["depth_fraction"], seed=seed)
    raise ValueError(kind)


def auroc_at(abs_err, score, thr):
    labels = (abs_err > thr).astype(int)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return np.nan
    return float(roc_auc_score(labels, score))


def main():
    logger = setup_logger("06c_reliability", log_dir=str(PROJECT_ROOT / "logs"))
    set_seed(42)
    cfg = load_config("config/config.yaml")
    ens_cfg = cfg["ensemble"]
    gene_frac, cell_frac = ens_cfg["gene_sampling_fraction"], ens_cfg["cell_sampling_fraction"]
    base_col = cfg["data"]["sc_reference"]["cell_type_column"]

    proc = PROJECT_ROOT / "data" / "processed"
    out_dir = PROJECT_ROOT / "results" / "stress_marker_5types"
    paths = resolve_inputs("markers", "5type", proc)
    gene_panel = [l.strip() for l in open(paths["panel_file"]) if l.strip()]

    import scanpy as sc
    adata = sc.read_h5ad(proc / "cell_pool_reference.h5ad")
    ctcol = resolve_cell_type_column(adata, "5type", base_col=base_col)
    cell_types = sorted(adata.obs[ctcol].unique().tolist())
    Xc = pd.read_csv(paths["test_file"], index_col=0)[gene_panel]
    y = pd.read_csv(paths["true_test"], index_col=0)[CANON]; y.index.name = "sample_id"
    true_order = list(y.index)
    arrs, tots = build_panel_celltype_arrays(adata, ctcol, cell_types, gene_panel, logger)

    SCORES = ["mean_std", "max_std", "mean_interval_width", "max_interval_width",
              "mean_disagreement", "max_disagreement"]
    rel_rows, rej_rows = [], []
    RETAIN = [0.9, 0.75, 0.5, 0.25]

    for name, kind, params, sev in TIER1:
        seed = 42 + abs(hash(name)) % 10000
        Xp = perturb(kind, Xc, params, seed)[gene_panel]
        preds = fast_ensemble(X_panel=Xp.values.astype(np.float64),
                              per_type_arrays=arrs, per_type_totals=tots,
                              cell_types=cell_types, n_panel_genes=len(gene_panel),
                              n_iterations=50, gene_frac=gene_frac, cell_frac=cell_frac,
                              noise_std=0.0, seed=seed, logger=logger, tag=name)
        summ = summarize(preds, cell_types, list(Xp.index))

        def wide(stat):
            return summ.pivot_table(index="sample_id", columns="cell_type",
                                    values=stat, aggfunc="first").loc[true_order, CANON]
        mean, std, width, dis = wide("mean"), wide("std"), wide("interval_width"), wide("disagreement")
        yt = y.loc[mean.index, mean.columns]
        assert list(yt.index) == list(mean.index) and list(yt.columns) == list(mean.columns)

        abs_err = np.abs(yt.values - mean.values).mean(axis=1)
        overall_mae = float(abs_err.mean())
        score_vecs = {
            "mean_std": std.mean(axis=1).values, "max_std": std.max(axis=1).values,
            "mean_interval_width": width.mean(axis=1).values, "max_interval_width": width.max(axis=1).values,
            "mean_disagreement": dis.mean(axis=1).values, "max_disagreement": dis.max(axis=1).values,
        }
        n = len(abs_err); k = max(1, n // 10)
        for sn in SCORES:
            sc_v = score_vecs[sn]
            order = np.argsort(-sc_v)  # high->low
            top10 = float(abs_err[order[:k]].mean()); bot10 = float(abs_err[order[-k:]].mean())
            pear = float(sps.pearsonr(sc_v, abs_err)[0]) if np.std(sc_v) > 0 else np.nan
            spear = float(sps.spearmanr(sc_v, abs_err)[0]) if np.std(sc_v) > 0 else np.nan
            rel_rows.append({
                "scenario": name, "score_name": sn,
                "pearson_score_mae": pear, "spearman_score_mae": spear,
                "top10_uncertain_mae": top10, "bottom10_uncertain_mae": bot10,
                "overall_mae": overall_mae, "top10_minus_bottom10": top10 - bot10,
                "auroc_fail_gt0.10": auroc_at(abs_err, sc_v, 0.10),
                "auroc_fail_gt0.15": auroc_at(abs_err, sc_v, 0.15),
                "failure_rate_gt0.10": float((abs_err > 0.10).mean()),
                "failure_rate_gt0.15": float((abs_err > 0.15).mean()),
            })
            # rejection both directions
            asc = np.argsort(sc_v)  # low->high
            for frac in RETAIN:
                nk = max(1, int(n * frac))
                rh = float(abs_err[asc[:nk]].mean())       # keep LOW score (reject high)
                rl = float(abs_err[order[:nk]].mean())     # keep HIGH score (reject low)
                rej_rows.append({
                    "scenario": name, "score_name": sn, "retained_fraction": frac,
                    "reject_high_mae": rh, "reject_low_mae": rl,
                    "reject_high_delta_vs_all": rh - overall_mae,
                    "reject_low_delta_vs_all": rl - overall_mae,
                })
        logger.info("[%s] done overall_mae=%.4f", name, overall_mae)

    rel = pd.DataFrame(rel_rows); rej = pd.DataFrame(rej_rows)
    rel.to_csv(out_dir / "reliability_score_diagnostics_tier1.csv", index=False)
    rej.to_csv(out_dir / "rejection_direction_diagnostics_tier1.csv", index=False)
    print("\n[SAVED]", out_dir / "reliability_score_diagnostics_tier1.csv")
    print("[SAVED]", out_dir / "rejection_direction_diagnostics_tier1.csv")

    # ── headline diagnostics ──
    print("\n=== Spearman(score, sample_MAE) by score (mean over scenarios) ===")
    print(rel.groupby("score_name")["spearman_score_mae"].mean().round(4).sort_values(ascending=False).to_string())
    print("\n=== top10_minus_bottom10 (mean over scenarios; >0 = uncertain are worse) ===")
    print(rel.groupby("score_name")["top10_minus_bottom10"].mean().round(4).sort_values(ascending=False).to_string())
    print("\n=== mean_std: reject@retain=0.5 both directions per scenario ===")
    sub = rej[(rej.score_name == "mean_std") & (rej.retained_fraction == 0.5)]
    print(sub[["scenario", "reject_high_mae", "reject_low_mae", "reject_high_delta_vs_all"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
