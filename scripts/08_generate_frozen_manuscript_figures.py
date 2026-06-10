#!/usr/bin/env python
"""
Generate manuscript figures from FROZEN results only.

NO new experiments, NO ensemble/conformal re-runs, NO downloads, NO Tier 3.
Reads only existing result CSVs; writes ONLY to results/figures_manuscript/.
Does not modify or overwrite any existing core result file.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.grid": True,
    "grid.alpha": 0.3, "axes.axisbelow": True,
})

R = PROJECT_ROOT / "results"
OUT = R / "figures_manuscript"
OUT.mkdir(parents=True, exist_ok=True)
CANON = ["Monocyte", "NK", "B", "DC", "T_cell"]
T_SUBSETS = ["CD4_T", "CD8_T", "other_T"]

_missing = []
_made = []


def need(path):
    p = R / path if not str(path).startswith(str(R)) else Path(path)
    if not Path(p).exists():
        _missing.append(str(path))
        return None
    return Path(p)


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    _made.append(stem)


def read(path):
    p = need(path)
    return pd.read_csv(p) if p is not None else None


# ── MAIN FIGURES ─────────────────────────────────────────────────────────────

def fig2C():
    """Per-cell-type CCC: Phase 2 NNLS vs Phase 3 ensemble."""
    p2 = read("nnls_marker_5types/marker_5type_summary.csv")
    p3 = read("ensemble_marker_5types/ensemble_accuracy.csv")
    if p2 is None or p3 is None:
        return
    p2 = p2.set_index("cell_type")["CCC"]
    p3 = p3[(p3.split == "test") & (p3.level == "per_cell_type")].set_index("cell_type")["CCC"]
    x = np.arange(len(CANON)); w = 0.38
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - w/2, [p2[c] for c in CANON], w, label="Phase 2 NNLS")
    ax.bar(x + w/2, [p3[c] for c in CANON], w, label="Phase 3 ensemble mean")
    ax.set_xticks(x); ax.set_xticklabels(CANON); ax.set_ylabel("CCC (test)")
    ax.set_ylim(0, 1); ax.set_title("Per-cell-type CCC: baseline vs ensemble")
    ax.legend()
    save(fig, "fig2C_per_celltype_CCC_phase2_vs_phase3")


def fig3B():
    """Per-cell-type 90% conformal coverage (clipped), nominal line at 0.90."""
    c = read("conformal_marker_5types/coverage_by_cell_type.csv")
    if c is None:
        return
    c90 = c[c.nominal_coverage == 0.90].set_index("cell_type")
    vals = [c90.loc[ct, "empirical_coverage_clip"] for ct in CANON]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(CANON, vals, color="#4C72B0")
    ax.axhline(0.90, color="k", ls="--", lw=1, label="nominal 0.90")
    ax.set_ylim(0.8, 1.0); ax.set_ylabel("Empirical coverage @90% (clipped)")
    ax.set_title("Per-cell-type conformal coverage"); ax.legend()
    save(fig, "fig3B_per_celltype_coverage90")


def _tier1_baseline():
    t1 = read("stress_marker_5types/stress_summary_tier1_corrected.csv")
    return t1[t1.scenario == "baseline"].iloc[0] if t1 is not None else None


def _refred():
    t2 = read("stress_marker_5types_tier2_subset/stress_summary_tier2_subset.csv")
    if t2 is None:
        return None
    return t2[t2.scenario == "reference_reduction_1"].iloc[0]


def fig5A():
    """baseline vs reference_reduction_1: MAE and CCC."""
    b, r = _tier1_baseline(), _refred()
    if b is None or r is None:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4))
    x = np.arange(2); w = 0.35
    ax.bar(x - w/2, [b["MAE"], r["MAE"]], w, label="MAE", color="#C44E52")
    ax.bar(x + w/2, [b["CCC"], r["CCC"]], w, label="CCC", color="#55A868")
    ax.set_xticks(x); ax.set_xticklabels(["baseline", "reference_reduction_1"])
    ax.set_ylabel("value"); ax.set_title("Accuracy under reference reduction")
    ax.legend()
    save(fig, "fig5A_reference_reduction_accuracy")


def fig5B():
    """baseline vs reference_reduction_1: coverage90 (nominal line)."""
    b, r = _tier1_baseline(), _refred()
    if b is None or r is None:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["baseline", "reference_reduction_1"],
           [b["coverage90_clip"], r["coverage90_clip"]], color="#4C72B0")
    ax.axhline(0.90, color="k", ls="--", lw=1, label="nominal 0.90")
    ax.set_ylim(0.8, 1.0); ax.set_ylabel("Coverage @90% (clipped)")
    ax.set_title("Coverage holds after recalibration"); ax.legend()
    save(fig, "fig5B_reference_reduction_coverage")


def fig5C():
    """baseline vs reference_reduction_1: uncertainty-error correlation drop."""
    b, r = _tier1_baseline(), _refred()
    if b is None or r is None:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["baseline", "reference_reduction_1"],
           [b["uncertainty_error_corr"], r["uncertainty_error_corr"]], color="#8172B3")
    ax.set_ylim(0, 0.7); ax.set_ylabel("Uncertainty-error correlation")
    ax.set_title("Reliability signal weakens (0.62 → 0.20)")
    save(fig, "fig5C_reference_reduction_uecorr_drop")


def fig5D():
    """reject_high delta@50% : baseline, dropout_high, reference_reduction_1."""
    t1 = read("stress_marker_5types/stress_summary_tier1_corrected.csv")
    r = _refred()
    if t1 is None or r is None:
        return
    b = t1[t1.scenario == "baseline"].iloc[0]
    dh = t1[t1.scenario == "dropout_high"].iloc[0]
    names = ["baseline", "dropout_high", "reference_reduction_1"]
    vals = [b["reject_high_delta_vs_all"], dh["reject_high_delta_vs_all"],
            r["reject_high_delta_vs_all"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, vals, color=["#55A868", "#C44E52", "#8172B3"])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("reject_high Δ vs all @retain50%")
    ax.set_title("Rejection benefit (negative = error drops)")
    ax.tick_params(axis="x", rotation=15)
    save(fig, "fig5D_reference_reduction_rejection")


# ── SUPPLEMENTARY FIGURES ────────────────────────────────────────────────────

def supp_s1A():
    """CCC comparison across configs: HVG7 / marker7 / marker5."""
    c = read("nnls_comparison/nnls_baseline_comparison.csv")
    if c is None:
        return
    c = c.set_index("baseline")
    order = ["HVG3000_7types", "marker_7types", "marker_5types"]
    order = [o for o in order if o in c.index]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(order, [c.loc[o, "test_CCC"] for o in order],
           color=["#C44E52", "#DD8452", "#55A868"])
    ax.set_ylim(0, 1); ax.set_ylabel("test CCC")
    ax.set_title("Configuration comparison (CCC)")
    ax.tick_params(axis="x", rotation=15)
    save(fig, "supp_s1A_config_CCC_comparison")


def supp_s1B():
    """7type predicted mean per cell type: CD8_T / other_T collapse to ~0."""
    p = read("nnls_marker_7types/predicted_proportions_test.csv")
    if p is None:
        return
    if p.columns[0].startswith("Unnamed") or p.iloc[:, 0].dtype == object:
        p = p.drop(columns=p.columns[0])
    p = p.select_dtypes(include=[np.number])
    means = p.mean().sort_values()
    colors = ["#C44E52" if c in T_SUBSETS else "#4C72B0" for c in means.index]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(means.index, means.values, color=colors)
    ax.set_ylabel("predicted mean proportion")
    ax.set_title("7-type collapse: CD8_T / other_T → ~0 (red)")
    ax.tick_params(axis="x", rotation=20)
    save(fig, "supp_s1B_7type_collapse_pred_mean")


def supp_s2C():
    """Reliability score comparison: mean Spearman & AUROC across scenarios."""
    d = read("stress_marker_5types/reliability_score_diagnostics_tier1.csv")
    if d is None:
        return
    agg = d.groupby("score_name").agg(
        spearman=("spearman_score_mae", "mean"),
        auroc=("auroc_fail_gt0.10", "mean")).reset_index()
    agg = agg.sort_values("spearman", ascending=False)
    x = np.arange(len(agg)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(x - w/2, agg["spearman"], w, label="mean Spearman(score, MAE)")
    ax.bar(x + w/2, agg["auroc"], w, label="mean AUROC(>0.10)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(agg["score_name"], rotation=20, ha="right")
    ax.set_title("Reliability score comparison (Tier 1 mean)"); ax.legend()
    save(fig, "supp_s2C_score_comparison")


def supp_s2D():
    """Rejection direction: reject_high vs reject_low delta @ retain~50% (mean_std)."""
    d = read("stress_marker_5types/rejection_direction_diagnostics_tier1.csv")
    if d is None:
        return
    d = d[d.score_name == "mean_std"].copy()
    # pick the retained_fraction closest to 0.5
    d["dist"] = (d["retained_fraction"] - 0.5).abs()
    d = d.sort_values("dist").groupby("scenario", as_index=False).first()
    d = d.sort_values("scenario")
    x = np.arange(len(d)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w/2, d["reject_high_delta_vs_all"], w, label="reject_high (keep confident)")
    ax.bar(x + w/2, d["reject_low_delta_vs_all"], w, label="reject_low (keep uncertain)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(d["scenario"], rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("Δ MAE vs all @retain50%")
    ax.set_title("Rejection direction (negative = error drops)"); ax.legend()
    save(fig, "supp_s2D_rejection_direction")


def supp_s3():
    """batch_shift subset: MAE and coverage90 vs baseline."""
    t2 = read("stress_marker_5types_tier2_subset/stress_summary_tier2_subset.csv")
    b = _tier1_baseline()
    if t2 is None or b is None:
        return
    bs = t2[t2.kind == "batch_shift"].set_index("scenario")
    names = ["baseline", "batch_shift_small", "batch_shift_large"]
    mae = [b["MAE"]] + [bs.loc[n, "MAE"] for n in names[1:] if n in bs.index]
    cov = [b["coverage90_clip"]] + [bs.loc[n, "coverage90_clip"] for n in names[1:] if n in bs.index]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(names, mae, color="#C44E52"); axes[0].set_ylabel("MAE")
    axes[0].set_title("MAE"); axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(names, cov, color="#4C72B0"); axes[1].axhline(0.90, color="k", ls="--", lw=1)
    axes[1].set_ylim(0.8, 1.0); axes[1].set_ylabel("coverage @90%")
    axes[1].set_title("Coverage (nominal 0.90)"); axes[1].tick_params(axis="x", rotation=20)
    fig.suptitle("batch_shift subset (additive shift weak on CPM scale)")
    save(fig, "supp_s3_batch_shift_subset")


def supp_s4():
    """Bug-guards table as markdown (table figure is awkward to typeset)."""
    md = OUT / "supp_s4_bug_guards_table.md"
    md.write_text(
        "# Supp S4 — Fixed bugs and guards\n\n"
        "| # | Bug | Guard |\n|---|---|---|\n"
        "| 1 | 10x debug data mistaken for results | debug = code-validation only; formal line uses Hao 2021 |\n"
        "| 2 | 7type CD8_T/other_T collapse | collapse to 5type (T_cell); 7type kept as diagnostic |\n"
        "| 3 | signature CSV read with index_col=0 drops a column | read without index_col; assert shape (445,5) |\n"
        "| 4 | pivot_table lexicographic sample_id row scramble | .loc[true_order, canonical_cols] + asserts |\n"
        "| 5 | cal true/pred NK/B column order mismatch | normalize to canonical order; backup preCANON |\n"
        "| 6 | rejection direction misread (NO-DROP) | emit reject_high & reject_low; primary=reject_high |\n"
        "| 7 | unaligned .values metrics | align_true_to_pred/align_and_assert before any metric |\n",
        encoding="utf-8")
    _made.append("supp_s4_bug_guards_table (md)")


def main():
    for fn in [fig2C, fig3B, fig5A, fig5B, fig5C, fig5D,
               supp_s1A, supp_s1B, supp_s2C, supp_s2D, supp_s3, supp_s4]:
        try:
            fn()
        except Exception as e:
            print(f"[ERROR] {fn.__name__}: {e}")

    print("\n=== generated ===")
    for s in _made:
        print("  [OK]", s)
    if _missing:
        print("\n=== missing inputs ===")
        for m in sorted(set(_missing)):
            print("  [MISSING]", m)
    else:
        print("\nNo missing inputs.")
    print(f"\nOutput dir: {OUT}")


if __name__ == "__main__":
    main()
