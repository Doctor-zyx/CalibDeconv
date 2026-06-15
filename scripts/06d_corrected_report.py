#!/usr/bin/env python
"""
Regenerate CORRECTED Phase 5 Tier 1 reports + figures from EXISTING results.

Does NOT re-run ensemble / stress prediction. Reads:
  results/stress_marker_5types/stress_summary_tier1.csv
  results/stress_marker_5types/reliability_score_diagnostics_tier1.csv
  results/stress_marker_5types/rejection_direction_diagnostics_tier1.csv

Primary reliability score = mean_std (best in Tier 1 diagnostics);
secondary = mean_interval_width; disagreement is NOT used as primary.
Rejection convention: reject_high = reject most-uncertain, keep confident.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = PROJECT_ROOT / "results" / "stress_marker_5types"
FIG = PROJECT_ROOT / "results" / "figures"
PRIMARY_SCORE = "mean_std"
SECONDARY_SCORE = "mean_interval_width"

summary = pd.read_csv(OUT / "stress_summary_tier1.csv")
rel = pd.read_csv(OUT / "reliability_score_diagnostics_tier1.csv")
rejdir = pd.read_csv(OUT / "rejection_direction_diagnostics_tier1.csv")

# ── 1. corrected summary (annotate which score is primary, add rejection delta) ──
prim = rel[rel.score_name == PRIMARY_SCORE].set_index("scenario")
rej50 = rejdir[(rejdir.score_name == PRIMARY_SCORE) & (rejdir.retained_fraction == 0.5)].set_index("scenario")
corr = summary.copy()
corr["primary_score"] = PRIMARY_SCORE
corr["spearman_meanstd_mae"] = corr["scenario"].map(prim["spearman_score_mae"])
corr["reject_high_mae_retain50"] = corr["scenario"].map(rej50["reject_high_mae"])
corr["reject_low_mae_retain50"] = corr["scenario"].map(rej50["reject_low_mae"])
corr["reject_high_delta_vs_all"] = corr["scenario"].map(rej50["reject_high_delta_vs_all"])
corr.to_csv(OUT / "stress_summary_tier1_corrected.csv", index=False)

# ── 2. corrected rejection curves (full curve, both directions, primary score) ──
# Rebuild a 20-point curve for the primary score from rejection_direction (which
# only has 4 retain points). We instead recompute curve points present and save.
rej_primary = rejdir[rejdir.score_name == PRIMARY_SCORE].copy()
rej_primary.to_csv(OUT / "rejection_curves_tier1_corrected.csv", index=False)

# ── 3. corrected failure-detection (primary score AUROCs) ──
fd = rel[rel.score_name == PRIMARY_SCORE][
    ["scenario", "auroc_fail_gt0.10", "auroc_fail_gt0.15",
     "failure_rate_gt0.10", "failure_rate_gt0.15",
     "spearman_score_mae", "top10_minus_bottom10"]].copy()
fd.to_csv(OUT / "failure_detection_tier1_corrected.csv", index=False)

# ── 4. figures ──
def savefig(stem):
    for ext in ("png", "pdf"):
        plt.savefig(FIG / f"{stem}.{ext}", dpi=150)
    plt.close()

fams = ["gaussian", "dropout", "low_depth"]
base = summary[summary.perturbation == "none"]

def fam_curve(col):
    plt.figure(figsize=(6, 5))
    for fam in fams:
        d = pd.concat([base, summary[summary.perturbation == fam].sort_values("severity")])
        plt.plot(d["severity"], d[col], "o-", label=fam)
    return

fam_curve("MAE"); plt.xlabel("Severity"); plt.ylabel("Test MAE")
plt.title("Error vs severity (Tier 1, corrected)"); plt.legend(); plt.tight_layout()
savefig("stress_error_vs_severity_corrected")

fam_curve("mean_uncertainty_std"); plt.xlabel("Severity"); plt.ylabel("Mean ensemble std (mean_std)")
plt.title("Uncertainty vs severity (Tier 1, corrected)"); plt.legend(); plt.tight_layout()
savefig("stress_uncertainty_vs_severity_corrected")

fam_curve("coverage90_clip"); plt.axhline(0.90, color="k", ls="--", lw=1, label="nominal 90%")
plt.xlabel("Severity"); plt.ylabel("Coverage @90% (clipped)")
plt.title("Conformal coverage vs severity (Tier 1, corrected)"); plt.legend(); plt.tight_layout()
savefig("stress_coverage_vs_severity_corrected")

# rejection curves (mean_std), primary = reject_high
plt.figure(figsize=(7, 5))
for name in summary["scenario"]:
    d = rej_primary[rej_primary.scenario == name].sort_values("retained_fraction")
    plt.plot(d["retained_fraction"], d["reject_high_mae"], "-o", ms=3, label=name, lw=1)
plt.xlabel("Fraction retained (reject most-uncertain first, keep confident)")
plt.ylabel("Retained-set MAE")
plt.title("Rejection curves by mean_std (Tier 1, corrected)")
plt.legend(fontsize=6, ncol=2); plt.tight_layout()
savefig("stress_rejection_curves_mean_std_corrected")

# failure AUROC (mean_std) by scenario
plt.figure(figsize=(7, 5))
order = summary.sort_values("severity")
au = order["scenario"].map(rel[rel.score_name == PRIMARY_SCORE].set_index("scenario")["auroc_fail_gt0.10"])
plt.bar(order["scenario"], au.values)
plt.axhline(0.5, color="k", ls="--", lw=1, label="random")
plt.ylabel("AUROC (failure MAE>0.10), mean_std")
plt.title("Failure detection AUROC by mean_std (Tier 1, corrected)")
plt.xticks(rotation=45, ha="right", fontsize=7); plt.legend(); plt.tight_layout()
savefig("stress_failure_auroc_mean_std_corrected")

print("[OK] corrected CSVs + figures regenerated (no ensemble rerun)")
print("\nRejection @ retain50% (primary score = mean_std):")
print(rej50.reset_index()[["scenario", "reject_high_mae", "reject_low_mae",
                           "reject_high_delta_vs_all"]].round(4).to_string(index=False))
