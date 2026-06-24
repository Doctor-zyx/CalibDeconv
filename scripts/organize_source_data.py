"""Organize statistical source data tables for journal submission."""
import shutil, os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)

out = os.path.join("submission_package", "submission_ready", "Source_Data")
os.makedirs(out, exist_ok=True)

def cpdir(name):
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    return d

def cp(src, dst_dir, rename=None):
    fname = rename if rename else os.path.basename(src)
    shutil.copy2(src, os.path.join(dst_dir, fname))

# === Table 1 source data ===
t1 = cpdir("Table1_benchmark_summary")
cp("results/nnls_marker_5types/accuracy_metrics.csv", t1, "nnls_baseline_accuracy_overall.csv")
cp("results/nnls_marker_5types/accuracy_per_celltype.csv", t1, "nnls_baseline_accuracy_per_celltype.csv")
cp("results/ensemble_marker_5types/ensemble_accuracy.csv", t1, "ensemble_accuracy.csv")
cp("results/ensemble_marker_5types/uncertainty_error_correlation.csv", t1, "ensemble_uncertainty_error_correlation.csv")
cp("results/conformal_marker_5types/coverage_by_nominal.csv", t1, "conformal_coverage_by_nominal.csv")
cp("results/conformal_marker_5types/coverage_by_cell_type.csv", t1, "conformal_coverage_by_celltype.csv")
cp("results/conformal_marker_5types/calibration_summary.csv", t1, "conformal_calibration_summary.csv")
cp("results/conformal_marker_5types/interval_width_summary.csv", t1, "conformal_interval_width_summary.csv")

# === Figure 2: Point estimate + uncertainty ===
f2 = cpdir("Figure2_accuracy_uncertainty")
cp("data/processed/true_proportions_test_5type.csv", f2, "true_proportions_test.csv")
cp("results/nnls_marker_5types/predicted_proportions_test.csv", f2, "nnls_predicted_proportions_test.csv")
cp("results/nnls_marker_5types/metrics_test.csv", f2, "nnls_baseline_metrics_test.csv")
cp("results/nnls_marker_5types/marker_5type_summary.csv", f2, "nnls_baseline_per_celltype_summary.csv")
cp("results/ensemble_marker_5types/predicted_proportions_test_ensemble.csv", f2, "ensemble_predicted_proportions_test.csv")
cp("results/ensemble_marker_5types/ensemble_summary_test.csv", f2, "ensemble_summary_test.csv")
cp("results/ensemble_marker_5types/ensemble_accuracy.csv", f2, "ensemble_accuracy_metrics.csv")
cp("results/ensemble_marker_5types/uncertainty_error_correlation.csv", f2, "uncertainty_error_correlation.csv")

# === Figure 3: Conformal calibration ===
f3 = cpdir("Figure3_conformal_calibration")
cp("data/processed/true_proportions_test_5type.csv", f3, "true_proportions_test.csv")
cp("results/conformal_marker_5types/coverage_by_nominal.csv", f3, "coverage_by_nominal.csv")
cp("results/conformal_marker_5types/coverage_by_cell_type.csv", f3, "coverage_by_celltype.csv")
cp("results/conformal_marker_5types/interval_width_summary.csv", f3, "interval_width_summary.csv")
cp("results/conformal_marker_5types/intervals_test_clipped.csv", f3, "intervals_test_clipped.csv")
cp("results/conformal_marker_5types/conformal_quantiles.csv", f3, "conformal_quantiles.csv")
cp("results/conformal_marker_5types/nonconformity_scores.csv", f3, "nonconformity_scores.csv")

# === Figure 4: Stress testing ===
f4 = cpdir("Figure4_stress_testing")
cp("results/stress_marker_5types/stress_summary_tier1_corrected.csv", f4, "stress_summary.csv")
cp("results/stress_marker_5types/failure_detection_tier1_corrected.csv", f4, "failure_detection_auroc.csv")
cp("results/stress_marker_5types/rejection_curves_tier1_corrected.csv", f4, "rejection_curves.csv")
cp("results/stress_marker_5types/stress_per_celltype_tier1.csv", f4, "stress_per_celltype.csv")

# === Figure 5: Reference reduction / ablation ===
f5 = cpdir("Figure5_reference_ablation")
cp("results/phase7_ablation/ablation_metrics.csv", f5, "ablation_metrics.csv")
cp("results/phase7_ablation/coverage_ablation.csv", f5, "coverage_ablation.csv")
cp("results/phase7_ablation/uncertainty_error_ablation.csv", f5, "uncertainty_error_ablation.csv")
cp("results/stress_marker_5types_tier2_subset/stress_summary_tier2_subset.csv", f5, "stress_summary_tier2_subset.csv")

# === Supplementary Figure S1: Config comparison ===
s1 = cpdir("FigS1_config_comparison")
cp("results/nnls_markers/comparison_summary.csv", s1, "config_comparison_summary.csv")
cp("results/nnls_marker_7types/accuracy_per_celltype.csv", s1, "marker_7type_accuracy_per_celltype.csv")
cp("results/nnls_marker_7types/predicted_proportions_test.csv", s1, "marker_7type_predicted_proportions_test.csv")

# === Supplementary Figure S2: Baseline comparison ===
s2 = cpdir("FigS2_baseline_comparison")
cp("results/phase7_baseline_comparison/baseline_metrics.csv", s2, "baseline_metrics.csv")
cp("results/phase7_baseline_comparison/per_celltype_metrics.csv", s2, "baseline_per_celltype_metrics.csv")
cp("results/stress_marker_5types/reliability_score_diagnostics_tier1.csv", s2, "reliability_score_diagnostics.csv")
cp("results/stress_marker_5types/rejection_direction_diagnostics_tier1.csv", s2, "rejection_direction_diagnostics.csv")

# === Supplementary Figure S3: Ablation analyses ===
s3 = cpdir("FigS3_ablation_analyses")
cp("results/phase7_adaptive_conformal/adaptive_conformal_comparison.csv", s3, "adaptive_conformal_comparison.csv")
cp("results/nnls_comparison/nnls_baseline_comparison.csv", s3, "nnls_baseline_comparison.csv")
cp("results/phase7_ablation/ablation_metrics.csv", s3, "module_ablation_metrics.csv")
cp("results/stress_marker_5types_tier2_subset/stress_summary_tier2_subset.csv", s3, "batch_shift_summary.csv")

# === Supplementary Figure S4: External PBMC 3k ===
s4 = cpdir("FigS4_external_PBMC3k")
cp("results/phase7_external_pseudobulk/external_pseudobulk_metrics.csv", s4, "external_pbmc3k_metrics.csv")
cp("results/phase7_external_pseudobulk/external_pseudobulk_per_celltype.csv", s4, "external_pbmc3k_per_celltype.csv")
cp("results/phase7_external_pseudobulk/predicted_proportions.csv", s4, "external_pbmc3k_predicted_proportions.csv")
cp("results/phase7_external_pseudobulk/ground_truth.csv", s4, "external_pbmc3k_ground_truth.csv")
cp("results/phase7_external_pseudobulk/ensemble_std.csv", s4, "external_pbmc3k_ensemble_std.csv")

# === Supplementary Figure S5: GSE107572 ===
s5 = cpdir("FigS5_GSE107572")
cp("results/phase6b_gse107572/metrics_overall.csv", s5, "gse107572_metrics_overall.csv")
cp("results/phase6b_gse107572/metrics_per_celltype.csv", s5, "gse107572_metrics_per_celltype.csv")
cp("results/phase6b_gse107572/predicted_proportions.csv", s5, "gse107572_predicted_proportions.csv")
cp("results/phase6b_gse107572/ground_truth_5type.csv", s5, "gse107572_ground_truth.csv")
cp("results/phase6b_gse107572/ensemble_std.csv", s5, "gse107572_ensemble_std.csv")

# === Supplementary Figure S6: GSE60424 whole blood ===
s6 = cpdir("FigS6_GSE60424_whole_blood")
cp("results/phase6b_gse60424/metrics_overall.csv", s6, "gse60424_metrics_overall.csv")
cp("results/phase6b_gse60424/predicted_proportions_4class.csv", s6, "gse60424_predicted_proportions.csv")
cp("results/phase6b_gse60424/ground_truth_4class.csv", s6, "gse60424_ground_truth.csv")
cp("results/phase6b_gse60424/sample_diagnostics.csv", s6, "gse60424_sample_diagnostics.csv")
cp("results/phase6b_gse60424/predicted_proportions_ensemble_4class.csv", s6, "gse60424_predicted_proportions_ensemble.csv")
cp("results/phase6b_gse60424/ensemble_std_4class.csv", s6, "gse60424_ensemble_std.csv")

# === Supplementary Figure S7: Neutrophil OOD ===
s7 = cpdir("FigS7_neutrophil_OOD")
cp("results/phase7_neutrophil_ood/neutrophil_contamination_metrics.csv", s7, "neutrophil_contamination_metrics.csv")

# === Reference construction (frozen signature + markers) ===
ref = cpdir("Reference_and_signature")
cp("results/nnls_marker_5types/signature_matrix_markers.csv", ref, "frozen_signature_matrix_445genes_5types.csv")
cp("data/processed/pool_celltype_distribution.csv", ref, "reference_pool_celltype_distribution.csv")
cp("results/data_prep/donor_counts.csv", ref, "donor_cell_counts.csv")
cp("results/data_prep/cell_type_counts.csv", ref, "cell_type_counts.csv")
cp("results/data_prep/split_preview_donor_assignment.csv", ref, "donor_pool_assignment.csv")
cp("data/processed/selected_genes_markers.txt", ref, "frozen_445_marker_genes.txt")

# Print final structure
print("=== Source_Data/ final structure ===")
for root, dirs, files in os.walk(out):
    level = root.replace(out, "").count(os.sep)
    indent = "  " * level
    dirname = os.path.basename(root)
    n_files = len(files)
    print(f"{indent}{dirname}/  ({n_files} files)")
    subindent = "  " * (level + 1)
    for f in sorted(files):
        fpath = os.path.join(root, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"{subindent}{f}  ({size_kb:.0f} KB)")

# Count totals
total_files = sum(len(files) for _, _, files in os.walk(out))
total_dirs = sum(1 for _ in os.walk(out)) - 1
print(f"\nTotal: {total_files} files in {total_dirs} subdirectories")
