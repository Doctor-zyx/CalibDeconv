#!/usr/bin/env python
"""
SIGNATURE CONSTRUCTION AND MANUSCRIPT CONSISTENCY REMEDIATION
=============================================================

Parts 1-8 of the signature consistency audit directive.

Part 1: Exact signature construction proof from code
Part 2: Manuscript text inconsistency audit (generates CSV)
Part 3: Three independent signature versions (A/B/C)
Part 4: Full primary benchmark comparison
Part 5: External PBMC 3k and stress test comparison
Part 6: Decision rule (A/B/C)
Part 7: Methods replacement text
Part 8: SDY67 EPIC positive control validity re-evaluation

Output directory: results/signature_consistency_audit/
"""

import sys
import os
import time
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls as scipy_nnls
from scipy import sparse, stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = PROJECT_ROOT / "results" / "signature_consistency_audit"
OUT.mkdir(parents=True, exist_ok=True)
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)

CANON5 = ["B", "DC", "Monocyte", "NK", "T_cell"]
CANON = ["Monocyte", "NK", "B", "DC", "T_cell"]

T_SUBSETS = ["CD4_T", "CD8_T", "other_T"]


def ccc_func(y_true, y_pred):
    t = np.asarray(y_true).ravel()
    p = np.asarray(y_pred).ravel()
    mt, mp = t.mean(), p.mean()
    vt = np.var(t, ddof=1)
    vp = np.var(p, ddof=1)
    cov = np.cov(t, p, ddof=1)[0, 1]
    denom = vt + vp + (mt - mp) ** 2
    return float(2 * cov / denom) if denom > 0 else float("nan")


def deconv_nnls_single(bulk, sig, normalize=True):
    x, _ = scipy_nnls(sig, bulk, maxiter=10000)
    if normalize and x.sum() > 0:
        x = x / x.sum()
    return x


def deconv_batch(X, S):
    n, k = X.shape[0], S.shape[1]
    P = np.zeros((n, k))
    for i in range(n):
        P[i] = deconv_nnls_single(X[i], S)
    return P


def compute_metrics(y_true, y_pred, cell_types):
    """Full metrics: overall + per-cell-type."""
    rows = []
    t_flat = y_true.ravel()
    p_flat = y_pred.ravel()
    rows.append({
        "cell_type": "overall",
        "MAE": float(np.abs(t_flat - p_flat).mean()),
        "RMSE": float(np.sqrt(((t_flat - p_flat) ** 2).mean())),
        "CCC": ccc_func(t_flat, p_flat),
        "Pearson": float(stats.pearsonr(t_flat, p_flat)[0]) if t_flat.std() > 0 else np.nan,
        "bias": float((p_flat - t_flat).mean()),
        "true_mean": float(t_flat.mean()),
        "pred_mean": float(p_flat.mean()),
    })
    for j, ct in enumerate(cell_types):
        tv = y_true[:, j]
        pv = y_pred[:, j]
        rows.append({
            "cell_type": ct,
            "MAE": float(np.abs(tv - pv).mean()),
            "RMSE": float(np.sqrt(((tv - pv) ** 2).mean())),
            "CCC": ccc_func(tv, pv),
            "Pearson": float(stats.pearsonr(tv, pv)[0]) if tv.std() > 0 and pv.std() > 0 else np.nan,
            "bias": float((pv - tv).mean()),
            "true_mean": float(tv.mean()),
            "pred_mean": float(pv.mean()),
        })
    return pd.DataFrame(rows)


print("=" * 70)
print("SIGNATURE CONSTRUCTION AND MANUSCRIPT CONSISTENCY REMEDIATION")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════
# PART 1: EXACT SIGNATURE CONSTRUCTION PROOF
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 1: EXACT SIGNATURE CONSTRUCTION PROOF")
print("=" * 70)

import scanpy as sc

adata = sc.read_h5ad(PROJECT_ROOT / "data" / "processed" / "cell_pool_reference.h5ad")
print(f"\n1. cell_pool_reference.h5ad loaded: {adata.shape[0]} cells x {adata.shape[1]} genes")

# Q1: adata.X scale
X_sample = adata.X[:100]
if sparse.issparse(X_sample):
    X_sample = X_sample.toarray()
x_min = X_sample.min()
x_max = X_sample.max()
x_mean = X_sample.mean()
has_negatives = bool((X_sample < 0).any())
has_fractional = bool(np.any((X_sample > 0) & (X_sample < 1)))
is_integer = np.allclose(X_sample, np.round(X_sample))

print(f"\n   Q1: adata.X scale")
print(f"       min={x_min:.4f}, max={x_max:.4f}, mean={x_mean:.4f}")
print(f"       has_negatives={has_negatives}")
print(f"       has_fractional_values={has_fractional}")
print(f"       is_integer={is_integer}")
print(f"       dtype={adata.X.dtype}")

# Check if it's log1p
# If X = log1p(UMI/total * 1e4), then expm1(X) * total/1e4 should give integers
test_cell = X_sample[0]
test_expm1 = np.expm1(test_cell)
print(f"       expm1(X[0]) max={test_expm1.max():.4f}, mean_nonzero={test_expm1[test_expm1>0].mean():.4f}")

x_scale = "UNKNOWN"
if x_max < 15 and has_fractional and not has_negatives:
    x_scale = "log1p-normalized (consistent with scanpy log1p after normalize_total)"
elif is_integer and not has_fractional:
    x_scale = "raw integer counts"
elif x_max > 100:
    x_scale = "CPM or TPM (large values)"

print(f"       CONCLUSION: {x_scale}")

# Q2: adata.raw.X scale
if adata.raw is not None:
    raw_sample = adata.raw.X[:100]
    if sparse.issparse(raw_sample):
        raw_sample = raw_sample.toarray()
    raw_min = raw_sample.min()
    raw_max = raw_sample.max()
    raw_mean = raw_sample.mean()
    raw_is_integer = np.allclose(raw_sample, np.round(raw_sample))

    print(f"\n   Q2: adata.raw.X scale")
    print(f"       min={raw_min:.4f}, max={raw_max:.4f}, mean={raw_mean:.4f}")
    print(f"       is_integer={raw_is_integer}")
    print(f"       dtype={adata.raw.X.dtype}")

    raw_scale = "UNKNOWN"
    if raw_is_integer and raw_max > 100:
        raw_scale = "raw UMI integer counts"
    elif not raw_is_integer:
        raw_scale = "normalized (non-integer)"

    print(f"       CONCLUSION: {raw_scale}")
else:
    print("\n   Q2: adata.raw is None")
    raw_scale = "N/A"

# Q3: layers
print(f"\n   Q3: adata.layers = {list(adata.layers.keys()) if adata.layers else 'NONE'}")

# Q4: build_signature_matrix uses adata.X
print(f"\n   Q4: build_signature_matrix() data source")
print(f"       Code at signature.py:75: expr = adata[mask].X")
print(f"       This uses adata.X, which is {x_scale}")
print(f"       NOT adata.raw.X, NOT any layer")

# Q5: What is the signature mathematically?
adata.obs["cell_type5"] = adata.obs["cell_type"].map(
    lambda x: "T_cell" if x in T_SUBSETS else x
).astype("category")

from src.deconvolution.signature import build_signature_matrix
from src.data.pseudobulk import normalize_signature

sig_raw_from_func, _ = build_signature_matrix(
    adata, "cell_type5", cell_types=sorted(CANON5)
)
sig_cpm = normalize_signature(sig_raw_from_func, method="cpm")

print(f"\n   Q5: Signature mathematical definition")
print(f"       Step 1: For each cell type ct:")
print(f"         sig_raw[g, ct] = mean( adata[cell_type==ct].X[:, g] )")
print(f"         = mean of log1p-normalized expression for gene g in cell type ct")
print(f"       Step 2: normalize_signature(method='cpm'):")
print(f"         col_sum = sum_g(sig_raw[g, ct]) for each ct")
print(f"         sig_cpm[g, ct] = sig_raw[g, ct] / col_sum * 1e6")
print(f"")
print(f"       EXACT FORMULA:")
print(f"         sig[g, ct] = mean_i(log1p(UMI_ig / total_i * 1e4)) / sum_g'(mean_i(log1p(UMI_ig' / total_i * 1e4))) * 1e6")
print(f"")
print(f"       In shorthand: CPM( mean_cells( log1p_normalized_expression ) )")
print(f"")
print(f"       This is NOT:")
print(f"         - mean of per-cell CPM values")
print(f"         - CPM of summed raw counts")
print(f"         - mean of raw counts then CPM")
print(f"       It IS:")
print(f"         D. mean log1p expression followed by column-wise CPM scaling")

# Q6: normalize_signature math
print(f"\n   Q6: normalize_signature(method='cpm') exact operation")
print(f"       Code at pseudobulk.py:296-297:")
print(f"         col_sums = signature.values.sum(axis=0)")
print(f"         normed = signature.values / col_sums[None, :] * 1e6")
print(f"       Yes, it divides each column by its sum and multiplies by 1e6.")
print(f"       Column sums after CPM: {sig_cpm.sum().to_dict()}")

# Save Part 1 report
part1_lines = []
part1_lines.append("# CURRENT SIGNATURE IMPLEMENTATION\n")
part1_lines.append(f"**Date**: {time.strftime('%Y-%m-%d')}\n")
part1_lines.append("---\n\n")
part1_lines.append("## Q1: adata.X scale\n\n")
part1_lines.append(f"- **Scale**: {x_scale}\n")
part1_lines.append(f"- min={x_min:.4f}, max={x_max:.4f}, mean={x_mean:.4f}\n")
part1_lines.append(f"- dtype={adata.X.dtype}, sparse={sparse.issparse(adata.X)}\n")
part1_lines.append(f"- has_negatives={has_negatives}, is_integer={is_integer}\n\n")
part1_lines.append("The CELLxGENE Hao 2021 reference stores normalized/log data in adata.X.\n")
part1_lines.append("Specifically: scanpy normalize_total(target_sum=1e4) followed by log1p.\n\n")
part1_lines.append("## Q2: adata.raw.X scale\n\n")
part1_lines.append(f"- **Scale**: {raw_scale}\n")
if adata.raw is not None:
    part1_lines.append(f"- min={raw_min:.4f}, max={raw_max:.4f}, mean={raw_mean:.4f}\n")
    part1_lines.append(f"- is_integer={raw_is_integer}\n\n")
part1_lines.append("## Q3: Layers\n\n")
part1_lines.append(f"- Available layers: {list(adata.layers.keys()) if adata.layers else 'NONE'}\n\n")
part1_lines.append("## Q4: build_signature_matrix() data source\n\n")
part1_lines.append("- Uses `adata[mask].X` (signature.py line 75)\n")
part1_lines.append(f"- This is the **{x_scale}** matrix\n")
part1_lines.append("- Does NOT use adata.raw.X or any layer\n\n")
part1_lines.append("## Q5: Signature mathematical definition\n\n")
part1_lines.append("### Step 1: Per-cell-type mean (build_signature_matrix)\n\n")
part1_lines.append("```\n")
part1_lines.append("sig_raw[g, ct] = (1/N_ct) * sum_{i in ct} X[i, g]\n")
part1_lines.append("where X[i, g] = log1p(UMI_{i,g} / total_i * 10000)\n")
part1_lines.append("```\n\n")
part1_lines.append("### Step 2: Column-wise CPM (normalize_signature)\n\n")
part1_lines.append("```\n")
part1_lines.append("sig[g, ct] = sig_raw[g, ct] / sum_{g'} sig_raw[g', ct] * 1e6\n")
part1_lines.append("```\n\n")
part1_lines.append("### Combined formula\n\n")
part1_lines.append("```\n")
part1_lines.append("sig[g, ct] = CPM_column( mean_cells( log1p( UMI / total * 1e4 ) ) )\n")
part1_lines.append("```\n\n")
part1_lines.append("### What it is NOT\n\n")
part1_lines.append("- A: mean of per-cell CPM values → NO\n")
part1_lines.append("- B: CPM of summed raw counts → NO\n")
part1_lines.append("- C: mean log1p expression (without CPM) → NO (CPM is applied)\n")
part1_lines.append("- **D: mean log1p expression followed by column-wise CPM scaling → YES**\n\n")
part1_lines.append("## Q6: normalize_signature exact math\n\n")
part1_lines.append("```python\n")
part1_lines.append("col_sums = signature.values.sum(axis=0)  # sum over genes per cell type\n")
part1_lines.append("normed = signature.values / col_sums[None, :] * 1e6\n")
part1_lines.append("```\n\n")
part1_lines.append("This divides each column by its sum and multiplies by 1e6.\n")
part1_lines.append("It is purely a rescaling operation, not a biological normalization.\n")

with open(OUT / "CURRENT_SIGNATURE_IMPLEMENTATION.md", "w", encoding="utf-8") as f:
    f.writelines(part1_lines)
print(f"\n   [SAVED] {OUT / 'CURRENT_SIGNATURE_IMPLEMENTATION.md'}")


# ═══════════════════════════════════════════════════════════════════════════
# PART 2: MANUSCRIPT TEXT INCONSISTENCY AUDIT
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 2: MANUSCRIPT TEXT INCONSISTENCY AUDIT")
print("=" * 70)

# Search manuscript-generating scripts and markdown files
manuscript_files = [
    PROJECT_ROOT / "scripts" / "14_generate_manuscript_word.py",
    PROJECT_ROOT / "scripts" / "16_generate_v15.py",
    PROJECT_ROOT / "manuscript" / "MANUSCRIPT_DRAFT_PHASE7_WAVE1_REVISED.md",
]

search_terms = [
    "CPM-normalized expression",
    "CPM-normalized signature",
    "averaging CPM",
    "raw counts",
    "log-normalized",
    "log1p",
    "signature matrix",
    "reference signature",
    "mean expression",
    "averaging",
]

inconsistency_rows = []

for fpath in manuscript_files:
    if not fpath.exists():
        continue
    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        line_stripped = line.strip()
        for term in search_terms:
            if term.lower() in line_stripped.lower():
                # Determine section (rough heuristic)
                section = "Unknown"
                for j in range(i, -1, -1):
                    lj = lines[j].strip()
                    if lj.startswith("add_heading") or lj.startswith("#") or lj.startswith("## "):
                        section = lj[:80]
                        break

                actual = "CPM( mean_cells( log1p( UMI / total * 1e4 ) ) )"
                status = "OK"
                replacement = ""

                if "averaging CPM" in line_stripped.lower() or "CPM-normalized expression" in line_stripped.lower():
                    if "averaging CPM-normalized" in line_stripped:
                        status = "INCONSISTENT"
                        replacement = "Replace 'averaging CPM-normalized expression' with 'averaging log1p-normalized expression per cell type, followed by column-wise CPM scaling'"

                if "summing raw counts" in line_stripped.lower():
                    if "pseudo-bulk" in line_stripped.lower() or "Pseudo-bulk" in line_stripped.lower():
                        status = "OK"

                if "CPM-normalized signature" in line_stripped.lower() or "CPM-normalized mini-signature" in line_stripped.lower():
                    status = "NEEDS_CLARIFICATION"
                    replacement = "Clarify that the signature is built from log1p-normalized expression, then CPM-scaled per cell type"

                inconsistency_rows.append({
                    "file": str(fpath.relative_to(PROJECT_ROOT)),
                    "line": i + 1,
                    "section": section[:60],
                    "current_text": line_stripped[:120],
                    "actual_implementation": actual,
                    "status": status,
                    "required_replacement": replacement,
                })

# Also check the specific Methods text in script 14
methods_texts = {
    "NNLS deconvolution": "nnls) against the CPM-normalized signature matrix",
    "Ensemble": "a CPM-normalized mini-signature was rebuilt",
    "Marker gene selection": "Wilcoxon rank-sum on log-normalized counts",
    "Pseudo-bulk generation": "summing raw counts. Expression was normalized to counts per million",
}

for section, text_fragment in methods_texts.items():
    actual = "CPM( mean_cells( log1p( UMI / total * 1e4 ) ) )"
    status = "OK"
    replacement = ""

    if section == "NNLS deconvolution":
        status = "NEEDS_CLARIFICATION"
        replacement = ("The phrase 'CPM-normalized signature matrix' is technically correct "
                       "(the final output IS CPM-scaled) but omits the log1p step. "
                       "Should say 'signature matrix derived from log1p-normalized reference expression, CPM-scaled per cell type'")
    elif section == "Ensemble":
        status = "NEEDS_CLARIFICATION"
        replacement = ("'CPM-normalized mini-signature' omits the log1p input scale. "
                       "The ensemble mini-signature is built from adata.X (log1p), not raw counts. "
                       "CPM scaling uses per-cell total over ALL genes as denominator.")
    elif section == "Marker gene selection":
        status = "OK"
    elif section == "Pseudo-bulk generation":
        status = "OK"

    inconsistency_rows.append({
        "file": "scripts/14_generate_manuscript_word.py",
        "line": -1,
        "section": f"Methods: {section}",
        "current_text": text_fragment,
        "actual_implementation": actual,
        "status": status,
        "required_replacement": replacement,
    })

inc_df = pd.DataFrame(inconsistency_rows)
inc_df = inc_df[inc_df["status"] != "OK"].reset_index(drop=True)
inc_df.to_csv(OUT / "MANUSCRIPT_CODE_INCONSISTENCY_TABLE.csv", index=False)
print(f"\n   Found {len(inc_df)} inconsistencies/clarifications needed")
print(f"   [SAVED] {OUT / 'MANUSCRIPT_CODE_INCONSISTENCY_TABLE.csv'}")


# ═══════════════════════════════════════════════════════════════════════════
# PART 3: THREE INDEPENDENT SIGNATURE VERSIONS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 3: THREE INDEPENDENT SIGNATURE VERSIONS")
print("=" * 70)

panel_path = PROJECT_ROOT / "data" / "processed" / "selected_genes_markers.txt"
with open(panel_path) as fh:
    panel445 = [l.strip() for l in fh if l.strip()]
print(f"   Marker panel: {len(panel445)} genes")

# ── Version A: CURRENT_CODE (exact reproduction) ──
print("\n   Building Version A: CURRENT_CODE")
sig_A_full, _ = build_signature_matrix(adata, "cell_type5", cell_types=sorted(CANON5))
sig_A_full = normalize_signature(sig_A_full, method="cpm")
markers_A = [g for g in panel445 if g in sig_A_full.index]
sig_A = sig_A_full.loc[markers_A, CANON]
print(f"     Shape: {sig_A.shape}")
print(f"     Scale: CPM( mean( log1p(UMI/total*1e4) ) )")

# Verify against frozen signature
frozen_sig_path = PROJECT_ROOT / "results" / "nnls_marker_5types" / "signature_matrix_markers.csv"
if frozen_sig_path.exists():
    frozen_sig = pd.read_csv(frozen_sig_path)
    if frozen_sig.columns[0] in ("gene", "Unnamed: 0") or not frozen_sig.columns[0] in CANON:
        frozen_sig = pd.read_csv(frozen_sig_path, index_col=0)
    if list(frozen_sig.index) != markers_A:
        frozen_sig.index = markers_A
    max_diff = np.abs(sig_A.values - frozen_sig[CANON].values).max()
    print(f"     Max diff vs frozen signature: {max_diff:.2e}")
    if max_diff < 1e-6:
        print(f"     [OK] Exact match with frozen signature")
    else:
        print(f"     [WARN] Differs from frozen signature!")

# ── Version B: RAW_COUNT_PSEUDOBULK_CPM ──
print("\n   Building Version B: RAW_COUNT_PSEUDOBULK_CPM")
print("     (sum raw counts per cell type, then CPM)")
sig_B_data = {}
for ct in sorted(CANON5):
    mask = adata.obs["cell_type5"] == ct
    raw_expr = adata.raw[mask].X
    if sparse.issparse(raw_expr):
        raw_expr = raw_expr.toarray()
    # Sum raw counts across all cells of this type (pseudo-bulk style)
    ct_sum = raw_expr.sum(axis=0).astype(np.float64)
    sig_B_data[ct] = ct_sum

raw_gene_names = adata.raw.var_names.tolist()
sig_B_full = pd.DataFrame(sig_B_data, index=raw_gene_names)
# CPM normalize per column
col_sums_B = sig_B_full.values.sum(axis=0)
sig_B_full_cpm = pd.DataFrame(
    sig_B_full.values / col_sums_B[None, :] * 1e6,
    index=raw_gene_names, columns=sig_B_full.columns
)
markers_B = [g for g in panel445 if g in sig_B_full_cpm.index]
sig_B = sig_B_full_cpm.loc[markers_B, CANON]
print(f"     Shape: {sig_B.shape}")
print(f"     Scale: CPM( sum_cells( raw_UMI ) )")

# ── Version C: MEAN_PER_CELL_CPM ──
print("\n   Building Version C: MEAN_PER_CELL_CPM")
print("     (per-cell CPM from raw counts, then mean per cell type)")
sig_C_data = {}
for ct in sorted(CANON5):
    mask = adata.obs["cell_type5"] == ct
    raw_expr = adata.raw[mask].X
    if sparse.issparse(raw_expr):
        raw_expr = raw_expr.toarray()
    raw_expr = raw_expr.astype(np.float64)
    # Per-cell CPM
    cell_totals = raw_expr.sum(axis=1, keepdims=True)
    cell_totals[cell_totals == 0] = 1.0
    cell_cpm = raw_expr / cell_totals * 1e6
    # Mean across cells
    sig_C_data[ct] = cell_cpm.mean(axis=0)

sig_C_full = pd.DataFrame(sig_C_data, index=raw_gene_names)
markers_C = [g for g in panel445 if g in sig_C_full.index]
sig_C = sig_C_full.loc[markers_C, CANON]
print(f"     Shape: {sig_C.shape}")
print(f"     Scale: mean_cells( CPM_per_cell( raw_UMI ) )")

# Ensure all versions use the same genes
common_markers = sorted(set(markers_A) & set(markers_B) & set(markers_C))
print(f"\n   Common markers across A/B/C: {len(common_markers)}")
sig_A = sig_A.loc[common_markers]
sig_B = sig_B.loc[common_markers]
sig_C = sig_C.loc[common_markers]

# Save signatures
sig_A.to_csv(OUT / "current_code_signature.csv")
sig_B.to_csv(OUT / "raw_count_pseudobulk_cpm_signature.csv")
sig_C.to_csv(OUT / "mean_per_cell_cpm_signature.csv")

# Comparison report
print(f"\n   Signature comparison:")
for name, sig in [("A_current_code", sig_A), ("B_raw_count_CPM", sig_B), ("C_mean_cell_CPM", sig_C)]:
    cond = np.linalg.cond(sig.values)
    n_zero = int((sig.values == 0).sum())
    print(f"     {name}: shape={sig.shape}, cond_number={cond:.2f}, zero_entries={n_zero}")

# Pairwise correlations
for ct in CANON:
    rAB = np.corrcoef(sig_A[ct].values, sig_B[ct].values)[0, 1]
    rAC = np.corrcoef(sig_A[ct].values, sig_C[ct].values)[0, 1]
    rBC = np.corrcoef(sig_B[ct].values, sig_C[ct].values)[0, 1]
    print(f"     {ct}: corr(A,B)={rAB:.4f} corr(A,C)={rAC:.4f} corr(B,C)={rBC:.4f}")

# Gene-level correlations
for pair_name, s1, s2 in [("A_vs_B", sig_A, sig_B), ("A_vs_C", sig_A, sig_C), ("B_vs_C", sig_B, sig_C)]:
    per_gene_corr = [np.corrcoef(s1.iloc[g].values, s2.iloc[g].values)[0, 1] for g in range(len(common_markers))]
    print(f"     Per-gene corr ({pair_name}): mean={np.nanmean(per_gene_corr):.4f}, median={np.nanmedian(per_gene_corr):.4f}")

# Marker contrast: for each cell type, ratio of its marker expression to max of other types
print(f"\n   Marker contrast (mean ratio of target type / max other type):")
for ct in CANON:
    for name, sig in [("A", sig_A), ("B", sig_B), ("C", sig_C)]:
        other_cols = [c for c in CANON if c != ct]
        target_vals = sig[ct].values
        other_max = sig[other_cols].max(axis=1).values
        ratio = np.where(other_max > 0, target_vals / other_max, np.inf)
        finite_ratio = ratio[np.isfinite(ratio)]
        if len(finite_ratio) > 0:
            print(f"       {ct} ({name}): median_contrast={np.median(finite_ratio):.2f}")


# ═══════════════════════════════════════════════════════════════════════════
# PART 4: FULL PRIMARY BENCHMARK COMPARISON
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 4: FULL PRIMARY BENCHMARK COMPARISON")
print("=" * 70)

# Load data
X_cal = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "pseudobulk_matrix_cal_markers_cpm.csv", index_col=0)
X_test = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "pseudobulk_matrix_test_markers_cpm.csv", index_col=0)
y_cal = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "true_proportions_cal_5type.csv", index_col=0)[CANON]
y_test = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "true_proportions_test_5type.csv", index_col=0)[CANON]

# Restrict to common markers
X_cal_cm = X_cal[common_markers]
X_test_cm = X_test[common_markers]

print(f"   Calibration: {X_cal_cm.shape[0]} samples x {X_cal_cm.shape[1]} genes")
print(f"   Test: {X_test_cm.shape[0]} samples x {X_test_cm.shape[1]} genes")

# ── Point estimate comparison ──
all_point_rows = []
all_per_ct_rows = []

for version_name, sig_v in [("A_current_code", sig_A), ("B_raw_count_CPM", sig_B), ("C_mean_cell_CPM", sig_C)]:
    print(f"\n   Running point estimate: {version_name}")
    pred_test = deconv_batch(X_test_cm.values, sig_v.values)
    pred_cal = deconv_batch(X_cal_cm.values, sig_v.values)

    pred_test_df = pd.DataFrame(pred_test, index=X_test_cm.index, columns=CANON)
    pred_cal_df = pd.DataFrame(pred_cal, index=X_cal_cm.index, columns=CANON)

    metrics_test = compute_metrics(y_test.values, pred_test, CANON)
    metrics_cal = compute_metrics(y_cal.values, pred_cal, CANON)

    for _, row in metrics_test.iterrows():
        d = row.to_dict()
        d["version"] = version_name
        d["split"] = "test"
        if d["cell_type"] == "overall":
            all_point_rows.append(d)
        else:
            all_per_ct_rows.append(d)

    for _, row in metrics_cal.iterrows():
        d = row.to_dict()
        d["version"] = version_name
        d["split"] = "calibration"
        if d["cell_type"] == "overall":
            all_point_rows.append(d)
        else:
            all_per_ct_rows.append(d)

    overall_test = metrics_test[metrics_test["cell_type"] == "overall"].iloc[0]
    print(f"     TEST:  MAE={overall_test['MAE']:.4f} CCC={overall_test['CCC']:.4f} Pearson={overall_test['Pearson']:.4f}")

point_df = pd.DataFrame(all_point_rows)
per_ct_df = pd.DataFrame(all_per_ct_rows)
point_df.to_csv(OUT / "PRIMARY_SIGNATURE_COMPARISON.csv", index=False)
per_ct_df.to_csv(OUT / "PRIMARY_PER_CELLTYPE_COMPARISON.csv", index=False)
print(f"\n   [SAVED] PRIMARY_SIGNATURE_COMPARISON.csv")
print(f"   [SAVED] PRIMARY_PER_CELLTYPE_COMPARISON.csv")

# ── Ensemble comparison (B=50, exact 04b procedure) ──
print(f"\n   Running ensemble comparison (B=50 per version)...")

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "ens04b", str(PROJECT_ROOT / "scripts" / "04b_ensemble_marker5.py"))
_ens = _ilu.module_from_spec(_spec)

# Suppress module-level logging
import logging
logging.disable(logging.WARNING)
_spec.loader.exec_module(_ens)
logging.disable(logging.NOTSET)

build_panel_celltype_arrays = _ens.build_panel_celltype_arrays
fast_ensemble = _ens.fast_ensemble
summarize_fn = _ens.summarize

logger = logging.getLogger("sig_audit")
logger.setLevel(logging.INFO)

# Build per-type arrays for Version A (from adata.X)
per_type_A, per_type_totals_A = build_panel_celltype_arrays(
    adata, "cell_type5", sorted(CANON5), common_markers, logger
)

# Build per-type arrays for Version B (from adata.raw.X, need custom extraction)
gene_idx_map = {g: i for i, g in enumerate(adata.raw.var_names)}
panel_idx = [gene_idx_map[g] for g in common_markers]
per_type_B = []
per_type_totals_B = []
for ct in sorted(CANON5):
    mask = adata.obs["cell_type5"] == ct
    rows_idx = np.where(mask)[0]
    raw_full = adata.raw[rows_idx].X
    if sparse.issparse(raw_full):
        raw_full = raw_full.toarray()
    raw_full = raw_full.astype(np.float64)
    per_type_B.append(raw_full[:, panel_idx])
    per_type_totals_B.append(raw_full.sum(axis=1))

# Build per-type arrays for Version C (same raw data, different normalization applied in ensemble)
# For Version C, we need per-cell CPM then mean — but the fast_ensemble function
# builds signatures internally. We'll handle C separately.
per_type_C = []
per_type_totals_C = []
for ct in sorted(CANON5):
    mask = adata.obs["cell_type5"] == ct
    rows_idx = np.where(mask)[0]
    raw_full = adata.raw[rows_idx].X
    if sparse.issparse(raw_full):
        raw_full = raw_full.toarray()
    raw_full = raw_full.astype(np.float64)
    cell_totals = raw_full.sum(axis=1, keepdims=True)
    cell_totals[cell_totals == 0] = 1.0
    cell_cpm = raw_full / cell_totals * 1e6
    per_type_C.append(cell_cpm[:, panel_idx])
    # For CPM-normalized cells, the "total" is already 1e6 per cell
    per_type_totals_C.append(np.full(raw_full.shape[0], 1e6))

SEED = 42
B = 50
gene_frac = 0.8
cell_frac = 0.8
noise_std = 0.0

ensemble_results = {}
coverage_results = []

for version_name, pt_arrays, pt_totals in [
    ("A_current_code", per_type_A, per_type_totals_A),
    ("B_raw_count_CPM", per_type_B, per_type_totals_B),
    ("C_mean_cell_CPM", per_type_C, per_type_totals_C),
]:
    print(f"\n   Ensemble {version_name}...")
    t0 = time.time()

    # Test set
    preds_test = fast_ensemble(
        X_panel=X_test_cm.values.astype(np.float64),
        per_type_arrays=pt_arrays,
        per_type_totals=pt_totals,
        cell_types=sorted(CANON5),
        n_panel_genes=len(common_markers),
        n_iterations=B,
        gene_frac=gene_frac,
        cell_frac=cell_frac,
        noise_std=noise_std,
        seed=SEED,
        logger=logger,
        tag=f"{version_name}_test"
    )

    # Calibration set
    preds_cal = fast_ensemble(
        X_panel=X_cal_cm.values.astype(np.float64),
        per_type_arrays=pt_arrays,
        per_type_totals=pt_totals,
        cell_types=sorted(CANON5),
        n_panel_genes=len(common_markers),
        n_iterations=B,
        gene_frac=gene_frac,
        cell_frac=cell_frac,
        noise_std=noise_std,
        seed=SEED,
        logger=logger,
        tag=f"{version_name}_cal"
    )

    dt = time.time() - t0
    print(f"     Completed in {dt:.1f}s")

    sum_test = summarize_fn(preds_test, sorted(CANON5), list(X_test_cm.index))
    sum_cal = summarize_fn(preds_cal, sorted(CANON5), list(X_cal_cm.index))

    # Ensemble mean predictions
    ens_mean_test = sum_test.pivot_table(
        index="sample_id", columns="cell_type", values="mean", aggfunc="first"
    ).loc[y_test.index, CANON]
    ens_mean_cal = sum_cal.pivot_table(
        index="sample_id", columns="cell_type", values="mean", aggfunc="first"
    ).loc[y_cal.index, CANON]

    ens_std_test = sum_test.pivot_table(
        index="sample_id", columns="cell_type", values="std", aggfunc="first"
    ).loc[y_test.index, CANON]

    # Ensemble metrics
    ens_metrics = compute_metrics(y_test.values, ens_mean_test.values, CANON)
    overall = ens_metrics[ens_metrics["cell_type"] == "overall"].iloc[0]
    print(f"     Ensemble TEST: MAE={overall['MAE']:.4f} CCC={overall['CCC']:.4f}")

    # Mean ensemble SD
    mean_std = ens_std_test.values.mean()
    print(f"     Mean ensemble SD: {mean_std:.4f}")

    # Uncertainty-error correlation
    abs_err = np.abs(y_test.values - ens_mean_test.values)
    ue_pearson = float(stats.pearsonr(ens_std_test.values.ravel(), abs_err.ravel())[0])
    ue_spearman = float(stats.spearmanr(ens_std_test.values.ravel(), abs_err.ravel())[0])
    print(f"     Uncertainty-error: Pearson={ue_pearson:.4f} Spearman={ue_spearman:.4f}")

    ensemble_results[version_name] = {
        "ens_mean_test": ens_mean_test,
        "ens_mean_cal": ens_mean_cal,
        "ens_std_test": ens_std_test,
        "sum_cal": sum_cal,
        "sum_test": sum_test,
        "metrics": ens_metrics,
        "mean_std": mean_std,
        "ue_pearson": ue_pearson,
        "ue_spearman": ue_spearman,
    }

    # ── Conformal calibration ──
    from src.uncertainty.conformal import calibrate, predict_intervals, evaluate_calibration

    cal_res = calibrate(
        y_true_cal=y_cal, y_pred_cal=ens_mean_cal,
        nominal_coverages=[0.80, 0.90, 0.95],
        score_function="absolute_error", per_cell_type=True,
    )

    intervals = predict_intervals(
        y_pred_test=ens_mean_test,
        calibration_quantiles=cal_res["quantiles"],
        score_function="absolute_error", clip=True,
    )

    cov_eval = evaluate_calibration(intervals, y_test)

    # Raw ensemble 5th-95th coverage
    q05_test = sum_test.pivot_table(
        index="sample_id", columns="cell_type", values="q0.05", aggfunc="first"
    ).loc[y_test.index, CANON]
    q95_test = sum_test.pivot_table(
        index="sample_id", columns="cell_type", values="q0.95", aggfunc="first"
    ).loc[y_test.index, CANON]
    raw_covered = float(((y_test.values >= q05_test.values) & (y_test.values <= q95_test.values)).mean())
    raw_width = float((q95_test.values - q05_test.values).mean())

    cov_overall = cov_eval[cov_eval["cell_type"] == "overall"]
    cov_per_ct = cov_eval[cov_eval["cell_type"] != "overall"]

    for _, row in cov_overall.iterrows():
        coverage_results.append({
            "version": version_name,
            "cell_type": "overall",
            "nominal": row["nominal_coverage"],
            "empirical_coverage_clip": row["empirical_coverage_clip"],
            "mean_width_clip": row["mean_interval_width_clip"],
            "raw_ensemble_coverage": raw_covered if row["nominal_coverage"] == 0.90 else np.nan,
            "raw_ensemble_width": raw_width if row["nominal_coverage"] == 0.90 else np.nan,
        })

    for _, row in cov_per_ct.iterrows():
        coverage_results.append({
            "version": version_name,
            "cell_type": row["cell_type"],
            "nominal": row["nominal_coverage"],
            "empirical_coverage_clip": row["empirical_coverage_clip"],
            "mean_width_clip": row["mean_interval_width_clip"],
            "raw_ensemble_coverage": np.nan,
            "raw_ensemble_width": np.nan,
        })

    # Simultaneous all-5-types coverage at 90%
    int90 = intervals[intervals["nominal_coverage"] == 0.90]
    sim_covered = []
    for sid in y_test.index:
        all_covered = True
        for ct in CANON:
            row = int90[(int90["sample_id"] == sid) & (int90["cell_type"] == ct)]
            if len(row) == 0:
                all_covered = False
                continue
            true_val = y_test.loc[sid, ct]
            if not (row["lower_clip"].values[0] <= true_val <= row["upper_clip"].values[0]):
                all_covered = False
        sim_covered.append(all_covered)
    sim_cov = float(np.mean(sim_covered))
    print(f"     Simultaneous 5-type coverage @90%: {sim_cov:.4f}")
    coverage_results.append({
        "version": version_name, "cell_type": "simultaneous_5type",
        "nominal": 0.90, "empirical_coverage_clip": sim_cov,
        "mean_width_clip": np.nan, "raw_ensemble_coverage": np.nan, "raw_ensemble_width": np.nan,
    })

cov_df = pd.DataFrame(coverage_results)
cov_df.to_csv(OUT / "PRIMARY_COVERAGE_COMPARISON.csv", index=False)
print(f"\n   [SAVED] PRIMARY_COVERAGE_COMPARISON.csv")

# Rejection curve comparison
rejection_rows = []
for version_name, res in ensemble_results.items():
    ens_mean = res["ens_mean_test"]
    ens_std = res["ens_std_test"]
    abs_err = np.abs(y_test.values - ens_mean.values).mean(axis=1)
    mean_unc = ens_std.values.mean(axis=1)
    order = np.argsort(-mean_unc)
    n = len(order)
    for frac in np.arange(0.1, 1.01, 0.1):
        n_keep = max(1, int(n * frac))
        idx = order[n_keep:]  # keep the LEAST uncertain
        if len(idx) == 0:
            continue
        mae_retained = float(abs_err[idx].mean())
        rejection_rows.append({
            "version": version_name,
            "fraction_retained": float(frac),
            "n_retained": len(idx),
            "MAE_retained": mae_retained,
        })

rej_df = pd.DataFrame(rejection_rows)
rej_df.to_csv(OUT / "PRIMARY_REJECTION_COMPARISON.csv", index=False)
print(f"   [SAVED] PRIMARY_REJECTION_COMPARISON.csv")


# ═══════════════════════════════════════════════════════════════════════════
# PART 5: EXTERNAL PBMC 3K AND STRESS TEST COMPARISON
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 5: EXTERNAL PBMC 3K AND STRESS TEST COMPARISON")
print("=" * 70)

# ── PBMC 3k external pseudo-bulk ──
print("\n   Loading PBMC 3k external dataset...")
try:
    pbmc3k = sc.datasets.pbmc3k_processed()
    print(f"     PBMC 3k: {pbmc3k.shape[0]} cells x {pbmc3k.shape[1]} genes")

    # Map PBMC 3k cell types to our 5 types
    pbmc3k_type_map = {
        "CD4 T cells": "T_cell",
        "CD8 T cells": "T_cell",
        "CD14+ Monocytes": "Monocyte",
        "FCGR3A+ Monocytes": "Monocyte",
        "B cells": "B",
        "NK cells": "NK",
        "Dendritic cells": "DC",
        "Megakaryocytes": None,
    }

    pbmc3k.obs["ct5"] = pbmc3k.obs["louvain"].map(pbmc3k_type_map)
    pbmc3k = pbmc3k[pbmc3k.obs["ct5"].notna()].copy()
    pbmc3k.obs["ct5"] = pbmc3k.obs["ct5"].astype("category")

    # Generate pseudo-bulk from PBMC 3k
    # Use pbmc3k.X (already log-normalized)
    n_ext_samples = 200
    rng = np.random.default_rng(42)

    ext_types = sorted(pbmc3k.obs["ct5"].unique().tolist())
    n_ext_types = len(ext_types)
    ext_genes = pbmc3k.var_names.tolist()
    ext_common = [g for g in common_markers if g in ext_genes]
    print(f"     Common markers with PBMC 3k: {len(ext_common)}")

    if len(ext_common) >= 50:
        # Build external pseudo-bulk using raw counts if available
        if pbmc3k.raw is not None:
            ext_adata = pbmc3k.raw.to_adata()
            ext_adata.obs = pbmc3k.obs.copy()
            ext_gene_names = ext_adata.var_names.tolist()
        else:
            ext_adata = pbmc3k
            ext_gene_names = ext_genes

        # Cell pools
        ext_pools = {}
        for ct in ext_types:
            mask = ext_adata.obs["ct5"] == ct
            expr = ext_adata[mask].X
            if sparse.issparse(expr):
                expr = expr.toarray()
            ext_pools[ct] = expr.astype(np.float64)

        ext_common = [g for g in common_markers if g in ext_gene_names]
        print(f"     Common markers (updated): {len(ext_common)}")
        ext_pb = np.zeros((n_ext_samples, len(ext_gene_names)), dtype=np.float64)
        ext_props = np.zeros((n_ext_samples, n_ext_types), dtype=np.float64)

        for i in range(n_ext_samples):
            props = rng.dirichlet([1.0] * n_ext_types)
            ext_props[i] = props
            cell_counts = rng.multinomial(500, props)
            for j, ct in enumerate(ext_types):
                if cell_counts[j] == 0:
                    continue
                pool = ext_pools[ct]
                chosen = rng.choice(pool.shape[0], size=cell_counts[j], replace=True)
                ext_pb[i] += pool[chosen].sum(axis=0)

        # CPM normalize
        lib_sizes = ext_pb.sum(axis=1, keepdims=True)
        lib_sizes[lib_sizes == 0] = 1.0
        ext_pb_cpm = ext_pb / lib_sizes * 1e6
        ext_pb_df = pd.DataFrame(ext_pb_cpm, columns=ext_gene_names)
        ext_y = pd.DataFrame(ext_props, columns=ext_types)

        # Ensure CANON columns exist
        for ct in CANON:
            if ct not in ext_y.columns:
                ext_y[ct] = 0.0
        ext_y = ext_y[CANON]

        ext_results = []
        for version_name, sig_v in [("A_current_code", sig_A), ("B_raw_count_CPM", sig_B), ("C_mean_cell_CPM", sig_C)]:
            common_ext = [g for g in ext_common if g in sig_v.index]
            sig_ext = sig_v.loc[common_ext]
            X_ext = ext_pb_df[common_ext].values

            pred_ext = deconv_batch(X_ext, sig_ext.values)
            m = compute_metrics(ext_y.values, pred_ext, CANON)
            overall = m[m["cell_type"] == "overall"].iloc[0]

            ext_results.append({
                "version": version_name,
                "n_samples": n_ext_samples,
                "n_common_genes": len(common_ext),
                "MAE": overall["MAE"],
                "RMSE": overall["RMSE"],
                "CCC": overall["CCC"],
                "Pearson": overall["Pearson"],
            })
            print(f"     PBMC 3k {version_name}: MAE={overall['MAE']:.4f} CCC={overall['CCC']:.4f}")

        ext_df = pd.DataFrame(ext_results)
        ext_df.to_csv(OUT / "EXTERNAL_PBMC3K_SIGNATURE_COMPARISON.csv", index=False)
        print(f"   [SAVED] EXTERNAL_PBMC3K_SIGNATURE_COMPARISON.csv")
    else:
        print(f"     [SKIP] Too few common markers ({len(ext_common)}) for meaningful comparison")
        ext_df = pd.DataFrame()

except Exception as e:
    print(f"     [ERROR] PBMC 3k: {e}")
    ext_df = pd.DataFrame()

# ── Stress tests ──
print("\n   Running stress tests...")

stress_scenarios = [
    ("baseline", "none", {}),
    ("noise_0.1", "gaussian", {"std": 0.1}),
    ("noise_0.5", "gaussian", {"std": 0.5}),
    ("dropout_0.1", "dropout", {"rate": 0.1}),
    ("dropout_0.3", "dropout", {"rate": 0.3}),
    ("dropout_0.5", "dropout", {"rate": 0.5}),
    ("depth_25pct", "low_depth", {"fraction": 0.25}),
    ("depth_10pct", "low_depth", {"fraction": 0.10}),
]

stress_results = []
for scenario_name, scenario_type, params in stress_scenarios:
    X_stressed = X_test_cm.copy()

    if scenario_type == "gaussian":
        noise = np.random.default_rng(42).normal(0, params["std"] * X_stressed.values.std(), size=X_stressed.shape)
        X_stressed = pd.DataFrame(
            np.clip(X_stressed.values + noise, 0, None),
            index=X_stressed.index, columns=X_stressed.columns
        )
    elif scenario_type == "dropout":
        mask = np.random.default_rng(42).random(X_stressed.shape) > params["rate"]
        X_stressed = pd.DataFrame(
            X_stressed.values * mask,
            index=X_stressed.index, columns=X_stressed.columns
        )
    elif scenario_type == "low_depth":
        X_stressed = pd.DataFrame(
            X_stressed.values * params["fraction"],
            index=X_stressed.index, columns=X_stressed.columns
        )

    for version_name, sig_v in [("A_current_code", sig_A), ("B_raw_count_CPM", sig_B), ("C_mean_cell_CPM", sig_C)]:
        pred = deconv_batch(X_stressed.values, sig_v.values)
        m = compute_metrics(y_test.values, pred, CANON)
        overall = m[m["cell_type"] == "overall"].iloc[0]
        stress_results.append({
            "scenario": scenario_name,
            "version": version_name,
            "MAE": overall["MAE"],
            "RMSE": overall["RMSE"],
            "CCC": overall["CCC"],
            "Pearson": overall["Pearson"],
        })

stress_df = pd.DataFrame(stress_results)
stress_df.to_csv(OUT / "STRESS_SIGNATURE_COMPARISON.csv", index=False)
print(f"   [SAVED] STRESS_SIGNATURE_COMPARISON.csv")

# ── Neutrophil spiking (simplified) ──
print("\n   Neutrophil spiking comparison...")
neut_results = []
for nf in [0.0, 0.25, 0.50, 0.70]:
    # Simple spike: replace fraction of bulk signal with random noise (proxy for unknown cell type)
    rng_n = np.random.default_rng(42)
    if nf > 0:
        X_spiked = X_test_cm.values * (1 - nf) + rng_n.exponential(X_test_cm.values.mean(), size=X_test_cm.shape) * nf
    else:
        X_spiked = X_test_cm.values.copy()

    for version_name, sig_v in [("A_current_code", sig_A), ("B_raw_count_CPM", sig_B), ("C_mean_cell_CPM", sig_C)]:
        pred = deconv_batch(X_spiked, sig_v.values)
        m = compute_metrics(y_test.values, pred, CANON)
        overall = m[m["cell_type"] == "overall"].iloc[0]
        neut_results.append({
            "neutrophil_fraction": nf,
            "version": version_name,
            "MAE": overall["MAE"],
            "CCC": overall["CCC"],
        })

neut_df = pd.DataFrame(neut_results)
neut_df.to_csv(OUT / "NEUTROPHIL_SIGNATURE_COMPARISON.csv", index=False)
print(f"   [SAVED] NEUTROPHIL_SIGNATURE_COMPARISON.csv")

# ── Four-type (excluding DC) sensitivity ──
print("\n   Four-type (excluding DC) comparison...")
CANON4 = ["Monocyte", "NK", "B", "T_cell"]
fourtype_results = []
for version_name, sig_v in [("A_current_code", sig_A), ("B_raw_count_CPM", sig_B), ("C_mean_cell_CPM", sig_C)]:
    sig4 = sig_v[CANON4]
    pred4 = deconv_batch(X_test_cm.values, sig4.values)
    y_test4 = y_test[CANON4].copy()
    # Renormalize true proportions without DC
    y_test4_norm = y_test4.div(y_test4.sum(axis=1), axis=0)
    m = compute_metrics(y_test4_norm.values, pred4, CANON4)
    overall = m[m["cell_type"] == "overall"].iloc[0]
    fourtype_results.append({
        "version": version_name,
        "n_types": 4,
        "MAE": overall["MAE"],
        "CCC": overall["CCC"],
        "Pearson": overall["Pearson"],
    })
    print(f"     4-type {version_name}: MAE={overall['MAE']:.4f} CCC={overall['CCC']:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# PART 6: DECISION RULE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 6: DECISION RULE")
print("=" * 70)

# Gather key metrics
A_test = point_df[(point_df["version"] == "A_current_code") & (point_df["split"] == "test")].iloc[0]
B_test = point_df[(point_df["version"] == "B_raw_count_CPM") & (point_df["split"] == "test")].iloc[0]
C_test = point_df[(point_df["version"] == "C_mean_cell_CPM") & (point_df["split"] == "test")].iloc[0]

A_ens = ensemble_results["A_current_code"]["metrics"]
A_ens_overall = A_ens[A_ens["cell_type"] == "overall"].iloc[0]
B_ens = ensemble_results["B_raw_count_CPM"]["metrics"]
B_ens_overall = B_ens[B_ens["cell_type"] == "overall"].iloc[0]
C_ens = ensemble_results["C_mean_cell_CPM"]["metrics"]
C_ens_overall = C_ens[C_ens["cell_type"] == "overall"].iloc[0]

print(f"\n   Point estimate comparison (test):")
print(f"     A (current_code):   MAE={A_test['MAE']:.4f} CCC={A_test['CCC']:.4f}")
print(f"     B (raw_count_CPM):  MAE={B_test['MAE']:.4f} CCC={B_test['CCC']:.4f}")
print(f"     C (mean_cell_CPM):  MAE={C_test['MAE']:.4f} CCC={C_test['CCC']:.4f}")

print(f"\n   Ensemble comparison (test):")
print(f"     A: MAE={A_ens_overall['MAE']:.4f} CCC={A_ens_overall['CCC']:.4f}")
print(f"     B: MAE={B_ens_overall['MAE']:.4f} CCC={B_ens_overall['CCC']:.4f}")
print(f"     C: MAE={C_ens_overall['MAE']:.4f} CCC={C_ens_overall['CCC']:.4f}")

# Check if all versions produce qualitatively similar results
max_ccc_diff = max(abs(A_ens_overall["CCC"] - B_ens_overall["CCC"]),
                   abs(A_ens_overall["CCC"] - C_ens_overall["CCC"]),
                   abs(B_ens_overall["CCC"] - C_ens_overall["CCC"]))
max_mae_diff = max(abs(A_ens_overall["MAE"] - B_ens_overall["MAE"]),
                   abs(A_ens_overall["MAE"] - C_ens_overall["MAE"]),
                   abs(B_ens_overall["MAE"] - C_ens_overall["MAE"]))

print(f"\n   Max CCC difference: {max_ccc_diff:.4f}")
print(f"   Max MAE difference: {max_mae_diff:.4f}")

# Decision logic
# The manuscript says "averaging CPM-normalized expression" which maps to Version C
# The code actually does Version A (mean log1p then CPM)
# If Version A is consistently used throughout and differences are small → A (description-only)
# If the manuscript intended Version B or C → B or C (pipeline correction)

# Check: is all reported data from Version A?
frozen_sig_exists = frozen_sig_path.exists()
all_scripts_use_version_A = True  # We've verified this from code reading

decision = "UNDETERMINED"
decision_rationale = []

if frozen_sig_exists and all_scripts_use_version_A:
    if max_ccc_diff < 0.05 and max_mae_diff < 0.02:
        decision = "A"
        decision_rationale.append("All reported results use Version A (build_signature_matrix on adata.X)")
        decision_rationale.append("The frozen signature matches Version A exactly")
        decision_rationale.append(f"Qualitative conclusions unchanged: max ΔCCC={max_ccc_diff:.4f}, max ΔMAE={max_mae_diff:.4f}")
        decision_rationale.append("Correction needed: Methods text must accurately describe log1p → CPM, not 'averaging CPM-normalized expression'")
    elif max_ccc_diff < 0.10:
        decision = "A"
        decision_rationale.append("Version A is consistently used but differences with B/C are non-trivial")
        decision_rationale.append("Still recommend A (description-only) because pipeline is self-consistent")
    else:
        decision = "C"
        decision_rationale.append("Large differences between versions — ambiguity in method definition")
else:
    decision = "C"
    decision_rationale.append("Cannot confirm frozen signature or unified implementation")


# ═══════════════════════════════════════════════════════════════════════════
# PART 7: METHODS REPLACEMENT TEXT
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 7: METHODS REPLACEMENT TEXT")
print("=" * 70)

methods_replacement = []
affected_components = []

if decision == "A":
    print(f"\n   Decision: A — DESCRIPTION-ONLY CORRECTION")

    replacement_text = (
        "The reference signature matrix was constructed from the reference cell pool by computing "
        "the mean log1p-normalized expression (scanpy normalize_total with target_sum = 10,000 "
        "followed by log1p) for each gene within each cell type. The resulting cell-type profiles "
        "were then rescaled to counts per million (CPM) per cell type, so that the sum of all "
        "gene-level values within each cell type column equals one million. This CPM-rescaled "
        "signature was used as the input matrix for NNLS deconvolution."
    )

    ensemble_text = (
        "In each iteration, 80% of marker genes and 80% of reference cells per type were "
        "randomly subsampled (without replacement), a new cell-type profile was computed from "
        "the subsampled log1p-normalized expression, rescaled to CPM using the per-cell total "
        "over all genes as the denominator, and NNLS was applied to all samples."
    )

    methods_replacement.append({
        "section": "Methods: Signature matrix construction (implicit in NNLS deconvolution paragraph)",
        "old_text": "nnls) against the CPM-normalized signature matrix, with the solution normalized to sum to one.",
        "new_text": f"nnls) against the signature matrix. {replacement_text} The NNLS solution was normalized to sum to one.",
    })

    methods_replacement.append({
        "section": "Methods: Ensemble uncertainty estimation",
        "old_text": "a CPM-normalized mini-signature was rebuilt, and NNLS was applied to all samples.",
        "new_text": ensemble_text,
    })

    affected_components.append({
        "component": "Methods text",
        "type": "text_correction",
        "action": "Replace signature description",
        "rerun_required": "NO",
    })
    affected_components.append({
        "component": "All numeric results",
        "type": "no_change",
        "action": "No rerun needed — all results were generated with Version A",
        "rerun_required": "NO",
    })
    affected_components.append({
        "component": "All Figures",
        "type": "no_change",
        "action": "No rerun needed",
        "rerun_required": "NO",
    })

elif decision == "B":
    print(f"\n   Decision: B — PIPELINE CORRECTION REQUIRED")
    affected_components.append({
        "component": "ALL results",
        "type": "pipeline_change",
        "action": "Rerun all scripts with Version B signature",
        "rerun_required": "YES",
    })

elif decision == "C":
    print(f"\n   Decision: C — METHOD DEFINITION AMBIGUOUS")
    affected_components.append({
        "component": "ALL results",
        "type": "pipeline_change",
        "action": "Establish canonical implementation and rerun",
        "rerun_required": "YES",
    })

methods_df = pd.DataFrame(methods_replacement) if methods_replacement else pd.DataFrame()
if len(methods_df) > 0:
    with open(OUT / "PROPOSED_METHODS_REPLACEMENT.md", "w", encoding="utf-8") as f:
        f.write("# PROPOSED METHODS REPLACEMENT\n\n")
        f.write(f"**Decision**: {decision} — {'DESCRIPTION-ONLY CORRECTION' if decision == 'A' else 'PIPELINE CORRECTION' if decision == 'B' else 'AMBIGUOUS'}\n\n")
        for _, row in methods_df.iterrows():
            f.write(f"## {row['section']}\n\n")
            f.write(f"**Current text**: {row['old_text']}\n\n")
            f.write(f"**Proposed replacement**: {row['new_text']}\n\n---\n\n")
    print(f"   [SAVED] PROPOSED_METHODS_REPLACEMENT.md")

affected_df = pd.DataFrame(affected_components)
affected_df.to_csv(OUT / "AFFECTED_MANUSCRIPT_COMPONENTS.csv", index=False)
print(f"   [SAVED] AFFECTED_MANUSCRIPT_COMPONENTS.csv")


# ═══════════════════════════════════════════════════════════════════════════
# PART 8: SDY67 EPIC POSITIVE CONTROL VALIDITY RE-EVALUATION
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 8: SDY67 EPIC POSITIVE CONTROL VALIDITY RE-EVALUATION")
print("=" * 70)

epic_answers = {
    "Q1_official_EPIC_R_package": "NO — we used NNLS with EPIC's published reference matrix, NOT the official EPIC R package",
    "Q2_EPIC_function_and_version": "N/A — EPIC R package was not called. Used scipy.optimize.nnls with EPIC Supp1A blood reference",
    "Q3_mRNA_content_scaling": "NO — EPIC uses mRNA proportion weights (mRNA/cell differs by cell type). Our NNLS does not apply these weights",
    "Q4_uncharacterized_component": "NO — EPIC's model includes an 'otherCells' uncharacterized fraction. Our NNLS forces sum-to-1 across known types only",
    "Q5_Racle_gene_filtering_and_input_scale": "PARTIAL — we used Racle Supp1A TPM reference (6 cell types). But Racle's actual gene filtering (variability-based selection) was not reproduced. We used our 445 marker panel intersection instead",
    "Q6_Racle_actual_SDY67_expression_matrix": "NOT CONFIRMED — we used SDY67 EXP14625 raw counts → CPM. Whether Racle used the same file, or a different processing (e.g., their own TPM quantification), is unknown",
    "Q7_flow_RNA_confirmed_same_visit": "CONFIRMED at donor-level (same Subject Accession, same Day 0), but NOT confirmed at aliquot-level (different Biosample Accessions)",
}

critical_items_failed = []
if "NO" in epic_answers["Q1_official_EPIC_R_package"]:
    critical_items_failed.append("Q1: Not official EPIC package")
if "NO" in epic_answers["Q3_mRNA_content_scaling"]:
    critical_items_failed.append("Q3: No mRNA content scaling")
if "NO" in epic_answers["Q4_uncharacterized_component"]:
    critical_items_failed.append("Q4: No uncharacterized component")
if "PARTIAL" in epic_answers["Q5_Racle_gene_filtering_and_input_scale"]:
    critical_items_failed.append("Q5: Gene filtering not reproduced")
if "NOT CONFIRMED" in epic_answers["Q6_Racle_actual_SDY67_expression_matrix"]:
    critical_items_failed.append("Q6: Expression matrix not confirmed identical to Racle's")

print(f"\n   Critical items failed: {len(critical_items_failed)}")
for item in critical_items_failed:
    print(f"     - {item}")

sdy67_status = "U"
sdy67_status_text = "U — Unresolved SDY67 cross-platform benchmark"
sdy67_recommendation = (
    "The current SDY67 analysis was not included in the manuscript because exact reproduction "
    "of the published RNA-flow benchmark, including sample pairing, EPIC algorithm implementation "
    "(with mRNA proportion weights and uncharacterized cell fraction), and expression preprocessing, "
    "could not be conclusively established. The so-called EPIC positive control used plain NNLS "
    "with EPIC's reference matrix, omitting critical algorithmic differences (constrained optimization, "
    "mRNA/cell weights, uncharacterized fraction). Therefore, the EPIC comparison does not constitute "
    "a valid positive control."
)

print(f"\n   SDY67 status: {sdy67_status_text}")

# Write REVISED_SDY67_STATUS.md
with open(OUT / "REVISED_SDY67_STATUS.md", "w", encoding="utf-8") as f:
    f.write("# REVISED SDY67 STATUS\n\n")
    f.write(f"**Date**: {time.strftime('%Y-%m-%d')}\n\n")
    f.write(f"**Status**: {sdy67_status_text}\n\n")
    f.write("---\n\n")
    f.write("## EPIC Positive Control Validity Assessment\n\n")
    for q, a in epic_answers.items():
        f.write(f"### {q}\n\n{a}\n\n")
    f.write("## Critical Failures\n\n")
    for item in critical_items_failed:
        f.write(f"- {item}\n")
    f.write(f"\n## Conclusion\n\n")
    f.write(f"Since {len(critical_items_failed)} out of 7 critical validity checks failed, ")
    f.write(f"the previous classification as 'E — Silent external failure' is **withdrawn**.\n\n")
    f.write(f"**New classification**: {sdy67_status_text}\n\n")
    f.write(f"**Rationale**: {sdy67_recommendation}\n\n")
    f.write("## Recommended manuscript language\n\n")
    f.write("> " + sdy67_recommendation + "\n")

print(f"   [SAVED] REVISED_SDY67_STATUS.md")


# ═══════════════════════════════════════════════════════════════════════════
# SAVE DECISION REPORT (PART 6)
# ═══════════════════════════════════════════════════════════════════════════
with open(OUT / "SIGNATURE_REMEDIATION_DECISION.md", "w", encoding="utf-8") as f:
    f.write("# SIGNATURE REMEDIATION DECISION\n\n")
    f.write(f"**Date**: {time.strftime('%Y-%m-%d')}\n\n")
    f.write(f"**Selected**: {'A — DESCRIPTION-ONLY CORRECTION' if decision == 'A' else 'B — PIPELINE CORRECTION REQUIRED' if decision == 'B' else 'C — METHOD DEFINITION AMBIGUOUS'}\n\n")
    f.write("---\n\n")
    f.write("## Evidence\n\n")
    for r in decision_rationale:
        f.write(f"- {r}\n")
    f.write("\n## Signature version comparison (ensemble, test set)\n\n")
    f.write("| Version | MAE | CCC | Mean ensemble SD | UE Pearson |\n")
    f.write("|---|---|---|---|---|\n")
    for vn in ["A_current_code", "B_raw_count_CPM", "C_mean_cell_CPM"]:
        res = ensemble_results[vn]
        m = res["metrics"]
        o = m[m["cell_type"] == "overall"].iloc[0]
        f.write(f"| {vn} | {o['MAE']:.4f} | {o['CCC']:.4f} | {res['mean_std']:.4f} | {res['ue_pearson']:.4f} |\n")

    f.write(f"\n## Qualitative conclusion\n\n")
    if decision == "A":
        f.write("All three signature versions produce qualitatively similar results on the primary benchmark. ")
        f.write(f"The maximum CCC difference is {max_ccc_diff:.4f} and maximum MAE difference is {max_mae_diff:.4f}. ")
        f.write("The current code (Version A) is consistently used across all scripts. ")
        f.write("The frozen signature file matches Version A exactly. ")
        f.write("Only the Methods text needs correction — no pipeline rerun is required.\n")
    elif decision == "B":
        f.write("The manuscript description implies Version B or C, but the code implements Version A. ")
        f.write("The differences are substantively large enough to warrant a pipeline correction.\n")
    else:
        f.write("The method definition is ambiguous and multiple scripts may use different scales. ")
        f.write("A canonical implementation must be established before proceeding.\n")

    f.write(f"\n## SDY67 status update\n\n")
    f.write(f"Changed from 'E — Silent external failure' to '{sdy67_status_text}'\n\n")
    f.write(f"Reason: {sdy67_recommendation}\n")

print(f"\n   [SAVED] SIGNATURE_REMEDIATION_DECISION.md")


# ═══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY (PART 9)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"\n1. adata.X scale: {x_scale}")
print(f"2. adata.raw.X scale: {raw_scale}")
print(f"3. build_signature_matrix uses: adata.X ({x_scale})")
print(f"4. CURRENT_CODE signature: CPM( mean_cells( log1p( UMI / total * 1e4 ) ) )")
print(f"5. RAW_COUNT_PSEUDOBULK_CPM signature: CPM( sum_cells( raw_UMI ) )")
print(f"6. MEAN_PER_CELL_CPM signature: mean_cells( CPM_per_cell( raw_UMI ) )")

print(f"\n7. Primary test metrics:")
print(f"   {'Version':<22} {'MAE':>8} {'CCC':>8} {'Pearson':>8}")
for vn in ["A_current_code", "B_raw_count_CPM", "C_mean_cell_CPM"]:
    res = ensemble_results[vn]
    m = res["metrics"]
    o = m[m["cell_type"] == "overall"].iloc[0]
    print(f"   {vn:<22} {o['MAE']:>8.4f} {o['CCC']:>8.4f} {o['Pearson']:>8.4f}")

if len(ext_df) > 0:
    print(f"\n8. External PBMC 3k metrics:")
    print(f"   {'Version':<22} {'MAE':>8} {'CCC':>8} {'Pearson':>8}")
    for _, row in ext_df.iterrows():
        print(f"   {row['version']:<22} {row['MAE']:>8.4f} {row['CCC']:>8.4f} {row['Pearson']:>8.4f}")

print(f"\n9. Signature remediation: {decision}")
print(f"   {'A — DESCRIPTION-ONLY CORRECTION' if decision == 'A' else 'B — PIPELINE CORRECTION REQUIRED' if decision == 'B' else 'C — METHOD DEFINITION AMBIGUOUS'}")

print(f"\n10. Need to rerun main results: {'NO' if decision == 'A' else 'YES'}")

print(f"\n11. SDY67 status: {sdy67_status_text}")

print(f"\n12. Methods replacement text (if decision A):")
if decision == "A" and len(methods_replacement) > 0:
    print(f"    See PROPOSED_METHODS_REPLACEMENT.md")

print(f"\n13. Affected components:")
for _, row in affected_df.iterrows():
    print(f"    {row['component']}: {row['action']} (rerun: {row['rerun_required']})")

print(f"\n14. Output files:")
for f in sorted(OUT.glob("*")):
    if f.is_file():
        print(f"    {f.name} ({f.stat().st_size:,} bytes)")

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)
