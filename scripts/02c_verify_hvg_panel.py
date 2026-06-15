from pathlib import Path
import json
import pandas as pd
import scanpy as sc
import numpy as np

PROJECT = Path("d:/方法学论文/calibdeconv")
P = PROJECT / "data" / "processed"

panel_file = P / "selected_genes_hvg3000.txt"
cal_file = P / "pseudobulk_matrix_cal_hvg3000_cpm.csv"
test_file = P / "pseudobulk_matrix_test_hvg3000_cpm.csv"
meta_file = P / "run_metadata.json"
ref_file = P / "cell_pool_reference.h5ad"

print("=" * 70)
print("HVG PANEL FINAL VERIFICATION")
print("=" * 70)

# Load selected gene panel
with open(panel_file, "r", encoding="utf-8") as f:
    panel = [line.strip() for line in f if line.strip()]

print("\n1. First 10 genes in selected_genes_hvg3000.txt:")
for i, gene in enumerate(panel[:10], start=1):
    print(f"   {i:02d}. {gene}")

print(f"\n2. Gene panel count: {len(panel)}")
assert len(panel) == 3000, f"Expected 3000 genes, got {len(panel)}"

# Load pseudo-bulk matrices
cal = pd.read_csv(cal_file, index_col=0)
test = pd.read_csv(test_file, index_col=0)

print("\n3. Matrix dimensions:")
print(f"   Calibration matrix: {cal.shape[0]} samples x {cal.shape[1]} genes")
print(f"   Test matrix:        {test.shape[0]} samples x {test.shape[1]} genes")

assert cal.shape == (500, 3000), f"Unexpected cal shape: {cal.shape}"
assert test.shape == (500, 3000), f"Unexpected test shape: {test.shape}"

# Check gene order
print("\n4. Gene-set and gene-order consistency:")
print(f"   cal columns == panel:  {list(cal.columns) == panel}")
print(f"   test columns == panel: {list(test.columns) == panel}")
print(f"   cal columns == test columns: {list(cal.columns) == list(test.columns)}")

assert list(cal.columns) == panel, "Calibration matrix columns do not match selected gene panel order."
assert list(test.columns) == panel, "Test matrix columns do not match selected gene panel order."
assert list(cal.columns) == list(test.columns), "Calibration/test gene order mismatch."

# Check numeric validity
print("\n5. Numeric QC:")
for name, df in [("cal", cal), ("test", test)]:
    has_na = df.isna().any().any()
    has_inf = np.isinf(df.to_numpy()).any()
    has_neg = (df.to_numpy() < 0).any()
    print(f"   {name}: NA={has_na}, inf={has_inf}, negative={has_neg}")
    assert not has_na, f"{name} matrix contains NA."
    assert not has_inf, f"{name} matrix contains inf."
    assert not has_neg, f"{name} matrix contains negative values."

# Check reference gene availability
adata_ref = sc.read_h5ad(ref_file)
ref_genes = set(adata_ref.var_names)
panel_in_ref = [g for g in panel if g in ref_genes]
missing = [g for g in panel if g not in ref_genes]

print("\n6. Reference gene availability:")
print(f"   Panel genes in reference var_names: {len(panel_in_ref)}/{len(panel)}")
assert len(panel_in_ref) == len(panel), f"Missing genes in reference: {missing[:20]}"

# Check metadata
print("\n7. Metadata check:")
assert meta_file.exists(), "run_metadata.json missing."
with open(meta_file, "r", encoding="utf-8") as f:
    meta = json.load(f)

required_fields = [
    "n_hvg",
    "gene_selection_source",
    "gene_panel_file",
    "analysis_gene_set",
]
for field in required_fields:
    print(f"   {field}: {meta.get(field)}")

assert meta.get("n_hvg") == 3000, "metadata n_hvg is not 3000."
assert meta.get("gene_selection_source") == "reference_pool_only", "HVG source must be reference_pool_only."
assert meta.get("analysis_gene_set") == "hvg3000", "analysis_gene_set must be hvg3000."

print("\n8. File existence:")
for fp in [panel_file, cal_file, test_file, meta_file, ref_file]:
    print(f"   [OK] {fp}")
    assert fp.exists(), f"Missing file: {fp}"

print("\n[PASS] HVG PANEL FINAL VERIFICATION PASSED")
print("=" * 70)
