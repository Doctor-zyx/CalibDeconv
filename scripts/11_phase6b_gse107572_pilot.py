#!/usr/bin/env python
"""
Phase 6B Stage 1 — GSE107572 technical pilot.

Runs the FROZEN CalibDeconv pipeline (marker-5type signature + ensemble) on the
public real-bulk RNA-seq + flow-cytometry dataset GSE107572 (Finotello 2019).
No retraining, no signature change. n=9 → conformal treated as exploratory only.

GT mapping to 5 classes (neutrophil/PMN spike excluded, then renormalized):
  T cell  = cd4+ t + cd8+ t + tregs
  B       = b cells
  NK      = natural killer cells
  Monocyte= monocytes
  DC      = myeloid dendritic cells
"""
import gzip, re, sys
from pathlib import Path
import numpy as np, pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.deconvolution.signature import build_signature_matrix
from src.data.pseudobulk import normalize_signature
from src.deconvolution.nnls import deconvolve_batch
from scipy import stats

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("ens04b", str(PROJECT_ROOT / "scripts" / "04b_ensemble_marker5.py"))
_ens = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_ens)

CANON = ["Monocyte", "NK", "B", "DC", "T_cell"]
DD = PROJECT_ROOT / "data" / "real_bulk" / "gse107572"
OUT = PROJECT_ROOT / "results" / "phase6b_gse107572"; OUT.mkdir(parents=True, exist_ok=True)


def ccc(yt, yp):
    yt = np.asarray(yt).ravel(); yp = np.asarray(yp).ravel()
    mt, mp = yt.mean(), yp.mean(); vt, vp = yt.var(), yp.var()
    cov = ((yt-mt)*(yp-mp)).mean(); d = vt+vp+(mt-mp)**2
    return float(2*cov/d) if d > 0 else float("nan")


# ── 1. parse flow GT, map to 5 classes, renormalize (exclude neutrophils) ──
rows, title = {}, None
with gzip.open(DD/"series_matrix.txt.gz", "rt", encoding="utf-8", errors="ignore") as f:
    for line in f:
        if line.startswith("!Sample_title"):
            title = [x.strip().strip('"') for x in line.rstrip().split("\t")[1:]]
        if line.startswith("!Sample_characteristics_ch1") and ":" in line:
            cells = [x.strip().strip('"') for x in line.rstrip().split("\t")[1:]]
            key = cells[0].split(":")[0].strip()
            if key in ["natural killer cells","b cells","myeloid dendritic cells","tregs",
                       "monocytes","neutrophils","cd8+ t cells","cd4+ t cells"]:
                rows[key] = [float(c.split(":")[1]) for c in cells]
donors = [re.search(r"donor (\d+)", t).group(1) for t in title]
g = pd.DataFrame(rows, index=donors)
gt = pd.DataFrame({
    "Monocyte": g["monocytes"], "NK": g["natural killer cells"], "B": g["b cells"],
    "DC": g["myeloid dendritic cells"],
    "T_cell": g["cd4+ t cells"] + g["cd8+ t cells"] + g["tregs"],
}, index=donors)[CANON]
gt = gt.div(gt.sum(axis=1), axis=0)   # renormalize over 5 PBMC classes
gt.index.name = "donor"

# ── 2. load TPM, map columns -> donor ──
tpm = pd.read_csv(DD/"tpm.txt.gz", sep="\t", index_col=0)
col2donor = {c: re.search(r"pbmc_(\d+)", c).group(1) for c in tpm.columns}
tpm = tpm.rename(columns=col2donor)
tpm = tpm.loc[:, ~tpm.columns.duplicated()]
bulk = tpm[gt.index.tolist()].T          # donors × genes
bulk.index.name = "donor"

# ── 3. frozen signature: build from reference pool, CPM, subset to common markers ──
import scanpy as sc
panel = [l.strip() for l in open(PROJECT_ROOT/"data/processed/selected_genes_markers.txt") if l.strip()]
adata = sc.read_h5ad(PROJECT_ROOT/"data/processed/cell_pool_reference.h5ad")
adata.obs["cell_type5"] = adata.obs["cell_type"].map(
    lambda x: "T_cell" if x in ["CD4_T","CD8_T","other_T"] else x).astype("category")
cts = sorted(adata.obs["cell_type5"].unique())
sig_full, _ = build_signature_matrix(adata, "cell_type5", cell_types=cts)
sig_full = normalize_signature(sig_full, method="cpm")
common = [g for g in panel if g in sig_full.index and g in bulk.columns]
sig = sig_full.loc[common, CANON]
Xb = bulk[common]
print(f"[overlap] panel={len(panel)}  usable markers (sig∩bulk)={len(common)} ({100*len(common)/len(panel):.1f}%)")

# ── 4. point estimate (NNLS) ──
pred = deconvolve_batch(Xb, sig, verbose=False)[CANON]
pred.index = Xb.index

# ── 5. ensemble uncertainty (frozen B=50 core) on the same common-gene panel ──
arrs, tots = _ens.build_panel_celltype_arrays(adata, "cell_type5", cts, common, type("L",(),{"info":lambda *a:None})())
preds3 = _ens.fast_ensemble(Xb.values.astype(float), arrs, tots, cts, len(common),
                            n_iterations=50, gene_frac=0.8, cell_frac=0.8, noise_std=0.0,
                            seed=42, logger=type("L",(),{"info":lambda *a:None})(), tag="gse107572")
summ = _ens.summarize(preds3, cts, list(Xb.index))
std = summ.pivot_table(index="sample_id", columns="cell_type", values="std", aggfunc="first").loc[Xb.index, CANON]

# ── 6. metrics ──
yt = gt.loc[pred.index, CANON]
assert list(yt.index) == list(pred.index) and list(yt.columns) == CANON
A, B = yt.values, pred.values
overall = {
    "MAE": float(np.abs(A-B).mean()), "RMSE": float(np.sqrt(((A-B)**2).mean())),
    "Pearson": float(stats.pearsonr(A.ravel(), B.ravel())[0]),
    "Spearman": float(stats.spearmanr(A.ravel(), B.ravel())[0]),
    "CCC": ccc(A, B), "n": len(yt),
}
per = []
for ct in CANON:
    t, p = yt[ct].values, pred[ct].values
    per.append({"cell_type": ct, "MAE": float(np.abs(t-p).mean()),
                "Pearson": float(stats.pearsonr(t,p)[0]) if t.std()>0 and p.std()>0 else np.nan,
                "Spearman": float(stats.spearmanr(t,p)[0]) if t.std()>0 and p.std()>0 else np.nan,
                "true_mean": float(t.mean()), "pred_mean": float(p.mean())})
# uncertainty-error correlation (per-sample mean std vs mean abs error)
abs_err = np.abs(A-B).mean(axis=1)
rel = std.mean(axis=1).values
ue = float(stats.pearsonr(rel, abs_err)[0]) if np.std(rel)>0 else np.nan
ue_s = float(stats.spearmanr(rel, abs_err)[0]) if np.std(rel)>0 else np.nan

# ── 7. save ──
gt.to_csv(OUT/"ground_truth_5type.csv")
pred.to_csv(OUT/"predicted_proportions.csv")
std.to_csv(OUT/"ensemble_std.csv")
pd.DataFrame([overall]).to_csv(OUT/"metrics_overall.csv", index=False)
pd.DataFrame(per).to_csv(OUT/"metrics_per_celltype.csv", index=False)
pd.DataFrame([{"ue_pearson": ue, "ue_spearman": ue_s, "n": len(yt)}]).to_csv(OUT/"uncertainty_error.csv", index=False)

print("\n=== OVERALL ===")
for k,v in overall.items(): print(f"  {k}: {v:.4f}" if isinstance(v,float) else f"  {k}: {v}")
print("\n=== PER CELL TYPE ===")
print(pd.DataFrame(per).round(4).to_string(index=False))
print(f"\nuncertainty-error corr: Pearson={ue:.4f} Spearman={ue_s:.4f} (n={len(yt)})")
print(f"\nGT row-sum (renorm) min/mean/max: {gt.sum(1).min():.3f}/{gt.sum(1).mean():.3f}/{gt.sum(1).max():.3f}")
print(f"pred row-sum min/mean/max: {pred.sum(1).min():.3f}/{pred.sum(1).mean():.3f}/{pred.sum(1).max():.3f}")
print(f"saved to {OUT}")
