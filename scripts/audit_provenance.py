#!/usr/bin/env python
"""READ-ONLY provenance audit. Writes NOTHING. Copies/deletes NOTHING."""
import glob, os, hashlib, json, datetime
import numpy as np
import pandas as pd

def mtime(p):
    return datetime.datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d %H:%M:%S")

def fsize(p):
    return os.path.getsize(p)

def fhash(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()[:10]

CTS5 = ["T_cell", "B", "NK", "Monocyte", "DC"]

def ccc(yt, yp):
    yt = np.asarray(yt).ravel(); yp = np.asarray(yp).ravel()
    mt, mp = yt.mean(), yp.mean(); vt, vp = yt.var(), yp.var()
    cov = ((yt-mt)*(yp-mp)).mean(); d = vt+vp+(mt-mp)**2
    return 2*cov/d if d>0 else float("nan")

def pear(yt, yp):
    yt=np.asarray(yt).ravel(); yp=np.asarray(yp).ravel()
    if yt.std()==0 or yp.std()==0: return float("nan")
    return np.corrcoef(yt,yp)[0,1]

print("="*78)
print("SECTION 1: ALL signature_matrix*.csv")
print("="*78)
for p in sorted(glob.glob("**/signature_matrix*.csv", recursive=True)):
    df = pd.read_csv(p, index_col=0)
    cols = list(df.columns)
    flag = ""
    if df.shape == (445,5): flag="[445x5 OK]"
    elif df.shape==(445,4): flag="[445x4 *** 4-COL ***]"
    elif "marker" in p: flag=f"[{df.shape}]"
    print(f"\n{p}  {flag}")
    print(f"   mtime={mtime(p)} size={fsize(p)} hash={fhash(p)} shape={df.shape}")
    print(f"   columns={cols}")
    print(f"   T_cell={'T_cell' in cols} B={'B' in cols} NK={'NK' in cols} "
          f"Monocyte={'Monocyte' in cols} DC={'DC' in cols}")
    print(f"   first_3_genes={list(df.index[:3])}")

print("\n" + "="*78)
print("SECTION 2: ALL predicted_proportions_test.csv (+ cal where relevant)")
print("="*78)
for p in sorted(glob.glob("**/predicted_proportions*test*.csv", recursive=True)):
    df = pd.read_csv(p)
    if df.columns[0].startswith("Unnamed") or df.iloc[:,0].dtype == object:
        df = df.drop(columns=df.columns[0])
    df = df.select_dtypes(include=[np.number])
    cols = list(df.columns)
    zerocol = [c for c in cols if df[c].abs().max() < 1e-9]
    print(f"\n{p}")
    print(f"   mtime={mtime(p)} size={fsize(p)} hash={fhash(p)} shape={df.shape}")
    print(f"   columns={cols}")
    print(f"   has_Monocyte={'Monocyte' in cols} n_cols={len(cols)}")
    means = {c: round(float(df[c].mean()),4) for c in cols}
    print(f"   pred_mean={means}")
    print(f"   constant_zero_cols={zerocol if zerocol else 'none'}")

print("\n" + "="*78)
print("SECTION 3: metrics / summary / comparison files")
print("="*78)
for p in sorted(set(glob.glob("**/metrics_test.csv", recursive=True)
                    + glob.glob("**/marker_5type_summary.csv", recursive=True)
                    + glob.glob("**/ensemble_accuracy.csv", recursive=True)
                    + glob.glob("**/nnls_baseline_comparison.csv", recursive=True)
                    + glob.glob("**/comparison_summary.csv", recursive=True))):
    df = pd.read_csv(p)
    print(f"\n{p}")
    print(f"   mtime={mtime(p)} shape={df.shape} cols={list(df.columns)}")
    print(df.round(4).to_string(index=False))

print("\n" + "="*78)
print("SECTION 4: Recompute metrics for current nnls_marker_5types prediction")
print("="*78)
yt = pd.read_csv("data/processed/true_proportions_test_5type.csv", index_col=0)
pp = pd.read_csv("results/nnls_marker_5types/predicted_proportions_test.csv")
if pp.columns[0].startswith("Unnamed") or pp.iloc[:,0].dtype == object:
    pp = pp.drop(columns=pp.columns[0])
print(f"true cols : {list(yt.columns)}")
print(f"pred cols : {list(pp.columns)}")
common = [c for c in yt.columns if c in pp.columns]
missing_true = set(yt.columns) - set(pp.columns)
missing_pred = set(pp.columns) - set(yt.columns)
print(f"common={common}  only_in_true={missing_true}  only_in_pred={missing_pred}")
if set(yt.columns) == set(pp.columns):
    cols = list(yt.columns)
    A = yt[cols].reset_index(drop=True).values
    B = pp[cols].reset_index(drop=True).values
    print(f"\nOVERALL: MAE={np.abs(A-B).mean():.4f} RMSE={np.sqrt(((A-B)**2).mean()):.4f} "
          f"Pearson={pear(A,B):.4f} CCC={ccc(A,B):.4f}")
    print("Per-cell-type:")
    for j,c in enumerate(cols):
        print(f"   {c:<10} MAE={np.abs(A[:,j]-B[:,j]).mean():.4f} CCC={ccc(A[:,j],B[:,j]):.4f}")

print("\n" + "="*78)
print("SECTION 6: Root-cause check — re-read the 4-col signature with index_col=0 vs not")
print("="*78)
sp = "results/nnls_marker_5types/signature_matrix_markers.csv"
raw = pd.read_csv(sp)              # no index_col
print(f"raw read (no index_col): shape={raw.shape} cols={list(raw.columns)}")
print(f"   first row: {raw.iloc[0].to_dict()}")
with open(sp) as fh:
    head = [next(fh) for _ in range(2)]
print(f"   file header line: {head[0].strip()}")
print(f"   file row 1      : {head[1].strip()}")


