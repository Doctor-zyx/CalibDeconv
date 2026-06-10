from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path("d:/方法学论文/calibdeconv")
NN = PROJECT / "results" / "nnls"
P = PROJECT / "data" / "processed"

# File paths
sig_path = NN / "signature_matrix_hvg3000.csv"
pred_cal_path = NN / "predicted_proportions_cal.csv"
pred_test_path = NN / "predicted_proportions_test.csv"
y_cal_path = P / "true_proportions_cal.csv"
y_test_path = P / "true_proportions_test.csv"


def ccc(y_true, y_pred):
    # Lin's concordance correlation coefficient (flattened)
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    mt, mp = yt.mean(), yp.mean()
    vt, vp = yt.var(), yp.var()
    cov = ((yt - mt) * (yp - mp)).mean()
    denom = vt + vp + (mt - mp) ** 2
    return 2 * cov / denom if denom > 0 else float("nan")


def pearson(y_true, y_pred):
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    if yt.std() == 0 or yp.std() == 0:
        return float("nan")
    return np.corrcoef(yt, yp)[0, 1]


def metrics_block(name, y_true, y_pred):
    # Predictions have no sample index; true has one. Align by ROW POSITION
    # (same generation order) and select columns by NAME (orders differ).
    cols = list(y_pred.columns)
    yt = y_true.reset_index(drop=True)[cols]
    yp = y_pred.reset_index(drop=True)[cols]
    diff = yt.values - yp.values
    mae = np.abs(diff).mean()
    rmse = np.sqrt((diff ** 2).mean())
    pr = pearson(yt.values, yp.values)
    cc = ccc(yt.values, yp.values)
    print(f"\n--- {name} overall ---")
    print(f"   MAE     = {mae:.4f}")
    print(f"   RMSE    = {rmse:.4f}")
    print(f"   Pearson = {pr:.4f}")
    print(f"   CCC     = {cc:.4f}")
    return cols


print("=" * 70)
print("PHASE 2 NNLS DIAGNOSTICS")
print("=" * 70)

# Load all
sig = pd.read_csv(sig_path, index_col=0)
# Predictions saved WITHOUT a sample-id index -> read columns directly
pred_cal = pd.read_csv(pred_cal_path)
pred_test = pd.read_csv(pred_test_path)
# True proportions saved WITH a sample-id index column
y_cal = pd.read_csv(y_cal_path, index_col=0)
y_test = pd.read_csv(y_test_path, index_col=0)

print("\n1. File paths and dimensions:")
print(f"   signature_matrix_hvg3000.csv : {sig.shape[0]} genes x {sig.shape[1]} cell types")
print(f"     -> {sig_path}")
print(f"   predicted_proportions_cal.csv : {pred_cal.shape[0]} x {pred_cal.shape[1]}")
print(f"     -> {pred_cal_path}")
print(f"   predicted_proportions_test.csv: {pred_test.shape[0]} x {pred_test.shape[1]}")
print(f"     -> {pred_test_path}")

print("\n2. Predicted proportions (test) first 5 rows:")
print(pred_test.head().round(4).to_string())

print("\n3. Row-sum statistics (each sample's predicted proportions):")
for name, df in [("cal", pred_cal), ("test", pred_test)]:
    rs = df.sum(axis=1)
    print(f"   {name}: min={rs.min():.4f} mean={rs.mean():.4f} max={rs.max():.4f}")

print("\n4. Numeric validity of predictions:")
for name, df in [("cal", pred_cal), ("test", pred_test)]:
    has_na = df.isna().any().any()
    has_inf = np.isinf(df.to_numpy()).any()
    has_neg = (df.to_numpy() < 0).any()
    print(f"   {name}: NA={has_na} inf={has_inf} negative={has_neg}")

print("\n5. Accuracy metrics:")
cols = metrics_block("CALIBRATION", y_cal, pred_cal)
metrics_block("TEST", y_test, pred_test)

print("\n6. Per-cell-type metrics (test set):")
print(f"   {'cell_type':<12} {'MAE':>8} {'CCC':>8} {'true_mean':>10} {'pred_mean':>10}")
pt = pred_test[cols].reset_index(drop=True)
yt = y_test.reset_index(drop=True)[cols]
for ct in cols:
    mae_ct = np.abs(yt[ct].values - pt[ct].values).mean()
    ccc_ct = ccc(yt[ct].values, pt[ct].values)
    print(f"   {ct:<12} {mae_ct:>8.4f} {ccc_ct:>8.4f} {yt[ct].mean():>10.4f} {pt[ct].mean():>10.4f}")

print("\n7. True vs predicted plot:")
fig = PROJECT / "results" / "figures" / "nnls_true_vs_predicted.png"
print(f"   {'[OK]' if fig.exists() else '[MISSING]'} {fig}")

print("\n" + "=" * 70)
