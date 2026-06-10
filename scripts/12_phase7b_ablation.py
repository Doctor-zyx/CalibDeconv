#!/usr/bin/env python
"""Phase 7B ablation study — 4 ablations on frozen test set."""
import numpy as np, pandas as pd, sys, os
from pathlib import Path
from scipy.optimize import nnls
from scipy import stats
import scanpy as sc
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.uncertainty.conformal import calibrate, predict_intervals, evaluate_calibration

CANON = ['Monocyte','NK','B','DC','T_cell']
os.makedirs('results/phase7_ablation', exist_ok=True)

panel445 = [l.strip() for l in open('data/processed/selected_genes_markers.txt') if l.strip()]
sig445 = pd.read_csv('results/nnls_marker_5types/signature_matrix_markers.csv')
sig445.index = panel445; sig445 = sig445[CANON]
Xt = pd.read_csv('data/processed/pseudobulk_matrix_test_markers_cpm.csv', index_col=0)[panel445]
Xc = pd.read_csv('data/processed/pseudobulk_matrix_cal_markers_cpm.csv', index_col=0)[panel445]
yt = pd.read_csv('data/processed/true_proportions_test_5type.csv', index_col=0)[CANON]
yc = pd.read_csv('data/processed/true_proportions_cal_5type.csv', index_col=0)[CANON]
pred_ens_t = pd.read_csv('results/ensemble_marker_5types/predicted_proportions_test_ensemble.csv', index_col=0)[CANON].loc[yt.index]
pred_ens_c = pd.read_csv('results/ensemble_marker_5types/predicted_proportions_cal_ensemble.csv', index_col=0)[CANON].loc[yc.index]
sum_test = pd.read_csv('results/ensemble_marker_5types/ensemble_summary_test.csv')

def deconv_nnls_batch(X, S):
    n, k = X.shape[0], S.shape[1]; P = np.zeros((n, k))
    for i in range(n):
        x, _ = nnls(S.values, X.values[i]); s = x.sum()
        P[i] = x / s if s > 0 else x
    return pd.DataFrame(P, index=X.index, columns=S.columns)

def ccc(t, p):
    t = t.ravel(); p = p.ravel(); mt, mp = t.mean(), p.mean()
    d = t.var() + p.var() + (mt - mp)**2
    return float(2 * ((t - mt) * (p - mp)).mean() / d) if d > 0 else np.nan

pred_nnls_t = deconv_nnls_batch(Xt, sig445)
pred_nnls_c = deconv_nnls_batch(Xc, sig445)
ablation_rows = []

# Ablation 1: NNLS vs ensemble
A1_nnls_mae = float(np.abs(yt.values - pred_nnls_t.values).mean())
A1_nnls_ccc = ccc(yt.values, pred_nnls_t.values)
A1_ens_mae = float(np.abs(yt.values - pred_ens_t.values).mean())
A1_ens_ccc = ccc(yt.values, pred_ens_t.values)
std_t = sum_test.pivot_table(index='sample_id', columns='cell_type', values='std', aggfunc='first').loc[yt.index, CANON]
ue_corr = float(stats.pearsonr(std_t.mean(axis=1).values, np.abs(yt.values - pred_ens_t.values).mean(axis=1))[0])
ablation_rows.append({'ablation':'1_point_estimate','setting':'NNLS_only','MAE':A1_nnls_mae,'CCC':A1_nnls_ccc,'ue_corr':np.nan,'coverage90':np.nan,'width90':np.nan})
ablation_rows.append({'ablation':'1_point_estimate','setting':'ensemble_mean','MAE':A1_ens_mae,'CCC':A1_ens_ccc,'ue_corr':ue_corr,'coverage90':np.nan,'width90':np.nan})
print(f'Abl1: NNLS MAE={A1_nnls_mae:.4f} CCC={A1_nnls_ccc:.4f} | Ens MAE={A1_ens_mae:.4f} CCC={A1_ens_ccc:.4f} ue_corr={ue_corr:.4f}')

# Ablation 2: raw interval vs conformal
q05_t = sum_test.pivot_table(index='sample_id', columns='cell_type', values='q0.05', aggfunc='first').loc[yt.index, CANON]
q95_t = sum_test.pivot_table(index='sample_id', columns='cell_type', values='q0.95', aggfunc='first').loc[yt.index, CANON]
raw_covered = float(((yt.values >= q05_t.values) & (yt.values <= q95_t.values)).mean())
raw_width = float((q95_t.values - q05_t.values).mean())
conf_cov = pd.read_csv('results/conformal_marker_5types/coverage_by_nominal.csv')
c90 = conf_cov[conf_cov.nominal_coverage == 0.90]
conf_coverage = float(c90['empirical_coverage_clip'].values[0])
conf_width = float(c90['mean_interval_width_clip'].values[0])
ablation_rows.append({'ablation':'2_interval','setting':'raw_ensemble_q05_q95','MAE':np.nan,'CCC':np.nan,'ue_corr':np.nan,'coverage90':raw_covered,'width90':raw_width})
ablation_rows.append({'ablation':'2_interval','setting':'conformal_calibrated','MAE':np.nan,'CCC':np.nan,'ue_corr':np.nan,'coverage90':conf_coverage,'width90':conf_width})
print(f'Abl2: raw cov={raw_covered:.4f} width={raw_width:.4f} | conformal cov={conf_coverage:.4f} width={conf_width:.4f}')

# Ablation 3: donor-aware (cal vs test gap as proxy)
cal_mae = float(np.abs(yc.values - pred_ens_c.values).mean())
test_mae = float(np.abs(yt.values - pred_ens_t.values).mean())
cal_ccc = ccc(yc.values, pred_ens_c.values)
test_ccc = ccc(yt.values, pred_ens_t.values)
ablation_rows.append({'ablation':'3_split','setting':'donor_aware_cal','MAE':cal_mae,'CCC':cal_ccc,'ue_corr':np.nan,'coverage90':np.nan,'width90':np.nan})
ablation_rows.append({'ablation':'3_split','setting':'donor_aware_test','MAE':test_mae,'CCC':test_ccc,'ue_corr':np.nan,'coverage90':np.nan,'width90':np.nan})
print(f'Abl3: cal MAE={cal_mae:.4f} CCC={cal_ccc:.4f} | test MAE={test_mae:.4f} CCC={test_ccc:.4f} (gap ΔMAE={cal_mae-test_mae:+.4f})')

# Ablation 4: marker number sensitivity
Xt_full = pd.read_csv('data/processed/pseudobulk_matrix_test_cpm.csv', index_col=0)
Xc_full = pd.read_csv('data/processed/pseudobulk_matrix_cal_cpm.csv', index_col=0)
for n_markers in [100, 200, 445]:
    sub_panel = panel445[:n_markers]
    common = [g for g in sub_panel if g in sig445.index and g in Xt_full.columns and g in Xc_full.columns]
    sig_sub = sig445.loc[common]; Xt_sub = Xt_full[common]; Xc_sub = Xc_full[common]
    pt = deconv_nnls_batch(Xt_sub, sig_sub)
    pc = deconv_nnls_batch(Xc_sub, sig_sub)
    mae_t = float(np.abs(yt.values - pt.values).mean())
    ccc_t = ccc(yt.values, pt.values)
    cal_res = calibrate(yc, pc, nominal_coverages=[0.90], score_function='absolute_error', per_cell_type=True)
    intervals = predict_intervals(pt, cal_res['quantiles'], score_function='absolute_error', clip=True)
    cov_df = evaluate_calibration(intervals, yt)
    cov90 = float(cov_df[cov_df.cell_type == 'overall']['empirical_coverage_clip'].values[0])
    w90 = float(intervals[intervals.nominal_coverage == 0.90]['interval_width_clip'].mean())
    ablation_rows.append({'ablation':'4_markers','setting':f'{n_markers}_markers','MAE':mae_t,'CCC':ccc_t,'ue_corr':np.nan,'coverage90':cov90,'width90':w90})
    print(f'Abl4 {n_markers}: MAE={mae_t:.4f} CCC={ccc_t:.4f} cov90={cov90:.4f} width={w90:.4f}')

abl_df = pd.DataFrame(ablation_rows)
abl_df.to_csv('results/phase7_ablation/ablation_metrics.csv', index=False)
cov_rows = abl_df[abl_df['coverage90'].notna()]
cov_rows.to_csv('results/phase7_ablation/coverage_ablation.csv', index=False)
ue_rows = abl_df[abl_df['ue_corr'].notna()]
ue_rows.to_csv('results/phase7_ablation/uncertainty_error_ablation.csv', index=False)
print('\nAll ablation CSVs saved.')
print(abl_df.round(4).to_string(index=False))
