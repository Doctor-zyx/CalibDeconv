#!/usr/bin/env python
"""Redraw all supplementary figures S2-S7 with unified style fixes."""
import pandas as pd, numpy as np, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.linewidth': 1.0,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'savefig.dpi': 300,
    'axes.grid': False
})

OUT = 'd:/方法学论文/calibdeconv/submission_package/Supplementary_Figures'
INK = '#333333'

def panel_label(ax, label):
    ax.text(-0.12, 1.06, label, transform=ax.transAxes, fontsize=12, fontweight='bold', color=INK)

def save(fig, name):
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(f'{OUT}/{name}.{ext}', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f'  {name} saved')

# ═══════════════════════════════════════════════════
# S2: Baseline comparison
# ═══════════════════════════════════════════════════
bl = pd.read_csv('d:/方法学论文/calibdeconv/results/phase7_baseline_comparison/baseline_metrics.csv')
fig, (axA, axB) = plt.subplots(1, 2, figsize=(9, 3.8))

methods_short = ['NNLS', 'CalibDeconv', 'OLS', 'local nu-SVR']
colors_bl = ['#7A9DB5', '#4C72B0', '#95B07A', '#C06060']

axA.bar(range(4), bl['MAE'], color=colors_bl, width=0.6, edgecolor='white', linewidth=0.8)
for i, v in enumerate(bl['MAE']): axA.text(i, v+0.004, f'{v:.3f}', ha='center', fontsize=7)
axA.set_xticks(range(4)); axA.set_xticklabels(methods_short, fontsize=7.5)
axA.set_ylabel('Test MAE (lower is better)', fontsize=8.5)
axA.set_title('Point-estimate MAE', fontsize=9.5, pad=8)
axA.yaxis.grid(True, alpha=0.2, color='#AAAAAA'); axA.set_axisbelow(True)
panel_label(axA, 'a')

axB.bar(range(4), bl['CCC'], color=colors_bl, width=0.6, edgecolor='white', linewidth=0.8)
for i, v in enumerate(bl['CCC']): axB.text(i, max(v,0)+0.02, f'{v:.3f}', ha='center', fontsize=7)
axB.set_xticks(range(4)); axB.set_xticklabels(methods_short, fontsize=7.5)
axB.set_ylabel('Test CCC (higher is better)', fontsize=8.5)
axB.set_title('Point-estimate CCC', fontsize=9.5, pad=8)
axB.set_ylim(-0.15, 1.0)
axB.yaxis.grid(True, alpha=0.2, color='#AAAAAA'); axB.set_axisbelow(True)
panel_label(axB, 'b')

fig.suptitle('Baseline method comparison on held-out pseudo-bulks (n = 500)', fontsize=10, y=0.98)
fig.tight_layout(rect=[0,0,1,0.94])
save(fig, 'FigS2_baseline_comparison')

# ═══════════════════════════════════════════════════
# S3: Ablation
# ═══════════════════════════════════════════════════
abl = pd.read_csv('d:/方法学论文/calibdeconv/results/phase7_ablation/ablation_metrics.csv')
fig, axes = plt.subplots(2, 2, figsize=(8.5, 7))
(axA, axB), (axC, axD) = axes

# a: NNLS vs ensemble CCC
a1 = abl[abl.ablation == '1_point_estimate']
axA.bar([0,1], a1['CCC'].values, color=['#7A9DB5','#4C72B0'], width=0.55, edgecolor='white')
for i, v in enumerate(a1['CCC'].values): axA.text(i, v+0.003, f'{v:.4f}', ha='center', fontsize=7)
axA.set_xticks([0,1]); axA.set_xticklabels(['NNLS only','Ensemble mean'], fontsize=8)
axA.set_ylabel('CCC', fontsize=9); axA.set_ylim(0, 1.0)
axA.set_title('NNLS vs ensemble (ΔCCC = 0.004)', fontsize=9, pad=8)
axA.yaxis.grid(True, alpha=0.2, color='#AAAAAA'); axA.set_axisbelow(True)
panel_label(axA, 'a')

# b: raw vs conformal coverage
a2 = abl[abl.ablation == '2_interval']
axB.bar([0,1], a2['coverage90'].values, color=['#C06060','#4C72B0'], width=0.55, edgecolor='white')
axB.axhline(0.90, ls='--', lw=1, color='#555555')
axB.text(1.6, 0.91, 'Nominal 90%', fontsize=6.5, color='#555555')
for i, v in enumerate(a2['coverage90'].values): axB.text(i, v+0.015, f'{v:.3f}', ha='center', fontsize=7)
axB.set_xticks([0,1]); axB.set_xticklabels(['Raw ensemble\n(q05–q95)','Conformal\ncalibrated'], fontsize=7.5)
axB.set_ylabel('Empirical coverage', fontsize=9); axB.set_ylim(0, 1.08)
axB.set_title('Raw vs conformal intervals at 90% nominal', fontsize=9, pad=8)
axB.yaxis.grid(True, alpha=0.2, color='#AAAAAA'); axB.set_axisbelow(True)
panel_label(axB, 'b')

# c: donor-aware cal vs test
a3 = abl[abl.ablation == '3_split']
axC.bar([0,1], a3['MAE'].values, color=['#9B7DB8','#4C72B0'], width=0.55, edgecolor='white')
for i, v in enumerate(a3['MAE'].values): axC.text(i, v+0.002, f'{v:.4f}', ha='center', fontsize=7)
axC.set_xticks([0,1]); axC.set_xticklabels(['Calibration donors\n(P2, P7)','Test donors\n(P1, P6)'], fontsize=7.5)
axC.set_ylabel('MAE', fontsize=9)
axC.set_title('Calibration–test donor gap', fontsize=9, pad=8)
axC.yaxis.grid(True, alpha=0.2, color='#AAAAAA'); axC.set_axisbelow(True)
panel_label(axC, 'c')

# d: marker number sensitivity
a4 = abl[abl.ablation == '4_markers']
x4 = [100, 200, 445]
axD.plot(x4, a4['CCC'].values, 'o-', color='#4C72B0', lw=1.3, ms=6, label='CCC')
axD2 = axD.twinx()
axD2.plot(x4, a4['coverage90'].values, 's--', color='#6AAF6A', lw=1.3, ms=6, label='Coverage')
axD2.axhline(0.90, ls=':', lw=0.8, color='#6AAF6A', alpha=0.5)
axD.set_xlabel('Number of markers', fontsize=9); axD.set_ylabel('CCC', color='#4C72B0', fontsize=9)
axD2.set_ylabel('Empirical coverage', color='#6AAF6A', fontsize=9)
axD.set_title('Marker-number sensitivity', fontsize=9, pad=8)
axD.legend(loc='lower right', fontsize=7, frameon=False)
axD2.legend(loc='center right', fontsize=7, frameon=False)
panel_label(axD, 'd')

fig.suptitle('Ablation analyses', fontsize=10, y=0.99)
fig.tight_layout(rect=[0,0,1,0.97])
save(fig, 'FigS3_ablation')

# ═══════════════════════════════════════════════════
# S4: External PBMC 3k (already good, just fix title + add DC note)
# ═══════════════════════════════════════════════════
CANON = ['Monocyte','NK','B','DC','T_cell']
CT_COLORS = {'Monocyte':'#0072B2','NK':'#009E73','B':'#D55E00','DC':'#CC79A7','T_cell':'#E69F00'}

yt4 = pd.read_csv('d:/方法学论文/calibdeconv/results/phase7_external_pseudobulk/ground_truth.csv', index_col=0)[CANON]
pred4 = pd.read_csv('d:/方法学论文/calibdeconv/results/phase7_external_pseudobulk/predicted_proportions.csv', index_col=0)[CANON]
per4 = pd.read_csv('d:/方法学论文/calibdeconv/results/phase7_external_pseudobulk/external_pseudobulk_per_celltype.csv').set_index('cell_type')
m4 = pd.read_csv('d:/方法学论文/calibdeconv/results/phase7_external_pseudobulk/external_pseudobulk_metrics.csv').iloc[0]
std4 = pd.read_csv('d:/方法学论文/calibdeconv/results/phase7_external_pseudobulk/ensemble_std.csv', index_col=0)[CANON]

fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.8))
(axA, axB, axE), (axC, axD, axF) = axes

for ct in CANON:
    axA.scatter(yt4[ct], pred4[ct], s=12, alpha=0.5, color=CT_COLORS[ct], edgecolors='none',
                label=ct.replace('_',' '))
axA.plot([0,1],[0,1],'--',color='#AAAAAA',lw=1)
axA.set_xlabel('True proportion'); axA.set_ylabel('Predicted proportion')
axA.set_xlim(0,1); axA.set_ylim(0,1)
axA.set_title('Observed vs predicted', fontsize=9)
axA.text(0.05, 0.92, f'MAE = {m4["MAE"]:.3f}\nCCC = {m4["CCC"]:.3f}\n200 mixtures, 5 types',
         transform=axA.transAxes, fontsize=7, va='top', bbox=dict(boxstyle='round',fc='white',ec='#CCC'))
axA.legend(fontsize=6, frameon=False, loc='lower right')
panel_label(axA, 'a')

axB.bar(range(5), per4.loc[CANON,'CCC'], color=[CT_COLORS[c] for c in CANON], width=0.6, edgecolor='white')
for i, v in enumerate(per4.loc[CANON,'CCC']): axB.text(i, v+0.02, f'{v:.2f}', ha='center', fontsize=6.5)
axB.set_xticks(range(5)); axB.set_xticklabels([c.replace('_',' ') for c in CANON], fontsize=7)
axB.set_ylabel('CCC'); axB.set_ylim(0, 1.1)
axB.set_title('Per-cell-type concordance (5-type)', fontsize=9)
axB.text(3, per4.loc['DC','CCC']+0.08, 'DC: 37\nsource cells', ha='center', fontsize=5.5, color='#888888')
panel_label(axB, 'b')

axC.bar(range(5), per4.loc[CANON,'MAE'], color=[CT_COLORS[c] for c in CANON], width=0.6, edgecolor='white')
for i, v in enumerate(per4.loc[CANON,'MAE']): axC.text(i, v+0.005, f'{v:.3f}', ha='center', fontsize=6.5)
axC.set_xticks(range(5)); axC.set_xticklabels([c.replace('_',' ') for c in CANON], fontsize=7)
axC.set_ylabel('MAE'); axC.set_title('Per-cell-type MAE', fontsize=9)
panel_label(axC, 'c')

abs_err4 = np.abs(yt4.values - pred4.values).mean(axis=1)
mean_std4 = std4.mean(axis=1).values
axD.scatter(mean_std4, abs_err4, s=15, alpha=0.5, color='#4C72B0', edgecolors='white', linewidths=0.3)
r4 = stats.pearsonr(mean_std4, abs_err4)[0]
axD.set_xlabel('Mean ensemble SD'); axD.set_ylabel('Sample MAE')
axD.set_title('Uncertainty vs error', fontsize=9)
axD.text(0.05, 0.92, f'r = {r4:.3f}', transform=axD.transAxes, fontsize=7.5, va='top',
         bbox=dict(boxstyle='round',fc='white',ec='#CCC'))
panel_label(axD, 'd')

# e: Four-type sensitivity (excl DC)
CANON4 = ['Monocyte','NK','B','T_cell']
yt4_4 = yt4[CANON4].copy(); yt4_4 = yt4_4.div(yt4_4.sum(axis=1), axis=0)
pred4_4 = pred4[CANON4].copy(); pred4_4 = pred4_4.div(pred4_4.sum(axis=1), axis=0)
from scipy.optimize import nnls as _nnls_unused
def _ccc(t, p):
    t, p = np.asarray(t).ravel(), np.asarray(p).ravel()
    mt, mp = t.mean(), p.mean(); d = t.var() + p.var() + (mt-mp)**2
    return float(2*((t-mt)*(p-mp)).mean()/d) if d > 0 else np.nan
ccc4_per = [_ccc(yt4_4[c].values, pred4_4[c].values) for c in CANON4]
ccc4_overall = _ccc(yt4_4.values, pred4_4.values)
axE.bar(range(4), ccc4_per, color=[CT_COLORS[c] for c in CANON4], width=0.6, edgecolor='white')
for i, v in enumerate(ccc4_per): axE.text(i, v+0.01, f'{v:.3f}', ha='center', fontsize=6.5)
axE.axhline(ccc4_overall, ls='--', lw=1, color='#555555')
axE.text(3.3, ccc4_overall-0.03, f'overall = {ccc4_overall:.3f}', fontsize=6.5, color='#555555')
axE.set_xticks(range(4)); axE.set_xticklabels([c.replace('_',' ') for c in CANON4], fontsize=7)
axE.set_ylabel('CCC'); axE.set_ylim(0, 1.1)
axE.set_title('4-type sensitivity (excl DC)', fontsize=9)
panel_label(axE, 'e')

axF.axis('off')  # placeholder or leave blank

fig.suptitle('External 10x PBMC 3k pseudo-bulk validation (frozen pipeline, no retraining)', fontsize=10, y=0.99)
fig.tight_layout(rect=[0,0,1,0.97])
save(fig, 'FigS4_external_PBMC3k')

# ═══════════════════════════════════════════════════
# S5: GSE107572 (fix title, add "descriptive n=9")
# ═══════════════════════════════════════════════════
gt5 = pd.read_csv('d:/方法学论文/calibdeconv/results/phase6b_gse107572/ground_truth_5type.csv', index_col=0)[CANON]
pred5 = pd.read_csv('d:/方法学论文/calibdeconv/results/phase6b_gse107572/predicted_proportions.csv', index_col=0)[CANON]
m5 = pd.read_csv('d:/方法学论文/calibdeconv/results/phase6b_gse107572/metrics_overall.csv').iloc[0]
per5 = pd.read_csv('d:/方法学论文/calibdeconv/results/phase6b_gse107572/metrics_per_celltype.csv').set_index('cell_type')
std5 = pd.read_csv('d:/方法学论文/calibdeconv/results/phase6b_gse107572/ensemble_std.csv', index_col=0)[CANON]

fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.4))
(axA, axB), (axC, axD) = axes

for ct in CANON:
    axA.scatter(gt5[ct], pred5[ct], s=40, color=CT_COLORS[ct], alpha=0.8,
                label=ct.replace('_',' '), edgecolors='white', linewidths=0.5)
axA.plot([0,0.7],[0,0.7],'--',color='#AAAAAA',lw=1)
axA.set_xlabel('Flow cytometry fraction'); axA.set_ylabel('Predicted proportion')
axA.set_xlim(0,0.7); axA.set_ylim(0,0.7)
axA.set_title('Observed vs predicted (n = 9)', fontsize=9)
axA.text(0.05, 0.92, f'Pearson r = {m5["Pearson"]:.2f}\nMAE = {m5["MAE"]:.3f}',
         transform=axA.transAxes, fontsize=7.5, va='top', bbox=dict(boxstyle='round',fc='white',ec='#CCC'))
axA.legend(fontsize=6.5, frameon=False, loc='lower right')
panel_label(axA, 'a')

axB.bar(range(5), per5.loc[CANON,'MAE'], color=[CT_COLORS[c] for c in CANON], width=0.6, edgecolor='white')
for i, v in enumerate(per5.loc[CANON,'MAE']): axB.text(i, v+0.005, f'{v:.3f}', ha='center', fontsize=6.5)
axB.set_xticks(range(5)); axB.set_xticklabels([c.replace('_',' ') for c in CANON], fontsize=7)
axB.set_ylabel('MAE'); axB.set_title('Per-cell-type MAE', fontsize=9)
panel_label(axB, 'b')

pearsons5 = per5.loc[CANON,'Pearson'].values
axC.bar(range(5), pearsons5, color=[CT_COLORS[c] for c in CANON], width=0.6, edgecolor='white')
for i, v in enumerate(pearsons5):
    if not np.isnan(v): axC.text(i, max(v,0)+0.03, f'{v:.2f}', ha='center', fontsize=6.5)
    else: axC.text(i, 0.05, 'n/a', ha='center', fontsize=6, color='#999')
axC.axhline(0, color='k', lw=0.8); axC.set_ylim(-0.4, 1.0)
axC.set_xticks(range(5)); axC.set_xticklabels([c.replace('_',' ') for c in CANON], fontsize=7)
axC.set_ylabel('Pearson r'); axC.set_title('Per-cell-type correlation (descriptive, n = 9)', fontsize=9)
panel_label(axC, 'c')

abs_err5 = np.abs(gt5.values - pred5.values).mean(axis=1)
mean_std5 = std5.mean(axis=1).values
axD.scatter(mean_std5, abs_err5, s=50, color='#4C72B0', alpha=0.8, edgecolors='white', linewidths=0.6)
r5 = stats.pearsonr(mean_std5, abs_err5)[0]
axD.set_xlabel('Mean ensemble SD'); axD.set_ylabel('Sample MAE')
axD.set_title('Uncertainty vs error (n = 9)', fontsize=9)
axD.text(0.05, 0.92, f'r = {r5:.2f}', transform=axD.transAxes, fontsize=8, va='top',
         bbox=dict(boxstyle='round',fc='white',ec='#CCC'))
panel_label(axD, 'd')

fig.suptitle('GSE107572 real-bulk PBMC feasibility analysis (n = 9)', fontsize=10, y=0.99)
fig.tight_layout(rect=[0,0,1,0.97])
save(fig, 'FigS5_GSE107572_feasibility')

# ═══════════════════════════════════════════════════
# S6: Whole-blood OOD
# ═══════════════════════════════════════════════════
CANON4 = ['Monocyte','NK','B','T_cell']
gt6 = pd.read_csv('d:/方法学论文/calibdeconv/results/phase6b_gse60424/ground_truth_4class.csv', index_col=0)[CANON4]
pred6 = pd.read_csv('d:/方法学论文/calibdeconv/results/phase6b_gse60424/predicted_proportions_ensemble_4class.csv', index_col=0)[CANON4]
diag6 = pd.read_csv('d:/方法学论文/calibdeconv/results/phase6b_gse60424/sample_diagnostics.csv')

fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.4))
(axA, axB), (axC, axD) = axes

CT4_COLORS = {'Monocyte':'#0072B2','NK':'#009E73','B':'#D55E00','T_cell':'#E69F00'}
for ct in CANON4:
    axA.scatter(gt6[ct], pred6[ct], s=35, alpha=0.7, color=CT4_COLORS[ct],
                label=ct.replace('_',' '), edgecolors='white', linewidths=0.5)
axA.plot([0,1],[0,1],'--',color='#AAAAAA',lw=1)
axA.set_xlabel('FACS-derived fraction'); axA.set_ylabel('Predicted proportion')
axA.set_xlim(0,1); axA.set_ylim(0,1)
axA.set_title('Observed vs predicted (4-class, n = 14)', fontsize=9)
axA.text(0.05, 0.92, 'MAE = 0.332\nCCC = 0.021', transform=axA.transAxes, fontsize=7.5, va='top',
         bbox=dict(boxstyle='round',fc='white',ec='#CCC'))
axA.legend(fontsize=6.5, frameon=False, loc='lower right')
panel_label(axA, 'a')

# Per-type MAE
per_mae6 = [np.abs(gt6[ct].values-pred6[ct].values).mean() for ct in CANON4]
axB.bar(range(4), per_mae6, color=[CT4_COLORS[c] for c in CANON4], width=0.6, edgecolor='white')
for i, v in enumerate(per_mae6): axB.text(i, v+0.01, f'{v:.2f}', ha='center', fontsize=7)
axB.set_xticks(range(4)); axB.set_xticklabels([c.replace('_',' ') for c in CANON4], fontsize=7.5)
axB.set_ylabel('MAE'); axB.set_title('Per-cell-type MAE', fontsize=9)
axB.text(0.5, 0.92, 'Monocyte overestimation\nT cell, B underestimation',
         transform=axB.transAxes, ha='center', fontsize=6.5, color='#C62828', va='top')
panel_label(axB, 'b')

# UE scatter (inverted)
axC.scatter(diag6['mean_std'], diag6['sample_MAE'], s=40, color='#4C72B0', alpha=0.7, edgecolors='white')
r6 = stats.pearsonr(diag6['mean_std'], diag6['sample_MAE'])
axC.set_xlabel('Mean ensemble SD'); axC.set_ylabel('Sample MAE')
axC.set_title('Inverted uncertainty-error coupling', fontsize=9)
axC.text(0.05, 0.92, f'r = {r6[0]:.2f} (p = {r6[1]:.3f})\nDomain shift inverts\nreliability signal',
         transform=axC.transAxes, fontsize=7, va='top',
         bbox=dict(boxstyle='round',fc='#FFF0F0',ec='#E57373'))
panel_label(axC, 'c')

# Mean comparison
means_true = gt6.mean(); means_pred = pred6.mean()
x6 = np.arange(4); w = 0.38
axD.bar(x6-w/2, means_true, w, label='Observed', color='#B0B0B0', edgecolor='white')
axD.bar(x6+w/2, means_pred, w, label='Predicted', color=[CT4_COLORS[c] for c in CANON4], edgecolor='white')
axD.set_xticks(x6); axD.set_xticklabels([c.replace('_',' ') for c in CANON4], fontsize=7.5)
axD.set_ylabel('Mean fraction'); axD.set_title('Systematic bias under domain shift', fontsize=9)
axD.legend(fontsize=7, frameon=False)
panel_label(axD, 'd')

fig.suptitle('GSE60424 whole-blood out-of-domain analysis (n = 14, 4-class)', fontsize=10, y=0.99)
fig.tight_layout(rect=[0,0,1,0.97])
save(fig, 'FigS6_whole_blood_OOD')

# ═══════════════════════════════════════════════════
# S7: Neutrophil mechanism
# ═══════════════════════════════════════════════════
df7 = pd.read_csv('d:/方法学论文/calibdeconv/results/phase7_neutrophil_ood/neutrophil_contamination_metrics.csv')
nf = df7['neutro_frac'].values * 100

fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.5))
((axA,axB,axC),(axD,axE,axF)) = axes
BLUE, RED, GREEN, PURPLE = '#4C72B0', '#C06060', '#6AAF6A', '#8172B3'

axA.plot(nf, df7['MAE'], 'o-', color=RED, lw=1.3, ms=6, label='MAE')
axA2 = axA.twinx()
axA2.plot(nf, df7['CCC'], 's--', color=BLUE, lw=1.3, ms=6, label='CCC')
axA.set_xlabel('Neutrophil contamination (%)'); axA.set_ylabel('MAE', color=RED)
axA2.set_ylabel('CCC', color=BLUE); axA2.set_ylim(-0.2, 1.0)
axA.set_title('Accuracy degrades with contamination', fontsize=9)
axA.legend(loc='upper left', fontsize=7, frameon=False)
axA2.legend(loc='center right', fontsize=7, frameon=False)
panel_label(axA, 'a')

axB.plot(nf, df7['pred_Monocyte_mean'], 'o-', color='#0072B2', lw=1.3, ms=6)
axB.axhline(df7['pred_Monocyte_mean'].iloc[0], ls=':', lw=1, color='#888')
axB.text(50, df7['pred_Monocyte_mean'].iloc[0]+0.03, 'True PBMC monocyte fraction', fontsize=6, color='#888')
axB.set_xlabel('Neutrophil contamination (%)'); axB.set_ylabel('Predicted monocyte fraction')
axB.set_title('Monocyte absorbs neutrophil signal', fontsize=9); axB.set_ylim(0, 1.0)
panel_label(axB, 'b')

axC.plot(nf, df7['pred_Tcell_mean'], 'o-', color='#E69F00', lw=1.3, ms=6, label='T cell')
axC.plot(nf, df7['pred_B_mean'], 's--', color='#D55E00', lw=1.3, ms=6, label='B cell')
axC.set_xlabel('Neutrophil contamination (%)'); axC.set_ylabel('Predicted fraction')
axC.set_title('T and B cell predictions collapse', fontsize=9); axC.set_ylim(0, 0.5)
axC.legend(fontsize=7, frameon=False)
panel_label(axC, 'c')

axD.plot(nf, df7['mean_ensemble_std'], 'o-', color=PURPLE, lw=1.3, ms=6)
axD.axhline(df7['mean_ensemble_std'].iloc[0], ls=':', lw=1, color='#888')
axD.set_xlabel('Neutrophil contamination (%)'); axD.set_ylabel('Mean ensemble SD')
axD.set_title('Ensemble SD decreases (false confidence)', fontsize=9)
panel_label(axD, 'd')

axE.plot(nf, df7['ue_corr'], 'o-', color=GREEN, lw=1.3, ms=6)
axE.axhline(0, ls='-', lw=0.8, color='k')
axE.set_xlabel('Neutrophil contamination (%)'); axE.set_ylabel('UE correlation (Pearson r)')
axE.set_title('Uncertainty-error coupling is unstable', fontsize=9)
axE.set_ylim(-0.8, 0.8)
panel_label(axE, 'e')

axF.plot(nf, df7['coverage90'], 'o-', color=BLUE, lw=1.3, ms=6)
axF.axhline(0.90, ls='--', lw=1, color='k')
axF.text(5, 0.91, 'Nominal 90%', fontsize=6.5)
axF.set_xlabel('Neutrophil contamination (%)'); axF.set_ylabel('Empirical coverage')
axF.set_title('Coverage under neutrophil contamination', fontsize=9)
axF.set_ylim(0.85, 0.95)
panel_label(axF, 'f')

fig.suptitle('Neutrophil out-of-domain mechanism: unmodeled cell type produces false confidence', fontsize=10, y=0.99)
fig.tight_layout(rect=[0,0,1,0.97])
save(fig, 'FigS7_neutrophil_mechanism')

print('\nAll supplementary figures (S2-S7) redrawn with unified style.')
