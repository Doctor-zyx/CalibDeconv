#!/usr/bin/env python
"""Generate DECODE-inspired Figure 1 for CalibDeconv."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 7,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    'savefig.dpi': 300
})

fig, ax = plt.subplots(figsize=(14, 8.5))
ax.set_xlim(0, 140); ax.set_ylim(0, 85)
ax.axis('off')

# Colors
C_REF, C_CAL, C_TEST = '#4393C3', '#F4A582', '#92C5DE'
C_MONO, C_NK, C_B, C_DC, C_T = '#0072B2', '#009E73', '#D55E00', '#CC79A7', '#E69F00'
INK = '#333333'

def rbox(x, y, w, h, fc='white', ec='#BBBBBB', lw=1.0, label='', fs=7, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle='round,pad=0.3,rounding_size=0.8',
        fc=fc, ec=ec, lw=lw, zorder=2))
    if label:
        ax.text(x+w/2, y+h/2, label, ha='center', va='center',
                fontsize=fs, color=INK, fontweight='bold' if bold else 'normal', zorder=3)

def arr(x1, y1, x2, y2, color='#666666', lw=1.2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle='-|>', color=color, lw=lw))

def stage_label(x, y, text):
    ax.text(x, y, text, fontsize=9, fontweight='bold', color=INK, va='bottom')

def stage_bg(x, y, w, h, fc='#F8F9FA'):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle='round,pad=0.2,rounding_size=1.2',
        fc=fc, ec='#E0E0E0', lw=0.8, zorder=0))

def cells(cx, cy, n=10, spread=2.0):
    colors = [C_MONO, C_NK, C_B, C_DC, C_T, C_T, C_T, C_MONO, C_NK, C_B, C_T, C_T]
    rng = np.random.default_rng(7)
    for i in range(n):
        dx = (rng.random()-0.5) * spread
        dy = (rng.random()-0.5) * spread * 0.8
        ax.add_patch(Circle((cx+dx, cy+dy), 0.4, fc=colors[i%len(colors)], ec='white', lw=0.3, zorder=5))

# ════════ Panel a: Reference + donor split ════════
stage_bg(1, 48, 30, 34)
stage_label(2, 83, 'a')
ax.text(16, 82, 'Reference and donor-aware split', fontsize=8, ha='center', color=INK)

# scRNA cluster
cells(8, 72, n=15, spread=3.5)
ax.text(8, 66.5, 'PBMC scRNA-seq\n(Hao 2021, 8 donors)', ha='center', fontsize=6, color='#555555')

# Split boxes
for label, color, yy in [('Reference pool\nP3, P4, P5, P8', C_REF, 58),
                          ('Calibration pool\nP2, P7', C_CAL, 52),
                          ('Test pool\nP1, P6', C_TEST, 46-4)]:
    rbox(15, yy, 14, 5, fc=color, ec='#777777', label=label, fs=6)

arr(11, 68, 15, 62); arr(11, 68, 15, 56); arr(11, 68, 15, 47)

# ════════ Panel b: Marker + Pseudo-bulk ════════
stage_bg(33, 48, 30, 34)
stage_label(34, 83, 'b')
ax.text(48, 82, 'Marker signature and pseudo-bulk', fontsize=8, ha='center', color=INK)

rbox(35, 72, 12, 6, fc='#E3F2FD', ec='#1976D2', label='DE markers\n(445 genes)', fs=6.5)
rbox(50, 72, 12, 6, fc='#FFF3E0', ec='#E65100', label='Signature\nmatrix', fs=6.5)
arr(47, 75, 50, 75)

rbox(35, 58, 26, 10, fc='white', ec='#AAAAAA')
ax.text(48, 65, 'Pseudo-bulk generation', ha='center', fontsize=7.5, fontweight='bold', color=INK)
ax.text(48, 62, 'Dirichlet(1.0) → 500 cells → sum → CPM', ha='center', fontsize=5.5, color='#666666')
ax.text(48, 59.5, '500 cal + 500 test mixtures', ha='center', fontsize=5.5, color='#666666')

# Cell type dots + legend
for i, (ct, col) in enumerate(zip(['Mono','NK','B','DC','T cell'], [C_MONO,C_NK,C_B,C_DC,C_T])):
    ax.add_patch(Circle((37+i*5.5, 51), 0.7, fc=col, ec='white', lw=0.4, zorder=5))
    ax.text(37+i*5.5, 49.2, ct, ha='center', fontsize=5, color=INK)

arr(41, 72, 41, 68.5); arr(48, 72, 48, 68.5)

# ════════ Panel c: NNLS + Ensemble ════════
stage_bg(1, 5, 38, 40)
stage_label(2, 46, 'c')
ax.text(20, 45, 'NNLS and bootstrap ensemble uncertainty', fontsize=8, ha='center', color=INK)

rbox(3, 33, 10, 6, fc='#F5F5F5', ec='#999999', label='Bulk\nmixture', fs=6.5)
rbox(16, 31, 10, 9, fc='#E3F2FD', ec='#1976D2', label='NNLS\n(frozen)', fs=7.5, bold=True)

# Ensemble stack
for j in range(3):
    rbox(29+j*0.7, 33-j*0.5, 8, 7, fc='#E8F5E9', ec='#66BB6A', lw=0.7)
ax.text(33.5, 38, 'Ensemble', ha='center', fontsize=7, fontweight='bold', color='#2E7D32')
ax.text(33.5, 35.5, 'B = 50', ha='center', fontsize=6.5, color='#2E7D32')
ax.text(33.5, 33.5, '80% genes\n80% ref cells', ha='center', fontsize=5.5, color='#555555')

# Outputs
rbox(4, 10, 14, 8, fc='#FFF8E1', ec='#F9A825', label='Ensemble mean\n(point estimate)', fs=6.5)
rbox(21, 10, 14, 8, fc='#FCE4EC', ec='#C62828', label='Ensemble SD\n(uncertainty)', fs=6.5)
ax.text(28, 7.5, 'UE-corr = 0.627', ha='center', fontsize=5.5, style='italic', color='#C62828')

arr(13, 36, 16, 36); arr(26, 36, 29, 36)
arr(21, 31, 11, 18.5); arr(33.5, 32.5, 28, 18.5)

# ════════ Panel d: Conformal calibration ════════
stage_bg(41, 5, 42, 40)
stage_label(42, 46, 'd')
ax.text(62, 45, 'Split conformal calibration', fontsize=8, ha='center', color=INK)

rbox(43, 33, 14, 8, fc=C_CAL, ec='#BF360C', label='Cal-set scores\n|y − ŷ|', fs=6.5)
rbox(60, 33, 10, 8, fc='#F3E5F5', ec='#6A1B9A', label='Quantile\nq̂₉₀', fs=7, bold=True)
rbox(43, 14, 18, 11, fc='#E8F5E9', ec='#2E7D32', lw=1.5)
ax.text(52, 22.5, 'Calibrated intervals', ha='center', fontsize=7.5, fontweight='bold', color='#1B5E20')
ax.text(52, 19.5, '[ŷ − q̂,  ŷ + q̂] ∩ [0, 1]', ha='center', fontsize=7, color=INK)
ax.text(52, 16.5, '90% nominal → 95% empirical', ha='center', fontsize=6, color='#2E7D32')

# Raw ensemble (crossed out)
rbox(64, 14, 16, 11, fc='#FFEBEE', ec='#E53935', lw=1.2)
ax.text(72, 22, 'Raw ensemble', ha='center', fontsize=6.5, color='#C62828')
ax.text(72, 19, '5th–95th %ile', ha='center', fontsize=6, color='#C62828')
ax.text(72, 16, '39% coverage ✗', ha='center', fontsize=6.5, fontweight='bold', color='#C62828')

arr(57, 37, 60, 37); arr(65, 33, 65, 25.5); arr(52, 33, 52, 25.5)

# ════════ Panel e: Domain boundary ════════
stage_bg(86, 5, 52, 77)
stage_label(87, 83, 'e')
ax.text(112, 82, 'Domain validity and boundary', fontsize=8, ha='center', color=INK)

# In-domain
rbox(89, 55, 46, 21, fc='#E8F5E9', ec='#43A047', lw=1.8)
ax.text(112, 73.5, 'In-domain (PBMC)', ha='center', fontsize=8.5, fontweight='bold', color='#2E7D32')
ax.text(112, 69.5, '● Prediction intervals empirically valid', ha='center', fontsize=6.5, color='#333333')
ax.text(112, 66.5, '● Uncertainty correlates with error (r = 0.63)', ha='center', fontsize=6.5, color='#333333')
ax.text(112, 63.5, '● Rejection of uncertain samples reduces MAE', ha='center', fontsize=6.5, color='#333333')
ax.text(112, 60.5, '● External pseudo-bulk generalization (CCC 0.87)', ha='center', fontsize=6.5, color='#333333')
ax.text(112, 57.5, '● Stress-robust under moderate perturbation', ha='center', fontsize=6.5, color='#333333')

# Domain boundary line
ax.plot([89, 135], [52.5, 52.5], ls='--', lw=2, color='#FF6F00', zorder=3)
ax.text(112, 53.5, '— domain boundary —', ha='center', fontsize=7, color='#FF6F00', fontweight='bold')

# Out-of-domain
rbox(89, 8, 46, 42, fc='#FFF3E0', ec='#E65100', lw=1.8)
ax.text(112, 47, 'Out-of-domain (whole blood / neutrophil)', ha='center', fontsize=8.5, fontweight='bold', color='#BF360C')
ax.text(112, 43, '✗ Unmodeled neutrophils dominate signal', ha='center', fontsize=6.5, color='#C62828')
ax.text(112, 40, '✗ Monocyte absorbs neutrophil expression', ha='center', fontsize=6.5, color='#C62828')
ax.text(112, 37, '✗ T cell and B cell collapse to zero', ha='center', fontsize=6.5, color='#C62828')
ax.text(112, 34, '✗ Ensemble SD decreases → false confidence', ha='center', fontsize=6.5, color='#C62828')
ax.text(112, 31, '✗ UE-correlation inverts (r = −0.61)', ha='center', fontsize=6.5, color='#C62828')
ax.text(112, 27, 'Mechanism: deterministic NNLS under\ndominant unmodeled component', ha='center', fontsize=6, style='italic', color='#555555')
ax.text(112, 21, '→ Explicit domain check required\n    before interval interpretation', ha='center', fontsize=6.5, fontweight='bold', color='#BF360C')

# Connecting arrows between stages
arr(29, 62, 35, 62)   # a → b
arr(48, 58, 20, 42)   # b → c (pseudo-bulk)
arr(37, 12, 43, 14)   # c → d (uncertainty to conformal)
arr(80, 20, 89, 20)   # d → e (intervals to domain)
arr(80, 65, 89, 65)   # to in-domain

out = 'd:/方法学论文/calibdeconv/results/figures_publication_draft'
fig.savefig(f'{out}/fig1_nature_style_v2.png', bbox_inches='tight', dpi=300)
fig.savefig(f'{out}/fig1_nature_style_v2.pdf', bbox_inches='tight')
fig.savefig(f'{out}/fig1_nature_style_v2.svg', bbox_inches='tight')
plt.close()
print('Figure 1 Nature-style v2 saved (png/pdf/svg)')
