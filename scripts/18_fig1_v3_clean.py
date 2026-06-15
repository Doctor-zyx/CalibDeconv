#!/usr/bin/env python
"""Figure 1 v3: Clean 4-panel Nature Methods workflow (revised per feedback)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import numpy as np

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 7,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    'savefig.dpi': 300
})

fig, ax = plt.subplots(figsize=(13, 7.5))
ax.set_xlim(0, 130); ax.set_ylim(0, 75)
ax.axis('off')

# Colors
C_REF, C_CAL, C_TEST = '#4393C3', '#F4A582', '#92C5DE'
C_MONO, C_NK, C_B, C_DC, C_T = '#0072B2', '#009E73', '#D55E00', '#CC79A7', '#E69F00'
INK = '#333333'

def rbox(x, y, w, h, fc='white', ec='#BBBBBB', lw=1.0, zorder=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle='round,pad=0.2,rounding_size=0.7',
        fc=fc, ec=ec, lw=lw, zorder=zorder))

def txt(x, y, s, fs=7, ha='center', va='center', color=INK, bold=False, italic=False):
    ax.text(x, y, s, fontsize=fs, ha=ha, va=va, color=color,
            fontweight='bold' if bold else 'normal',
            fontstyle='italic' if italic else 'normal', zorder=5)

def arr(x1, y1, x2, y2, color='#777777', lw=1.3):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, mutation_scale=12))

def stage_bg(x, y, w, h):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle='round,pad=0.2,rounding_size=1.0',
        fc='#F7F8FA', ec='#E0E0E0', lw=0.7, zorder=0))

def cells(cx, cy, n=10, spread=2.2):
    colors = [C_T, C_T, C_T, C_MONO, C_MONO, C_NK, C_B, C_B, C_DC, C_T, C_NK, C_T]
    rng = np.random.default_rng(7)
    for i in range(n):
        dx = (rng.random()-0.5) * spread * 1.2
        dy = (rng.random()-0.5) * spread * 0.9
        ax.add_patch(Circle((cx+dx, cy+dy), 0.45, fc=colors[i%len(colors)],
                            ec='white', lw=0.3, zorder=5))

# ═══════════════════════════════════════════════════════════════
# 2×2 layout:  a (top-left) → b (top-right)
#              ↓                ↓
#              c (bot-left) → d (bot-right)
# ═══════════════════════════════════════════════════════════════

# ──── Panel a: Donor-aware split (top-left) ────
stage_bg(1, 40, 60, 33)
txt(4, 72, 'a', fs=12, bold=True, ha='left')

# scRNA cluster
cells(12, 60, n=14, spread=3.2)
txt(12, 54, 'PBMC CITE-seq\nreference (8 donors)', fs=6.5, color='#555555')

# Split into 3 pools
rbox(25, 63, 14, 5.5, fc=C_REF, ec='#2166AC')
txt(32, 65.8, 'Reference pool', fs=6.5, bold=True, color='white')
txt(32, 63.8, 'P3, P4, P5, P8', fs=5.5, color='white')

rbox(25, 55, 14, 5.5, fc=C_CAL, ec='#B2182B')
txt(32, 57.8, 'Calibration pool', fs=6.5, bold=True)
txt(32, 55.8, 'P2, P7', fs=5.5)

rbox(25, 47, 14, 5.5, fc=C_TEST, ec='#4393C3')
txt(32, 49.8, 'Test pool', fs=6.5, bold=True)
txt(32, 47.8, 'P1, P6', fs=5.5)

# Arrows from cells to pools
arr(16, 62, 25, 66); arr(16, 60, 25, 58); arr(16, 58, 25, 50)

# Right side of panel a: what each pool does
txt(47, 66, '→ marker selection\n   + signature', fs=6, ha='left', color='#555555')
txt(47, 58, '→ calibration\n   mixtures', fs=6, ha='left', color='#555555')
txt(47, 50, '→ held-out test\n   mixtures', fs=6, ha='left', color='#555555')

txt(32, 43, 'No donor overlap between pools', fs=6, italic=True, color='#888888')

# ──── Panel b: Signature + pseudo-bulk (top-right) ────
stage_bg(64, 40, 64, 33)
txt(67, 72, 'b', fs=12, bold=True, ha='left')

# Signature matrix
rbox(68, 58, 17, 11, fc='#E3F2FD', ec='#1565C0')
txt(76.5, 66, '445 DE markers', fs=7, bold=True, color='#1565C0')
txt(76.5, 63, 'Five-type\nsignature matrix', fs=6.5)
txt(76.5, 59.5, '(from ref pool only)', fs=5.5, color='#777777')

# Pseudo-bulk
rbox(89, 58, 17, 11, fc='#FFF8E1', ec='#F9A825')
txt(97.5, 66, 'Pseudo-bulk', fs=7, bold=True, color='#E65100')
txt(97.5, 63, 'Dirichlet(1.0)\n500 cells/mix\nsum → CPM', fs=5.5)

# 5 cell-type legend
for i, (ct, col) in enumerate(zip(['Mono','NK','B','DC','T cell'], [C_MONO,C_NK,C_B,C_DC,C_T])):
    ax.add_patch(Circle((72+i*7, 44.5), 0.8, fc=col, ec='white', lw=0.4, zorder=5))
    txt(72+i*7, 42.5, ct, fs=5.5)

# Arrow sig → pseudo-bulk connection
rbox(109, 58, 16, 11, fc='#F3E5F5', ec='#7B1FA2')
txt(117, 66, 'Known true', fs=6.5, bold=True, color='#4A148C')
txt(117, 63, 'proportions', fs=6.5, color='#4A148C')
txt(117, 60, '(ground truth)', fs=5.5, color='#777777')

arr(85, 63, 89, 63); arr(106, 63, 109, 63)

# ──── Panel c: NNLS + Ensemble (bottom-left) ────
stage_bg(1, 2, 60, 35)
txt(4, 36, 'c', fs=12, bold=True, ha='left')

# Input bulk
rbox(5, 20, 11, 8, fc='#FAFAFA', ec='#AAAAAA')
txt(10.5, 25.5, 'Bulk', fs=7)
txt(10.5, 22, 'mixture', fs=7)

# NNLS box
rbox(20, 18, 12, 12, fc='#E3F2FD', ec='#1565C0', lw=1.3)
txt(26, 26.5, 'NNLS', fs=8.5, bold=True, color='#0D47A1')
txt(26, 23.5, '(frozen)', fs=6.5, color='#555555')
txt(26, 20.5, 'non-negative\nleast squares', fs=5.5, color='#777777')

# Ensemble (stacked boxes)
for j in range(4):
    rbox(36+j*0.6, 20-j*0.4, 9, 9, fc='#E8F5E9', ec='#4CAF50', lw=0.6)
txt(41, 27, 'Bootstrap', fs=7, bold=True, color='#2E7D32')
txt(41, 24.5, 'ensemble', fs=7, bold=True, color='#2E7D32')
txt(41, 22, 'B = 50', fs=6.5, color='#2E7D32')

# Outputs
rbox(5, 5, 14, 8, fc='#FFF8E1', ec='#F9A825')
txt(12, 10.5, 'Ensemble mean', fs=6.5, bold=True)
txt(12, 7.5, '(point estimate)', fs=5.5, color='#777777')

rbox(22, 5, 14, 8, fc='#FCE4EC', ec='#C62828')
txt(29, 10.5, 'Ensemble SD', fs=6.5, bold=True, color='#C62828')
txt(29, 7.5, '(uncertainty)', fs=5.5, color='#777777')

# Subsampling note
rbox(48, 13, 11, 13, fc='white', ec='#AAAAAA', lw=0.7)
txt(53.5, 23, '80%', fs=7, bold=True, color='#555555')
txt(53.5, 20.5, 'genes', fs=6, color='#555555')
txt(53.5, 17.5, '80%', fs=7, bold=True, color='#555555')
txt(53.5, 15, 'ref cells', fs=6, color='#555555')

arr(16, 24, 20, 24); arr(32, 24, 36, 24)
arr(26, 18, 12, 13.5); arr(41, 19.5, 29, 13.5)
arr(46, 22, 48, 22)

# ──── Panel d: Conformal calibration (bottom-right) ────
stage_bg(64, 2, 64, 35)
txt(67, 36, 'd', fs=12, bold=True, ha='left')

# Cal set → scores → quantile
rbox(67, 22, 14, 10, fc=C_CAL, ec='#BF360C')
txt(74, 29, 'Calibration', fs=6.5, bold=True)
txt(74, 26.5, 'set scores', fs=6.5)
txt(74, 24, '|y − ŷ|', fs=7, color='#333333')

rbox(85, 22, 12, 10, fc='#F3E5F5', ec='#6A1B9A', lw=1.3)
txt(91, 29, 'Conformal', fs=7, bold=True, color='#4A148C')
txt(91, 26, 'quantile', fs=7, color='#4A148C')
txt(91, 23.5, 'q̂', fs=9, color='#4A148C')

# Calibrated intervals
rbox(101, 22, 24, 10, fc='#E8F5E9', ec='#2E7D32', lw=1.5)
txt(113, 29, 'Calibrated prediction intervals', fs=7, bold=True, color='#1B5E20')
txt(113, 26, 'ŷ ± q̂ , clipped to [0, 1]', fs=7, color='#333333')
txt(113, 23.5, 'marginal coverage per cell type', fs=5.5, color='#555555')

arr(81, 27, 85, 27); arr(97, 27, 101, 27)

# Raw ensemble warning (small box, not a full panel)
rbox(101, 5, 24, 12, fc='#FFF3E0', ec='#E65100', lw=1.0)
txt(113, 14.5, 'Raw ensemble (5th–95th)', fs=6, color='#BF360C')
txt(113, 11.5, 'under-covers severely', fs=6, color='#BF360C')
txt(113, 8.5, '→ conformal step essential', fs=6, italic=True, color='#BF360C')

# Small OOD warning note (replaces old panel e)
rbox(67, 5, 30, 12, fc='#FFEBEE', ec='#E53935', lw=0.8)
txt(82, 14, 'Domain note', fs=6, bold=True, color='#C62828')
txt(82, 11, 'Valid within calibrated PBMC domain', fs=5.5, color='#333333')
txt(82, 8.5, 'OOD (e.g. whole blood): domain check required', fs=5.5, color='#C62828')
txt(82, 6, 'before interval interpretation', fs=5.5, color='#C62828')

# ──── Main flow arrows between panels ────
# a → b (top)
arr(60, 58, 68, 58)
# a → c (down-left)
arr(30, 40, 26, 30.5)
# b → d (down-right)
arr(97, 40, 91, 32.5)
# c → d (bottom horizontal)
arr(59, 12, 67, 12)

out = 'd:/方法学论文/calibdeconv/results/figures_publication_draft'
fig.savefig(f'{out}/fig1_nature_style_v3.png', bbox_inches='tight', dpi=300)
fig.savefig(f'{out}/fig1_nature_style_v3.pdf', bbox_inches='tight')
fig.savefig(f'{out}/fig1_nature_style_v3.svg', bbox_inches='tight')
plt.close()
print('Figure 1 v3 (4-panel, clean) saved')
