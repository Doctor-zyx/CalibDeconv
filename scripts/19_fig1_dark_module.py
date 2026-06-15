#!/usr/bin/env python
"""Figure 1 v4: Dark-module BioRender-inspired style matching user reference."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 7,
    'figure.facecolor': '#F5F7FA', 'savefig.facecolor': '#F5F7FA',
    'savefig.dpi': 300
})

fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 140); ax.set_ylim(0, 90)
ax.axis('off')
ax.set_facecolor('#F5F7FA')

# Colors - dark module style
DARK1 = '#1A237E'   # deep indigo
DARK2 = '#283593'   # indigo
DARK3 = '#303F9F'   # lighter indigo
DARK4 = '#1B5E20'   # dark green
DARK5 = '#4A148C'   # deep purple
DARK6 = '#BF360C'   # deep orange
ACCENT1 = '#42A5F5' # light blue accent
ACCENT2 = '#66BB6A' # green accent
ACCENT3 = '#AB47BC' # purple accent
ACCENT4 = '#FF7043' # orange accent
WHITE = '#FFFFFF'
LGRAY = '#E8EAF6'
ARROW_C = '#546E7A'

def dark_module(x, y, w, h, color=DARK1, title='', subtitle='', detail=''):
    """Draw a dark rounded module with white text."""
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle='round,pad=0.3,rounding_size=1.0',
        fc=color, ec='#1A1A2E', lw=1.2, zorder=3))
    cy = y + h/2
    if title and subtitle and detail:
        ax.text(x+w/2, cy+h*0.22, title, ha='center', va='center',
                fontsize=8, fontweight='bold', color=WHITE, zorder=4)
        ax.text(x+w/2, cy-h*0.05, subtitle, ha='center', va='center',
                fontsize=6.5, color='#B3E5FC', zorder=4)
        ax.text(x+w/2, cy-h*0.28, detail, ha='center', va='center',
                fontsize=5.5, color='#90CAF9', zorder=4)
    elif title and subtitle:
        ax.text(x+w/2, cy+h*0.15, title, ha='center', va='center',
                fontsize=8.5, fontweight='bold', color=WHITE, zorder=4)
        ax.text(x+w/2, cy-h*0.15, subtitle, ha='center', va='center',
                fontsize=6.5, color='#B3E5FC', zorder=4)
    elif title:
        ax.text(x+w/2, cy, title, ha='center', va='center',
                fontsize=9, fontweight='bold', color=WHITE, zorder=4)

def light_module(x, y, w, h, color='#E3F2FD', ec='#1565C0', title='', subtitle=''):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle='round,pad=0.2,rounding_size=0.8',
        fc=color, ec=ec, lw=1.0, zorder=3))
    cy = y + h/2
    if title and subtitle:
        ax.text(x+w/2, cy+h*0.15, title, ha='center', va='center',
                fontsize=7, fontweight='bold', color='#1A237E', zorder=4)
        ax.text(x+w/2, cy-h*0.18, subtitle, ha='center', va='center',
                fontsize=5.5, color='#555555', zorder=4)
    elif title:
        ax.text(x+w/2, cy, title, ha='center', va='center',
                fontsize=7.5, fontweight='bold', color='#1A237E', zorder=4)

def thick_arrow(x1, y1, x2, y2, color=ARROW_C):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle='-|>', color=color, lw=2.0,
                        mutation_scale=15, connectionstyle='arc3,rad=0'))

def panel_label(x, y, text):
    ax.text(x, y, text, fontsize=13, fontweight='bold', color='#1A237E',
            va='center', ha='left', zorder=10)

# ════════════════════════════════════════════════════
# Panel a: Data and donor-aware split (top-left)
# ════════════════════════════════════════════════════
panel_label(2, 87, 'a')

dark_module(5, 72, 22, 12, DARK1,
            'PBMC CITE-seq Reference',
            'Hao et al. 2021',
            '158K cells, 8 donors')

# Three pool modules
dark_module(32, 78, 18, 7, '#1565C0',
            'Reference Pool', 'P3, P4, P5, P8')
dark_module(32, 69, 18, 7, '#E65100',
            'Calibration Pool', 'P2, P7')
dark_module(32, 60, 18, 7, '#00695C',
            'Test Pool', 'P1, P6')

# Arrows from reference to pools
thick_arrow(27, 80, 32, 82)
thick_arrow(27, 78, 32, 73)
thick_arrow(27, 76, 32, 64)

# Note
ax.text(41, 57, 'No donor overlap', fontsize=6, ha='center',
        style='italic', color='#666666')

# ════════════════════════════════════════════════════
# Panel b: Signature + pseudo-bulk (top-right)
# ════════════════════════════════════════════════════
panel_label(55, 87, 'b')

dark_module(58, 76, 20, 10, DARK2,
            'Marker Selection',
            '445 DE genes',
            'from reference pool only')

dark_module(82, 76, 20, 10, DARK3,
            'Signature Matrix',
            '445 genes x 5 types',
            'CPM-normalized')

dark_module(58, 61, 44, 11, '#004D40',
            'Pseudo-bulk Generation',
            'Dirichlet(1.0) proportions, 500 cells/mixture',
            '500 cal + 500 test mixtures, sum counts, CPM')

# Cell type dots
cell_colors = [('#0072B2','Mono'), ('#009E73','NK'), ('#D55E00','B'),
               ('#CC79A7','DC'), ('#E69F00','T cell')]
for i, (col, name) in enumerate(cell_colors):
    ax.add_patch(Circle((62+i*9, 58.5), 1.0, fc=col, ec='white', lw=0.5, zorder=5))
    ax.text(62+i*9, 56, name, ha='center', fontsize=5.5, color='#333333')

thick_arrow(78, 81, 82, 81)
thick_arrow(80, 76, 80, 72.5)

# ════════════════════════════════════════════════════
# Panel c: NNLS + Ensemble (bottom-left)
# ════════════════════════════════════════════════════
panel_label(2, 50, 'c')

dark_module(5, 33, 16, 12, DARK1,
            'NNLS',
            '(frozen estimator)',
            'non-negative LS')

# Ensemble stack effect
for j in range(3):
    ax.add_patch(FancyBboxPatch((25+j*0.8, 35-j*0.5), 16, 10,
        boxstyle='round,pad=0.2,rounding_size=0.8',
        fc='#1B5E20', ec='#2E7D32', lw=0.8, alpha=0.7, zorder=2+j))
dark_module(28, 33, 16, 10, DARK4,
            'Bootstrap Ensemble',
            'B = 50 iterations',
            '80% genes, 80% cells')

# Outputs
light_module(5, 10, 18, 10, '#FFF8E1', '#F9A825',
             'Ensemble Mean', '(point estimate)')
light_module(27, 10, 18, 10, '#FCE4EC', '#C62828',
             'Ensemble SD', '(uncertainty score)')

thick_arrow(21, 39, 25, 39)
thick_arrow(13, 33, 13, 20.5)
thick_arrow(36, 33, 36, 20.5)

# Input arrow
light_module(5, 46, 12, 5, '#ECEFF1', '#607D8B', 'Bulk mixture')
thick_arrow(11, 46, 11, 45.5)

# ════════════════════════════════════════════════════
# Panel d: Conformal calibration (bottom-right)
# ════════════════════════════════════════════════════
panel_label(55, 50, 'd')

dark_module(58, 35, 18, 12, DARK5,
            'Nonconformity',
            'Scores',
            'cal set |y − ŷ|')

dark_module(80, 35, 14, 12, '#6A1B9A',
            'Conformal',
            'Quantile q̂')

dark_module(98, 35, 26, 12, '#1B5E20',
            'Calibrated Intervals',
            'ŷ ± q̂, clipped to [0,1]',
            'marginal coverage per cell type')

# Warning boxes (small, at bottom)
light_module(58, 10, 24, 10, '#FFEBEE', '#E53935',
             'Raw ensemble: under-covers',
             '→ conformal step essential')

light_module(86, 10, 36, 10, '#FFF3E0', '#FF6F00',
             'Domain note: valid within PBMC',
             'OOD (whole blood): check before interpreting')

thick_arrow(76, 41, 80, 41)
thick_arrow(94, 41, 98, 41)

# ════════════════════════════════════════════════════
# Main flow arrows between panels
# ════════════════════════════════════════════════════
# a → b
thick_arrow(50, 78, 58, 78)
# b → c (signature feeds into NNLS)
thick_arrow(58, 61, 13, 45.5)
# c → d (ensemble outputs feed conformal)
thick_arrow(45, 15, 58, 35)
# a pools → b pseudo-bulk
thick_arrow(50, 66, 58, 66)

out = 'd:/方法学论文/calibdeconv/results/figures_publication_draft'
fig.savefig(f'{out}/fig1_dark_module_v4.png', bbox_inches='tight', dpi=300)
fig.savefig(f'{out}/fig1_dark_module_v4.pdf', bbox_inches='tight')
fig.savefig(f'{out}/fig1_dark_module_v4.svg', bbox_inches='tight')
plt.close()
print('Figure 1 v4 (dark-module style) saved')
