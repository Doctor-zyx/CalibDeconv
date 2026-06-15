#!/usr/bin/env python
"""
Figure 1 — CalibDeconv-v1 workflow schematic (pure matplotlib, no data, no runs).

Four horizontal panels A->B->C->D with connecting arrows. Saves PNG/PDF/SVG to
results/figures_manuscript/. No experiments, downloads, or external images.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "results" / "figures_manuscript"
OUT.mkdir(parents=True, exist_ok=True)

# Palette (muted, paper-friendly)
C_REF = "#4C72B0"     # reference / blue
C_CAL = "#DD8452"     # calibration / orange
C_TEST = "#C44E52"    # test / red
C_BG = "#F2F2F2"      # panel background
C_BOX = "#FFFFFF"     # inner box
C_ACCENT = "#55A868"  # green accent
EDGE = "#333333"

plt.rcParams.update({"font.size": 9.5, "svg.fonttype": "none"})


def panel(ax, x0, w, title, color):
    """Draw a rounded panel background spanning the full height, return bounds."""
    pad = 0.0
    box = FancyBboxPatch((x0, 0.06), w, 0.82,
                         boxstyle="round,pad=0.004,rounding_size=0.02",
                         linewidth=1.2, edgecolor=color, facecolor=C_BG, zorder=1)
    ax.add_patch(box)
    ax.text(x0 + w / 2, 0.93, title, ha="center", va="center",
            fontsize=11, fontweight="bold", color=color, zorder=3)
    return x0, x0 + w


def box(ax, cx, cy, w, h, lines, fc=C_BOX, ec=EDGE, fs=8.5, bold_first=False):
    b = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                       boxstyle="round,pad=0.003,rounding_size=0.012",
                       linewidth=1.0, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(b)
    if isinstance(lines, str):
        lines = [lines]
    n = len(lines)
    for i, ln in enumerate(lines):
        yy = cy + (n - 1) / 2 * 0.032 - i * 0.032
        fw = "bold" if (bold_first and i == 0) else "normal"
        ax.text(cx, yy, ln, ha="center", va="center", fontsize=fs,
                fontweight=fw, zorder=3)


def arrow(ax, x0, y0, x1, y1, color=EDGE, lw=1.6):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                 mutation_scale=14, linewidth=lw, color=color, zorder=4))


fig, ax = plt.subplots(figsize=(14, 4.2))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

W = 0.225
gap = 0.018
x = 0.01
xs = []
for i in range(4):
    xs.append(x)
    x += W + gap

# ── Panel A: reference + donor-aware split ──
aL, aR = panel(ax, xs[0], W, "A  Donor-aware split", C_REF)
cx = xs[0] + W / 2
box(ax, cx, 0.78, W * 0.8, 0.10, ["Hao 2021 PBMC", "scRNA / CITE-seq"], bold_first=True)
box(ax, cx, 0.60, W * 0.82, 0.10, ["Reference donors", "P3 P4 P5 P8"], fc="#E8EEF7", ec=C_REF)
box(ax, cx, 0.43, W * 0.82, 0.10, ["Calibration donors", "P2 P7"], fc="#FBEbDD", ec=C_CAL)
box(ax, cx, 0.26, W * 0.82, 0.10, ["Test donors", "P1 P6"], fc="#F7E3E4", ec=C_TEST)
ax.text(cx, 0.135, "leakage-safe: whole donors disjoint", ha="center", va="center",
        fontsize=7.5, style="italic", color="#555")

# ── Panel B: pseudo-bulk + marker signature ──
bL, bR = panel(ax, xs[1], W, "B  Pseudo-bulk & markers", C_ACCENT)
cx = xs[1] + W / 2
box(ax, cx, 0.78, W * 0.84, 0.10, ["Reference-pool markers", "445 genes (CPM)"], bold_first=True)
box(ax, cx, 0.60, W * 0.84, 0.10, ["Pseudo-bulk from", "held-out donors"])
box(ax, cx, 0.40, W * 0.84, 0.135,
    ["5 cell types:", "Monocyte  NK  B", "DC  T_cell"], fc="#EAF3EE", ec=C_ACCENT, bold_first=True)
ax.text(cx, 0.20, "7-type T-subtype collapse:", ha="center", va="center",
        fontsize=7.5, style="italic", color="#555")
ax.text(cx, 0.165, "diagnostic only, not primary", ha="center", va="center",
        fontsize=7.5, style="italic", color="#555")

# ── Panel C: deconvolution + ensemble ──
cL, cR = panel(ax, xs[2], W, "C  NNLS + ensemble", C_CAL)
cx = xs[2] + W / 2
box(ax, cx, 0.78, W * 0.8, 0.10, ["NNLS", "point estimate"], bold_first=True)
box(ax, cx, 0.585, W * 0.86, 0.14,
    ["Ensemble (B=50):", "gene subsampling", "cell/ref perturbation"], fc="#FBEbDD", ec=C_CAL)
box(ax, cx, 0.34, W * 0.86, 0.155,
    ["Outputs:", "ensemble mean / std", "interval width", "mean_std score"],
    fc="#FFFFFF", ec=EDGE, bold_first=True)

# ── Panel D: conformal + reliability ──
dL, dR = panel(ax, xs[3], W, "D  Conformal & reliability", C_TEST)
cx = xs[3] + W / 2
box(ax, cx, 0.79, W * 0.86, 0.10, ["Calibration:", "nonconformity scores"], bold_first=True)
box(ax, cx, 0.62, W * 0.86, 0.10, ["Intervals @ 80/90/95%"], fc="#F7E3E4", ec=C_TEST)
box(ax, cx, 0.45, W * 0.86, 0.10, ["Stress testing +", "uncertainty rejection"])
box(ax, cx, 0.26, W * 0.86, 0.115,
    ["coverage · intervals", "reliability diagnostics"], fc="#FFFFFF", ec=EDGE, bold_first=True)

# ── Connecting arrows A->B->C->D ──
for i in range(3):
    arrow(ax, xs[i] + W + 0.001, 0.47, xs[i + 1] - 0.001, 0.47, color="#444", lw=2.0)

fig.text(0.5, 0.005,
         "Figure 1. Overview of the CalibDeconv-v1 workflow.",
         ha="center", fontsize=8.5, style="italic")

for ext in ("png", "pdf", "svg"):
    fig.savefig(OUT / f"fig1_calibdeconv_workflow_schematic.{ext}",
                bbox_inches="tight", dpi=200)
plt.close(fig)
print("[OK] fig1 saved (png/pdf/svg) to", OUT)
