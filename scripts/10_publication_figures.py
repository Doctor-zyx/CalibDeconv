#!/usr/bin/env python
"""
Publication-grade redraw of Figure 1 (workflow schematic) and Figure 2
(accuracy + uncertainty), from FROZEN results only. No new experiments,
no downloads. Writes ONLY to results/figures_publication_draft/.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ── unified publication style ────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",      # Arial-like, always available
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.linewidth": 1.0, "lines.linewidth": 1.1,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.dpi": 300, "figure.dpi": 150,
})

# muted, non-default palette (Okabe–Ito-ish), one color per cell type
CT_COLORS = {
    "Monocyte": "#0072B2", "NK": "#009E73", "B": "#D55E00",
    "DC": "#CC79A7", "T_cell": "#E69F00",
}
CANON = ["Monocyte", "NK", "B", "DC", "T_cell"]
GREY = "#9A9A9A"
INK = "#222222"

R = PROJECT_ROOT / "results"
OUT = R / "figures_publication_draft"
OUT.mkdir(parents=True, exist_ok=True)
_made = []


def save(fig, stem):
    wrote, locked = [], []
    for ext in ("png", "pdf", "svg"):
        try:
            fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight")
            wrote.append(ext)
        except PermissionError:
            locked.append(ext)
    plt.close(fig)
    _made.append(stem + (f"  [LOCKED: {','.join(locked)}]" if locked else ""))


def panel_label(ax, s, dx=-0.18, dy=1.04):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=12, fontweight="bold",
            va="top", ha="left", color=INK)


# ── FIGURE 2 ─────────────────────────────────────────────────────────────────

def figure2():
    yt = pd.read_csv(R / "data_processed_true.csv") if False else \
        pd.read_csv(PROJECT_ROOT / "data" / "processed" / "true_proportions_test_5type.csv", index_col=0)[CANON]
    pe = pd.read_csv(R / "ensemble_marker_5types" / "predicted_proportions_test_ensemble.csv", index_col=0)[CANON]
    p2 = pd.read_csv(R / "nnls_marker_5types" / "marker_5type_summary.csv").set_index("cell_type")
    ea = pd.read_csv(R / "ensemble_marker_5types" / "ensemble_accuracy.csv")
    ea_pt = ea[(ea.split == "test") & (ea.level == "per_cell_type")].set_index("cell_type")
    ea_ov = ea[(ea.split == "test") & (ea.level == "overall")].iloc[0]
    m2 = pd.read_csv(R / "nnls_marker_5types" / "metrics_test.csv").iloc[0]
    ue = pd.read_csv(R / "ensemble_marker_5types" / "uncertainty_error_correlation.csv")
    ue_t = ue[ue.split == "test"].set_index("cell_type")

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))
    (axA, axB), (axC, axD) = axes

    # A — ensemble true vs predicted scatter, colored by cell type
    for ct in CANON:
        axA.scatter(yt[ct], pe[ct], s=7, alpha=0.45, color=CT_COLORS[ct],
                    edgecolors="none", label=ct.replace("_", " "))
    axA.plot([0, 1], [0, 1], ls="--", lw=1.0, color=GREY, zorder=0)
    axA.set_xlim(0, 1); axA.set_ylim(0, 1)
    axA.set_xlabel("True proportion"); axA.set_ylabel("Ensemble predicted")
    axA.set_title("Ensemble accuracy (test)", fontsize=9)
    axA.text(0.04, 0.92, f"MAE = {ea_ov['MAE']:.3f}\nCCC = {ea_ov['CCC']:.3f}",
             transform=axA.transAxes, fontsize=7.5, va="top",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GREY, lw=0.8))
    axA.legend(loc="lower right", frameon=False, handletextpad=0.2,
               borderpad=0.2, labelspacing=0.25, markerscale=1.4)
    axA.grid(False)

    # B — per-cell-type CCC: baseline vs ensemble
    x = np.arange(len(CANON)); w = 0.38
    b2 = [p2.loc[c, "CCC"] for c in CANON]
    b3 = [ea_pt.loc[c, "CCC"] for c in CANON]
    axB.bar(x - w/2, b2, w, color="#B0B0B0", label="NNLS baseline")
    axB.bar(x + w/2, b3, w, color="#4C72B0", label="Ensemble")
    for xi, v in zip(x - w/2, b2):
        axB.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=6.2, color=INK)
    for xi, v in zip(x + w/2, b3):
        axB.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=6.2, color=INK)
    axB.set_xticks(x); axB.set_xticklabels([c.replace("_", " ") for c in CANON], rotation=20, ha="right")
    axB.set_ylim(0, 1.18); axB.set_ylabel("CCC")
    axB.set_title("Per-cell-type concordance", fontsize=9)
    axB.legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0),
               handletextpad=0.4, columnspacing=1.2)
    axB.yaxis.grid(True, color="#E6E6E6", lw=0.8); axB.set_axisbelow(True)
    axB.xaxis.grid(False)

    # C — overall metrics baseline vs ensemble (MAE, RMSE, CCC)
    mets = ["MAE", "RMSE", "CCC"]
    base = [m2["MAE"], m2["RMSE"], m2["CCC"]]
    ens = [ea_ov["MAE"], ea_ov["RMSE"], ea_ov["CCC"]]
    xc = np.arange(len(mets)); w = 0.38
    axC.bar(xc - w/2, base, w, color="#B0B0B0", label="NNLS baseline")
    axC.bar(xc + w/2, ens, w, color="#4C72B0", label="Ensemble")
    for xi, v in zip(xc - w/2, base):
        axC.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=6.2)
    for xi, v in zip(xc + w/2, ens):
        axC.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=6.2)
    axC.set_xticks(xc); axC.set_xticklabels(mets)
    axC.set_ylim(0, 1.0); axC.set_ylabel("Value")
    axC.set_title("Overall accuracy preserved", fontsize=9)
    axC.legend(loc="upper left", frameon=False)
    axC.yaxis.grid(True, color="#E6E6E6", lw=0.8); axC.set_axisbelow(True)
    axC.xaxis.grid(False)

    # D — per-cell-type uncertainty-error correlation
    r = [ue_t.loc[c, "pearson_r"] for c in CANON]
    colors = [CT_COLORS[c] for c in CANON]
    axD.bar(np.arange(len(CANON)), r, color=colors, width=0.62)
    overall_r = float(ue_t.loc["overall", "pearson_r"])
    axD.axhline(overall_r, ls="--", lw=1.0, color=GREY)
    axD.text(0.02, overall_r + 0.02, f"pooled cell-level r = {overall_r:.3f}",
             transform=axD.get_yaxis_transform(), fontsize=7, color=INK)
    axD.axhline(0, color=INK, lw=0.8)
    axD.set_xticks(np.arange(len(CANON))); axD.set_xticklabels([c.replace("_", " ") for c in CANON], rotation=20, ha="right")
    axD.set_ylim(-0.25, 0.75); axD.set_ylabel("Pearson r (std vs |error|)")
    axD.set_title("Uncertainty–error association varies by lineage", fontsize=9)
    axD.yaxis.grid(True, color="#E6E6E6", lw=0.8); axD.set_axisbelow(True)
    axD.xaxis.grid(False)

    for ax, lab in zip([axA, axB, axC, axD], ["A", "B", "C", "D"]):
        panel_label(ax, lab)
    fig.tight_layout(w_pad=2.0, h_pad=2.4)
    save(fig, "fig2_accuracy_uncertainty_publication")


# ── FIGURE 1 (workflow schematic) ────────────────────────────────────────────

PANEL_BAR = ["#0072B2", "#009E73", "#E69F00", "#CC79A7"]  # muted top bars


def _box(ax, cx, cy, w, h, text, fc="white", ec="#BDBDBD", fs=7.0):
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.9",
                 fc=fc, ec=ec, lw=1.0))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            color=INK, linespacing=1.3)


def _varrow(ax, cx, y0, y1):
    ax.add_patch(FancyArrowPatch((cx, y0), (cx, y1), arrowstyle="-|>",
                 mutation_scale=8, lw=1.0, color=GREY))


def _harrow(ax, x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                 mutation_scale=10, lw=1.1, color=GREY))


def figure1():
    # Wide, two-column-style canvas with generous breathing room.
    fig, ax = plt.subplots(figsize=(11.6, 4.5))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    # Uniform 3-step modules. Panel A collapses the donor pools into one compact
    # table box so all four panels share the same three-box rhythm.
    panels = [
        ("A", "Reference &\ndonor-aware\nsplit",
         ["PBMC scRNA-seq\n(Hao et al. 2021)",
          "Donor-aware split",
          "Reference: P3 P4 P5 P8\nCalibration: P2 P7\nTest: P1 P6"]),
        ("B", "Pseudo-bulk &\nmarkers",
         ["Pseudo-bulk mixtures\n(held-out donors)",
          "445 marker genes\n(reference-pool DE)",
          "5 cell types\nMonocyte · NK · B\nDC · T_cell"]),
        ("C", "Deconvolution &\nensemble",
         ["NNLS point\nestimate",
          "Ensemble  B = 50\n(gene + cell\nsubsampling)",
          "mean · std · width\nmean_std score"]),
        ("D", "Conformal &\nreliability",
         ["Split conformal\n(calibration set)",
          "Prediction intervals\n80 / 90 / 95 %",
          "Stress test &\nuncertainty rejection"]),
    ]

    # ── uniform panel geometry ──
    L, Rm, gap = 3.0, 3.0, 6.0          # left/right margins, inter-panel gap
    pw = (100 - L - Rm - 3 * gap) / 4   # identical panel width
    card_top, card_bot = 94.0, 5.0
    bar_h = 1.8
    title_y = 85.5                      # title anchor (centered), clear of color bar
    bw = pw - 2.4                       # box width (padding inside card)

    # Shared 3-step vertical grid: identical box CENTERS for every panel.
    bh, bg = 17.0, 7.0                  # uniform box height + gap
    box_centers = [70.0, 70.0 - (bh + bg), 70.0 - 2 * (bh + bg)]  # 70, 46, 22

    centers = []
    for i, (lab, title, boxes) in enumerate(panels):
        px = L + i * (pw + gap)
        cx = px + pw / 2
        centers.append((px, cx))

        # panel card
        ax.add_patch(FancyBboxPatch((px, card_bot), pw, card_top - card_bot,
                     boxstyle="round,pad=0.02,rounding_size=1.2",
                     fc="#FCFCFC", ec="#DcDcDc", lw=1.0))
        # aligned color bar at the very top of the card
        ax.add_patch(plt.Rectangle((px + 0.4, card_top - bar_h - 0.4),
                     pw - 0.8, bar_h, fc=PANEL_BAR[i], ec="none"))
        # panel letter, top-left, outside the card
        ax.text(px - 0.2, 98.0, lab, fontsize=15, fontweight="bold",
                va="center", ha="left", color=INK)
        # title (centered; A is 3-line, B-D are 2-line) — kept narrow so width
        # stays well within ~85% of the card and never touches the border.
        ax.text(cx, title_y, title, fontsize=8.5, fontweight="bold",
                ha="center", va="center", color=INK, linespacing=1.25)

        # three boxes on the shared grid
        for j, (yc, txt) in enumerate(zip(box_centers, boxes)):
            _box(ax, cx, yc, bw, bh, txt, fs=7.5)
            if j > 0:
                _varrow(ax, cx, box_centers[j-1] - bh/2, yc + bh/2)

    # horizontal connector arrows centered in each gap, at mid box-grid height
    mid = box_centers[1]
    for i in range(3):
        px_i, _ = centers[i]
        px_j, _ = centers[i + 1]
        _harrow(ax, px_i + pw + 0.6, px_j - 0.6, mid)

    save(fig, "fig1_workflow_publication")


# ── FIGURE 3 (conformal calibration, 2x2) ────────────────────────────────────

DISPLAY = {"Monocyte": "Monocyte", "NK": "NK", "B": "B", "DC": "DC", "T_cell": "T cell"}


def figure3():
    cd = R / "conformal_marker_5types"
    cov_nom = pd.read_csv(cd / "coverage_by_nominal.csv")
    cov_ct = pd.read_csv(cd / "coverage_by_cell_type.csv")
    cov_ct = cov_ct[cov_ct.nominal_coverage == 0.90].set_index("cell_type")
    iv = pd.read_csv(cd / "intervals_test_clipped.csv")
    iv90 = iv[iv.nominal_coverage == 0.90]
    es = pd.read_csv(R / "ensemble_marker_5types" / "ensemble_summary_test.csv")
    ens_w = es.groupby("cell_type")["interval_width"].mean()
    conf_w = iv90.groupby("cell_type")["interval_width_clip"].mean()

    labels = [DISPLAY[c] for c in CANON]
    colors = [CT_COLORS[c] for c in CANON]
    n_test = 500

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.6))
    (axA, axB), (axC, axD) = axes

    # ── A. Coverage calibration (clipped only; raw≈clip) ──
    nom = cov_nom["nominal_coverage"].values
    clip = cov_nom["empirical_coverage_clip"].values
    lo, hi = 0.76, 0.99
    axA.plot([lo, hi], [lo, hi], ls="--", lw=1.0, color="#C8C8C8", zorder=1)
    axA.text(0.945, 0.905, "ideal", color="#B5B5B5", fontsize=7, rotation=45,
             ha="center", va="center", rotation_mode="anchor")
    axA.plot(nom, clip, "-", lw=1.3, color="#0072B2", zorder=3)
    axA.scatter(nom, clip, s=42, color="#0072B2", edgecolors="white",
                linewidths=0.8, zorder=4)
    # direct label on the blue curve, above the curve (open area)
    axA.text(0.915, 0.967, "Conformal", color="#0072B2", fontsize=8,
             fontweight="bold", ha="center", va="bottom")
    for x, y in zip(nom, clip):
        axA.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                     xytext=(9, -10), fontsize=7, color=INK)
    axA.set_xlim(lo, hi); axA.set_ylim(lo, hi); axA.set_aspect("equal")
    axA.set_xticks([0.80, 0.90, 0.95]); axA.set_yticks([0.80, 0.90, 0.95])
    axA.set_xlabel("Nominal coverage"); axA.set_ylabel("Empirical coverage")
    axA.set_title("Coverage calibration", fontsize=9)
    axA.text(0.928, 0.822, "conservative\ncoverage", fontsize=6.8, color="#9A9A9A",
             style="italic", ha="center", va="center")
    axA.grid(False)

    # ── B. Coverage by cell type (lollipop + binomial CI) ──
    x = np.arange(len(CANON))
    p = np.array([cov_ct.loc[c, "empirical_coverage_clip"] for c in CANON])
    ci = 1.96 * np.sqrt(p * (1 - p) / n_test)
    axB.axhline(0.90, ls="--", lw=1.0, color="#777777", zorder=1)
    axB.text(len(CANON) - 0.5, 0.905, "Nominal 90%", fontsize=7, color="#777777",
             ha="right", va="bottom")
    for xi, pi, ci_i, col in zip(x, p, ci, colors):
        axB.plot([xi, xi], [0.86, pi], lw=1.4, color=col, zorder=2, alpha=0.55)
        axB.errorbar(xi, pi, yerr=ci_i, fmt="o", ms=7, color=col,
                     ecolor=col, elinewidth=1.1, capsize=3, zorder=3,
                     markeredgecolor="white", markeredgewidth=0.8)
        axB.annotate(f"{pi:.3f}", (xi, pi + ci_i), textcoords="offset points",
                     xytext=(0, 5), ha="center", fontsize=7, color=INK)
    axB.set_xticks(x); axB.set_xticklabels(labels)
    axB.set_ylim(0.86, 1.03); axB.set_ylabel("Empirical coverage @ 90%")
    axB.set_title("Coverage by cell type", fontsize=9)
    axB.yaxis.grid(True, color="#ECECEC", lw=0.8); axB.set_axisbelow(True)
    axB.xaxis.grid(False)

    # ── C. Interval width by cell type (colored box + soft jitter) ──
    data = [iv90[iv90.cell_type == c]["interval_width_clip"].values for c in CANON]
    bp = axC.boxplot(data, positions=x, widths=0.55, patch_artist=True,
                     showfliers=False, medianprops=dict(color=INK, lw=1.2),
                     whiskerprops=dict(color="#888888", lw=1.0),
                     capprops=dict(color="#888888", lw=1.0),
                     boxprops=dict(lw=1.0))
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col); patch.set_alpha(0.40); patch.set_edgecolor(col)
    rng = np.random.default_rng(0)
    for xi, vals, col in zip(x, data, colors):
        jx = xi + (rng.random(len(vals)) - 0.5) * 0.28
        axC.scatter(jx, vals, s=5, color=col, alpha=0.18, edgecolors="none", zorder=3)
    axC.set_xticks(x); axC.set_xticklabels(labels)
    axC.set_ylim(0.14, 0.45)
    axC.set_ylabel("90% interval width"); axC.set_title("Interval width by cell type", fontsize=9)
    axC.yaxis.grid(True, color="#ECECEC", lw=0.8); axC.set_axisbelow(True)
    axC.xaxis.grid(False)

    # ── D. Conformal expansion of uncertainty (paired dots) ──
    xr, xc = 0.0, 1.0
    for c, col in zip(CANON, colors):
        ew, cw = ens_w[c], conf_w[c]
        axD.plot([xr, xc], [ew, cw], "-", lw=1.0, color=col, alpha=0.7, zorder=2)
        axD.scatter([xr], [ew], s=40, color="#B0B0B0", edgecolors="white",
                    linewidths=0.8, zorder=3)
        axD.scatter([xc], [cw], s=46, color=col, edgecolors="white",
                    linewidths=0.8, zorder=4)
        axD.annotate(f"{DISPLAY[c]}  ×{cw/ew:.1f}", (xc, cw),
                     textcoords="offset points", xytext=(9, 0), va="center",
                     fontsize=7, color=col)
    axD.set_xlim(-0.4, 1.9); axD.set_xticks([xr, xc])
    axD.set_xticklabels(["Raw ensemble\nspread", "Conformal\ninterval"])
    axD.set_ylim(0, max(conf_w.max(), ens_w.max()) * 1.15)
    axD.set_ylabel("Mean interval width")
    axD.set_title("Raw → conformal expansion", fontsize=9)
    axD.text(0.5, 0.97, "Raw coverage = 0.390 → Conformal coverage = 0.950",
             transform=axD.transAxes, ha="center", va="top", fontsize=7,
             color=INK, style="italic")
    axD.yaxis.grid(True, color="#ECECEC", lw=0.8); axD.set_axisbelow(True)
    axD.xaxis.grid(False)

    for ax, lab in zip([axA, axB, axC, axD], ["A", "B", "C", "D"]):
        panel_label(ax, lab)
    fig.tight_layout(w_pad=2.4, h_pad=2.6)
    save(fig, "fig3_conformal_publication")


# ── FIGURE 4 (stress tests, 2x2) ─────────────────────────────────────────────

PERT_COLOR = {"gaussian": "#4C72B0", "dropout": "#C44E52", "low_depth": "#55A868"}
PERT_LABEL = {"gaussian": "Gaussian noise", "dropout": "Dropout", "low_depth": "Low depth"}
BASE_GREY = "#4D4D4D"


def figure4():
    s = pd.read_csv(R / "stress_marker_5types" / "stress_summary_tier1_corrected.csv")
    rej = pd.read_csv(R / "stress_marker_5types" / "rejection_curves_tier1_corrected.csv")
    rej = rej[rej.score_name == "mean_std"]

    base = s[s.perturbation == "none"].iloc[0]
    fams = ["gaussian", "dropout", "low_depth"]

    def series(fam, ycol):
        d = s[s.perturbation == fam].sort_values("severity")
        # anchor each family at the shared baseline (severity 0)
        xs = [0.0] + d["severity"].tolist()
        ys = [base[ycol]] + d[ycol].tolist()
        return xs, ys

    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.6))
    (axA, axB), (axC, axD) = axes

    # ── A. Prediction error under perturbation ──
    for fam in fams:
        xs, ys = series(fam, "MAE")
        axA.plot(xs, ys, "-o", lw=1.3, ms=5, color=PERT_COLOR[fam],
                 label=PERT_LABEL[fam], markeredgecolor="white", markeredgewidth=0.6)
    axA.set_xlabel("Perturbation severity"); axA.set_ylabel("Test MAE")
    axA.set_title("Prediction error under perturbation", fontsize=9)
    axA.legend(frameon=False, loc="upper left", handletextpad=0.5)
    axA.yaxis.grid(True, color="#ECECEC", lw=0.8); axA.set_axisbelow(True)
    axA.xaxis.grid(False)

    # ── B. Ensemble uncertainty under perturbation ──
    for fam in fams:
        xs, ys = series(fam, "mean_uncertainty_std")
        axB.plot(xs, ys, "-o", lw=1.3, ms=5, color=PERT_COLOR[fam],
                 label=PERT_LABEL[fam], markeredgecolor="white", markeredgewidth=0.6)
    axB.set_xlabel("Perturbation severity"); axB.set_ylabel("Mean ensemble SD")
    axB.set_title("Ensemble uncertainty under perturbation", fontsize=9)
    axB.legend(frameon=False, loc="upper left", handletextpad=0.5)
    axB.yaxis.grid(True, color="#ECECEC", lw=0.8); axB.set_axisbelow(True)
    axB.xaxis.grid(False)

    # ── C. Conformal coverage under perturbation ──
    axC.axhspan(0.80, 0.90, color="#F2D9D9", alpha=0.5, zorder=0)  # under-coverage cue
    for fam in fams:
        xs, ys = series(fam, "coverage90_clip")
        axC.plot(xs, ys, "-o", lw=1.3, ms=5, color=PERT_COLOR[fam],
                 label=PERT_LABEL[fam], markeredgecolor="white", markeredgewidth=0.6, zorder=3)
    axC.axhline(0.90, ls="--", lw=1.0, color="#555555", zorder=2)
    axC.text(0.02, 0.902, "Nominal 90%", fontsize=7, color="#555555", va="bottom")
    axC.set_ylim(0.80, 0.97)
    axC.set_xlabel("Perturbation severity"); axC.set_ylabel("Coverage @ 90%")
    axC.set_title("Conformal coverage under perturbation", fontsize=9)
    axC.legend(frameon=False, loc="lower left", handletextpad=0.5)
    axC.yaxis.grid(True, color="#ECECEC", lw=0.8); axC.set_axisbelow(True)
    axC.xaxis.grid(False)

    # ── D. Uncertainty-guided rejection (representative curves) ──
    # Same color map as A-C: Gaussian blue, Dropout red, Low depth green,
    # Baseline neutral dark grey. Label stagger follows true endpoint order
    # (baseline sits just above low-depth) to avoid crossed/mislabeled curves.
    reps = [("dropout_high", PERT_COLOR["dropout"], "Dropout (high)", 0),
            ("gaussian_noise_high", PERT_COLOR["gaussian"], "Gaussian noise (high)", 0),
            ("baseline", BASE_GREY, "Baseline", 6),
            ("low_depth_high", PERT_COLOR["low_depth"], "Low depth (high)", -7)]
    for sc, col, lab, dy in reps:
        d = rej[rej.scenario == sc].sort_values("retained_fraction")
        axD.plot(d["retained_fraction"], d["reject_high_mae"], "-o", lw=1.3, ms=4,
                 color=col, markeredgecolor="white", markeredgewidth=0.5)
        x_end = d.retained_fraction.max()
        y_end = d[d.retained_fraction == x_end]["reject_high_mae"].values[0]
        axD.annotate(lab, (x_end, y_end), textcoords="offset points", xytext=(7, dy),
                     ha="left", va="center", fontsize=6.8, color=col, fontweight="bold")
    axD.set_xlim(0.15, 1.55)
    axD.set_ylim(0.055, 0.116)          # headroom above the flat dropout curve
    axD.set_xticks([0.2, 0.4, 0.6, 0.8])
    axD.set_xlabel("Fraction retained"); axD.set_ylabel("Retained-set MAE")
    axD.set_title("Uncertainty-guided rejection", fontsize=9)
    # rejection-direction cue in the clean band above all curves
    axD.annotate("", xy=(0.10, 0.90), xytext=(0.40, 0.90), xycoords=axD.transAxes,
                 arrowprops=dict(arrowstyle="->", color="#999999", lw=1.0))
    axD.text(0.43, 0.90, "More rejection", transform=axD.transAxes,
             fontsize=7, color="#888888", ha="left", va="center")
    axD.yaxis.grid(True, color="#ECECEC", lw=0.8); axD.set_axisbelow(True)
    axD.xaxis.grid(False)

    for ax, lab in zip([axA, axB, axC, axD], ["A", "B", "C", "D"]):
        panel_label(ax, lab)
    fig.tight_layout(w_pad=2.6, h_pad=2.8)
    save(fig, "fig4_stress_publication")


# ── FIGURE 5 (reference reduction, 2x2) ──────────────────────────────────────

C_BASE = "#5B6770"     # neutral gray-blue baseline
C_REF = "#7E4FA0"      # reference reduction = purple
C_DROP = "#C44E52"     # high dropout = red


def figure5():
    import matplotlib.gridspec as gridspec
    t1 = pd.read_csv(R / "stress_marker_5types" / "stress_summary_tier1_corrected.csv").set_index("scenario")
    t2 = pd.read_csv(R / "stress_marker_5types_tier2_subset" / "stress_summary_tier2_subset.csv").set_index("scenario")
    b, dh, rr = t1.loc["baseline"], t1.loc["dropout_high"], t2.loc["reference_reduction_1"]
    n_test = 500

    fig = plt.figure(figsize=(7.6, 6.6))
    outer = gridspec.GridSpec(2, 2, figure=fig, wspace=0.38, hspace=0.42)

    # ── A. Accuracy (two sub-axes: MAE | CCC) ──
    gsA = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0, 0], wspace=0.62)
    axA1 = fig.add_subplot(gsA[0]); axA2 = fig.add_subplot(gsA[1])
    conds = ["Baseline\n(5-type)", "DC-removed\n(4-type)"]
    ccols = [C_BASE, C_REF]
    for ax, vals, ylab, ttl in [
            (axA1, [b.MAE, rr.MAE], "Test MAE", "MAE"),
            (axA2, [b.CCC, rr.CCC], "CCC", "CCC")]:
        ax.bar([0, 1], vals, width=0.6, color=ccols, edgecolor="white", linewidth=0.8)
        for xi, v in zip([0, 1], vals):
            ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=6.8, color=INK)
        ax.set_xticks([0, 1]); ax.set_xticklabels(conds, fontsize=6.8)
        ax.set_ylabel(ylab); ax.set_title(ttl, fontsize=8)
        ax.set_ylim(0, max(vals) * 1.22)
        ax.yaxis.grid(True, color="#ECECEC", lw=0.8); ax.set_axisbelow(True); ax.xaxis.grid(False)
    axA1.text(1.35, 1.24, "Accuracy under DC-removed reference ablation", transform=axA1.transAxes,
              fontsize=9, fontweight="bold", ha="center", va="bottom", color=INK)

    # ── B. Coverage after recalibration (lollipop + binomial CI) ──
    axB = fig.add_subplot(outer[0, 1])
    cov = [b.coverage90_clip, rr.coverage90_clip]
    p = np.array(cov); ci = 1.96 * np.sqrt(p * (1 - p) / n_test)
    axB.axhline(0.90, ls="--", lw=1.0, color="#777777", zorder=1)
    axB.text(-0.45, 0.904, "Nominal 90%", fontsize=7, color="#777777", ha="left", va="bottom")
    for xi, pi, ci_i, col in zip([0, 1], p, ci, ccols):
        axB.plot([xi, xi], [0.88, pi], lw=1.4, color=col, alpha=0.55, zorder=2)
        axB.errorbar(xi, pi, yerr=ci_i, fmt="o", ms=8, color=col, ecolor=col,
                     elinewidth=1.1, capsize=3, markeredgecolor="white",
                     markeredgewidth=0.8, zorder=3)
        axB.annotate(f"{pi:.3f}", (xi, pi + ci_i), textcoords="offset points",
                     xytext=(0, 5), ha="center", fontsize=7, color=INK)
    axB.set_xlim(-0.5, 1.5); axB.set_xticks([0, 1]); axB.set_xticklabels(conds)
    axB.set_ylim(0.88, 1.0); axB.set_ylabel("Coverage @ 90%")
    axB.set_title("Coverage after recalibration", fontsize=9)
    axB.yaxis.grid(True, color="#ECECEC", lw=0.8); axB.set_axisbelow(True); axB.xaxis.grid(False)

    # ── C. Reliability signal (dumbbell) ──
    axC = fig.add_subplot(outer[1, 0])
    ue = [b.uncertainty_error_corr, rr.uncertainty_error_corr]
    axC.plot([0, 1], ue, "-", lw=1.6, color="#BBBBBB", zorder=1)
    for xi, v, col in zip([0, 1], ue, ccols):
        axC.scatter(xi, v, s=80, color=col, edgecolors="white", linewidths=0.9, zorder=3)
        axC.annotate(f"{v:.2f}", (xi, v), textcoords="offset points",
                     xytext=(0, 11), ha="center", fontsize=8, color=col, fontweight="bold")
    axC.set_xlim(-0.5, 1.5); axC.set_xticks([0, 1]); axC.set_xticklabels(conds)
    axC.set_ylim(0, 0.75); axC.set_ylabel("Sample-level Pearson r\n(mean SD vs sample MAE)")
    axC.set_title("Reliability signal weakens under\nreference reduction", fontsize=9)
    axC.yaxis.grid(True, color="#ECECEC", lw=0.8); axC.set_axisbelow(True); axC.xaxis.grid(False)

    # ── D. Error reduction after rejection (positive = improvement) ──
    axD = fig.add_subplot(outer[1, 1])
    names = ["Baseline", "DC-removed\n(4-type)", "High\ndropout"]
    cols = [C_BASE, C_REF, C_DROP]
    impr = [-b.reject_high_delta_vs_all, -rr.reject_high_delta_vs_all, -dh.reject_high_delta_vs_all]
    axD.bar(np.arange(3), impr, width=0.62, color=cols, edgecolor="white", linewidth=0.8)
    for xi, v in zip(np.arange(3), impr):
        axD.text(xi, v + 0.0004, f"{v:.3f}", ha="center", va="bottom", fontsize=7, color=INK)
    axD.axhline(0, color=INK, lw=0.8)
    axD.set_xticks(np.arange(3)); axD.set_xticklabels(names)
    axD.set_ylim(0, max(impr) * 1.2)
    axD.set_ylabel("MAE reduction after rejection")
    axD.set_title("Error reduction after rejection", fontsize=9)
    axD.yaxis.grid(True, color="#ECECEC", lw=0.8); axD.set_axisbelow(True); axD.xaxis.grid(False)

    # consistent panel letters just outside each panel's upper-left (figure coords)
    for ax, lab in [(axA1, "A"), (axB, "B"), (axC, "C"), (axD, "D")]:
        pos = ax.get_position()
        fig.text(pos.x0 - 0.052, pos.y1 + 0.028, lab, fontsize=13,
                 fontweight="bold", va="bottom", ha="left", color=INK)

    save(fig, "fig5_reference_reduction_publication")


def main():
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()


if __name__ == "__main__":
    main()

