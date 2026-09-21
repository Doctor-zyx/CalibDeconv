#!/usr/bin/env python
"""
GSE107572 real-bulk PBMC descriptive analysis.

Scores the frozen CalibDeconv pipeline against the nine GSE107572 PBMC donors
with matched flow-cytometry fractions, on the native five-type basis used
throughout the manuscript. Two outputs:

  1. Point-estimation agreement (MAE, CCC, Pearson, RMSE), overall and per
     cell type.
  2. Descriptive coverage of the frozen conformal intervals, at each nominal
     level and under the Bonferroni-adjusted family-wise extension.

This is scoring, not calibration. The conformal radii are the ones already
frozen from the pseudo-bulk calibration set (500 samples, absolute-error
nonconformity scores) and are re-used here unchanged; nothing is re-fitted,
re-selected or tuned on real-bulk data, and the GSE107572 ground truth enters
only in the final metric and coverage counts.

The coverage figures are descriptive only. Split conformal guarantees marginal
coverage under exchangeability between the calibration and target samples.
Cross-cohort transfer to an independent cohort on a different platform plainly
violates that assumption, so no coverage guarantee carries over; the numbers
quantify how the frozen intervals behave off-assumption, at n = 9.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
R = PROJECT_ROOT / 'results'
OUT = R / 'gse107572_real_bulk'
CANON = ['Monocyte', 'NK', 'B', 'DC', 'T_cell']
K = len(CANON)
NOMINALS = [0.80, 0.90, 0.95]
FAMILY_TARGET = 0.90


def ccc(x, y):
    """Lin's concordance correlation coefficient."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mx, my = x.mean(), y.mean()
    return 2 * np.mean((x - mx) * (y - my)) / (x.var() + y.var() + (mx - my) ** 2)


def radii(scores, alpha):
    """Frozen split-conformal radius: ceil((n+1)*(1-alpha))-th order statistic."""
    n = len(scores)
    order = int(np.ceil((n + 1) * (1 - alpha)))
    if order > n:
        raise RuntimeError(f'order statistic {order} exceeds calibration n={n}')
    return np.array([np.sort(scores[c].values)[order - 1] for c in CANON]), order


def main():
    gt = pd.read_csv(R / 'phase6b_gse107572/ground_truth_5type.csv', index_col=0)
    pred = pd.read_csv(R / 'phase6b_gse107572/predicted_proportions.csv',
                       index_col=0)
    gt.index = gt.index.astype(str)
    pred.index = pred.index.astype(str)
    pred = pred.loc[gt.index]
    gt, pred = gt[CANON], pred[CANON]
    n = len(gt)

    # ---- point-estimation agreement
    t, p = gt.values.ravel(), pred.values.ravel()
    overall = pd.DataFrame([dict(
        cohort='GSE107572', basis='5type', n_donors=n, n_types=K,
        MAE=np.abs(t - p).mean(), CCC=ccc(t, p),
        Pearson=np.corrcoef(t, p)[0, 1],
        RMSE=np.sqrt(((t - p) ** 2).mean()))])

    per_type = pd.DataFrame([dict(
        cohort='GSE107572', basis='5type', cell_type=c,
        true_mean=gt[c].mean(), pred_mean=pred[c].mean(),
        mean_signed_deviation=pred[c].mean() - gt[c].mean(),
        MAE=np.abs(gt[c] - pred[c]).mean()) for c in CANON])

    # ---- descriptive coverage of the frozen conformal radii
    scores = pd.read_csv(R / 'conformal_marker_5types/nonconformity_scores.csv',
                         index_col=0)[CANON]
    rows = []
    specs = ([('marginal_conformal', nom, 1 - nom) for nom in NOMINALS]
             + [('bonferroni_simultaneous_conformal', FAMILY_TARGET,
                 (1 - FAMILY_TARGET) / K)])
    for tag, nom, alpha in specs:
        rad, order = radii(scores, alpha)
        lo = np.clip(pred.values - rad, 0, 1)
        hi = np.clip(pred.values + rad, 0, 1)
        cov = (gt.values >= lo) & (gt.values <= hi)
        joint = cov.all(axis=1)
        rows.append(dict(cohort='GSE107572', basis='5type', interval=tag,
                         nominal=nom, order_statistic=order, n_donors=n,
                         marginal=cov.mean(), simultaneous=joint.mean(),
                         simultaneous_n=int(joint.sum()),
                         mean_width=(hi - lo).mean()))
    coverage = pd.DataFrame(rows)

    OUT.mkdir(parents=True, exist_ok=True)
    overall.to_csv(OUT / 'metrics_overall.csv', index=False)
    per_type.to_csv(OUT / 'metrics_per_celltype.csv', index=False)
    coverage.to_csv(OUT / 'conformal_coverage.csv', index=False)

    pd.set_option('display.width', 200)
    print(f'GSE107572 donors: {n}   cell types: {K}   '
          f'calibration samples: {len(scores)}\n')
    print(overall.to_string(index=False))
    print()
    print(per_type.to_string(index=False))
    print()
    print(coverage.to_string(index=False))
    print(f'\nwrote 3 CSVs to {OUT.relative_to(PROJECT_ROOT)}')


if __name__ == '__main__':
    main()
