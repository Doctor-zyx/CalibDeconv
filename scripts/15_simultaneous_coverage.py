#!/usr/bin/env python
"""
Composition-level simultaneous coverage for the frozen CalibDeconv outputs.

Post-hoc scoring of frozen result CSVs already on disk. Two quantities:

  1. Cell-type-wise (marginal) conformal intervals are calibrated. How often is
     the full 5-component composition vector covered *simultaneously*?
  2. Under a Bonferroni-adjusted family-wise extension, what is the simultaneous
     coverage, and what is the cost in interval width?

The family-wise construction is Bonferroni: for a family-wise level 1-alpha over
K=5 cell types, each cell type is calibrated at 1-alpha/K. The conformal radius
is the ceil((n+1)*(1-alpha/K))-th order statistic of the calibration-set
absolute-error nonconformity scores.

The construction is calibration-set-defined: alpha, K, the score function and
the order statistic determine the radius from calibration data alone, and test
truth enters only in the final scoring step. That is a statement about which
data the computation reads, not about when the analysis was planned.

No other simultaneous construction is computed here. Alternatives (global
max-score, standardised max-score, simplex-aware regions) are not reported.

The script also emits the pairwise phi correlations between per-cell-type miss
indicators. These are descriptive: they exist so that the report's claim about
the relationship between simultaneous coverage and the product of the marginals
is reproducible, and specifically so that the observed numerical agreement is not
mis-read as evidence that the miss events are independent.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
R = PROJECT_ROOT / 'results'
OUT = R / 'simultaneous_coverage'
CANON = ['Monocyte', 'NK', 'B', 'DC', 'T_cell']
K = len(CANON)
NOMINALS = [0.80, 0.90, 0.95]
FAMILY_TARGET = 0.90          # family-wise simultaneous level
ALPHA_K = (1 - FAMILY_TARGET) / K


def wilson(k, n, z=1.959963984540054):
    """Wilson score interval -- appropriate here because coverage runs near 1."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load():
    truth = pd.read_csv(
        PROJECT_ROOT / 'data/processed/true_proportions_test_5type.csv',
        index_col=0)[CANON]

    iv = pd.read_csv(R / 'conformal_marker_5types/intervals_test_clipped.csv')
    ens = pd.read_csv(R / 'ensemble_marker_5types/ensemble_summary_test.csv')
    scores = pd.read_csv(
        R / 'conformal_marker_5types/nonconformity_scores.csv',
        index_col=0)[CANON]
    return truth, iv, ens, scores


def wide(df, col, index):
    return df.pivot(index='sample_id', columns='cell_type')[col].loc[index, CANON]


def score(truth, lower, upper):
    cov = (truth.values >= lower) & (truth.values <= upper)
    return cov, cov.all(axis=1)


def main():
    truth, iv, ens, scores = load()
    idx = truth.index
    n_test = len(idx)
    rows, per_type_rows = [], []
    cov_by_nominal = {}

    # ---- raw bootstrap ensemble q05-q95 (never designed for any coverage target)
    lo = wide(ens, 'q0.05', idx).values
    hi = wide(ens, 'q0.95', idx).values
    cov, joint = score(truth, lo, hi)
    lo_ci, hi_ci = wilson(joint.sum(), n_test)
    rows.append(dict(interval='raw_bootstrap_q05_q95', nominal=np.nan,
                     target_type='none', marginal=cov.mean(),
                     simultaneous=joint.mean(), simultaneous_n=int(joint.sum()),
                     simultaneous_lo=lo_ci, simultaneous_hi=hi_ci,
                     mean_width=(hi - lo).mean(),
                     product_of_marginals=np.prod(cov.mean(axis=0))))

    # ---- marginal conformal at each nominal level, as reported in v1.1
    for nom in NOMINALS:
        sub = iv[iv.nominal_coverage == nom]
        L = wide(sub, 'lower_clip', idx).values
        U = wide(sub, 'upper_clip', idx).values
        cov, joint = score(truth, L, U)
        lo_ci, hi_ci = wilson(joint.sum(), n_test)
        rows.append(dict(interval='marginal_conformal', nominal=nom,
                         target_type='marginal_per_cell_type',
                         marginal=cov.mean(), simultaneous=joint.mean(),
                         simultaneous_n=int(joint.sum()),
                         simultaneous_lo=lo_ci, simultaneous_hi=hi_ci,
                         mean_width=(U - L).mean(),
                         product_of_marginals=np.prod(cov.mean(axis=0))))
        if nom == FAMILY_TARGET:
            for i, ct in enumerate(CANON):
                per_type_rows.append(dict(
                    interval='marginal_conformal', cell_type=ct,
                    coverage=cov[:, i].mean(), n_miss=int((~cov[:, i]).sum()),
                    mean_width=(U - L)[:, i].mean(),
                    true_mean=truth[ct].mean()))
            marg_cov = cov
        cov_by_nominal[nom] = cov

    # ---- Bonferroni-adjusted family-wise extension (calibration-set-defined)
    n_cal = len(scores)
    order = int(np.ceil((n_cal + 1) * (1 - ALPHA_K)))
    if order > n_cal:
        raise RuntimeError(
            f'Bonferroni order statistic {order} exceeds calibration size '
            f'{n_cal}; the target is not attainable at this sample size.')
    radius = np.array([np.sort(scores[ct].values)[order - 1] for ct in CANON])

    centre = wide(iv[iv.nominal_coverage == FAMILY_TARGET], 'mean', idx).values
    Lb = np.clip(centre - radius, 0, 1)
    Ub = np.clip(centre + radius, 0, 1)
    cov_b, joint_b = score(truth, Lb, Ub)
    lo_ci, hi_ci = wilson(joint_b.sum(), n_test)
    rows.append(dict(interval='bonferroni_simultaneous_conformal',
                     nominal=FAMILY_TARGET, target_type='family_wise',
                     marginal=cov_b.mean(), simultaneous=joint_b.mean(),
                     simultaneous_n=int(joint_b.sum()),
                     simultaneous_lo=lo_ci, simultaneous_hi=hi_ci,
                     mean_width=(Ub - Lb).mean(),
                     product_of_marginals=np.prod(cov_b.mean(axis=0))))
    for i, ct in enumerate(CANON):
        per_type_rows.append(dict(
            interval='bonferroni_simultaneous_conformal', cell_type=ct,
            coverage=cov_b[:, i].mean(), n_miss=int((~cov_b[:, i]).sum()),
            mean_width=(Ub - Lb)[:, i].mean(), true_mean=truth[ct].mean()))

    # ---- miss-pattern decomposition: which cell types drive the joint failure
    pattern_rows = []
    for tag, c in (('marginal_conformal', marg_cov),
                   ('bonferroni_simultaneous_conformal', cov_b)):
        nmiss = (~c).sum(axis=1)
        for k in range(K + 1):
            if (nmiss == k).sum():
                pattern_rows.append(dict(interval=tag, n_types_missed=k,
                                         n_samples=int((nmiss == k).sum())))
        for i, ct in enumerate(CANON):
            only = ((~c[:, i]) & (nmiss == 1)).sum()
            pattern_rows.append(dict(interval=tag, n_types_missed='only_' + ct,
                                     n_samples=int(only)))

    # ---- miss-indicator dependence: descriptive, and deliberately reported
    # because "simultaneous ~= product of marginals" is NOT evidence of
    # independence. Offsetting positive and negative dependencies reproduce the
    # same product.
    corr_rows = []
    targets = ([(f'marginal_conformal@{n:.2f}', cov_by_nominal[n])
                for n in NOMINALS]
               + [('bonferroni_simultaneous_conformal', cov_b)])
    for tag, c in targets:
        miss = (~c).astype(float)
        for i, j in itertools.combinations(range(K), 2):
            a, b = miss[:, i], miss[:, j]
            phi = (np.nan if a.std() == 0 or b.std() == 0
                   else float(np.corrcoef(a, b)[0, 1]))
            corr_rows.append(dict(interval=tag, type_a=CANON[i], type_b=CANON[j],
                                  phi=phi, n_both_miss=int((a * b).sum())))

    OUT.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    pd.DataFrame(corr_rows).to_csv(OUT / 'miss_correlations.csv', index=False)
    pd.DataFrame([dict(cell_type=ct, alpha_k=ALPHA_K, order_statistic=order,
                       n_cal=n_cal, radius=radius[i])
                  for i, ct in enumerate(CANON)]).to_csv(
        OUT / 'bonferroni_radii.csv', index=False)
    summary.to_csv(OUT / 'coverage_summary.csv', index=False)
    pd.DataFrame(per_type_rows).to_csv(OUT / 'coverage_by_cell_type.csv',
                                       index=False)
    pd.DataFrame(pattern_rows).to_csv(OUT / 'miss_patterns.csv', index=False)

    pd.set_option('display.width', 200)
    print(f'test mixtures: {n_test}   cell types: {K}   '
          f'calibration samples: {n_cal}')
    print(f'Bonferroni: alpha_k={ALPHA_K:.4f}  order statistic={order}\n')
    print(summary[['interval', 'nominal', 'marginal', 'simultaneous',
                   'simultaneous_n', 'product_of_marginals',
                   'mean_width']].to_string(index=False))

    print('\nWilson 95% intervals for simultaneous coverage '
          '(quote these, do not round by hand):')
    for _, r in summary.iterrows():
        nom = '   -' if pd.isna(r.nominal) else f'{r.nominal:.2f}'
        print(f'  {r.interval:34s} {nom}  {r.simultaneous * 100:5.1f}%  '
              f'[{r.simultaneous_lo * 100:.1f}, {r.simultaneous_hi * 100:.1f}]')

    cdf = pd.DataFrame(corr_rows)
    print('\nMiss-indicator phi correlations, largest |phi| per interval '
          '(evidence AGAINST independence):')
    for tag, grp in cdf.groupby('interval', sort=False):
        top = grp.reindex(grp.phi.abs().sort_values(ascending=False).index).head(3)
        s = '; '.join(f'{r.type_a}x{r.type_b} {r.phi:+.3f}' for _, r in top.iterrows())
        print(f'  {tag:34s} {s}')

    print(f'\nwrote 5 CSVs to {OUT.relative_to(PROJECT_ROOT)}')


if __name__ == '__main__':
    main()
