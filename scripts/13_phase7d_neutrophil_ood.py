#!/usr/bin/env python
"""Phase 7D: Neutrophil OOD mechanism simulation."""
import numpy as np, pandas as pd, sys, os, importlib.util, gzip
from pathlib import Path
from scipy.optimize import nnls
from scipy import stats
import scanpy as sc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.deconvolution.signature import build_signature_matrix
from src.data.pseudobulk import normalize_signature
from src.uncertainty.conformal import calibrate, predict_intervals, evaluate_calibration

PROJECT = Path(__file__).resolve().parents[1]
os.makedirs(PROJECT / 'results' / 'phase7_neutrophil_ood', exist_ok=True)
CANON = ['Monocyte', 'NK', 'B', 'DC', 'T_cell']
NEUTRO_FRACS = [0.0, 0.10, 0.25, 0.50, 0.70]
N_SAMPLES = 150  # per contamination level
CELLS_PER_SAMPLE = 500
SEED = 42

# Import ensemble core
spec = importlib.util.spec_from_file_location('e', str(PROJECT / 'scripts' / '04b_ensemble_marker5.py'))
_ens = importlib.util.module_from_spec(spec); spec.loader.exec_module(_ens)
logger = type('L', (), {'info': lambda *a: None})()


def ccc(t, p):
    t = t.ravel(); p = p.ravel(); mt, mp = t.mean(), p.mean()
    d = t.var() + p.var() + (mt - mp) ** 2
    return float(2 * ((t - mt) * (p - mp)).mean() / d) if d > 0 else np.nan


# ── 1. Load frozen reference + signature ──
panel = [l.strip() for l in open(PROJECT / 'data/processed/selected_genes_markers.txt') if l.strip()]
ref = sc.read_h5ad(PROJECT / 'data/processed/cell_pool_reference.h5ad')
ref.obs['ct5'] = ref.obs['cell_type'].map(
    lambda x: 'T_cell' if x in ['CD4_T', 'CD8_T', 'other_T'] else x).astype('category')
sig_full, _ = build_signature_matrix(ref, 'ct5', cell_types=sorted(ref.obs['ct5'].unique()))
sig_full = normalize_signature(sig_full, method='cpm')

# ── 2. Build neutrophil profile from GSE60424 sorted samples ──
# Map Ensembl → HGNC using our reference
ens2hgnc = dict(zip(ref.var['ensembl_id'], ref.var_names))
expr = pd.read_csv(PROJECT / 'data/real_bulk/gse60424/counts.txt.gz', sep='\t', index_col=0, compression='gzip')

# Identify neutrophil libs
lines_d, titles_all = {}, []
with gzip.open(str(PROJECT / 'data/real_bulk/gse60424/series_matrix.txt.gz'), 'rt', errors='ignore') as f:
    for line in f:
        if line.startswith('!Sample_title'):
            titles_all = [x.strip().strip('"') for x in line.rstrip().split('\t')[1:]]
        elif line.startswith('!Sample_characteristics_ch1'):
            cells = [x.strip().strip('"') for x in line.rstrip().split('\t')[1:]]
            key = cells[0].split(':')[0].strip()
            if key not in lines_d: lines_d[key] = []
            lines_d[key].append(cells)
gsm_data = [{} for _ in range(len(titles_all))]
for key, rows in lines_d.items():
    for row in rows:
        for i, cell in enumerate(row):
            parts = cell.split(':', 1)
            if len(parts) == 2: gsm_data[i][parts[0].strip()] = parts[1].strip()
neutro_libs = [titles_all[i] for i in range(len(titles_all)) if gsm_data[i].get('celltype') == 'Neutrophils']

# Average neutrophil profile, map to HGNC, normalize to CPM
neutro_raw = expr[neutro_libs].mean(axis=1)  # mean across 20 donors
neutro_raw.index = neutro_raw.index.map(lambda x: ens2hgnc.get(x, None))
neutro_raw = neutro_raw[neutro_raw.index.notna()]
neutro_raw = neutro_raw[~neutro_raw.index.duplicated(keep='first')]
neutro_cpm = neutro_raw / neutro_raw.sum() * 1e6  # CPM
overlap = [g for g in panel if g in sig_full.index and g in neutro_cpm.index]
print(f'Neutro profile: {len(neutro_raw)} genes; marker overlap: {len(overlap)}/445')

# Restrict everything to overlapping markers
sig = sig_full.loc[overlap, CANON]
neutro_vec = neutro_cpm.loc[overlap].values  # (n_markers,) CPM

# ── 3. Generate PBMC pseudo-bulk + ensemble arrays ──
arrs, tots = _ens.build_panel_celltype_arrays(ref, 'ct5', sorted(ref.obs['ct5'].unique()), overlap, logger)
# Also need PBMC pseudo-bulk from Hao test set restricted to overlap
Xt_full = pd.read_csv(PROJECT / 'data/processed/pseudobulk_matrix_test_cpm.csv', index_col=0)
Xt_panel = Xt_full[[g for g in overlap if g in Xt_full.columns]]
# Use first N_SAMPLES from frozen test as base PBMC pseudo-bulk
Xbase = Xt_panel.iloc[:N_SAMPLES].values  # (N_SAMPLES, n_markers)
yt = pd.read_csv(PROJECT / 'data/processed/true_proportions_test_5type.csv', index_col=0)[CANON].iloc[:N_SAMPLES]
print(f'Base PBMC pseudo-bulk: {Xbase.shape}, true props: {yt.shape}')

# ── 4. Simulate contamination levels ──
all_rows = []
for nf in NEUTRO_FRACS:
    # Mix: (1-nf)*PBMC + nf*neutro
    X_contam = (1 - nf) * Xbase + nf * neutro_vec[None, :]
    X_df = pd.DataFrame(X_contam, index=yt.index, columns=overlap)

    # NNLS point estimate
    P = np.zeros((N_SAMPLES, len(CANON)))
    for i in range(N_SAMPLES):
        x, _ = nnls(sig.values, X_contam[i])
        s = x.sum(); P[i] = x / s if s > 0 else x
    pred = pd.DataFrame(P, index=yt.index, columns=CANON)

    # Ensemble (B=30 for speed)
    preds_3d = _ens.fast_ensemble(X_contam.astype(float), arrs, tots,
                                  sorted(ref.obs['ct5'].unique()), len(overlap),
                                  n_iterations=30, gene_frac=0.8, cell_frac=0.8,
                                  noise_std=0.0, seed=SEED, logger=logger,
                                  tag=f'nf{int(nf*100)}')
    summ = _ens.summarize(preds_3d, sorted(ref.obs['ct5'].unique()), list(yt.index))
    ens_mean = summ.pivot_table(index='sample_id', columns='cell_type', values='mean', aggfunc='first').loc[yt.index, CANON]
    ens_std = summ.pivot_table(index='sample_id', columns='cell_type', values='std', aggfunc='first').loc[yt.index, CANON]

    # Metrics (against ORIGINAL PBMC true proportions — the "known answer" before contamination)
    A, B = yt.values, ens_mean.values
    mae = float(np.abs(A - B).mean())
    cc = ccc(A, B)
    abs_err = np.abs(A - B).mean(axis=1)
    rel_std = ens_std.mean(axis=1).values
    ue = float(stats.pearsonr(rel_std, abs_err)[0]) if np.std(rel_std) > 0 else np.nan

    # Conformal (calibrate on first half, test on second)
    n_cal = N_SAMPLES // 2
    cal_res = calibrate(yt.iloc[:n_cal], ens_mean.iloc[:n_cal],
                        nominal_coverages=[0.90], score_function='absolute_error', per_cell_type=True)
    intervals = predict_intervals(ens_mean.iloc[n_cal:], cal_res['quantiles'],
                                  score_function='absolute_error', clip=True)
    cov_df = evaluate_calibration(intervals, yt.iloc[n_cal:])
    cov90 = float(cov_df[(cov_df.cell_type == 'overall') & (cov_df.nominal_coverage == 0.90)]['empirical_coverage_clip'].values[0])
    w90 = float(intervals[intervals.nominal_coverage == 0.90]['interval_width_clip'].mean())

    row = {'neutro_frac': nf, 'MAE': mae, 'CCC': cc,
           'pred_Monocyte_mean': float(ens_mean['Monocyte'].mean()),
           'pred_Tcell_mean': float(ens_mean['T_cell'].mean()),
           'pred_B_mean': float(ens_mean['B'].mean()),
           'pred_NK_mean': float(ens_mean['NK'].mean()),
           'mean_ensemble_std': float(rel_std.mean()),
           'ue_corr': ue, 'coverage90': cov90, 'width90': w90}
    all_rows.append(row)
    print(f'nf={nf:.2f}: MAE={mae:.4f} CCC={cc:.4f} pred_Mono={row["pred_Monocyte_mean"]:.3f} '
          f'std={rel_std.mean():.4f} ue={ue:.3f} cov90={cov90:.3f}')

df = pd.DataFrame(all_rows)
df.to_csv(PROJECT / 'results/phase7_neutrophil_ood/neutrophil_contamination_metrics.csv', index=False)
print('\nSaved metrics.')
print(df.round(4).to_string(index=False))
