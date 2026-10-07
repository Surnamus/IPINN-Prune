"""Friedman then paired Wilcoxon/Holm, using identical (seed, sparsity) blocks."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from experiment_config import SEEDS, SPARSITIES, ARMS, ANGLE_ARMS, result_paths

ROOT = Path(__file__).resolve().parent


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1., np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]))
    return adjusted


def load_all(directory):
    paths = result_paths(directory)
    if not paths:
        raise ValueError(f'No results in {directory}; run the experiment sweep first')
    rows = []
    for p in paths:
        row = json.loads(p.read_text())
        folder = p.parent.parent.name
        if folder.startswith('seed') and folder[4:].isdigit() and row.get('seed') != int(folder[4:]):
            raise ValueError(f'Seed metadata/directory mismatch: {p}')
        rows.append(row)
    df = pd.DataFrame(rows)
    required = {'arm', 'seed', 'sparsity', 'abs_err_lbfgs', 'stage', 'protocol'}
    if not required <= set(df):
        raise ValueError(f'Missing fields: {required-set(df)}')
    if df.duplicated(['arm', 'seed', 'sparsity']).any():
        raise ValueError('Duplicate (arm, seed, sparsity) results')
    if not df.stage.eq('lbfgs').all():
        raise ValueError('Statistics require completed LBFGS runs; smoke/train-only results are excluded')
    protocols = {json.dumps(v, sort_keys=True) for v in df.protocol}
    if len(protocols) != 1:
        raise ValueError('Cannot pool different budgets/datasets/implementations')
    if not set(df.arm) <= set((*ARMS, 'dense')) or not set(df.seed) <= set(SEEDS):
        raise ValueError('Unexpected arms/seeds')
    sparse = df[df.arm != 'dense'].copy()
    if not sparse.sparsity.isin(SPARSITIES).all():
        raise ValueError('Unexpected sparsity levels')
    errors = pd.to_numeric(sparse.abs_err_lbfgs, errors='raise')
    if not (np.isfinite(errors) & (errors >= 0)).all():
        raise ValueError('Errors must be finite and nonnegative')
    sparse['zero_error'] = errors.eq(0)
    sparse['y'] = np.log10(errors.clip(lower=np.finfo(float).tiny))
    return sparse, df


def paired_wide(df, arms):
    # No pivot_table averaging: duplicates are a design error.
    return df.pivot(index=['seed', 'sparsity'], columns='arm', values='y').reindex(columns=arms).dropna()


def friedman(wide):
    n, k = wide.shape
    if n < 3 or k < 3:
        return dict(n_blocks=n, n_arms=k, status='insufficient', statistic=None, p=None, kendall_W=None)
    values = wide.to_numpy()
    if np.all(values == values[:, :1]):
        statistic, p = 0., 1.
    else:
        result = stats.friedmanchisquare(*[wide[c].to_numpy() for c in wide])
        statistic, p = float(result.statistic), float(result.pvalue)
    return dict(n_blocks=n, n_arms=k, status='ok', statistic=statistic, p=p,
                kendall_W=statistic/(n*(k-1)))


def contrasts(wide, pairs, family, omnibus=None):
    rows = []
    for a, b in pairs:
        d = (wide[a]-wide[b]).to_numpy()
        if len(d) < 3:
            continue
        nonzero = d[d != 0]
        p = 1. if not len(nonzero) else float(stats.wilcoxon(d, zero_method='wilcox', alternative='two-sided').pvalue)
        ranks = stats.rankdata(np.abs(nonzero))
        effect = 0. if not len(nonzero) else float((ranks[nonzero > 0].sum()-ranks[nonzero < 0].sum())/ranks.sum())
        rows.append(dict(family=family, a=a, b=b, n_blocks=len(d), p=p,
                         mean_diff_log10=float(d.mean()), median_diff_log10=float(np.median(d)),
                         rank_biserial=effect))
    table = pd.DataFrame(rows)
    if len(table):
        table['p_holm'] = holm(table.p)
        table['reject_h0'] = (table.p_holm < .05) & (omnibus is None or (omnibus.get('p') is not None and omnibus['p'] < .05))
        table['omnibus_gate'] = omnibus is not None
    return table


def seed_bootstrap_ci(group, n_boot=10000):
    # Resample complete seeds (all sparsities), preserving within-seed dependence.
    by_seed = [g.y.to_numpy() for _, g in group.groupby('seed')]
    if len(by_seed) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(2022)
    medians = [np.median(np.concatenate([by_seed[i] for i in rng.integers(len(by_seed), size=len(by_seed))]))
               for _ in range(n_boot)]
    return tuple(np.quantile(medians, [.025, .975]))


def analyze(directory=ROOT, output=ROOT/'statistics'):
    df, all_rows = load_all(directory)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    expected = pd.MultiIndex.from_product([SEEDS, SPARSITIES], names=['seed', 'sparsity'])
    coverage = df.pivot(index=['seed', 'sparsity'], columns='arm', values='y').reindex(index=expected, columns=ARMS)
    coverage.notna().to_csv(output/'coverage.csv')
    missing = int(coverage.isna().sum().sum())
    print(f'Complete angle blocks: {len(paired_wide(df, ANGLE_ARMS))}/{len(SEEDS)*len(SPARSITIES)}; missing arm/block cells: {missing}')
    omnis, tables = [], []
    # Four angles, all six pairwise comparisons with one Holm family.
    angle_wide = paired_wide(df, ANGLE_ARMS)
    omni = friedman(angle_wide)
    omnis.append(dict(family='angle_four', **omni))
    tables.append(contrasts(angle_wide, list(itertools.combinations(ANGLE_ARMS, 2)), 'angle_four', omni))
    # All planned ablation contrasts share a single Holm family.
    ablation_arms = (*ANGLE_ARMS, 'm2', 'random')
    w = paired_wide(df, ablation_arms)
    pairs = [('th45', 'th0'), ('th45', 'th90'), ('th45', 'th135'), ('th45', 'm2')]
    pairs += [(a, 'random') for a in (*ANGLE_ARMS, 'm2')]
    tables.append(contrasts(w, pairs, 'ablation'))
    # Sensitivity: average over sparsities, then pair the independent seeds.
    seed_means = df.groupby(['seed', 'arm']).agg(y=('y', 'mean'), n_s=('sparsity', 'nunique')).reset_index()
    seed_means = seed_means[seed_means.n_s == len(SPARSITIES)]
    independent = ANGLE_ARMS
    sw = seed_means.pivot(index='seed', columns='arm', values='y').reindex(columns=independent).dropna()
    omni = friedman(sw)
    omnis.append(dict(family='angle_seed_sensitivity', **omni))
    tables.append(contrasts(sw, list(itertools.combinations(independent, 2)), 'angle_seed_sensitivity', omni))
    pd.DataFrame(omnis).to_csv(output/'friedman.csv', index=False)
    usable = [t for t in tables if len(t)]
    (pd.concat(usable, ignore_index=True) if usable else pd.DataFrame()).to_csv(output/'wilcoxon_holm.csv', index=False)
    summaries = []
    for (arm, sparsity), g in df.groupby(['arm', 'sparsity']):
        lo, hi = seed_bootstrap_ci(g)
        summaries.append(dict(arm=arm, sparsity=sparsity, n_seeds=g.seed.nunique(),
                              median_log10_abs_err=g.y.median(), ci_lo=lo, ci_hi=hi))
    pd.DataFrame(summaries).to_csv(output/'summary_by_sparsity.csv', index=False)
    cols = [c for c in ('nnz', 'real_sparsity', 'dense_B', 'ideal_B', 'csr_B', 'mask_B',
                        'model_and_masks_B', 'checkpoint_B', 'gpu_peak_allocated_B',
                        'gpu_peak_reserved_B', 'training_seconds') if c in all_rows]
    if cols:
        all_rows.groupby(['arm', 'sparsity'])[cols].mean().to_csv(output/'footprint.csv')
    diagnostics = [c for c in ('arm', 'seed', 'sparsity', 'turnover', 'growth_events',
                                'pruned_max_abs_after_lbfgs', 'reused_from') if c in df]
    df[diagnostics].to_csv(output/'mask_diagnostics.csv', index=False)
    report = dict(missing_cells=missing, zero_errors=int(df.zero_error.sum()),
                  interpretation='Failure to reject H0 is not proof of no angle effect. Blocks sharing a seed may be dependent; consult seed sensitivity. Friedman p uses an asymptotic approximation.',
                  omnibus=omnis)
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(pd.DataFrame(omnis).to_string(index=False))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, default=ROOT)
    p.add_argument('--output', type=Path, default=ROOT/'statistics')
    args = p.parse_args()
    analyze(args.results, args.output)
