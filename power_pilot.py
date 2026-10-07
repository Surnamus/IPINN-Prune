"""Exploratory pilot variability, 7.2% accuracy acceptance, and power scenarios.

This is not a guarantee or a preregistered final power calculation. Angle-effect
scenarios are error ratios, distinct from the absolute 7.2% acceptance threshold.
Simulation uses independent seed aggregates with a correlated Gaussian model
on log10 errors fitted to six seeds; sparsities never count as independent seeds.
"""
import argparse
import itertools
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats


def adjusted(p):
    order = np.argsort(p, axis=1)
    sorted_p = np.take_along_axis(p, order, axis=1)
    sorted_adj = np.minimum(1., np.maximum.accumulate(sorted_p*np.arange(p.shape[1], 0, -1), axis=1))
    result = np.empty_like(p)
    np.put_along_axis(result, order, sorted_adj, axis=1)
    return result


def analyze(results, output, simulations=2000, threshold=7.2, target_reduction_pct=None):
    output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(p.read_text()) for p in results.glob('seed*/results_v2/*.json')]
    df = pd.DataFrame(rows)
    if not len(df) or not df.stage.eq('lbfgs').all():
        raise ValueError('Requires final pilot results')
    df['relative_error_pct'] = 100*df.abs_err_lbfgs/(.01/math.pi)
    df['acceptable'] = df.relative_error_pct <= threshold
    summaries = []
    for (arm, sparsity), group in df.groupby(['arm', 'sparsity']):
        n, k = len(group), int(group.acceptable.sum())
        summaries.append(dict(arm=arm, sparsity=sparsity, n_seeds=n,
                              mean_error_pct=group.relative_error_pct.mean(),
                              sd_error_pp=group.relative_error_pct.std(ddof=1),
                              median_error_pct=group.relative_error_pct.median(),
                              min_error_pct=group.relative_error_pct.min(),
                              max_error_pct=group.relative_error_pct.max(),
                              n_acceptable=k, pass_fraction=k/n,
                              pass_ci95_lo=0 if k == 0 else stats.beta.ppf(.025, k, n-k+1),
                              pass_ci95_hi=1 if k == n else stats.beta.ppf(.975, k+1, n-k),
                              pass_lower95_one_sided=0 if k == 0 else stats.beta.ppf(.05, k, n-k+1)))
    pd.DataFrame(summaries).to_csv(output/'accuracy_threshold_and_variability.csv', index=False)
    arms = ['th0', 'th45', 'th90', 'th135', 'm2', 'random']
    sparse = df[df.arm.isin(arms)].copy()
    sparse['log_error'] = np.log10(sparse.abs_err_lbfgs.clip(lower=np.finfo(float).tiny))
    g = sparse.groupby(['seed', 'arm']).agg(y=('log_error', 'mean'), n=('sparsity', 'nunique'))
    g = g[g.n.eq(10)].reset_index()
    wide = g.pivot(index='seed', columns='arm', values='y').reindex(columns=arms).dropna()
    if len(wide) < 6:
        raise ValueError('Need six complete independent pilot seeds')
    wide.to_csv(output/'seed_log_error_means.csv')
    covariance = np.cov(wide.to_numpy(), rowvar=False)
    rng = np.random.default_rng(2022)
    angle_pairs = list(itertools.combinations(range(4), 2))
    ablation_pairs = [(1, 0), (1, 2), (1, 3), (1, 4)] + [(i, 5) for i in range(5)]
    power_rows = []
    for scale in [1., 1.5]:
        for n in [6, 8, 9, 10, 12, 15, 19, 25, 30, 40, 50, 75, 100]:
            noise = rng.multivariate_normal(np.zeros(6), covariance*scale**2, size=(simulations, n))
            for ratio in ([1., 1/(1-target_reduction_pct/100)] if target_reduction_pct is not None else [1., 1.2, 1.5, 2.]):
                samples = noise.copy()
                samples[:, :, 1] -= math.log10(ratio)
                omnibus = stats.friedmanchisquare(*(samples[:, :, i] for i in range(4)), axis=1).pvalue
                for family, pairs in [('angle_four', angle_pairs), ('ablation', ablation_pairs)]:
                    p = np.stack([stats.wilcoxon(samples[:, :, a]-samples[:, :, b], axis=1,
                                  alternative='two-sided', method='exact').pvalue for a, b in pairs], axis=1)
                    rejection = adjusted(p) < .05
                    if family == 'angle_four':
                        rejection &= omnibus[:, None] < .05
                    # For ratio>1 only contrasts involving th45 have injected effects.
                    relevant = [j for j, pair in enumerate(pairs) if ratio == 1 or 1 in pair]
                    success = rejection[:, relevant].any(axis=1)
                    probability = float(success.mean())
                    power_rows.append(dict(family=family, n_seeds=n, error_ratio=ratio,
                                           variability_scale=scale, simulations=simulations,
                                           metric='null_any_rejection_rate' if ratio == 1 else 'power_any_true_contrast',
                                           probability=probability,
                                           monte_carlo_se=math.sqrt(probability*(1-probability)/simulations),
                                           power_all_true_contrasts=float(rejection[:, relevant].all(axis=1).mean()) if ratio != 1 else None,
                                           power_th45_vs_m2=float(rejection[:, pairs.index((1, 4))].mean()) if family == 'ablation' else None))
    pd.DataFrame(power_rows).to_csv(output/'power_scenarios.csv', index=False)
    report = dict(pilot_seeds=list(map(int, wide.index)), threshold_relative_error_pct=threshold,
                  results_above_threshold=int((~df.acceptable).sum()), results_total=len(df), target_reduction_pct=target_reduction_pct,
                  interpretations=[
                      '7.2% is an accuracy threshold, not an angle-effect size or equivalence margin.',
                      'Power scenarios inject a uniform th45 error reduction by a specified ratio.',
                      'Scenarios pool sparsities only within seed; seed is the independent unit.',
                      'Gaussian log-error covariance fitted to six seeds is uncertain; scales 1 and 1.5 are sensitivity scenarios, not confidence limits.',
                      'Friedman uses its asymptotic approximation with four arms; small-n estimates need caution.',
                      'Power means detecting at least one truly affected planned contrast, not all contrasts.',
                      'Results are exploratory planning estimates, not guaranteed adequacy or evidence of equivalence.'])
    (output/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--simulations', type=int, default=2000)
    p.add_argument('--target-reduction-pct', type=float)
    args = p.parse_args()
    if args.target_reduction_pct is not None and not 0 < args.target_reduction_pct < 100:
        p.error('target reduction must be between 0 and 100')
    analyze(args.results, args.output, args.simulations, target_reduction_pct=args.target_reduction_pct)
