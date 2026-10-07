"""Read-only model analysis: acceptance, seed-level ablations and equivalence.

Inference uses equally weighted sparsity log-errors averaged within independent
seeds. Exact signed-rank tests assume independent, symmetric seed differences.
Bootstrap intervals are descriptive percentile intervals from paired seeds.
All rules are exploratory: thresholds were selected after initial results.
"""
import argparse
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from scipy import stats
from experiment_config import SEEDS, SPARSITIES, ANGLE_ARMS, ARMS
from stats_analysis import load_all, holm, friedman


def signed_p(values, alternative='two-sided'):
    values = np.asarray(values, float)
    values = values[values != 0]
    if not len(values):
        return 1.
    if len(values) > 20:
        raise ValueError('Exhaustive test limited to 20 nonzero independent pairs')
    ranks = stats.rankdata(np.abs(values))
    signs = ((np.arange(2**len(values))[:, None] >> np.arange(len(values))) & 1)
    possible = signs @ ranks
    observed = ranks[values > 0].sum()
    lower, upper = float((possible <= observed).mean()), float((possible >= observed).mean())
    return lower if alternative == 'less' else upper if alternative == 'greater' else min(1., 2*min(lower, upper))


def paired_summary(values):
    rng = np.random.default_rng(2022)
    draws = rng.choice(values, size=(20000, len(values)), replace=True)
    lo, hi = np.quantile(np.median(draws, axis=1), [.025, .975])
    nonzero = values[values != 0]
    ranks = stats.rankdata(np.abs(nonzero))
    effect = 0. if not len(nonzero) else float((ranks[nonzero > 0].sum()-ranks[nonzero < 0].sum())/ranks.sum())
    return dict(n_seeds=len(values), median_log10_ratio=float(np.median(values)),
                median_error_ratio=float(10**np.median(values)), ratio_ci95_lo=float(10**lo),
                ratio_ci95_hi=float(10**hi), rank_biserial=effect)


def finalize(results, criteria_path, pilot_source, analysis_output=None, skip_power=False):
    criteria = json.loads(criteria_path.read_text())
    out = analysis_output or results/'final_statistics';out.mkdir(parents=True,exist_ok=True)
    (out/'criteria.json').write_text(json.dumps(criteria, indent=2))
    df, all_rows = load_all(results)
    expected = {(a, seed, s) for a in ARMS for seed in SEEDS for s in SPARSITIES}
    actual = set(zip(df.arm, df.seed, df.sparsity))
    if actual != expected:
        raise ValueError(f'Incomplete design: missing {len(expected-actual)}, unexpected {len(actual-expected)}')
    sizes = pd.read_csv(results/'results_csv'/'final_model_sizes.csv')
    merged = all_rows.merge(sizes[['arm', 'seed', 'sparsity', 'dense_file_B', 'csr_file_B',
                                  'saving_pct', 'prediction_max_abs_diff', 'tensors_exact']],
                            on=['arm', 'seed', 'sparsity'], validate='one_to_one', how='left')
    if merged.dense_file_B.isna().any():
        raise ValueError('Missing model exports')
    merged['nu_relative_error_pct'] = 100*merged.abs_err_lbfgs/(.01/math.pi)
    merged['accuracy_pass'] = merged.nu_relative_error_pct <= criteria['nu_relative_error_pct_max']
    merged['storage_pass'] = (merged.saving_pct > criteria['compressed_file_saving_pct_min']) if criteria.get('storage_strict_greater', False) else (merged.saving_pct >= criteria['compressed_file_saving_pct_min'])
    merged['export_pass'] = merged.tensors_exact.eq(True) & merged.prediction_max_abs_diff.eq(0)
    merged['joint_pass'] = merged.accuracy_pass & merged.storage_pass & merged.export_pass
    columns = ['arm', 'seed', 'sparsity', 'nu_relative_error_pct', 'dense_file_B', 'csr_file_B',
               'saving_pct', 'accuracy_pass', 'storage_pass', 'export_pass', 'joint_pass']
    merged[columns].to_csv(out/'criteria_per_run.csv', index=False)
    candidates = merged[merged.arm.ne('dense')]
    summary = candidates.groupby(['arm', 'sparsity']).agg(
        n_seeds=('seed', 'nunique'), n_joint_pass=('joint_pass', 'sum'),
        all_observed_seeds_pass=('joint_pass', 'all'),
        median_error_pct=('nu_relative_error_pct', 'median'), median_saving_pct=('saving_pct', 'median')).reset_index()
    summary.to_csv(out/'criteria_by_sparsity.csv', index=False)
    baseline = candidates[candidates.arm.eq('m2')][['seed', 'sparsity', 'nu_relative_error_pct']].rename(columns={'nu_relative_error_pct': 'm2_error_pct'})
    benefit = candidates[candidates.arm.isin(ANGLE_ARMS)][['arm','seed','sparsity','nu_relative_error_pct']].merge(baseline,on=['seed','sparsity'],validate='many_to_one')
    benefit['reduction_pct'] = np.where(benefit.m2_error_pct.gt(0), 100*(1-benefit.nu_relative_error_pct/benefit.m2_error_pct), np.nan)
    benefit['practical_benefit_pass'] = benefit.reduction_pct >= criteria['model1_relative_error_reduction_vs_model2_pct_min']
    benefit.to_csv(out/'model1_benefit_per_run.csv', index=False)
    benefit.groupby(['arm','sparsity']).agg(n_seeds=('seed','nunique'),n_pass=('practical_benefit_pass','sum'),all_seeds_pass=('practical_benefit_pass','all'),median_reduction_pct=('reduction_pct','median')).to_csv(out/'model1_benefit_summary.csv')
    if df.zero_error.any():
        raise ValueError('Zero errors need a separate ratio analysis; thresholds exported, inference not silently floored')
    wide = df.groupby(['seed','arm']).y.mean().unstack('arm').reindex(index=SEEDS,columns=ARMS)
    if wide.isna().any().any():
        raise ValueError('Missing seed aggregate')
    wide.to_csv(out/'seed_aggregate_log_errors.csv')
    omnibus = friedman(wide[list(ANGLE_ARMS)])
    families = {
        'angles_seed': list(itertools.combinations(ANGLE_ARMS,2)),
        'ablations_seed': [('th45','th0'),('th45','th90'),('th45','th135'),('th45','m2')]+[(a,'random') for a in (*ANGLE_ARMS,'m2')]
    }
    tables = []
    for family,pairs in families.items():
        rows = []
        for a,b in pairs:
            d=(wide[a]-wide[b]).to_numpy()
            rows.append(dict(family=family,a=a,b=b,p=signed_p(d),**paired_summary(d)))
        table=pd.DataFrame(rows);table['p_holm']=holm(table.p)
        table['reject_equal_errors']=table.p_holm.lt(criteria['alpha'])
        if family=='angles_seed':
            table['reject_equal_errors'] &= omnibus['p'] < criteria['alpha']
        tables.append(table)
    pd.concat(tables).to_csv(out/'seed_wilcoxon_holm.csv',index=False)
    margin = criteria['angle_equivalence_margin']; eq=[]
    for a,b in families['angles_seed']:
        d=(wide[a]-wide[b]).to_numpy()
        lower=signed_p(d-margin['log10_ratio_lower'],'greater')
        upper=signed_p(d-margin['log10_ratio_upper'],'less')
        eq.append(dict(a=a,b=b,p_lower=lower,p_upper=upper,p_tost=max(lower,upper),**paired_summary(d)))
    eq=pd.DataFrame(eq);eq['p_tost_holm']=holm(eq.p_tost)
    eq['equivalence_confirmed']=eq.p_tost_holm.lt(criteria['alpha'])
    eq.to_csv(out/'angle_equivalence_seed.csv',index=False)
    superiority=[]; boundary=math.log10(1-criteria['model1_relative_error_reduction_vs_model2_pct_min']/100)
    for a in ANGLE_ARMS:
        d=(wide[a]-wide['m2']).to_numpy()
        superiority.append(dict(a=a,b='m2',p_above_10pct_benefit=signed_p(d-boundary,'less'),**paired_summary(d)))
    superiority=pd.DataFrame(superiority);superiority['p_holm']=holm(superiority.p_above_10pct_benefit)
    superiority['benefit_above_10pct_confirmed']=superiority.p_holm.lt(criteria['alpha'])
    superiority.to_csv(out/'model1_10pct_superiority_seed.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5))
    for arm,g in summary.groupby('arm'):
        g=g.sort_values('sparsity');ax.plot(g.sparsity,100*g.n_joint_pass/g.n_seeds,marker='o',label=arm)
    ax.set(xlabel='Target sparsity',ylabel='Observed seeds passing accuracy and storage criteria (%)',ylim=(-5,105))
    ax.legend();fig.tight_layout();fig.savefig(out/'joint_criteria_pass_rates.png',dpi=220);plt.close(fig)
    for reduction in ([] if skip_power else [10,20,30]):
        subprocess.run([sys.executable,str(Path(__file__).with_name('power_pilot.py')),
                        '--results',str(pilot_source),'--output',str(out/f'power_{reduction}pct'),
                        '--simulations','4000','--target-reduction-pct',str(reduction)],check=True)
    report=dict(n_seeds=len(SEEDS),sparse_runs=len(df),dense_baselines=int(all_rows.arm.eq('dense').sum()),
                n_joint_pass=int(candidates.joint_pass.sum()), angle_seed_friedman=omnibus,
                all_angle_pairs_equivalent=bool(eq.equivalence_confirmed.all()),
                criteria_definition='Exploratory, after initial results; not preregistered',
                limitations=[
                    'Seed-level inference averages log-errors equally over ten sparsities; it does not establish equivalence at every sparsity.',
                    'Exact signed-rank inference assumes independent symmetric seed-level differences; small n limits power.',
                    '95% bootstrap intervals are descriptive and pointwise, not simultaneous inference intervals.',
                    'Equivalence uses two one-sided exact signed-rank tests and Holm across six pairs; nonsignificance of difference is not equivalence.',
                    'Superiority tests evidence beyond the 10% boundary; power scenarios instead assess detecting a 10%, 20%, or 30% effect against zero difference.',
                    'All observed seeds passing is not a guarantee for unseen seeds.',
                    'Compressed file savings do not establish training or inference GPU memory savings.',
                    'No claim of achieved 80% power is made.'])
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--criteria',type=Path,required=True)
    p.add_argument('--pilot-source',type=Path,required=True)
    p.add_argument('--analysis-output',type=Path)
    p.add_argument('--status-file',type=Path)
    p.add_argument('--skip-power',action='store_true')
    p.add_argument('--wait',action='store_true')
    args=p.parse_args();status=args.status_file or args.results/'final_analysis_status.json'
    def write(state,**details):
        tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(dict(state=state,pid=os.getpid(),updated_unix=time.time(),**details),indent=2));tmp.replace(status)
    try:
        if args.wait:
            write('waiting_for_exports')
            while True:
                previous=json.loads((args.results/'extension_status.json').read_text())
                if previous['state']=='complete':break
                if previous['state']=='failed':raise RuntimeError('Extension failed')
                time.sleep(20)
        write('processing')
        report=finalize(args.results,args.criteria,args.pilot_source,args.analysis_output,args.skip_power)
        write('complete',n_seeds=report['n_seeds'],sparse_runs=report['sparse_runs'])
        print(json.dumps(report,indent=2))
    except Exception as error:
        write('failed',error=str(error));raise


if __name__=='__main__':
    main()
