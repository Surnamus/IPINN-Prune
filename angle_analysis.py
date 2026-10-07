"""Angle profiles: paired permutations preserve each seed's sparsity curve.

Tests are exploratory and assume exchangeability of angle labels within seed
under the global null. Post-hoc tests assume symmetric seed differences.
No model, loss, or score is changed. No absence of effect is inferred from p>alpha.
"""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from experiment_config import SEEDS, SPARSITIES, ANGLE_ARMS
from stats_analysis import load_all, holm
from final_analysis import signed_p, paired_summary


def permutation_profile(y, permutations=100000, seed=20221007):
    n,s,k=y.shape
    ranks=stats.rankdata(y,axis=2)
    tie_sum=np.zeros(s)
    for i in range(n):
        for j in range(s):
            counts=np.unique(y[i,j],return_counts=True)[1]
            tie_sum[j]+=np.sum(counts**3-counts)
    correction=1-tie_sum/(n*(k**3-k))
    def statistic(r):
        sums=r.sum(axis=-3)
        value=12/(n*k*(k+1))*(sums**2).sum(axis=-1)-3*n*(k+1)
        return np.where(correction>0,value/np.where(correction>0,correction,1),0)
    observed=statistic(ranks)
    choices=np.array(list(itertools.permutations(range(k))))
    rng=np.random.default_rng(seed);local=np.zeros(s,dtype=int);global_count=0
    for start in range(0,permutations,1000):
        batch=min(1000,permutations-start)
        order=choices[rng.integers(len(choices),size=(batch,n))]
        permuted=np.take_along_axis(ranks[None,:,:,:],order[:,:,None,:],axis=-1)
        simulated=statistic(permuted)
        local+=(simulated>=observed[None,:]-1e-12).sum(axis=0)
        global_count+=int((simulated.sum(axis=1)>=observed.sum()-1e-12).sum())
    return observed,(local+1)/(permutations+1),(global_count+1)/(permutations+1)


def analyze(results,criteria,permutations=100000):
    output=results/'angle_analysis';output.mkdir(exist_ok=True)
    df,_=load_all(results)
    frame=df[df.arm.isin(ANGLE_ARMS)]
    y=np.empty((len(SEEDS),len(SPARSITIES),len(ANGLE_ARMS)))
    for i,seed in enumerate(SEEDS):
        for j,s in enumerate(SPARSITIES):
            for a,arm in enumerate(ANGLE_ARMS):
                values=frame[(frame.seed==seed)&(frame.sparsity==s)&(frame.arm==arm)].y
                if len(values)!=1:raise ValueError('Missing or duplicate angle cell')
                y[i,j,a]=values.iloc[0]
    observed,p,global_p=permutation_profile(y,permutations)
    local=pd.DataFrame(dict(sparsity=SPARSITIES,n_seeds=len(SEEDS),friedman_statistic=observed,
                            permutation_p=p,kendall_W=observed/(len(SEEDS)*3)))
    local['p_holm_10_sparsities']=holm(p)
    local['angle_difference_detected']=local.p_holm_10_sparsities.lt(criteria['alpha'])
    local.to_csv(output/'friedman_permutation_by_sparsity.csv',index=False)
    rows=[];margin=criteria['angle_equivalence_margin']
    for j,s in enumerate(SPARSITIES):
        for a,b in itertools.combinations(range(4),2):
            d=y[:,j,a]-y[:,j,b]
            rows.append(dict(sparsity=s,a=ANGLE_ARMS[a],b=ANGLE_ARMS[b],
                p_difference=signed_p(d),
                p_equivalence=max(signed_p(d-margin['log10_ratio_lower'],'greater'),
                                  signed_p(d-margin['log10_ratio_upper'],'less')),
                **paired_summary(d)))
    contrasts=pd.DataFrame(rows)
    contrasts['p_difference_holm_60']=holm(contrasts.p_difference)
    contrasts['difference_detected']=contrasts.p_difference_holm_60.lt(criteria['alpha'])
    contrasts['p_equivalence_holm_60']=holm(contrasts.p_equivalence)
    contrasts['equivalence_confirmed']=contrasts.p_equivalence_holm_60.lt(criteria['alpha'])
    contrasts.to_csv(output/'paired_angle_contrasts_by_sparsity.csv',index=False)
    summary=[];rng=np.random.default_rng(2022)
    for j,s in enumerate(SPARSITIES):
        for a,arm in enumerate(ANGLE_ARMS):
            values=y[:,j,a]
            samples=rng.choice(values,size=(20000,len(SEEDS)),replace=True)
            lo,hi=np.quantile(np.median(samples,axis=1),[.025,.975])
            summary.append(dict(arm=arm,sparsity=s,median_log10_abs_error=np.median(values),ci95_lo=lo,ci95_hi=hi))
    summary=pd.DataFrame(summary);summary.to_csv(output/'angle_profiles_seed_intervals.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5))
    for arm,g in summary.groupby('arm'):
        g=g.sort_values('sparsity');ax.plot(g.sparsity,g.median_log10_abs_error,marker='o',label=arm)
        ax.fill_between(g.sparsity,g.ci95_lo,g.ci95_hi,alpha=.12)
    ax.set(xlabel='Target sparsity',ylabel='Median log10 absolute viscosity error',
           title='Angle profiles; pointwise 95% paired-seed bootstrap intervals')
    ax.legend();fig.tight_layout();fig.savefig(output/'angle_profiles.png',dpi=220);plt.close(fig)
    report=dict(n_seeds=len(SEEDS),n_sparsities=len(SPARSITIES),permutations=permutations,
        global_sum_friedman_statistic=float(observed.sum()),global_angle_profile_p=float(global_p),
        global_angle_effect_detected=bool(global_p<criteria['alpha']),
        n_sparsities_detected=int(local.angle_difference_detected.sum()),
        n_pairwise_differences_detected=int(contrasts.difference_detected.sum()),
        n_equivalences_confirmed=int(contrasts.equivalence_confirmed.sum()),
        limits=[
            'Angle labels are permuted once per seed and that permutation is shared over all ten sparsities; within-seed dependence is retained.',
            'Global statistic tests angle profile differences, not a separate pure angle-by-sparsity interaction null.',
            'Local omnibus tests use Holm over ten sparsities; pairwise difference and equivalence tests each use Holm over sixty contrasts.',
            'Eight independent seeds cannot attain Holm significance in the family of sixty exact pairwise tests, even in the most extreme case.',
            'Bootstrap intervals are descriptive pointwise intervals, not simultaneous intervals.',
            'These exploratory analyses were specified after initial results; no claim of absence of effect follows from nonsignificance.'])
    (output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True);p.add_argument('--criteria',type=Path,required=True)
    p.add_argument('--permutations',type=int,default=100000)
    a=p.parse_args();analyze(a.results,json.loads(a.criteria.read_text()),a.permutations)
