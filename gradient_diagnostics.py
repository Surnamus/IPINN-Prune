"""Measure endpoint gradient scales on final models, without changing weights.

Fixed observations match each run's LBFGS subset. These measurements diagnose
final endpoints only, not the historical gradients at topology update steps.
No normalization, drop/growth change, or new training is performed.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from cfdsolver import get_static_dataset
from experiment_config import result_paths
from train_dst import IPINN, loss_terms
from pruningalg import angle_coefficients


def norm(parts):
    return math.sqrt(sum(float(p.detach().double().square().sum()) for p in parts))


def diagnostics(results,dataset):
    torch.set_num_threads(1)
    output=results/'gradient_diagnostics';output.mkdir(exist_ok=True)
    data=get_static_dataset(str(dataset),device='cpu')
    data={k:(v.detach() if torch.is_tensor(v) else v) for k,v in data.items()}
    dataset_hash=hashlib.sha256(dataset.read_bytes()).hexdigest()
    rows,layers=[],[]
    for path in result_paths(results):
        row=json.loads(path.read_text())
        if row['protocol']['dataset_sha256']!=dataset_hash:raise ValueError('Dataset differs')
        ck=torch.load(row['checkpoint'],map_location='cpu',weights_only=False)
        model=IPINN();model.load_state_dict(ck['model_state_dict']);model.eval()
        raw=ck['raw_nu'].detach().clone().requires_grad_()
        weights=[p for p in model.parameters() if p.ndim==2]
        masks=ck['pruner_state_dict']['backward_masks']
        rng=torch.Generator().manual_seed(row['seed']+200000)
        idx=torch.randperm(len(data['dataIn']),generator=rng)[:row['protocol']['lbfgs_points']]
        ic,bc,pde,obs=loss_terms(model,raw,data,idx)
        gp=torch.autograd.grad(ic+bc+pde,weights,retain_graph=True)
        ge=torch.autograd.grad(pde,weights,retain_graph=True)
        gd=torch.autograd.grad(obs,weights)
        phys,data_norm,pde_norm=norm(gp),norm(gd),norm(ge)
        physics_flat=torch.cat([g.detach().flatten().double() for g in gp])
        data_flat=torch.cat([g.detach().flatten().double() for g in gd])
        gradient_cosine=float(torch.dot(physics_flat,data_flat)/(phys*data_norm)) if phys*data_norm else np.nan
        common=dict(arm=row['arm'],seed=row['seed'],sparsity=row['sparsity'])
        rows.append(dict(**common,loss_ic=float(ic.detach()),loss_bc=float(bc.detach()),
                         loss_pde=float(pde.detach()),loss_data=float(obs.detach()),
                         grad_physics_l2=phys,grad_pde_only_l2=pde_norm,grad_data_l2=data_norm,
                         physics_data_norm_ratio=phys/data_norm if data_norm else np.nan,
                         physics_data_gradient_cosine=gradient_cosine))
        a,b=angle_coefficients(row['angle']) if row['arm'].startswith('th') else (1.,1.)
        for i,(w,m,p,d,e) in enumerate(zip(weights,masks,gp,gd,ge)):
            if m is None: m=torch.ones_like(w,dtype=torch.bool)
            p,d,e=p.detach(),d.detach(),e.detach()
            weighted_p=(a*w.detach()*p)[m];weighted_d=(b*w.detach()*d)[m]
            np_,nd=norm([weighted_p]),norm([weighted_d])
            layers.append(dict(**common,layer=i,topology_updates_enabled=(row['arm'] not in ['random','dense'] and masks[i] is not None),
                               n_active=int(m.sum()),n_inactive=int((~m).sum()),
                               physics_active_l2=norm([p[m]]),data_active_l2=norm([d[m]]),
                               physics_inactive_l2=norm([p[~m]]),data_inactive_l2=norm([d[~m]]),
                               pde_inactive_l2=norm([e[~m]]),drop_physics_component_l2=np_,
                               drop_data_component_l2=nd,drop_mixed_component_l2=norm([weighted_p+weighted_d]),
                               drop_physics_share_norm=np_/(np_+nd) if np_+nd else np.nan))
        if len(rows)%40==0: print('Measured',len(rows),'final models',flush=True)
    pd.DataFrame(rows).to_csv(output/'gradient_scales_per_run.csv',index=False)
    pd.DataFrame(layers).to_csv(output/'gradient_scales_per_layer.csv',index=False)
    frame=pd.DataFrame(rows)
    frame.groupby(['arm','sparsity']).agg(n_seeds=('seed','nunique'),
           median_norm_ratio=('physics_data_norm_ratio','median'),
           median_gradient_cosine=('physics_data_gradient_cosine','median')).to_csv(output/'summary.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5))
    for arm,g in frame.groupby('arm'):
        med=g.groupby('sparsity').physics_data_norm_ratio.median()
        ax.plot(med.index,med.values,marker='o',label=arm)
    ax.axhline(1,color='gray',linestyle='--');ax.set_yscale('log')
    ax.set(xlabel='Target sparsity',ylabel='Median ||gradient physics|| / ||gradient data||',
           title='Final endpoints; physics = IC + BC + PDE')
    ax.legend();fig.tight_layout();fig.savefig(output/'gradient_norm_ratios.png',dpi=220);plt.close(fig)
    (output/'report.json').write_text(json.dumps(dict(n_models=len(rows),
        interpretation='Endpoint diagnosis only; no claim of balanced gradients during training. Norm ratios are not direct fractions of selected connections.',
        physics_definition='L_ic + L_bc + L_pde; PDE-only gradients also reported',
        sampling='Same seed+200000 fixed observation subset used for LBFGS',
        model_changes=False),indent=2))
    return len(rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--wait',action='store_true');a=p.parse_args()
    status=a.results/'gradient_diagnostics_status.json'
    def write(state,**details):
        tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(dict(state=state,pid=os.getpid(),updated_unix=time.time(),**details),indent=2));tmp.replace(status)
    try:
        if a.wait:
            write('waiting_for_supplement')
            while True:
                previous=json.loads((a.results/'supplement_checks/status.json').read_text())
                if previous['state']=='complete':break
                if previous['state']=='failed':raise RuntimeError('Supplement failed')
                time.sleep(20)
        write('processing');count=diagnostics(a.results,a.dataset);write('complete',n_models=count)
    except Exception as error:
        write('failed',error=str(error));raise
