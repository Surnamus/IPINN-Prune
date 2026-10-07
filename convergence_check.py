"""Additional fixed-topology optimization; original checkpoints are never written.

This is a diagnostic of endpoint stability, not a proof of global convergence.
Adam/LBFGS optimizer states are restarted; no topology changes are performed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import torch
from train_dst import IPINN, loss_terms, frozen_mask_max, synchronize, atomic_json
from cfdsolver import get_static_dataset


def check(result, dataset, output, adam_steps=2000, lbfgs_steps=1000, device='cuda'):
    row=json.loads(result.read_text());ident=result.stem
    report=output/'results'/f'{ident}.json'
    if report.exists():
        return json.loads(report.read_text())
    if hashlib.sha256(dataset.read_bytes()).hexdigest()!=row['protocol']['dataset_sha256']:
        raise ValueError('Dataset fingerprint mismatch')
    device=torch.device(device);torch.set_num_threads(1)
    ck=torch.load(row['checkpoint'],map_location=device,weights_only=False)
    model=IPINN().to(device);model.load_state_dict(ck['model_state_dict'])
    raw=torch.nn.Parameter(ck['raw_nu'].detach().clone())
    masks=[None if m is None else m.to(device) for m in ck['pruner_state_dict']['backward_masks']]
    weights=[p for p in model.parameters() if p.ndim==2]
    data=get_static_dataset(str(dataset),device=device)
    data={k:(v.detach() if torch.is_tensor(v) else v) for k,v in data.items()}
    generator=torch.Generator(device=device).manual_seed(row['seed']+200000)
    idx=torch.randperm(len(data['dataIn']),generator=generator,device=device)[:row['protocol']['lbfgs_points']]
    batch_rng=torch.Generator(device=device).manual_seed(row['seed']+300000)
    def gradients():
        with torch.no_grad():
            for w,m in zip(weights,masks):
                if m is not None and w.grad is not None:w.grad.mul_(m)
    def enforce():
        with torch.no_grad():
            for w,m in zip(weights,masks):
                if m is not None:w.mul_(m)
    def field_error():
        x=torch.as_tensor(data['X'].ravel(),device=device,dtype=torch.float32)[:,None]
        t=torch.as_tensor(data['T'].ravel(),device=device,dtype=torch.float32)[:,None]
        target=torch.as_tensor(data['vu'].ravel(),device=device,dtype=torch.float32)[:,None]
        error=0.
        with torch.no_grad():
            for start in range(0,len(x),8192):
                end=start+8192;error+=float((model(x[start:end],t[start:end])-target[start:end]).square().sum())
        return math.sqrt(error/float(target.square().sum()))
    before_nu=float(raw.exp().detach());before_loss=float(sum(loss_terms(model,raw,data,idx)).detach());before_field=field_error()
    synchronize(device);start=time.perf_counter()
    optimizer=torch.optim.Adam([*model.parameters(),raw],lr=1e-4)
    for step in range(adam_steps):
        optimizer.zero_grad(set_to_none=True)
        batch=torch.randperm(len(data['dataIn']),generator=batch_rng,device=device)[:row['protocol']['batch_size']]
        loss=sum(loss_terms(model,raw,data,batch));loss.backward();gradients()
        torch.nn.utils.clip_grad_norm_([*model.parameters(),raw],1.)
        optimizer.step();enforce()
        if (step+1)%1000==0:print(ident,'additional Adam',step+1,flush=True)
    del optimizer
    optimizer=torch.optim.LBFGS([*model.parameters(),raw],max_iter=lbfgs_steps,
                              history_size=row['protocol']['history_size'],line_search_fn='strong_wolfe')
    trace=[]
    def closure():
        optimizer.zero_grad(set_to_none=True);loss=sum(loss_terms(model,raw,data,idx))
        loss.backward();gradients();trace.append(dict(loss=float(loss.detach()),nu=float(raw.exp().detach())))
        return loss
    optimizer.step(closure);enforce();closure();synchronize(device)
    after_nu=float(raw.exp().detach());after_field=field_error()
    result_row=dict(arm=row['arm'],seed=row['seed'],sparsity=row['sparsity'],
                    source_checkpoint=row['checkpoint'],diagnostic='fixed topology; restarted optimizer states',
                    additional_adam_steps=adam_steps,additional_adam_lr=1e-4,additional_lbfgs_max_iter=lbfgs_steps,
                    actual_lbfgs_iterations=int(optimizer.state[next(model.parameters())].get('n_iter',0)),
                    before_nu=before_nu,after_nu=after_nu,
                    before_nu_error_pct=100*abs(before_nu-.01/math.pi)/(.01/math.pi),
                    after_nu_error_pct=100*abs(after_nu-.01/math.pi)/(.01/math.pi),
                    nu_change_pct_of_truth=100*abs(after_nu-before_nu)/(.01/math.pi),
                    before_fixed_loss=before_loss,after_fixed_loss=trace[-1]['loss'],
                    before_field_relative_l2=before_field,after_field_relative_l2=after_field,
                    masked_max_abs=frozen_mask_max(model,masks),
                    final_gradient_max=max(float(p.grad.detach().abs().max()) for p in [*model.parameters(),raw]),
                    seconds=time.perf_counter()-start,trace=trace)
    if result_row['masked_max_abs']!=0:raise ValueError('Inactive weights changed')
    destination=output/'checkpoints'/f'{ident}.pt';destination.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model_state_dict=model.state_dict(),raw_nu=raw.detach(),masks=masks,metadata=result_row),destination)
    report.parent.mkdir(parents=True,exist_ok=True);atomic_json(report,result_row)
    return result_row


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--adam-steps',type=int,default=2000);p.add_argument('--lbfgs-steps',type=int,default=1000)
    a=p.parse_args();check(a.result,a.dataset,a.output,a.adam_steps,a.lbfgs_steps,a.device)
