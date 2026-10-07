"""Shared Adam + fixed-mask LBFGS pipeline for all experimental arms."""
import argparse
import fcntl
import os
from contextlib import contextmanager
import hashlib
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from cfdsolver import get_static_dataset
from experiment_config import SEEDS, SPARSITIES, ARMS, canonical_arm, run_id, seed_directory
from pruningalg import RigLScheduler, angle_coefficients

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / 'datasets' / 'static_dataset.pt'
if not DEFAULT_DATASET.exists():
    DEFAULT_DATASET = ROOT.parent / 'sources' / 'static_dataset.pt'
if not DEFAULT_DATASET.exists():
    DEFAULT_DATASET = ROOT.parents[1] / 'sources' / 'static_dataset.pt'


class IPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
        nn.Linear(2, 20), nn.Tanh(),
        nn.Linear(20, 20), nn.Tanh(),
        nn.Linear(20, 20), nn.Tanh(),
        nn.Linear(20, 20), nn.Tanh(),
        nn.Linear(20, 20), nn.Tanh(),
        nn.Linear(20, 20), nn.Tanh(),
        nn.Linear(20, 20), nn.Tanh(),
        nn.Linear(20, 20), nn.Tanh(),
        nn.Linear(20, 1)
        )

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loss_terms(model, raw_nu, data, idx):
    def supervised(prefix):
        inputs, target = data[prefix+'In'], data[prefix+'Out']
        return (model(inputs[:, :1], inputs[:, 1:2])-target.reshape(-1, 1)).square().mean()
    xy = data['dataIn'][idx].detach()
    x, t = xy[:, :1].clone().requires_grad_(), xy[:, 1:2].clone().requires_grad_()
    u = model(x, t)
    ux, ut = torch.autograd.grad(u, (x, t), torch.ones_like(u), create_graph=True)
    uxx = torch.autograd.grad(ux, x, torch.ones_like(ux), create_graph=True)[0]
    pde = (ut+u*ux-raw_nu.exp()*uxx).square().mean()
    obs = (u-data['dataOut'][idx].reshape(-1, 1)).square().mean()
    return supervised('init'), supervised('bound'), pde, obs


def footprint(model, raw_nu, masks):
    params = [*model.parameters(), raw_nu]
    dense = sum(p.numel()*p.element_size() for p in params)
    dense_extra = sum(p.numel()*p.element_size() for p in params if p.ndim != 2)
    ideal, csr, nnz, total, mask_bytes = dense_extra, dense_extra, 0, 0, 0
    for w, mask in zip((p for p in model.parameters() if p.ndim == 2), masks):
        count = w.numel() if mask is None else int(mask.sum())
        total += w.numel()
        nnz += count
        ideal += count*w.element_size()
        csr += min(w.numel()*w.element_size(), count*(w.element_size()+4)+(w.shape[0]+1)*4)
        mask_bytes += 0 if mask is None else mask.numel()*mask.element_size()
    return dict(nnz=nnz, n_weights=total, real_sparsity=1-nnz/total,
                dense_B=dense, ideal_B=ideal, csr_B=csr, mask_B=mask_bytes,
                model_and_masks_B=dense+mask_bytes)


def frozen_mask_max(model, masks):
    values = [float(w[~m].detach().abs().max()) for w, m in zip(
        (p for p in model.parameters() if p.ndim == 2), masks)
              if m is not None and (~m).any()]
    return max(values, default=0.0)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def atomic_json(path, row):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(row, indent=2, allow_nan=False))
    tmp.replace(path)


def _train_one(arm, seed, sparsity, args):
    arm = canonical_arm(arm)
    if arm not in (*ARMS, 'dense'):
        raise ValueError('Unknown experimental arm')
    if seed not in SEEDS or (arm != 'dense' and sparsity not in SPARSITIES):
        raise ValueError('Seed/sparsity outside predeclared design')
    ident = run_id(arm, seed, sparsity)
    seed_root = seed_directory(args.output, seed)
    result_path = seed_root/'results_v2'/f'{ident}.json'
    ckpt_path = seed_root/'checkpoints'/f'{ident}.pt'
    # Fingerprint the protocol and dataset; never silently reuse incompatible results.
    protocol = dict(steps=args.steps, batch_size=args.batch_size, delta=args.delta,
                    lbfgs_iter=args.lbfgs_iter, lbfgs_points=args.lbfgs_points,
                    history_size=args.history_size, pruning_steps=args.pruning_steps,
                    dataset_sha256=args.dataset_sha256,
                    implementation='og-structure-current-behavior-v1')
    if result_path.exists() and not args.overwrite:
        old = json.loads(result_path.read_text())
        if old.get('protocol') != protocol or not ckpt_path.exists():
            raise ValueError(f'Incompatible/incomplete run {ident}; use a fresh output directory')
        return old
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable: check your PyTorch/ROCm installation')
    seed_all(seed)  # before model initialization and mask generation
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = time.perf_counter()
    data = get_static_dataset(path=str(args.dataset), device=device)
    data = {k: v.detach() for k, v in data.items() if torch.is_tensor(v)}
    model = IPINN().to(device)
    raw_nu = nn.Parameter(torch.tensor(math.log(.008), device=device))
    optimizer = torch.optim.Adam([*model.parameters(), raw_nu], lr=.001)
    lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=1e-4)
    pruner = RigLScheduler(model, optimizer, dense_allocation=1-sparsity,
                          T_end=args.pruning_steps, delta=args.delta,
                          static_topo=arm in ('random', 'dense'))
    initial_masks = [None if m is None else m.clone() for m in pruner.backward_masks]
    # Separate RNG keeps identical data batches across arms despite mask construction.
    batch_rng = torch.Generator(device=device).manual_seed(seed+100000)
    n = len(data['dataIn'])
    partial_path = seed_root/'checkpoints'/f'{ident}.partial.pt'
    adam_protocol = {k: v for k, v in protocol.items() if k not in ('lbfgs_iter', 'lbfgs_points', 'history_size')}
    cache_tag = hashlib.sha256(json.dumps(adam_protocol, sort_keys=True).encode()).hexdigest()[:16]
    cache_dir = seed_directory(args.adam_cache, seed) if args.adam_cache is not None else seed_root/'adam_cache'
    adam_path = cache_dir/f'{ident}_{cache_tag}.pt'
    adam_reused = False
    first_step, previous_seconds = 0, 0.
    previous_peaks = dict(gpu_peak_allocated_B=0, gpu_peak_reserved_B=0)
    if adam_path.exists() and not args.overwrite:
        cached = torch.load(adam_path, map_location=device, weights_only=False)
        if cached['protocol'] != adam_protocol:
            raise ValueError('Incompatible Adam cache')
        model.load_state_dict(cached['model_state_dict'])
        with torch.no_grad():
            raw_nu.copy_(cached['raw_nu'])
        pruner.load_state_dict(cached['pruner_state_dict'])
        initial_masks = cached['initial_masks']
        previous_seconds = cached['adam_seconds']
        previous_peaks = cached['gpu_peaks']
        first_step = args.steps
        adam_reused = True
        pruner.apply_mask_to_weights()
        del cached
        print(f'REUSED completed Adam phase: {ident}, steps={args.steps}', flush=True)
    elif partial_path.exists() and not args.overwrite:
        partial = torch.load(partial_path, map_location=device, weights_only=False)
        if partial['protocol'] != protocol:
            raise ValueError(f'Incompatible partial checkpoint: {ident}')
        model.load_state_dict(partial['model_state_dict'])
        with torch.no_grad():
            raw_nu.copy_(partial['raw_nu'])
        optimizer.load_state_dict(partial['optimizer_state_dict'])
        lr.load_state_dict(partial['lr_state_dict'])
        pruner.load_state_dict(partial['pruner_state_dict'])
        initial_masks = partial['initial_masks']
        batch_rng.set_state(partial['batch_rng_state'].cpu())
        first_step = partial['steps_done']
        previous_seconds = partial['adam_seconds']
        previous_peaks = partial.get('gpu_peaks', previous_peaks)
        pruner.apply_mask_to_weights()
        del partial
        print(f'RESUMED {ident} at Adam step {first_step}', flush=True)
    for step in range(first_step, args.steps):
        optimizer.zero_grad(set_to_none=True)
        idx = torch.randperm(n, generator=batch_rng, device=device)[:args.batch_size]
        ic, bc, pde, obs = loss_terms(model, raw_nu, data, idx)
        total = ic+bc+pde+obs
        due = pruner.due()
        if due:
            # No parameter hooks: score gradients at inactive positions stay dense.
            physics = torch.autograd.grad(ic+bc+pde, pruner.W, retain_graph=True)
            observed = torch.autograd.grad(obs, pruner.W, retain_graph=True)
            if arm == 'm2':
                drop = grow = [p+d for p, d in zip(physics, observed)]
            else:
                a, b = angle_coefficients(float(arm[2:]))
                drop = [a*p+b*d for p, d in zip(physics, observed)]
                grow = physics
        total.backward()  # graph retained only on topology update steps above
        pruner.step += 1
        if due:
            pruner.update(drop, grow)
        pruner.apply_mask_to_gradients()
        torch.nn.utils.clip_grad_norm_([*model.parameters(), raw_nu], 1.)
        optimizer.step()
        pruner.reset_momentum()
        pruner.apply_mask_to_weights()
        lr.step()
        if step % args.log_every == 0:
            print(f'{ident} {step+1}/{args.steps} loss={total.item():.6g} elapsed={previous_seconds+time.perf_counter()-started:.1f}s', flush=True)
        if (step+1) % 10000 == 0 or step+1 == args.steps:
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            temp = partial_path.with_suffix('.tmp')
            peaks = {name: max(previous_peaks[name], fn(device)) if device.type == 'cuda' else 0
                     for name, fn in [('gpu_peak_allocated_B', torch.cuda.max_memory_allocated),
                                      ('gpu_peak_reserved_B', torch.cuda.max_memory_reserved)]}
            torch.save(dict(protocol=protocol, steps_done=step+1,
                            model_state_dict=model.state_dict(), raw_nu=raw_nu.detach(),
                            optimizer_state_dict=optimizer.state_dict(), lr_state_dict=lr.state_dict(),
                            pruner_state_dict=pruner.state_dict(), initial_masks=initial_masks,
                            batch_rng_state=batch_rng.get_state(), gpu_peaks=peaks,
                            adam_seconds=previous_seconds+time.perf_counter()-started), temp)
            temp.replace(partial_path)
    if not adam_reused:
        adam_path.parent.mkdir(parents=True, exist_ok=True)
        cache_tmp = adam_path.with_suffix('.tmp')
        peaks = {name: max(previous_peaks[name], fn(device)) if device.type == 'cuda' else 0
                 for name, fn in [('gpu_peak_allocated_B', torch.cuda.max_memory_allocated),
                                  ('gpu_peak_reserved_B', torch.cuda.max_memory_reserved)]}
        torch.save(dict(protocol=adam_protocol, model_state_dict=model.state_dict(),
                        raw_nu=raw_nu.detach(), pruner_state_dict=pruner.state_dict(),
                        initial_masks=initial_masks, adam_seconds=previous_seconds+time.perf_counter()-started,
                        gpu_peaks=peaks), cache_tmp)
        cache_tmp.replace(adam_path)
    abs_err_train = abs(float(raw_nu.exp().detach())-.01/math.pi)
    # Release Adam state and last graph before measuring/refining with LBFGS.
    del optimizer, lr
    if first_step < args.steps:
        del total, ic, bc, pde, obs
    if 'drop' in locals():
        del drop, grow, physics, observed
    pruner.optimizer = None
    synchronize(device)
    adam_seconds = previous_seconds+time.perf_counter()-started
    masks = pruner.backward_masks
    lb_rng = torch.Generator(device=device).manual_seed(seed+200000)
    idx = torch.randperm(n, generator=lb_rng, device=device)[:args.lbfgs_points]
    lbfgs = torch.optim.LBFGS([*model.parameters(), raw_nu], lr=1., max_iter=args.lbfgs_iter,
                            history_size=args.history_size, line_search_fn='strong_wolfe')
    closures = 0
    lbfgs_trace = []
    def closure():
        nonlocal closures
        lbfgs.zero_grad(set_to_none=True)
        pruner.apply_mask_to_weights()
        loss = sum(loss_terms(model, raw_nu, data, idx))
        loss.backward()
        pruner.apply_mask_to_gradients()
        closures += 1
        lbfgs_trace.append(dict(closure=closures, iteration=int(lbfgs.state.get(next(model.parameters()), {}).get('n_iter', 0)), loss=float(loss.detach()), nu=float(raw_nu.exp().detach())))
        return loss
    if args.lbfgs_iter:
        lbfgs.step(closure)
    pruner.apply_mask_to_weights()
    lbfgs.zero_grad(set_to_none=True)
    final_terms = loss_terms(model, raw_nu, data, idx)
    final_loss = sum(final_terms)
    final_loss.backward()
    pruner.apply_mask_to_gradients()
    final_gradient_max = max(float(p.grad.detach().abs().max()) for p in [*model.parameters(), raw_nu] if p.grad is not None)
    losses_final = {name: float(value.detach()) for name, value in zip(('ic', 'bc', 'pde', 'data'), final_terms)}
    del final_loss, final_terms
    synchronize(device)
    elapsed = previous_seconds+time.perf_counter()-started
    changed = sum(int((a != b).sum()) for a, b in zip(initial_masks, masks) if a is not None)
    active = sum(int(a.sum()) for a in initial_masks if a is not None)
    union = sum(int((a | b).sum()) for a, b in zip(initial_masks, masks) if a is not None)
    intersection = sum(int((a & b).sum()) for a, b in zip(initial_masks, masks) if a is not None)
    row = dict(arm=arm, seed=seed, sparsity=sparsity,
               angle=float(arm[2:]) if arm.startswith('th') else None,
               stage='lbfgs' if args.lbfgs_iter else 'train', nu=float(raw_nu.exp().detach()),
               abs_err_train=abs_err_train, abs_err_lbfgs=abs(float(raw_nu.exp().detach())-.01/math.pi),
               turnover=1-intersection/union if union else 0., changed_mask_bits=changed,
               initial_active=active, growth_events=pruner.changed_connections,
               pruned_max_abs_after_lbfgs=frozen_mask_max(model, masks),
               adam_seconds=adam_seconds, training_seconds=elapsed, adam_reused=adam_reused,
               lbfgs_closures=closures, lbfgs_iterations=int(lbfgs.state.get(next(model.parameters()), {}).get('n_iter', 0)),
               losses_final=losses_final, final_loss=sum(losses_final.values()),
               final_gradient_max=final_gradient_max, lbfgs_trace=lbfgs_trace, protocol=protocol,
               gpu_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
               torch_version=torch.__version__, hip_version=torch.version.hip,
               gpu_peak_allocated_B=max(previous_peaks['gpu_peak_allocated_B'], torch.cuda.max_memory_allocated(device)) if device.type == 'cuda' else None,
               gpu_peak_reserved_B=max(previous_peaks['gpu_peak_reserved_B'], torch.cuda.max_memory_reserved(device)) if device.type == 'cuda' else None,
               **footprint(model, raw_nu, masks))
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_ckpt = ckpt_path.with_suffix('.tmp')
    torch.save(dict(model_state_dict=model.state_dict(), raw_nu=raw_nu.detach(),
                    nu=raw_nu.exp().detach(), pruner_state_dict=pruner.state_dict(),
                    initial_masks=initial_masks, metadata=row), tmp_ckpt)
    tmp_ckpt.replace(ckpt_path)
    row['checkpoint_B'] = ckpt_path.stat().st_size
    row['checkpoint'] = str(ckpt_path.resolve())
    atomic_json(result_path, row)
    partial_path.unlink(missing_ok=True)
    return row


@contextmanager
def exclusive_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Run already active: {path.name}') from exc
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def train_one(arm, seed, sparsity, args):
    ident = run_id(arm, seed, sparsity)
    with exclusive_lock(seed_directory(args.output, seed)/'.locks'/f'{ident}.lock'):
        return _train_one(arm, seed, sparsity, args)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', choices=(*ARMS, 'dense'), default='th45')
    p.add_argument('--seed', type=int, choices=SEEDS, default=2022)
    p.add_argument('--sparsity', type=float, default=.5)
    p.add_argument('--steps', type=int, default=10000)
    p.add_argument('--batch-size', type=int, default=6000)
    p.add_argument('--pruning-steps', type=int, default=None, help='Freeze topology after this step; default 75%% of Adam budget')
    p.add_argument('--delta', type=int, default=100)
    p.add_argument('--lbfgs-iter', type=int, default=2500)
    p.add_argument('--lbfgs-points', type=int, default=5000)
    p.add_argument('--history-size', type=int, default=50)
    p.add_argument('--log-every', type=int, default=1000)
    p.add_argument('--device', default='cuda')
    p.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    p.add_argument('--adam-cache', type=Path, default=None, help='Shared completed-Adam cache for budget comparisons; no LBFGS state is reused')
    p.add_argument('--output', type=Path, default=ROOT)
    p.add_argument('--overwrite', action='store_true')
    return p


def validate_args(args):
    if min(args.steps, args.batch_size, args.delta, args.lbfgs_points, args.history_size, args.log_every) < 1 or args.lbfgs_iter < 0:
        raise ValueError('Budgets must be positive; lbfgs-iter can be zero')
    if not args.dataset.is_file():
        raise FileNotFoundError(args.dataset)
    args.pruning_steps = args.pruning_steps if args.pruning_steps is not None else max(1, int(.75*args.steps))
    if not 1 <= args.pruning_steps <= args.steps:
        raise ValueError('Pruning steps must be in [1, Adam steps]')
    args.dataset_sha256 = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    args.output = args.output.resolve()
    return args


if __name__ == '__main__':
    args = validate_args(parser().parse_args())
    train_one(args.arm, args.seed, 0. if args.arm == 'dense' else args.sparsity, args)
