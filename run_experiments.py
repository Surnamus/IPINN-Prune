"""Bounded subprocess sweep; each run has its own RNG, log and artifact lock."""
import argparse
import json
import os
import subprocess
import signal
import sys
import time
from pathlib import Path
from experiment_config import SEEDS, SPARSITIES, ARMS, ANGLE_ARMS, run_id, seed_directory


def phase_for(arm):
    return 'model1' if arm in ANGLE_ARMS else {'m2': 'model2', 'random': 'random', 'dense': 'dense'}[arm]


def plan(seeds=SEEDS, include_dense=True):
    # All Model 1 runs must finish before any Model 2 run starts.
    for arm in ANGLE_ARMS:
        for sparsity in SPARSITIES:
            for seed in seeds:
                yield arm, seed, sparsity
    for arm in ('m2', 'random'):
        for sparsity in SPARSITIES:
            for seed in seeds:
                yield arm, seed, sparsity
    if include_dense:
        for seed in seeds:
            yield 'dense', seed, 0.


def worker_command(arm, seed, sparsity, args):
    cmd = [sys.executable, '-u', str(Path(__file__).with_name('train_dst.py')),
           '--arm', arm, '--seed', str(seed), '--sparsity', str(sparsity)]
    for key in ('steps', 'pruning_steps', 'batch_size', 'delta', 'lbfgs_iter',
                'lbfgs_points', 'history_size', 'log_every', 'device', 'dataset', 'output'):
        cmd.extend(['--'+key.replace('_', '-'), str(getattr(args, key))])
    if args.overwrite:
        cmd.append('--overwrite')
    return cmd


def sweep(args):
    from train_dst import atomic_json
    seeds = tuple(s for s in SEEDS if getattr(args, 'seed_start', SEEDS[0]) <= s <= getattr(args, 'seed_end', SEEDS[-1]))
    include_dense = not getattr(args, 'skip_dense', False)
    for seed in seeds:
        (seed_directory(args.output, seed)/'logs').mkdir(parents=True, exist_ok=True)
    jobs = list(plan(seeds, include_dense))
    manifest = dict(seeds=seeds, analysis_seeds=SEEDS, sparsities=SPARSITIES, angles=(0, 45, 90, 135),
                    n_jobs=len(jobs), workers=args.workers, steps=args.steps,
                    pruning_steps=args.pruning_steps, lbfgs_iter=args.lbfgs_iter,
                    batch_size=args.batch_size, lbfgs_points=args.lbfgs_points, model2_jobs=len(seeds)*len(SPARSITIES),
                    result_layout='seedNNNN/{checkpoints,results_v2,logs,adam_cache}',
                    phase_order=['model1', 'model2', 'random', 'dense'],
                    jobs=[dict(arm=a, seed=seed, sparsity=s) for a, seed, s in jobs])
    atomic_json(args.output/'manifest.json', manifest)
    active, failures, completed, cursor = {}, [], 0, 0
    current_phase, finished_phases = None, []
    launched_at = {}
    env = os.environ.copy()
    env.setdefault('OMP_NUM_THREADS', '1')
    env.setdefault('MKL_NUM_THREADS', '1')
    try:
        while cursor < len(jobs) or active:
            while cursor < len(jobs) and len(active) < args.workers:
                arm, seed, sparsity = jobs[cursor]
                phase = phase_for(arm)
                if active and phase != current_phase:
                    break  # do not fill a spare slot with the next phase early
                if phase != current_phase:
                    current_phase = phase
                    print(f'PHASE_STARTED {phase}', flush=True)
                ident = run_id(arm, seed, sparsity)
                log = (seed_directory(args.output, seed)/'logs'/f'{ident}.log').open('a', buffering=1)
                cmd = worker_command(arm, seed, sparsity, args)
                process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
                active[ident] = (process, log)
                launched_at[ident] = time.monotonic()
                cursor += 1
                print(f'LAUNCHED {ident} pid={process.pid}', flush=True)
            for ident, (process, log) in list(active.items()):
                code = process.poll()
                if code is None and time.monotonic()-launched_at[ident] > getattr(args, 'worker_timeout', 600):
                    print(f'TIMED_OUT {ident}', flush=True)
                    process.kill()
                    code = process.wait()
                if code is not None:
                    log.close()
                    del active[ident]
                    if code:
                        failures.append(dict(run=ident, exit_code=code))
                        print(f'FAILED {ident}; inspect seedNNNN/logs/{ident}.log', flush=True)
                    else:
                        completed += 1
                        print(f'FINISHED {ident} ({completed}/{len(jobs)})', flush=True)
            if not active and current_phase is not None and (cursor == len(jobs) or phase_for(jobs[cursor][0]) != current_phase):
                if current_phase not in finished_phases and not failures:
                    finished_phases.append(current_phase)
                    print(f'PHASE_FINISHED {current_phase}', flush=True)
            atomic_json(args.output/'progress.json', dict(
                supervisor_pid=os.getpid(), completed=completed, total=len(jobs),
                active={ident: p.pid for ident, (p, _) in active.items()},
                launched=cursor, failures=failures, phase=current_phase, completed_phases=finished_phases, updated_unix=time.time()))
            # Stop launching after an error instead of repeating 366 failures.
            if failures:
                cursor = len(jobs)
                break  # finally stops other workers, including a GPU-stalled worker
            if active:
                time.sleep(2)
    finally:
        for process, log in active.values():
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            log.close()
    if failures:
        raise RuntimeError(f'{len(failures)} worker(s) failed; logs are preserved. Resume after correcting the cause.')
    if args.analyze:
        from stats_analysis import analyze
        analyze(args.output, args.output/'statistics')


def main():
    from train_dst import parser, validate_args, exclusive_lock
    p = parser()
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--analyze', action='store_true')
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--worker-timeout', type=float, default=600)
    p.add_argument('--seed-start', type=int, default=SEEDS[0])
    p.add_argument('--seed-end', type=int, default=SEEDS[-1])
    p.add_argument('--skip-dense', action='store_true')
    args = p.parse_args()
    selected = tuple(s for s in SEEDS if args.seed_start <= s <= args.seed_end)
    if not selected or args.seed_start not in SEEDS or args.seed_end not in SEEDS:
        p.error('Seed range must be within the configured seeds')
    if args.workers < 1:
        p.error('workers must be >= 1')
    print(f'Seeds: {selected}\nSparsities: {SPARSITIES}\nAngles: 0, 45, 90, 135 (all trained separately)')
    print(f'{len(list(plan(selected, not args.skip_dense)))} jobs; Adam={args.steps}, pruning={args.pruning_steps or int(.75*args.steps)}, LBFGS max_iter={args.lbfgs_iter}; workers={args.workers}', flush=True)
    if args.plan_only:
        return
    validate_args(args)
    def stop(signum, frame):
        raise KeyboardInterrupt('Sweep stopped')
    signal.signal(signal.SIGTERM, stop)
    with exclusive_lock(args.output/'.sweep.lock'):
        sweep(args)


if __name__ == '__main__':
    main()
