"""Wait for the successful sweep, export CSVs, run tests, and save plots."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def run(output, dataset, device):
    from main import evaluate
    from mask_analysis import analyze as masks
    from stats_analysis import analyze
    from experiment_config import SEEDS
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd

    evaluate(output, dataset, device)
    masks(output)
    analyze(output, output/'statistics')
    script = Path(__file__).with_name('plot.py')
    for seed in SEEDS:
        subprocess.run([sys.executable, str(script), '--csv-dir',
                        str(output/f'seed{seed}'/'results_csv'), '--out-dir',
                        str(output/f'seed{seed}'/'plots')], check=True)
    # Aggregate the supplied plotter's tables by median across seeds.
    aggregate = output/'results_csv'
    for filename in ('table3_model1_detail.csv', 'table4_model2_detail.csv'):
        df = pd.concat([pd.read_csv(output/f'seed{s}'/'results_csv'/filename) for s in SEEDS])
        df.to_csv(aggregate/filename.replace('.csv', '_all_seeds.csv'), index=False)
        keys = ['Sparsity', 'Stage', 'Arm']
        summary = df.groupby(keys, as_index=False).agg(
            Angle=('Angle', 'first'), nu=('nu', 'median'), AbsErr=('AbsErr', 'median'),
            **{'RelErr_%': ('RelErr_%', 'median')})
        summary.to_csv(aggregate/filename, index=False)
    subprocess.run([sys.executable, str(script), '--csv-dir', str(aggregate),
                    '--out-dir', str(output/'plots')], check=True)
    summary = pd.read_csv(output/'statistics'/'summary_by_sparsity.csv')
    fig, ax = plt.subplots(figsize=(9, 5))
    for arm, g in summary.groupby('arm'):
        g = g.sort_values('sparsity')
        ax.plot(g.sparsity, g.median_log10_abs_err, marker='o', label=arm)
        ax.fill_between(g.sparsity, g.ci_lo, g.ci_hi, alpha=.12)
    ax.set(xlabel='Target sparsity', ylabel='Median log10 absolute viscosity error',
           title=f'{len(SEEDS)} seeds; shaded seed bootstrap 95% intervals')
    ax.legend(); fig.tight_layout()
    fig.savefig(output/'plots'/'ablation_seed_intervals.png', dpi=220)
    plt.close(fig)
    memory = pd.read_csv(output/'statistics'/'footprint.csv')
    fig, ax = plt.subplots(figsize=(9, 5))
    for column, label in [('dense_B', 'Dense parameters'),
                          ('model_and_masks_B', 'Dense parameters + masks'),
                          ('csr_B', 'Estimated CSR payload')]:
        g = memory[memory.arm.eq('th45')].sort_values('sparsity')
        ax.plot(g.sparsity, g[column]/1024, marker='o', label=label)
    ax.set(xlabel='Target sparsity', ylabel='KiB', title='Model1 45°: measured storage and estimated compression')
    ax.legend(); fig.tight_layout()
    fig.savefig(output/'plots'/'memory_footprint.png', dpi=220)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--wait', action='store_true')
    args = p.parse_args()
    status = args.output/'postprocess_status.json'
    def write(state, **details):
        temporary = status.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(state=state, pid=os.getpid(), updated_unix=time.time(), **details), indent=2))
        temporary.replace(status)
    try:
        if args.wait:
            write('waiting_for_sweep')
            while True:
                progress = json.loads((args.output/'progress.json').read_text())
                if progress['failures']:
                    raise RuntimeError('Training failed; refusing partial statistical analysis')
                if progress['completed'] == progress['total'] and not progress['active']:
                    # The supervisor runs its own statistics before exiting.
                    try:
                        os.kill(progress['supervisor_pid'], 0)
                    except ProcessLookupError:
                        break
                else:
                    try:
                        os.kill(progress['supervisor_pid'], 0)
                    except ProcessLookupError:
                        raise RuntimeError('Supervisor exited before completing the sweep')
                time.sleep(20)
        write('processing')
        run(args.output, args.dataset, args.device)
        write('complete')
        print('CSV export, statistics, mask checks, and plots complete.', flush=True)
    except Exception as error:
        write('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
