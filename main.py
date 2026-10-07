from experiment_config import result_paths, seed_directory
"""Evaluate the unified checkpoints and export legacy-compatible per-seed CSVs."""
import argparse
import csv
import json
from pathlib import Path
import torch
from train_dst import IPINN, ROOT, DEFAULT_DATASET
from cfdsolver import get_static_dataset

PINN = IPINN


def write_csv(path, headers, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)


def evaluate(output=ROOT, dataset=DEFAULT_DATASET, device='cpu'):
    output = Path(output)
    data = get_static_dataset(str(dataset), device=device)
    # Cached exact solution: avoid recomputing expensive quadrature for evaluation.
    x = torch.as_tensor(data['X'].ravel(), device=device, dtype=torch.float32)[:, None]
    t = torch.as_tensor(data['T'].ravel(), device=device, dtype=torch.float32)[:, None]
    target = torch.as_tensor(data['vu'].ravel(), device=device, dtype=torch.float32)[:, None]
    rows = []
    for path in result_paths(output):
        row = json.loads(path.read_text())
        model = IPINN().to(device)
        ck = torch.load(row['checkpoint'], map_location=device, weights_only=False)
        model.load_state_dict(ck['model_state_dict'])
        model.eval()
        error_sum = 0.
        with torch.no_grad():
            for start in range(0, len(x), 8192):
                stop = start+8192
                error_sum += float((model(x[start:stop], t[start:stop])-target[start:stop]).square().sum())
        rows.append({**row, 'field_mse_truth': error_sum/len(x)})
    if not rows:
        raise ValueError('No unified results; run run_experiments.py first')
    for seed in sorted({r['seed'] for r in rows}):
        for model_name, arms in [('model1', ('th0', 'th45', 'th90', 'th135')), ('model2', ('m2',))]:
            group = [r for r in rows if r['seed'] == seed and r['arm'] in arms]
            group.sort(key=lambda r: r['abs_err_lbfgs'])
            values = [[i+1, r['sparsity'], r['angle'], r['stage'], r['nu'],
                       r['abs_err_lbfgs'], r['abs_err_lbfgs']/(.01/torch.pi)*100,
                       seed, r['arm'], r['field_mse_truth']] for i, r in enumerate(group)]
            filename = 'table3_model1_detail.csv' if model_name == 'model1' else 'table4_model2_detail.csv'
            write_csv(seed_directory(output, seed)/'results_csv'/filename,
                      ['Rank', 'Sparsity', 'Angle', 'Stage', 'nu', 'AbsErr', 'RelErr_%', 'Seed', 'Arm', 'FieldMSE'], values)
    keys = ['arm', 'seed', 'sparsity', 'angle', 'stage', 'nu', 'abs_err_train',
            'abs_err_lbfgs', 'field_mse_truth', 'dense_B', 'ideal_B', 'csr_B',
            'model_and_masks_B', 'gpu_peak_allocated_B', 'gpu_peak_reserved_B']
    write_csv(output/'results_csv'/'all_results.csv', keys, [[r.get(k) for k in keys] for r in rows])
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=ROOT)
    p.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    p.add_argument('--device', default='cpu')
    p.add_argument('--stats', action='store_true')
    args = p.parse_args()
    evaluate(args.output, args.dataset, args.device)
    if args.stats:
        from stats_analysis import analyze
        analyze(args.output, args.output/'statistics')
