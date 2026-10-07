"""Export final inference models as dense or hybrid CSR NPZ; measure real files.

Both formats use uncompressed NPZ containers. CSR includes int32 column indices
and row pointers. Biases and raw_nu remain dense. Loading reconstructs dense
PyTorch tensors: these exports measure storage, not sparse training/inference RAM.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from experiment_config import result_paths, run_id
from train_dst import IPINN


def save_model(path, state, raw_nu, compressed):
    arrays, entries = {}, []
    for i, (name, tensor) in enumerate([*state.items(), ('raw_nu', raw_nu)]):
        a = tensor.detach().cpu().numpy().copy()
        entry = dict(name=name, shape=list(a.shape), key=f'a{i}', storage='dense')
        if compressed and a.ndim == 2:
            row, col = np.nonzero(a)
            values = a[row, col]
            columns = col.astype(np.int32)
            pointers = np.concatenate(([0], np.cumsum(np.bincount(row, minlength=a.shape[0])))).astype(np.int32)
            if values.nbytes + columns.nbytes + pointers.nbytes < a.nbytes:
                entry['storage'] = 'csr'
                arrays[f'a{i}_values'] = values
                arrays[f'a{i}_columns'] = columns
                arrays[f'a{i}_pointers'] = pointers
            else:
                arrays[f'a{i}'] = a
        else:
            arrays[f'a{i}'] = a
        entries.append(entry)
    payload = sum(a.nbytes for a in arrays.values())
    metadata = dict(version=1, architecture='IPINN_2_20x8_1_tanh', entries=entries)
    arrays['metadata'] = np.frombuffer(json.dumps(metadata, separators=(',', ':')).encode(), dtype=np.uint8)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream, **arrays)
    temporary.replace(path)
    return payload


def load_model(path):
    state = {}
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(archive['metadata'].tobytes())
        if metadata['version'] != 1:
            raise ValueError('Unsupported export version')
        for e in metadata['entries']:
            key = e['key']
            if e['storage'] == 'dense':
                a = archive[key].copy()
            else:
                values, columns, pointers = (archive[key+s] for s in ('_values', '_columns', '_pointers'))
                a = np.zeros(e['shape'], dtype=values.dtype)
                for row in range(a.shape[0]):
                    start, stop = pointers[row:row+2]
                    a[row, columns[start:stop]] = values[start:stop]
            state[e['name']] = torch.from_numpy(a)
    raw_nu = state.pop('raw_nu')
    model = IPINN()
    model.load_state_dict(state)
    return model.eval(), raw_nu


def export_all(results, export_root=None, limit=None):
    torch.set_num_threads(1)
    output = Path(export_root or results)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    x, t = torch.meshgrid(torch.linspace(-1, 1, 257), torch.linspace(0, 1, 101), indexing='ij')
    x, t = x.reshape(-1, 1), t.reshape(-1, 1)
    paths = result_paths(results)
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise ValueError('No final results found')
    for path in paths:
        row = json.loads(path.read_text())
        if row['stage'] != 'lbfgs':
            raise ValueError(f'Not a final model: {path}')
        ck = torch.load(row['checkpoint'], map_location='cpu', weights_only=False)
        original = IPINN().eval()
        original.load_state_dict(ck['model_state_dict'])
        folder = output/f"seed{row['seed']}"/'exported_models'
        folder.mkdir(parents=True, exist_ok=True)
        ident = run_id(row['arm'], row['seed'], row['sparsity'])
        dense, sparse = folder/f'{ident}_dense.npz', folder/f'{ident}_csr.npz'
        dense_payload = save_model(dense, original.state_dict(), ck['raw_nu'], False)
        csr_payload = save_model(sparse, original.state_dict(), ck['raw_nu'], True)
        loaded, nu = load_model(sparse)
        if not torch.equal(nu, ck['raw_nu'].cpu()) or any(
                not torch.equal(v, loaded.state_dict()[k]) for k, v in original.state_dict().items()):
            raise ValueError(f'Export changed model tensors: {ident}')
        max_error = 0.
        with torch.no_grad():
            for start in range(0, len(x), 8192):
                end = start+8192
                max_error = max(max_error, float((original(x[start:end], t[start:end])-loaded(x[start:end], t[start:end])).abs().max()))
        if max_error != 0:
            raise ValueError(f'Export changed predictions: {ident}')
        dense_bytes, csr_bytes = dense.stat().st_size, sparse.stat().st_size
        rows.append(dict(arm=row['arm'], seed=row['seed'], sparsity=row['sparsity'],
                         nu=float(nu.exp()), dense_payload_B=dense_payload, csr_payload_B=csr_payload,
                         dense_file_B=dense_bytes, csr_file_B=csr_bytes,
                         saving_pct=100*(1-csr_bytes/dense_bytes),
                         prediction_max_abs_diff=max_error, tensors_exact=True,
                         dense_file=str(dense), csr_file=str(sparse)))
    target = output/'results_csv'/'final_model_sizes.csv'
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    for arm in sorted({r['arm'] for r in rows}):
        group = [r for r in rows if r['arm'] == arm]
        levels = sorted({r['sparsity'] for r in group})
        ax.plot(levels, [np.median([r['saving_pct'] for r in group if r['sparsity'] == s]) for s in levels], marker='o', label=arm)
    ax.axhline(0, color='gray', linewidth=.7)
    ax.set(xlabel='Target sparsity', ylabel='Median file size saving (%)',
           title='Final models: hybrid CSR versus dense NPZ (including indices and metadata)')
    ax.legend(); fig.tight_layout()
    (output/'plots').mkdir(exist_ok=True)
    fig.savefig(output/'plots'/'final_model_storage_savings.png', dpi=220)
    plt.close(fig)
    print(f'Exported and verified {len(rows)} final models: {target}', flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--export-root', type=Path)
    p.add_argument('--limit', type=int)
    p.add_argument('--wait', action='store_true', help='Wait for queued CSV/statistics/plot postprocessing to complete')
    args = p.parse_args()
    status = (args.export_root or args.results)/'export_status.json'
    def write(state, **details):
        status.parent.mkdir(parents=True, exist_ok=True)
        temporary = status.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(state=state, pid=os.getpid(), updated_unix=time.time(), **details), indent=2))
        temporary.replace(status)
    try:
        write('waiting_for_postprocessing' if args.wait else 'exporting')
        while args.wait:
            previous = json.loads((args.results/'postprocess_status.json').read_text())
            if previous['state'] == 'complete':
                break
            if previous['state'] == 'failed':
                raise RuntimeError('Postprocessing failed; exports not started')
            try:
                os.kill(previous['pid'], 0)
            except ProcessLookupError:
                raise RuntimeError('Postprocessor stopped before completion')
            time.sleep(20)
        write('exporting')
        rows = export_all(args.results, args.export_root, args.limit)
        write('complete', models=len(rows))
    except Exception as error:
        write('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
