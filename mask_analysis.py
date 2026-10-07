from experiment_config import result_paths, seed_directory
"""Footprint and topology diagnostics from the unified JSON/checkpoint schema."""
import argparse
import json
from pathlib import Path
import pandas as pd
import torch
from train_dst import ROOT, IPINN, footprint, frozen_mask_max


def analyze(output=ROOT):
    rows = []
    for result in result_paths(output):
        row = json.loads(result.read_text())
        ck = torch.load(row['checkpoint'], map_location='cpu', weights_only=False)
        model = IPINN()
        model.load_state_dict(ck['model_state_dict'])
        masks = ck['pruner_state_dict']['backward_masks']
        initial = ck['initial_masks']
        intersection = sum(int((a & b).sum()) for a, b in zip(initial, masks) if a is not None)
        union = sum(int((a | b).sum()) for a, b in zip(initial, masks) if a is not None)
        rows.append({**row, **footprint(model, ck['raw_nu'], masks),
                     'turnover': 1-intersection/union if union else 0.,
                     'pruned_max_abs_after_lbfgs': frozen_mask_max(model, masks)})
    if not rows:
        raise ValueError('No unified results/checkpoints')
    frame = pd.DataFrame(rows)
    frame.drop(columns=['protocol'], errors='ignore').to_csv(output/'footprint.csv', index=False)
    frame[['arm', 'seed', 'sparsity', 'turnover', 'growth_events', 'pruned_max_abs_after_lbfgs']].to_csv(output/'turnover.csv', index=False)
    if not frame.loc[frame.arm.eq('random'), 'turnover'].eq(0).all():
        raise ValueError('Frozen random masks changed')
    if not frame.pruned_max_abs_after_lbfgs.eq(0).all():
        raise ValueError('Masked weights regrew during LBFGS')
    return frame


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=ROOT)
    analyze(p.parse_args().output)
