from pathlib import Path
"""Predeclared experimental design; shared by runner and statistics."""
SEEDS = tuple(range(2022, 2034))
# No prior sparsity grid was supplied: ten levels below full disconnection.
SPARSITIES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
ANGLES = (0, 45, 90, 135)
ANGLE_ARMS = tuple(f'th{a}' for a in ANGLES)
ARMS = (*ANGLE_ARMS, 'm2', 'random')


def canonical_arm(arm):
    return arm


def run_id(arm, seed, sparsity):
    return f'{arm}_seed{seed}_sparsity{sparsity:g}'


def seed_directory(output, seed):
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError('Unexpected seed')
    return Path(output)/f'seed{seed}'


def result_paths(output):
    output = Path(output)
    if output.name == 'results_v2':
        folders = [output]
    else:
        folders = [d for d in [output/'results_v2', *output.glob('seed*/results_v2')] if d.is_dir()]
        if not folders:
            folders = [output]  # explicit legacy result folder or synthetic test data
    return sorted(p for folder in folders for p in folder.glob('*.json'))
