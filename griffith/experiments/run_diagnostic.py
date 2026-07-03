"""
CLI entry point for a single elastic diagnostic experiment.

Usage:
    python experiments/run_diagnostic.py --coord_scheme both --precision fp32 --arch standard --seed 0

Results are written under experiments/results/ (never under griffith/results/,
which is reserved for the formal training entry point train.py), keeping
experiment artifacts fully separated from production checkpoints.
"""
import argparse
import csv
import os
import sys

import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from experiments.diagnostics import run_elastic_diagnostic
from experiments.metrics import compute_metrics, evaluate_on_grid

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
RUNS_DIR = os.path.join(RESULTS_DIR, 'runs')
METRICS_CSV = os.path.join(RESULTS_DIR, 'metrics.csv')

FIELDNAMES = [
    'tag', 'coord_scheme', 'precision', 'arch', 'seed',
    'peak_loc_err', 'peak_mag_ratio', 'l2_rel_err', 'bc_err', 'sym_err', 'equil_final',
]


def load_base_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_one(coord_scheme='both', precision='fp32', arch='standard', seed=0,
            steps=None, save_plot=True, cfg=None):
    """Run a single (coord_scheme, precision, arch, seed) configuration end-to-end."""
    cfg = cfg if cfg is not None else load_base_config()
    result = run_elastic_diagnostic(
        cfg, coord_scheme=coord_scheme, precision=precision, arch=arch, seed=seed, steps=steps)
    metrics = compute_metrics(result, cfg['physics'], cfg)

    tag = f"{coord_scheme}_{precision}_{arch}_seed{seed}"
    row = {'tag': tag, 'coord_scheme': coord_scheme, 'precision': precision,
           'arch': arch, 'seed': seed, **metrics}

    _append_metrics_row(row)
    if save_plot:
        _save_diagnostic_plot(result, cfg, tag)

    return row


def _append_metrics_row(row):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    file_exists = os.path.exists(METRICS_CSV)
    with open(METRICS_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _save_diagnostic_plot(result, cfg, tag):
    os.makedirs(RUNS_DIR, exist_ok=True)
    grid = evaluate_on_grid(result, cfg['physics'])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.pcolormesh(grid['X'], grid['Y'], grid['SIG'], cmap='jet', shading='auto')
    ax.set_title(f'sigma_yy  [{tag}]')
    ax.set_aspect('equal')
    fig.colorbar(im, ax=ax)
    out_path = os.path.join(RUNS_DIR, f'{tag}.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved figure -> {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a single elastic diagnostic experiment')
    parser.add_argument('--coord_scheme', choices=['cartesian', 'elliptic', 'both', 'williams'],
                         default='both')
    parser.add_argument('--precision', choices=['fp32', 'fp64'], default='fp32')
    parser.add_argument('--arch', choices=['standard', 'mixed'], default='standard')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=None)
    args = parser.parse_args()

    result_row = run_one(args.coord_scheme, args.precision, args.arch, args.seed, args.steps)
    print(result_row)
