"""
L5 tier-3 factor validation: Deep Energy Method (DEM) vs strong-form PDE
residual, at fixed coord_scheme=elliptic, arch=y_sym, w_bc_u=400 (current
production-recommended baseline; see docs/griffith-ablation-study.md \u00a79).

Motivation (docs/griffith-ablation-study.md \u00a711.3, candidate C3): the
strong-form u_net loss needs SECOND spatial derivatives (equilibrium PDE
residual = d(stress)/d(x) = d^2(u)/d(x)^2), which requires create_graph=True
double backward -- empirically the dominant per-iteration cost (see
docs/griffith-ablation-study.md \u00a712 speed measurements). DEM instead
minimizes the total potential energy functional directly (only FIRST
derivatives needed: strain = d(u)/d(x)), using PFPhysicsEvaluator's
compute_u_loss_dem(). It also naturally satisfies traction-free Neumann BCs
(right free edge) with no penalty term at all.

This script validates DEM two ways, at matched (coord_scheme, arch, w_bc_u):
  1. Physics correctness: does it converge to the same sigma_yy field as the
     strong-form baseline (peak_loc_err / peak_mag_ratio / l2_rel_err /
     sym_err all in the same ballpark)?
  2. Speed: wall-clock seconds per run (same steps count), to directly
     quantify the claimed speedup from dropping to first-derivative-only.

NOTE: 'equil_final' means something different for DEM rows (it is the final
total potential ENERGY value, not a PDE residual near 0) -- see
compute_u_loss_dem()'s docstring. Do not compare equil_final directly across
use_dem=True/False rows; use l2_rel_err / peak_loc_err / peak_mag_ratio /
sym_err / bc_err for the physics comparison, and train_seconds for the speed
comparison.

Usage (meant to be run on a server, not on this machine):
    python experiments/l5_dem_validation.py --seeds 0 1 2 3 4 5 6 7
"""
import argparse
import copy
import csv
import os
import sys
import time

import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from experiments.diagnostics import run_elastic_diagnostic
from experiments.metrics import compute_metrics

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
METRICS_CSV = os.path.join(RESULTS_DIR, 'l5_dem_validation_metrics.csv')

FIELDNAMES = [
    'tag', 'coord_scheme', 'precision', 'arch', 'w_bc_u', 'use_dem', 'seed',
    'peak_loc_err', 'peak_mag_ratio', 'l2_rel_err', 'bc_err', 'sym_err', 'equil_final',
    'train_seconds',
]


def load_base_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def make_tag(coord_scheme, precision, arch, w_bc_u, use_dem, seed):
    dem_tag = 'dem' if use_dem else 'strong'
    return f"{coord_scheme}_{precision}_{arch}_wbc{w_bc_u:g}_{dem_tag}_seed{seed}"


def load_completed_rows():
    rows = {}
    if not os.path.exists(METRICS_CSV):
        return rows
    with open(METRICS_CSV, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['seed'] = int(row['seed'])
            row['w_bc_u'] = float(row['w_bc_u'])
            row['use_dem'] = row['use_dem'] in ('True', 'true', '1')
            for col in ('peak_loc_err', 'peak_mag_ratio', 'l2_rel_err',
                        'bc_err', 'sym_err', 'equil_final', 'train_seconds'):
                row[col] = float(row[col])
            rows[row['tag']] = row
    return rows


def _append_metrics_row(row):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    file_exists = os.path.exists(METRICS_CSV)
    with open(METRICS_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run_one(coord_scheme, precision, arch, w_bc_u, use_dem, seed, steps, base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg['weighting']['w_bc_u'] = w_bc_u

    t0 = time.perf_counter()
    result = run_elastic_diagnostic(
        cfg, coord_scheme=coord_scheme, precision=precision, arch=arch,
        seed=seed, steps=steps, use_dem=use_dem)
    train_seconds = time.perf_counter() - t0

    metrics = compute_metrics(result, cfg['physics'], cfg)

    tag = make_tag(coord_scheme, precision, arch, w_bc_u, use_dem, seed)
    row = {'tag': tag, 'coord_scheme': coord_scheme, 'precision': precision,
           'arch': arch, 'w_bc_u': w_bc_u, 'use_dem': use_dem, 'seed': seed,
           **metrics, 'train_seconds': train_seconds}
    _append_metrics_row(row)
    return row


def run_sweep(coord_scheme, precision, arch, w_bc_u, seeds, steps=None, skip_existing=True):
    base_cfg = load_base_config()
    completed = load_completed_rows() if skip_existing else {}
    rows = []
    for use_dem in (False, True):
        for seed in seeds:
            tag = make_tag(coord_scheme, precision, arch, w_bc_u, use_dem, seed)
            if skip_existing and tag in completed:
                print(f"Skipping already-completed run: {tag}")
                rows.append(completed[tag])
                continue
            print(f"Running: {tag}")
            row = run_one(coord_scheme, precision, arch, w_bc_u, use_dem, seed, steps, base_cfg)
            rows.append(row)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='L5 tier-3 factor validation: Deep Energy Method (DEM) vs '
                     'strong-form PDE residual, at fixed coord_scheme=elliptic, '
                     'arch=y_sym, w_bc_u=400.')
    parser.add_argument('--coord_scheme', default='elliptic',
                         choices=['cartesian', 'elliptic', 'both', 'williams'])
    parser.add_argument('--precision', default='fp32', choices=['fp32', 'fp64'])
    parser.add_argument('--arch', default='y_sym', choices=['standard', 'y_sym', 'y_sym_hard'])
    parser.add_argument('--w_bc_u', type=float, default=400.0)
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(8)))
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--no_skip_existing', action='store_true')
    args = parser.parse_args()

    all_rows = run_sweep(
        coord_scheme=args.coord_scheme, precision=args.precision, arch=args.arch,
        w_bc_u=args.w_bc_u, seeds=args.seeds, steps=args.steps,
        skip_existing=not args.no_skip_existing)

    for use_dem in (False, True):
        group = [r for r in all_rows if r['use_dem'] == use_dem]
        if not group:
            continue
        import numpy as np
        print(f"\n{'='*70}\nuse_dem={use_dem}  (n={len(group)})\n{'='*70}")
        for col in ('peak_loc_err', 'peak_mag_ratio', 'l2_rel_err', 'bc_err',
                    'sym_err', 'equil_final', 'train_seconds'):
            vals = np.array([r[col] for r in group], dtype=float)
            print(f"  {col:16s}: mean={np.nanmean(vals):.4e}  std={np.nanstd(vals):.4e}")
