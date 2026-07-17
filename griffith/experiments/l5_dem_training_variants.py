"""
L5 tier-3 factor validation, stages 1-3: DEM training-loop variants inspired
by Manav et al. 2024 (Deep Ritz Method for phase-field fracture) -- see
docs/griffith-drm-insights-experiment-plan.md §2.1 / §3 (candidates O7, O8,
O9). All three factors are "drop-in" training-loop changes (no change to
network architecture or geometry/integration scheme) and compose freely with
each other and with use_dem in {False, True}:

  O7 (--optimizer)  : Adam -> RPROP
  O8 (--l2_reg)     : L2 weight regularization on u_net's parameters
  O9 (--log_energy) : log(energy) loss instead of raw energy (DEM only)
  C6 (--dem_quadratures) : 'mc' (random importance-weighted Monte Carlo,
                     default) vs 'deterministic' (tip-graded deterministic
                     quadrature grid, DEM only) -- see
                     docs/griffith-drm-insights-experiment-plan.md §2.2/§3
                     stage 5.

This is deliberately a SEPARATE script (and a separate results CSV) from
l5_dem_validation.py so the existing baseline data in
experiments/results/l5_dem_validation_metrics.csv (coord_scheme=elliptic,
arch=y_sym, w_bc_u=400, Adam, no log_energy, no l2_reg) is never touched --
it remains the fixed baseline this script's rows are compared against (see
docs/griffith-drm-insights-experiment-plan.md §一/§四 for the baseline table
and acceptance thresholds).

Usage (meant to be run on a server, not on this machine), e.g. stage 1 only:
    python experiments/l5_dem_training_variants.py --optimizers adam rprop \
        --seeds 0 1 2 3 4 5 6 7

Stage 2 (log-energy, DEM only) layered on top of the best stage-1 optimizer:
    python experiments/l5_dem_training_variants.py --optimizers rprop \
        --log_energy --dem_only --seeds 0 1 2 3 4 5 6 7

Stage 3 (L2 regularization sweep), layered on top of stage 1+2 winners:
    python experiments/l5_dem_training_variants.py --optimizers rprop \
        --log_energy --l2_regs 0 1e-6 1e-5 1e-4 --dem_only \
        --seeds 0 1 2 3

Stage 5 (C6: deterministic quadrature grid), layered on top of the stage
2 winner (log_energy=True), DEM only:
    python experiments/l5_dem_training_variants.py --log_energy \
        --dem_quadratures mc deterministic --dem_only \
        --seeds 0 1 2 3 4 5 6 7
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
METRICS_CSV = os.path.join(RESULTS_DIR, 'l5_dem_training_variants_metrics.csv')

FIELDNAMES = [
    'tag', 'coord_scheme', 'precision', 'arch', 'w_bc_u', 'use_dem', 'seed',
    'optimizer', 'log_energy', 'l2_reg', 'dem_quadrature',
    'peak_loc_err', 'peak_mag_ratio', 'l2_rel_err', 'bc_err', 'sym_err', 'equil_final',
    'train_seconds',
]


def load_base_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def make_tag(coord_scheme, precision, arch, w_bc_u, use_dem, seed, optimizer, log_energy, l2_reg,
             dem_quadrature='mc'):
    dem_tag = 'dem' if use_dem else 'strong'
    tag = f"{coord_scheme}_{precision}_{arch}_wbc{w_bc_u:g}_{dem_tag}_seed{seed}"
    if optimizer != 'adam':
        tag += f"_opt{optimizer}"
    if log_energy:
        tag += "_logE"
    if l2_reg > 0.0:
        tag += f"_l2reg{l2_reg:g}"
    if dem_quadrature != 'mc':
        tag += f"_quad{dem_quadrature}"
    return tag


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
            row['log_energy'] = row['log_energy'] in ('True', 'true', '1')
            row['l2_reg'] = float(row['l2_reg'])
            row['dem_quadrature'] = row.get('dem_quadrature') or 'mc'
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


def run_one(coord_scheme, precision, arch, w_bc_u, use_dem, seed, optimizer,
            log_energy, l2_reg, steps, base_cfg, dem_quadrature='mc'):
    cfg = copy.deepcopy(base_cfg)
    cfg['weighting']['w_bc_u'] = w_bc_u

    t0 = time.perf_counter()
    result = run_elastic_diagnostic(
        cfg, coord_scheme=coord_scheme, precision=precision, arch=arch,
        seed=seed, steps=steps, use_dem=use_dem, optimizer=optimizer,
        log_energy=log_energy, l2_reg=l2_reg, dem_quadrature=dem_quadrature)
    train_seconds = time.perf_counter() - t0

    metrics = compute_metrics(result, cfg['physics'], cfg)

    tag = make_tag(coord_scheme, precision, arch, w_bc_u, use_dem, seed,
                    optimizer, log_energy, l2_reg, dem_quadrature)
    row = {'tag': tag, 'coord_scheme': coord_scheme, 'precision': precision,
           'arch': arch, 'w_bc_u': w_bc_u, 'use_dem': use_dem, 'seed': seed,
           'optimizer': optimizer, 'log_energy': log_energy, 'l2_reg': l2_reg,
           'dem_quadrature': dem_quadrature,
           **metrics, 'train_seconds': train_seconds}
    _append_metrics_row(row)
    return row


def run_sweep(coord_scheme, precision, arch, w_bc_u, seeds, optimizers,
              log_energies, l2_regs, dem_only=False, steps=None, skip_existing=True,
              dem_quadratures=('mc',)):
    base_cfg = load_base_config()
    completed = load_completed_rows() if skip_existing else {}
    rows = []
    use_dem_choices = (True,) if dem_only else (False, True)
    for use_dem in use_dem_choices:
        for optimizer in optimizers:
            for log_energy in log_energies:
                if log_energy and not use_dem:
                    # compute_u_loss (strong-form) has no energy term to
                    # log-transform -- skip this invalid combination.
                    continue
                for l2_reg in l2_regs:
                    for dem_quadrature in dem_quadratures:
                        if dem_quadrature != 'mc' and not use_dem:
                            # deterministic quadrature (C6) only applies to
                            # the DEM energy term -- skip for strong-form.
                            continue
                        for seed in seeds:
                            tag = make_tag(coord_scheme, precision, arch, w_bc_u,
                                            use_dem, seed, optimizer, log_energy, l2_reg,
                                            dem_quadrature)
                            if skip_existing and tag in completed:
                                print(f"Skipping already-completed run: {tag}")
                                rows.append(completed[tag])
                                continue
                            print(f"Running: {tag}")
                            row = run_one(coord_scheme, precision, arch, w_bc_u, use_dem,
                                          seed, optimizer, log_energy, l2_reg, steps, base_cfg,
                                          dem_quadrature)
                            rows.append(row)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='L5 tier-3 factor validation, stages 1-3: DEM training-loop '
                     'variants (optimizer, log-energy loss, L2 weight regularization) '
                     'from docs/griffith-drm-insights-experiment-plan.md §2.1/§3.')
    parser.add_argument('--coord_scheme', default='elliptic',
                         choices=['cartesian', 'elliptic', 'both', 'williams'])
    parser.add_argument('--precision', default='fp32', choices=['fp32', 'fp64'])
    parser.add_argument('--arch', default='y_sym', choices=['standard', 'y_sym', 'y_sym_hard'])
    parser.add_argument('--w_bc_u', type=float, default=400.0)
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(8)))
    parser.add_argument('--optimizers', nargs='+', default=['adam'], choices=['adam', 'rprop'])
    parser.add_argument('--log_energy', action='store_true',
                         help='Also run the log-energy loss variant (DEM only; O9).')
    parser.add_argument('--l2_regs', nargs='+', type=float, default=[0.0],
                         help='L2 weight-regularization coefficients to sweep (O8).')
    parser.add_argument('--dem_quadratures', nargs='+', default=['mc'],
                         choices=['mc', 'deterministic'],
                         help="DEM energy-integral quadrature scheme(s) to sweep (C6): "
                              "'mc' (random importance-weighted Monte Carlo, default) or "
                              "'deterministic' (tip-graded deterministic quadrature grid). "
                              "'deterministic' is skipped for use_dem=False rows.")
    parser.add_argument('--dem_only', action='store_true',
                         help='Skip the use_dem=False (strong-form) rows entirely.')
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--no_skip_existing', action='store_true')
    args = parser.parse_args()

    log_energies = [False, True] if args.log_energy else [False]

    all_rows = run_sweep(
        coord_scheme=args.coord_scheme, precision=args.precision, arch=args.arch,
        w_bc_u=args.w_bc_u, seeds=args.seeds, optimizers=args.optimizers,
        log_energies=log_energies, l2_regs=args.l2_regs, dem_only=args.dem_only,
        steps=args.steps, skip_existing=not args.no_skip_existing,
        dem_quadratures=args.dem_quadratures)

    import numpy as np
    for use_dem in (False, True):
        for optimizer in args.optimizers:
            for log_energy in log_energies:
                for l2_reg in args.l2_regs:
                    for dem_quadrature in args.dem_quadratures:
                        group = [r for r in all_rows if r['use_dem'] == use_dem
                                  and r['optimizer'] == optimizer
                                  and r['log_energy'] == log_energy
                                  and r['l2_reg'] == l2_reg
                                  and r['dem_quadrature'] == dem_quadrature]
                        if not group:
                            continue
                        print(f"\n{'='*70}\nuse_dem={use_dem} optimizer={optimizer} "
                              f"log_energy={log_energy} l2_reg={l2_reg:g} "
                              f"dem_quadrature={dem_quadrature}  (n={len(group)})"
                              f"\n{'='*70}")
                        for col in ('peak_loc_err', 'peak_mag_ratio', 'l2_rel_err', 'bc_err',
                                    'sym_err', 'equil_final', 'train_seconds'):
                            vals = np.array([r[col] for r in group], dtype=float)
                            print(f"  {col:16s}: mean={np.nanmean(vals):.4e}  std={np.nanstd(vals):.4e}")
