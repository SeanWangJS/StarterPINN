"""
L4 tier-2 factor sweep: w_bc_u / w_pde_u weight ratio x seed, with
coord_scheme fixed to 'elliptic' and arch fixed to 'y_sym' (the current
production-recommended baseline; see docs/griffith-ablation-study.md \u00a79).

Motivation (docs/griffith-ablation-study.md \u00a79, y_sym n=8 vs n=8 comparison):
the architectural y-mirror-symmetry fix eliminated sym_err exactly and
significantly improved l2_rel_err (~30%), but left peak_loc_err and
peak_mag_ratio statistically unchanged from the standard baseline. The
mislocalization of the stress peak near the crack tip is therefore caused by
something else. The doc flags the w_bc_u/w_pde_u loss-weight ratio (outer
boundary-condition weight vs PDE-residual weight) as the next candidate: if
the outer BC loss dominates relative to the PDE residual, the network may be
under-weighting the residual near the crack tip relative to satisfying the
far-field boundary conditions, biasing where the stress concentration ends
up.

This script holds w_pde_u=1.0 fixed (production default) and sweeps w_bc_u
over a range around the production default (100.0), holding coord_scheme
='elliptic' and arch='y_sym' fixed, to see whether peak_loc_err / peak_mag_ratio
improve, and whether l2_rel_err / bc_err / sym_err regress.

This intentionally writes to its own CSV (l4_wbc_ratio_sweep_metrics.csv)
rather than appending to experiments/results/metrics.csv, for the same reason
as l4_wsym_sweep.py: the schemas differ (this one has a w_bc_u column) and
mixing them would make the two sweeps easy to misinterpret together.

Usage (meant to be run on a server, not on this machine):
    python experiments/l4_wbc_ratio_sweep.py \\
        --coord_scheme elliptic --precision fp32 --arch y_sym \\
        --w_bc_u_values 50 100 200 400 --seeds 0 1 2 3 4 5 6 7
"""
import argparse
import copy
import csv
import os
import sys

import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from experiments.diagnostics import run_elastic_diagnostic
from experiments.metrics import compute_metrics
from experiments.ablation_runner import summarize, print_summary

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
METRICS_CSV = os.path.join(RESULTS_DIR, 'l4_wbc_ratio_sweep_metrics.csv')

FIELDNAMES = [
    'tag', 'coord_scheme', 'precision', 'arch', 'w_bc_u', 'w_pde_u', 'seed',
    'peak_loc_err', 'peak_mag_ratio', 'l2_rel_err', 'bc_err', 'sym_err', 'equil_final',
]


def load_base_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def make_tag(coord_scheme, precision, arch, w_bc_u, w_pde_u, seed):
    return f"{coord_scheme}_{precision}_{arch}_wbc{w_bc_u:g}_wpde{w_pde_u:g}_seed{seed}"


def load_completed_rows():
    """Read l4_wbc_ratio_sweep_metrics.csv (if it exists) into a dict keyed by
    tag, so an interrupted sweep can be restarted and only re-run missing
    rows."""
    rows = {}
    if not os.path.exists(METRICS_CSV):
        return rows
    with open(METRICS_CSV, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['seed'] = int(row['seed'])
            row['w_bc_u'] = float(row['w_bc_u'])
            row['w_pde_u'] = float(row['w_pde_u'])
            for col in ('peak_loc_err', 'peak_mag_ratio', 'l2_rel_err',
                        'bc_err', 'sym_err', 'equil_final'):
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


def run_one(coord_scheme, precision, arch, w_bc_u, w_pde_u, seed, steps, base_cfg):
    """Run a single (coord_scheme, precision, arch, w_bc_u, w_pde_u, seed)
    configuration.

    base_cfg is deep-copied and its weighting.w_bc_u / weighting.w_pde_u
    overridden so the sweep never mutates the caller's config between
    iterations.
    """
    cfg = copy.deepcopy(base_cfg)
    cfg['weighting']['w_bc_u'] = w_bc_u
    cfg['weighting']['w_pde_u'] = w_pde_u

    result = run_elastic_diagnostic(
        cfg, coord_scheme=coord_scheme, precision=precision, arch=arch,
        seed=seed, steps=steps)
    metrics = compute_metrics(result, cfg['physics'], cfg)

    tag = make_tag(coord_scheme, precision, arch, w_bc_u, w_pde_u, seed)
    row = {'tag': tag, 'coord_scheme': coord_scheme, 'precision': precision,
           'arch': arch, 'w_bc_u': w_bc_u, 'w_pde_u': w_pde_u, 'seed': seed,
           **metrics}
    _append_metrics_row(row)
    return row


def run_sweep(coord_scheme, precisions, archs, w_bc_u_values, w_pde_u, seeds,
              steps=None, skip_existing=True):
    base_cfg = load_base_config()
    completed = load_completed_rows() if skip_existing else {}
    rows = []
    for precision in precisions:
        for arch in archs:
            for w_bc_u in w_bc_u_values:
                for seed in seeds:
                    tag = make_tag(coord_scheme, precision, arch, w_bc_u, w_pde_u, seed)
                    if skip_existing and tag in completed:
                        print(f"Skipping already-completed run: {tag}")
                        rows.append(completed[tag])
                        continue
                    print(f"Running: {tag}")
                    row = run_one(coord_scheme, precision, arch, w_bc_u, w_pde_u,
                                  seed, steps, base_cfg)
                    rows.append(row)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='L4 tier-2 sweep: w_bc_u/w_pde_u ratio x seed at fixed '
                     'coord_scheme=elliptic, arch=y_sym')
    parser.add_argument('--coord_scheme', default='elliptic',
                         choices=['cartesian', 'elliptic', 'both', 'williams'])
    parser.add_argument('--precision', nargs='+', default=['fp32'], choices=['fp32', 'fp64'])
    parser.add_argument('--arch', nargs='+', default=['y_sym'],
                         choices=['standard', 'mixed', 'y_sym'])
    parser.add_argument('--w_bc_u_values', nargs='+', type=float,
                         default=[50.0, 100.0, 200.0, 400.0])
    parser.add_argument('--w_pde_u', type=float, default=1.0,
                         help='Held fixed across the sweep (production default 1.0).')
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(8)))
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--no_skip_existing', action='store_true',
                         help='Re-run configurations even if already present in the CSV.')
    args = parser.parse_args()

    all_rows = run_sweep(
        coord_scheme=args.coord_scheme, precisions=args.precision, archs=args.arch,
        w_bc_u_values=args.w_bc_u_values, w_pde_u=args.w_pde_u, seeds=args.seeds,
        steps=args.steps, skip_existing=not args.no_skip_existing)

    summary = summarize(all_rows, group_keys=('precision', 'arch', 'w_bc_u'))
    print_summary(summary, group_keys=('precision', 'arch', 'w_bc_u'))
