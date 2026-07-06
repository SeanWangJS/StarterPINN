"""
L4 tier-2 factor sweep: w_sym (symmetry-axis BC weight) x seed, with
coord_scheme fixed to 'elliptic' (the winner of the L3 supervised-fit
comparison, see docs/griffith-ablation-study.md \u00a79).

Motivation (docs/griffith-ablation-study.md \u00a73, tier-2 factors + \u00a79 L4 row):
the existing elliptic rows in experiments/results/metrics.csv (w_sym=50, the
production default) show a very large `sym_err` (RMS up-down asymmetry of
sigma_yy), comparable in magnitude to the full stress scale, plus a seed with
a sign-flipped `peak_mag_ratio`. w_sym=50 was flagged as a suspected cause of
mislocalized stress concentration but was deferred instead of tested. This
script tests w_sym in {50, 10, 0} (production / reduced / disabled) over
several seeds, holding coord_scheme='elliptic' fixed, to see whether reducing
this soft-constraint weight improves peak_loc_err / sym_err without
regressing bc_err or l2_rel_err.

This intentionally writes to its own CSV (l4_wsym_sweep_metrics.csv) rather
than appending to experiments/results/metrics.csv, because that file's schema
(FIELDNAMES in run_diagnostic.py) has no w_sym column and its existing rows
all implicitly assume the production default w_sym=50 -- mixing schemas would
make the two sweeps easy to misinterpret together.

Usage (meant to be run on a server, not on this machine):
    python experiments/l4_wsym_sweep.py \\
        --coord_scheme elliptic --precision fp32 --arch standard \\
        --w_sym_values 50 10 0 --seeds 0 1 2 3 4 5 6 7
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
METRICS_CSV = os.path.join(RESULTS_DIR, 'l4_wsym_sweep_metrics.csv')

FIELDNAMES = [
    'tag', 'coord_scheme', 'precision', 'arch', 'w_sym', 'seed',
    'peak_loc_err', 'peak_mag_ratio', 'l2_rel_err', 'bc_err', 'sym_err', 'equil_final',
]


def load_base_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def make_tag(coord_scheme, precision, arch, w_sym, seed):
    return f"{coord_scheme}_{precision}_{arch}_wsym{w_sym:g}_seed{seed}"


def load_completed_rows():
    """Read l4_wsym_sweep_metrics.csv (if it exists) into a dict keyed by tag,
    so an interrupted sweep can be restarted and only re-run missing rows."""
    rows = {}
    if not os.path.exists(METRICS_CSV):
        return rows
    with open(METRICS_CSV, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['seed'] = int(row['seed'])
            row['w_sym'] = float(row['w_sym'])
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


def run_one(coord_scheme, precision, arch, w_sym, seed, steps, base_cfg):
    """Run a single (coord_scheme, precision, arch, w_sym, seed) configuration.

    base_cfg is deep-copied and its weighting.w_sym overridden so the sweep
    never mutates the caller's config between iterations.
    """
    cfg = copy.deepcopy(base_cfg)
    cfg['weighting']['w_sym'] = w_sym

    result = run_elastic_diagnostic(
        cfg, coord_scheme=coord_scheme, precision=precision, arch=arch,
        seed=seed, steps=steps)
    metrics = compute_metrics(result, cfg['physics'], cfg)

    tag = make_tag(coord_scheme, precision, arch, w_sym, seed)
    row = {'tag': tag, 'coord_scheme': coord_scheme, 'precision': precision,
           'arch': arch, 'w_sym': w_sym, 'seed': seed, **metrics}
    _append_metrics_row(row)
    return row


def run_sweep(coord_scheme, precisions, archs, w_sym_values, seeds, steps=None,
              skip_existing=True):
    base_cfg = load_base_config()
    completed = load_completed_rows() if skip_existing else {}
    rows = []
    for precision in precisions:
        for arch in archs:
            for w_sym in w_sym_values:
                for seed in seeds:
                    tag = make_tag(coord_scheme, precision, arch, w_sym, seed)
                    if skip_existing and tag in completed:
                        print(f"Skipping already-completed run: {tag}")
                        rows.append(completed[tag])
                        continue
                    print(f"Running: {tag}")
                    row = run_one(coord_scheme, precision, arch, w_sym, seed, steps, base_cfg)
                    rows.append(row)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='L4 tier-2 sweep: w_sym x seed at fixed coord_scheme (default elliptic)')
    parser.add_argument('--coord_scheme', default='elliptic',
                         choices=['cartesian', 'elliptic', 'both', 'williams'])
    parser.add_argument('--precision', nargs='+', default=['fp32'], choices=['fp32', 'fp64'])
    parser.add_argument('--arch', nargs='+', default=['standard'], choices=['standard', 'mixed'])
    parser.add_argument('--w_sym_values', nargs='+', type=float, default=[50.0, 10.0, 0.0])
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(8)))
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--no_skip_existing', action='store_true',
                         help='Re-run configurations even if already present in the CSV.')
    args = parser.parse_args()

    all_rows = run_sweep(
        coord_scheme=args.coord_scheme, precisions=args.precision, archs=args.arch,
        w_sym_values=args.w_sym_values, seeds=args.seeds, steps=args.steps,
        skip_existing=not args.no_skip_existing)

    summary = summarize(all_rows, group_keys=('precision', 'arch', 'w_sym'))
    print_summary(summary, group_keys=('precision', 'arch', 'w_sym'))
