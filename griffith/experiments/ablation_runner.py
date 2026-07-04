"""
Orchestrates the two-stage ablation matrix defined in
docs/griffith-ablation-study.md \u00a74:

    Stage 1: factor A (coord_scheme) x factor B (precision), arch fixed to
             'standard', run over several seeds each.
    Stage 2: best (coord_scheme, precision) from stage 1 x factor C (arch),
             run over several seeds each.

This module only calls into experiments/run_diagnostic.py's run_one() in a
loop; it never touches griffith/train.py or griffith/results/.
"""
import os
from collections import defaultdict

import numpy as np

from experiments.run_diagnostic import run_one, load_base_config, make_tag, load_completed_rows

COORD_SCHEMES = ['cartesian', 'elliptic', 'both', 'williams']
PRECISIONS = ['fp32', 'fp64']
SEEDS = [0, 1, 2]

NUMERIC_COLS = ['peak_loc_err', 'peak_mag_ratio', 'l2_rel_err', 'bc_err', 'sym_err', 'equil_final']


def run_stage1(seeds=SEEDS, steps=None, save_plot=False, skip_existing=True):
    """Sweep coord_scheme x precision at arch='standard'.

    If skip_existing=True (default), any (coord_scheme, precision, seed)
    combination already present in experiments/results/metrics.csv is loaded
    from disk instead of re-run, so an interrupted sweep can simply be
    restarted from scratch and it will only compute the missing rows.
    """
    cfg = load_base_config()
    completed = load_completed_rows() if skip_existing else {}
    rows = []
    for coord_scheme in COORD_SCHEMES:
        for precision in PRECISIONS:
            for seed in seeds:
                tag = make_tag(coord_scheme, precision, 'standard', seed)
                if skip_existing and tag in completed:
                    print(f"Skipping already-completed run: {tag}")
                    rows.append(completed[tag])
                    continue
                row = run_one(coord_scheme=coord_scheme, precision=precision,
                               arch='standard', seed=seed, steps=steps,
                               save_plot=save_plot, cfg=cfg)
                rows.append(row)
    return rows


def run_stage2(best_coord_scheme, best_precision, seeds=SEEDS, steps=None,
               save_plot=False, skip_existing=True):
    """Compare arch='standard' vs arch='mixed' at the best (coord_scheme, precision).

    Same skip_existing behavior as run_stage1.
    """
    cfg = load_base_config()
    completed = load_completed_rows() if skip_existing else {}
    rows = []
    for arch in ['standard', 'mixed']:
        for seed in seeds:
            tag = make_tag(best_coord_scheme, best_precision, arch, seed)
            if skip_existing and tag in completed:
                print(f"Skipping already-completed run: {tag}")
                rows.append(completed[tag])
                continue
            row = run_one(coord_scheme=best_coord_scheme, precision=best_precision,
                           arch=arch, seed=seed, steps=steps, save_plot=save_plot, cfg=cfg)
            rows.append(row)
    return rows


def summarize(rows, group_keys=('coord_scheme', 'precision')):
    """Aggregate mean/std of the six metrics per group, without a pandas dependency."""
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        groups[key].append(row)

    summary = {}
    for key, group_rows in groups.items():
        stats = {}
        for col in NUMERIC_COLS:
            vals = np.array([r[col] for r in group_rows], dtype=float)
            stats[col] = {'mean': float(np.nanmean(vals)), 'std': float(np.nanstd(vals))}
        summary[key] = stats
    return summary


def print_summary(summary, group_keys=('coord_scheme', 'precision')):
    header = " / ".join(group_keys)
    print(f"\n{'='*70}\nSummary ({header})\n{'='*70}")
    for key, stats in summary.items():
        print(f"\n[{key}]")
        for col, s in stats.items():
            print(f"  {col:16s}: mean={s['mean']:.4e}  std={s['std']:.4e}")


def pick_best(summary, metric='l2_rel_err'):
    """Return the group key with the lowest mean of `metric`."""
    return min(summary.items(), key=lambda kv: kv[1][metric]['mean'])[0]


if __name__ == '__main__':
    print("Running Stage 1: coord_scheme x precision (arch='standard')...")
    stage1_rows = run_stage1()
    summary1 = summarize(stage1_rows, group_keys=('coord_scheme', 'precision'))
    print_summary(summary1, group_keys=('coord_scheme', 'precision'))

    best_key = pick_best(summary1, metric='l2_rel_err')
    best_coord_scheme, best_precision = best_key
    print(f"\nBest (coord_scheme, precision) by l2_rel_err: {best_key}")

    print("\nRunning Stage 2: arch sweep at best (coord_scheme, precision)...")
    stage2_rows = run_stage2(best_coord_scheme, best_precision)
    summary2 = summarize(stage2_rows, group_keys=('arch',))
    print_summary(summary2, group_keys=('arch',))
