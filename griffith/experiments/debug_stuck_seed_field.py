"""
One-off diagnostic: reproduce a 'stuck-at-window-boundary' y_sym_hard seed
from l4_wbc_ratio_sweep_metrics.csv (w_bc_u=400, seed=0) at full production
training length, then plot the full sigma_yy field with:
  - the true crack tip (a, 0)
  - the restricted peak_loc_err search window (r = peak_search_radius)
  - the argmax found *inside* that window (what compute_metrics reports)
  - the argmax found over the *entire* domain (unrestricted), to see whether
    the window is genuinely excluding the artifact or just clipping it at
    its edge.

See docs/griffith-ablation-study.md \u00a79 for the investigation this
supports. Not part of the regular ablation pipeline -- run manually:

    python experiments/debug_stuck_seed_field.py --seed 0
    python experiments/debug_stuck_seed_field.py --seed 3
"""
import argparse
import copy
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from experiments.diagnostics import run_elastic_diagnostic
from experiments.metrics import evaluate_on_grid, compute_metrics

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
FIG_DIR = os.path.join(RESULTS_DIR, 'figures')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--w_bc_u', type=float, default=400.0)
    parser.add_argument('--arch', default='y_sym_hard', choices=['y_sym', 'y_sym_hard'])
    parser.add_argument('--steps', type=int, default=None,
                         help='Defaults to cfg.train.diag_steps (4000), matching the sweep.')
    parser.add_argument('--domain_points', type=int, default=None,
                         help='Override cfg.train.domain_points for a faster (approximate) repro.')
    parser.add_argument('--bc_points', type=int, default=None)
    parser.add_argument('--crack_face_points', type=int, default=None)
    parser.add_argument('--sym_axis_points', type=int, default=None)
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default.yaml')
    with open(config_path, 'r') as f:
        base_cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(base_cfg)
    cfg['weighting']['w_bc_u'] = args.w_bc_u
    if args.domain_points is not None:
        cfg['train']['domain_points'] = args.domain_points
    if args.bc_points is not None:
        cfg['train']['bc_points'] = args.bc_points
    if args.crack_face_points is not None:
        cfg['train']['crack_face_points'] = args.crack_face_points
    if args.sym_axis_points is not None:
        cfg['train']['sym_axis_points'] = args.sym_axis_points

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training arch={args.arch} seed={args.seed} w_bc_u={args.w_bc_u} on {device} ...")
    result = run_elastic_diagnostic(
        cfg, coord_scheme='elliptic', precision='fp32', arch=args.arch,
        seed=args.seed, steps=args.steps, device=device)

    metrics = compute_metrics(result, cfg['physics'], cfg)
    print("metrics (restricted search, matches CSV):", metrics)

    phys = cfg['physics']
    a_phys, W, H = phys['a'], phys['W'], phys['H']
    peak_search_radius = 0.5 * min(W / 2.0 - a_phys, H / 2.0)

    grid = evaluate_on_grid(result, phys)
    X, Y, SIG = grid['X'], grid['Y'], grid['SIG']

    # Restricted argmax (what compute_metrics reports)
    r_from_tip = np.sqrt((X - a_phys) ** 2 + Y ** 2)
    SIG_search = np.where(r_from_tip < peak_search_radius, SIG, -np.inf)
    idx_restricted = np.unravel_index(np.argmax(SIG_search), SIG_search.shape)
    xy_restricted = (X[idx_restricted], Y[idx_restricted])

    # Unrestricted (global) argmax, over the whole domain
    idx_global = np.unravel_index(np.argmax(SIG), SIG.shape)
    xy_global = (X[idx_global], Y[idx_global])

    print(f"true tip:            (a, 0) = ({a_phys}, 0)")
    print(f"restricted argmax:   {xy_restricted}  (SIG={SIG[idx_restricted]:.3f}), "
          f"dist_from_tip={np.hypot(xy_restricted[0]-a_phys, xy_restricted[1]):.6f}, "
          f"peak_search_radius={peak_search_radius}")
    print(f"unrestricted argmax: {xy_global}  (SIG={SIG[idx_global]:.3f}), "
          f"dist_from_tip={np.hypot(xy_global[0]-a_phys, xy_global[1]):.6f}")

    # ---- Plot 1: full field with markers ----
    E_prime = phys['E'] / (1 - phys['nu'] ** 2)
    v_max = cfg['train']['v_max']
    sigma_0 = E_prime * (v_max / (H / 2.0))
    vmax_ana, vmin_ana = sigma_0 * 5, -sigma_0

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    im0 = axes[0].pcolormesh(X, Y, SIG, cmap='jet', shading='auto', vmin=vmin_ana, vmax=vmax_ana)
    axes[0].set_title(rf'$\sigma_{{yy}}$ (clipped to analytical range), {args.arch} seed={args.seed}')
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(X, Y, SIG, cmap='jet', shading='auto')
    axes[1].set_title(rf'$\sigma_{{yy}}$ (full autoscaled range)')
    fig.colorbar(im1, ax=axes[1])

    for ax in axes:
        ax.set_aspect('equal')
        ax.plot(a_phys, 0.0, marker='*', color='white', markeredgecolor='black',
                markersize=18, label='true tip (a, 0)')
        circle = plt.Circle((a_phys, 0.0), peak_search_radius, fill=False,
                             color='black', linestyle='--', linewidth=1.5,
                             label=f'search radius r={peak_search_radius:g}')
        ax.add_patch(circle)
        ax.plot(*xy_restricted, marker='o', color='cyan', markeredgecolor='black',
                 markersize=10, label='restricted argmax')
        ax.plot(*xy_global, marker='X', color='magenta', markeredgecolor='black',
                 markersize=12, label='global argmax')
        ax.legend(loc='upper right', fontsize=8)

    os.makedirs(FIG_DIR, exist_ok=True)
    out_path = os.path.join(FIG_DIR, f'debug_stuck_{args.arch}_seed{args.seed}_wbc{args.w_bc_u:g}.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved field plot to {out_path}")

    # ---- Plot 2: SIG profile along the line from the tip toward the global argmax ----
    direction = np.array(xy_global) - np.array([a_phys, 0.0])
    dist = np.hypot(*direction)
    if dist > 1e-9:
        direction = direction / dist
        s_vals = np.linspace(0.0, dist, 300)
        line_x = a_phys + s_vals * direction[0]
        line_y = 0.0 + s_vals * direction[1]

        # Nearest-neighbor sample from the evaluated grid (cheap, good enough for a profile)
        flat_X, flat_Y, flat_SIG = X.flatten(), Y.flatten(), SIG.flatten()
        profile = np.empty_like(s_vals)
        for i, (lx, ly) in enumerate(zip(line_x, line_y)):
            j = np.argmin((flat_X - lx) ** 2 + (flat_Y - ly) ** 2)
            profile[i] = flat_SIG[j]

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.plot(s_vals, profile, marker='.')
        ax2.axvline(peak_search_radius, color='black', linestyle='--',
                    label=f'search radius r={peak_search_radius:g}')
        ax2.set_xlabel('distance from true tip, along tip->global-argmax direction')
        ax2.set_ylabel(r'$\sigma_{yy}$')
        ax2.set_title(f'Radial profile toward global argmax, {args.arch} seed={args.seed}')
        ax2.legend()
        out_path2 = os.path.join(
            FIG_DIR, f'debug_stuck_{args.arch}_seed{args.seed}_wbc{args.w_bc_u:g}_profile.png')
        plt.tight_layout()
        plt.savefig(out_path2, dpi=200)
        print(f"Saved radial profile plot to {out_path2}")
    else:
        print("global argmax coincides with the tip -- skipping radial profile plot.")


if __name__ == '__main__':
    main()
