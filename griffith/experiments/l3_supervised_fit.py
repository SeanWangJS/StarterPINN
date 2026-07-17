"""
L3 layer of the L1-L5 methodology (docs/griffith-ablation-study.md \u00a79):
"pure supervised MSE fitting" -- fix the network architecture (same
DisplacementNet, same hidden_layers/hidden_neurons/lr as production), feed it
a candidate coord_scheme's input features, and regress DIRECTLY onto the
analytical near-tip Williams displacement field with plain MSE. There is NO
PDE residual term and NO phase-field coupling here at all.

Why this level exists
----------------------
L1 (`debug_williams_sign.py` Part A, least squares) already showed the
Williams features f1..f4 can EXACTLY represent the correct field (a linear
algebra fact, no optimizer involved). L2 (Part B) showed those same features'
gradients blow up 64-86x near the tip. Neither test tells us whether a real
network, trained by Adam with the production learning rate, can actually
converge to that representable solution when the ONLY loss is a supervised
fit -- i.e. whether the failure mode previously seen at L4 (full elastic
diagnostic, PDE + BC losses) is caused by the coordinate features themselves,
or by their interaction with the PDE/BC loss weights. Skipping this level is
exactly what went wrong with the r_floor damping experiment (see docs \u00a79).

Usage
-----
    python experiments/l3_supervised_fit.py                         # all schemes
    python experiments/l3_supervised_fit.py --coord_scheme williams
    python experiments/l3_supervised_fit.py --coord_scheme williams --williams_r_floor 0.05
"""
import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.network import DisplacementNet, AdaptiveReLU
from src.evaluator import get_u_in_dim, williams_features, cartesian_to_elliptic
from experiments.debug_williams_sign import load_cfg, analytical_params, sample_near_tip_points
from experiments.diagnostics import set_seed

SCHEMES = ['cartesian', 'elliptic', 'both', 'williams']


def build_u_input(pts_nd, coord_scheme, williams_r_floor=0.0):
    """Mirror of PFPhysicsEvaluator._build_u_input, kept standalone here so
    this script has no dependency on any particular u_net/phi_net instance."""
    if coord_scheme == 'cartesian':
        return pts_nd
    mu, nu = cartesian_to_elliptic(pts_nd, 1.0)
    if coord_scheme == 'elliptic':
        return torch.cat([mu, nu], dim=1)
    if coord_scheme == 'both':
        return torch.cat([pts_nd, mu, nu], dim=1)
    if coord_scheme == 'williams':
        wf = williams_features(pts_nd, a_nd=1.0, r_floor=williams_r_floor)
        return torch.cat([pts_nd, wf], dim=1)
    raise ValueError(f"Unknown coord_scheme: '{coord_scheme}'")


def analytical_uv_nd(a_nd_pts, kappa, A_nd):
    """Analytical non-dimensional Williams displacement field, evaluated at
    non-dimensional (r, theta) already relative to the right tip (a_nd, 0)."""
    r_nd, theta = a_nd_pts
    u = A_nd * np.sqrt(r_nd) * (np.cos(theta / 2) * (kappa - 1)
                                 + 2 * np.sin(theta / 2) ** 2 * np.cos(theta / 2))
    v = A_nd * np.sqrt(r_nd) * (np.sin(theta / 2) * (kappa + 1)
                                 - 2 * np.cos(theta / 2) ** 2 * np.sin(theta / 2))
    return u, v


def get_A_nd_kappa(cfg):
    K_I, kappa, G, sigma_0, a_phys = analytical_params(cfg)
    A = K_I / (2 * G * np.sqrt(2 * np.pi))
    A_nd = A / np.sqrt(a_phys)   # non-dimensionalize: u_nd = u_phys / a_phys
    return A_nd, kappa


def sample_split(seed, n_r=40, n_theta=72, r_min_factor=1e-2, r_max_factor=0.3):
    """Deterministic grid for training + an independent random (interpolation,
    same r/theta range) sample for held-out testing. a_nd=1.0 throughout, i.e.
    everything is already in the same non-dimensional units the production
    evaluator uses (pts_nd, a_nd=1.0)."""
    x_tr, y_tr, r_tr, th_tr = sample_near_tip_points(
        1.0, n_r=n_r, n_theta=n_theta, r_min_factor=r_min_factor, r_max_factor=r_max_factor)

    rng = np.random.RandomState(seed + 12345)
    n_test = 2000
    r_te = rng.uniform(r_min_factor, r_max_factor, n_test)
    th_te = rng.uniform(-np.pi + 0.02, np.pi - 0.02, n_test)
    x_te = 1.0 + r_te * np.cos(th_te)
    y_te = r_te * np.sin(th_te)

    return (x_tr, y_tr, r_tr, th_tr), (x_te, y_te, r_te, th_te)


def rel_l2(pred, target):
    pred = pred.flatten()
    target = target.flatten()
    return float(np.sqrt(np.mean((pred - target) ** 2)) / (np.sqrt(np.mean(target ** 2)) + 1e-30))


def run_l3_fit(coord_scheme, cfg=None, williams_r_floor=0.0, seed=0, steps=3000,
                lr=None, verbose=True, activation='tanh', m0=1.0):
    cfg = cfg if cfg is not None else load_cfg()
    set_seed(seed)
    torch.set_default_dtype(torch.float64)

    mc = cfg['model']
    lr = lr if lr is not None else cfg['train']['u_lr']

    A_nd, kappa = get_A_nd_kappa(cfg)

    (x_tr, y_tr, r_tr, th_tr), (x_te, y_te, r_te, th_te) = sample_split(seed)
    u_tr, v_tr = analytical_uv_nd((r_tr, th_tr), kappa, A_nd)
    u_te, v_te = analytical_uv_nd((r_te, th_te), kappa, A_nd)

    pts_tr = torch.tensor(np.stack([x_tr, y_tr], axis=1))
    target_tr = torch.tensor(np.stack([u_tr, v_tr], axis=1))
    pts_te = torch.tensor(np.stack([x_te, y_te], axis=1))
    target_te = torch.tensor(np.stack([u_te, v_te], axis=1))

    if activation not in ('tanh', 'adaptive_relu'):
        raise ValueError(f"Unknown activation: '{activation}' (expected 'tanh' or 'adaptive_relu')")
    act = None if activation == 'tanh' else AdaptiveReLU(m0=m0)

    u_in_dim = get_u_in_dim(coord_scheme)
    u_net = DisplacementNet(in_dim=u_in_dim, out_dim=2,
                             hidden_layers=mc['hidden_layers'],
                             hidden_neurons=mc['hidden_neurons'], activation=act)
    opt = torch.optim.Adam(u_net.parameters(), lr=lr)

    tag = coord_scheme + (f"_rfloor{williams_r_floor:g}" if coord_scheme == 'williams' and williams_r_floor > 0 else "")
    tag = tag + (f"_actadaprelu_m0{m0:g}" if activation == 'adaptive_relu' else "")
    pbar = tqdm(range(steps), desc=f"[L3 {tag} seed={seed}]", leave=False, disable=not verbose)
    final_train_mse = None
    for step in pbar:
        opt.zero_grad()
        feat_tr = build_u_input(pts_tr, coord_scheme, williams_r_floor)
        pred_tr = u_net(feat_tr)
        loss = torch.mean((pred_tr - target_tr) ** 2)
        loss.backward()
        opt.step()
        final_train_mse = loss.item()
        if verbose and step % max(1, steps // 5) == 0:
            pbar.set_postfix(mse=final_train_mse)

    with torch.no_grad():
        feat_te = build_u_input(pts_te, coord_scheme, williams_r_floor)
        pred_te = u_net(feat_te).numpy()
    test_l2_rel_err = rel_l2(pred_te, target_te.numpy())

    return {
        'tag': tag, 'coord_scheme': coord_scheme, 'williams_r_floor': williams_r_floor,
        'seed': seed, 'train_final_mse': final_train_mse, 'test_l2_rel_err': test_l2_rel_err,
        'activation': activation, 'm0': m0,
    }


def main():
    parser = argparse.ArgumentParser(description='L3: pure supervised MSE fit of the '
                                                   'Williams near-tip field (no PDE, no phase field)')
    parser.add_argument('--coord_scheme', choices=SCHEMES + ['all'], default='all')
    parser.add_argument('--williams_r_floor', type=float, default=0.0)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--activations', choices=['tanh', 'adaptive_relu'], nargs='+', default=['tanh'],
                         help="O10 (see docs/griffith-drm-insights-experiment-plan.md stage 4): compare "
                              "'tanh' (default) against 'adaptive_relu' fitting capability at the L1 level.")
    parser.add_argument('--m0', type=float, default=2.0,
                         help='Initial learnable-slope value for adaptive_relu (ignored for tanh).')
    args = parser.parse_args()

    cfg = load_cfg()

    if args.coord_scheme == 'all':
        # cartesian / elliptic / both / williams (undamped) + williams with the
        # previously-tested r_floor=0.05 damping, for direct comparison.
        jobs = [(s, 0.0) for s in SCHEMES]
        jobs.append(('williams', 0.05))
    else:
        jobs = [(args.coord_scheme, args.williams_r_floor)]

    rows = []
    for coord_scheme, r_floor in jobs:
        for activation in args.activations:
            for seed in args.seeds:
                row = run_l3_fit(coord_scheme, cfg=cfg, williams_r_floor=r_floor, seed=seed,
                                  steps=args.steps, activation=activation, m0=args.m0)
                rows.append(row)
                print(f"  {row['tag']:24s} seed={seed}  train_mse={row['train_final_mse']:.4e}  "
                      f"test_l2_rel_err={row['test_l2_rel_err']:.4e}")

    print("\n" + "=" * 78)
    print("L3 summary (mean +/- std over seeds)")
    print("=" * 78)
    print(f"{'scheme':24s} {'train_mse (mean)':>18s} {'test_l2_rel_err (mean +/- std)':>30s}")
    tags_seen = []
    for row in rows:
        tag = row['tag']
        if tag in tags_seen:
            continue
        tags_seen.append(tag)
        group = [r for r in rows if r['tag'] == tag]
        mse_mean = np.mean([r['train_final_mse'] for r in group])
        err_mean = np.mean([r['test_l2_rel_err'] for r in group])
        err_std = np.std([r['test_l2_rel_err'] for r in group])
        print(f"{tag:24s} {mse_mean:18.4e} {err_mean:15.4e} +/- {err_std:.4e}")


if __name__ == '__main__':
    main()
