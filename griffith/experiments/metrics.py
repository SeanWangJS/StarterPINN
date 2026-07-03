"""
Quantitative metrics for the elastic diagnostic ablation study, matching
docs/griffith-ablation-study.md \u00a75:
  1. peak_loc_err   - distance from predicted sigma_yy argmax to true tip (a, 0)
  2. peak_mag_ratio - predicted / Irwin sigma_yy at r = 2*l_0 ahead of the tip
  3. l2_rel_err     - relative L2 error vs Westergaard, excluding a small tip disk
  4. bc_err         - final boundary-condition residual (top/bottom/right/sym/crack-face)
  5. sym_err        - RMS asymmetry of sigma_yy under y -> -y mirroring
  6. equil_final    - final equilibrium PDE residual (training stability proxy)
"""
import os
import sys

import numpy as np
import torch

_SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'scripts')
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from plot_analytical import westergaard_field, irwin_field  # noqa: E402


def evaluate_on_grid(result, phys, nx=150, ny=200):
    """
    Evaluate sigma_yy on the standard right-half grid and mirror it to the
    full domain (sigma_yy is symmetric about x=0).

    Returns a dict with full-domain X, Y, SIG plus the raw right-half grid
    (x_half, y_vals, SIG_half) for point-probing.
    """
    u_net = result['u_net']
    evaluator = result['evaluator']
    L0, E0 = result['L0'], result['E0']
    device = next(u_net.parameters()).device
    W, H = phys['W'], phys['H']

    x_half = np.linspace(0.0, W / 2, nx)
    y_vals = np.linspace(-H / 2, H / 2, ny)
    X_half, Y = np.meshgrid(x_half, y_vals)

    pts_phys = np.hstack([X_half.flatten()[:, None], Y.flatten()[:, None]])
    pts_nd = torch.tensor(pts_phys / L0, dtype=torch.get_default_dtype(), device=device)
    pts_nd.requires_grad_(True)

    with torch.set_grad_enabled(True):
        if hasattr(evaluator, 'compute_strains_and_stress'):
            eps_xx, eps_yy, eps_xy, _, _, _, _, sig_yy_nn, _ = \
                evaluator.compute_strains_and_stress(pts_nd)
            sig_yy_nd = sig_yy_nn
        else:
            eps_xx, eps_yy, eps_xy, _, u_nd, _, _, _, _ = evaluator.compute_strains(pts_nd)
            phi = torch.zeros_like(u_nd)   # elastic diagnostic: phi == 0 everywhere
            _, _, sig_yy_nd, _ = evaluator.compute_energy_and_stress(eps_xx, eps_yy, eps_xy, phi)

    ny_, nx_half = X_half.shape
    SIG_half = sig_yy_nd.detach().cpu().numpy().reshape(ny_, nx_half) * E0

    X_full = np.hstack([-np.fliplr(X_half)[:, :-1], X_half])
    Y_full = np.hstack([np.fliplr(Y)[:, :-1], Y])
    SIG_full = np.hstack([np.fliplr(SIG_half)[:, :-1], SIG_half])   # symmetric about x=0

    return {
        'X': X_full, 'Y': Y_full, 'SIG': SIG_full,
        'x_half': x_half, 'y_vals': y_vals, 'SIG_half': SIG_half,
    }


def compute_metrics(result, phys, cfg, tip_exclusion_factor=0.5):
    """Compute the six scalar metrics defined in docs/griffith-ablation-study.md \u00a75."""
    grid = evaluate_on_grid(result, phys)
    X, Y, SIG = grid['X'], grid['Y'], grid['SIG']
    a_phys, l_0 = phys['a'], phys['l_0']

    E_prime = phys['E'] / (1 - phys['nu'] ** 2)
    v_max = cfg['train']['v_max']
    sigma_0 = E_prime * (v_max / (phys['H'] / 2.0))

    # 1. Peak location error (distance from argmax to the true right tip)
    idx = np.unravel_index(np.argmax(SIG), SIG.shape)
    peak_xy = np.array([X[idx], Y[idx]])
    peak_loc_err = float(np.linalg.norm(peak_xy - np.array([a_phys, 0.0])))

    # 2. Peak magnitude ratio at r = 2*l_0 ahead of the right tip
    r_probe = 2.0 * l_0
    x_probe, y_probe = a_phys + r_probe, 0.0
    _, sig_yy_irwin_probe, _ = irwin_field(
        np.array([[x_probe]]), np.array([[y_probe]]), a_phys, sigma_0, finite_correction=False)
    i_row = int(np.argmin(np.abs(grid['y_vals'] - y_probe)))
    i_col = int(np.argmin(np.abs(grid['x_half'] - x_probe)))
    pred_probe = grid['SIG_half'][i_row, i_col]
    ana_probe = float(sig_yy_irwin_probe.item())
    peak_mag_ratio = float(pred_probe / ana_probe) if ana_probe != 0 else float('nan')

    # 3. Full-field relative L2 error vs Westergaard, excluding a disk around both tips
    _, sig_yy_ana, _ = westergaard_field(X, Y, a_phys, sigma_0)
    r_tip_right = np.sqrt((X - a_phys) ** 2 + Y ** 2)
    r_tip_left = np.sqrt((X + a_phys) ** 2 + Y ** 2)
    excl = (r_tip_right < tip_exclusion_factor * l_0) | (r_tip_left < tip_exclusion_factor * l_0)
    mask = ~excl
    l2_rel_err = float(
        np.linalg.norm(SIG[mask] - sig_yy_ana[mask]) / (np.linalg.norm(sig_yy_ana[mask]) + 1e-12)
    )

    # 4. Boundary-condition residual (from training, non-dimensional)
    bc_err = float(result.get('final_bc', float('nan')))

    # 5. Symmetry error under y -> -y mirroring
    sym_err = float(np.sqrt(np.mean((SIG - np.flipud(SIG)) ** 2)))

    # 6. Training stability: final equilibrium PDE residual
    equil_final = float(result.get('final_equil', float('nan')))

    return {
        'peak_loc_err': peak_loc_err,
        'peak_mag_ratio': peak_mag_ratio,
        'l2_rel_err': l2_rel_err,
        'bc_err': bc_err,
        'sym_err': sym_err,
        'equil_final': equil_final,
    }
