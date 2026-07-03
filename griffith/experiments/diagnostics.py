"""
Elastic-only diagnostic training (phi frozen to 0), factored out of train.py
so that the formal quasi-static coupled training entry point stays free of
experiment-only branches.

See docs/griffith-ablation-study.md for the overall ablation design and
docs/griffith-next-steps.md (step 1) for the diagnostic's original rationale.
"""
import os
import sys
import random

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.network import DisplacementNet, PhaseFieldNet
from src.evaluator import PFPhysicsEvaluator, get_u_in_dim


def set_seed(seed):
    """Seed all random sources used during sampling and initialization."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_networks(coord_scheme, arch, hidden_layers, hidden_neurons, device):
    """
    Construct (u_net, phi_net) for the requested architecture factor.

    arch == 'standard' : DisplacementNet outputs [u, v]              (factor C1)
    arch == 'mixed'    : MixedDisplacementNet outputs [u,v,sxx,syy,txy] (factor C2)
    """
    u_in_dim = get_u_in_dim(coord_scheme)
    if arch == 'mixed':
        from experiments.mixed_pinn_variant import MixedDisplacementNet
        u_net = MixedDisplacementNet(
            in_dim=u_in_dim, hidden_layers=hidden_layers, hidden_neurons=hidden_neurons
        ).to(device)
    elif arch == 'standard':
        u_net = DisplacementNet(
            in_dim=u_in_dim, out_dim=2,
            hidden_layers=hidden_layers, hidden_neurons=hidden_neurons
        ).to(device)
    else:
        raise ValueError(f"Unknown arch: '{arch}'")

    phi_net = PhaseFieldNet(
        in_dim=2, out_dim=1,
        hidden_layers=hidden_layers, hidden_neurons=hidden_neurons
    ).to(device)
    return u_net, phi_net


def run_elastic_diagnostic(cfg, coord_scheme='both', precision='fp32', arch='standard',
                            seed=0, steps=None, device=None):
    """
    Train the displacement network only (phi forced to 0 everywhere, i.e.
    g(phi)=1, fully elastic) and return the trained networks + evaluator +
    final loss values.

    This function never touches griffith/results/ (the production output
    directory); callers (e.g. experiments/run_diagnostic.py) are responsible
    for persisting whatever they need under experiments/results/.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.set_default_dtype(torch.float64 if precision == 'fp64' else torch.float32)
    set_seed(seed)

    phys = cfg['physics']
    E, nu, G_c, l_0, a_phys = phys['E'], phys['nu'], phys['G_c'], phys['l_0'], phys['a']
    W, H = phys['W'], phys['H']
    L0 = a_phys

    mc = cfg['model']
    u_net, phi_net = build_networks(
        coord_scheme, arch, mc['hidden_layers'], mc['hidden_neurons'], device)

    if arch == 'mixed':
        from experiments.mixed_pinn_variant import MixedPFEvaluator
        evaluator = MixedPFEvaluator(
            u_net, phi_net, E=E, nu=nu, G_c=G_c, l_0=l_0, a=a_phys, k=phys['k'],
            coord_scheme=coord_scheme
        ).to(device)
    else:
        evaluator = PFPhysicsEvaluator(
            u_net, phi_net, E=E, nu=nu, G_c=G_c, l_0=l_0, a=a_phys, k=phys['k'],
            coord_scheme=coord_scheme
        ).to(device)

    from src.geometry import GriffithGeometrySampler
    sampler = GriffithGeometrySampler(W=W, H=H, a=a_phys, device=device)

    tc = cfg['train']
    pts_phys = sampler.sample_fixed_domain(tc['domain_points'], l_0=l_0)
    bc_dict_phys = sampler.sample_boundaries(tc['bc_points'])
    crack_face_phys = sampler.sample_crack_face(tc['crack_face_points'])
    sym_axis_phys = sampler.sample_symmetry_axis(tc['sym_axis_points'])

    pts_nd = pts_phys / L0
    bc_dict_nd = {k: v / L0 for k, v in bc_dict_phys.items()}
    crack_face_nd = crack_face_phys / L0
    sym_axis_nd = sym_axis_phys / L0

    wc = cfg['weighting']
    opt_u = torch.optim.Adam(u_net.parameters(), lr=tc['u_lr'])

    n_steps = steps if steps is not None else tc.get('diag_steps', 4000)
    v_max_nd = tc['v_max'] / L0   # full load, no quasi-static ramp

    phi_net.eval()
    phi_net.forward = lambda x: torch.zeros(x.shape[0], 1, device=x.device)

    u_net.train()
    final_equil, final_bc = float('inf'), float('inf')
    tag = f"{coord_scheme}/{precision}/{arch} seed={seed}"
    pbar = tqdm(range(n_steps), desc=f"[Diag {tag}]", leave=False)
    for _ in pbar:
        opt_u.zero_grad()
        if arch == 'mixed':
            loss_u, loss_equil, bc_u, cf_u, sym_u, cons_u = evaluator.compute_u_loss_mixed(
                pts_nd, bc_dict_nd, v_max_nd, wc['w_pde_u'], wc['w_bc_u'],
                crack_face_nd=crack_face_nd, w_crack_face=wc['w_crack_face'],
                sym_axis_nd=sym_axis_nd, w_sym=wc['w_sym']
            )
        else:
            loss_u, loss_equil, bc_u, cf_u, sym_u = evaluator.compute_u_loss(
                pts_nd, bc_dict_nd, v_max_nd, wc['w_pde_u'], wc['w_bc_u'],
                crack_face_nd=crack_face_nd, w_crack_face=wc['w_crack_face'],
                sym_axis_nd=sym_axis_nd, w_sym=wc['w_sym']
            )
        loss_u.backward()
        opt_u.step()
        final_equil, final_bc = loss_equil, bc_u
        pbar.set_postfix({'Eq': f'{loss_equil:.2e}', 'BC': f'{bc_u:.2e}'})

    return {
        'u_net': u_net,
        'phi_net': phi_net,
        'evaluator': evaluator,
        'L0': L0,
        'E0': E,
        'final_equil': final_equil,
        'final_bc': final_bc,
    }
