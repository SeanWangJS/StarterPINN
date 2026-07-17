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

from src.network import DisplacementNet, PhaseFieldNet, AdaptiveReLU
from src.evaluator import PFPhysicsEvaluator, get_u_in_dim


def set_seed(seed):
    """Seed all random sources used during sampling and initialization."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_networks(coord_scheme, arch, hidden_layers, hidden_neurons, device,
                    activation='tanh', m0=1.0):
    """
    Construct (u_net, phi_net) for the requested architecture factor.

    arch == 'standard'   : DisplacementNet outputs [u, v]              (factor C1)
    arch == 'mixed'      : MixedDisplacementNet outputs [u,v,sxx,syy,txy] (factor C2)
    arch == 'y_sym'      : DisplacementNet outputs [u, v], but the evaluator
                           enforces sigma_yy(x,y) == sigma_yy(x,-y) by
                           construction -- see PFPhysicsEvaluator's
                           y_symmetric flag.
    arch == 'y_sym_hard' : same as 'y_sym', plus the evaluator's
                           hard_dirichlet flag also clamps v*(x,+-H*/2) to
                           exactly +-v_max*, removing loss_top/loss_bot from
                           the soft-penalty loss_bc entirely. See
                           docs/griffith-ablation-study.md §9.

    activation : 'tanh' (default) or 'adaptive_relu' (O10, see
        docs/griffith-drm-insights-experiment-plan.md §2.1 / stage 4). Both
        u_net and phi_net use the same activation choice. 'adaptive_relu'
        builds one independent AdaptiveReLU instance (own learnable m_k) per
        hidden layer, per network -- see src/network.py's MLP.
    m0 : initial value of each layer's learnable slope m_k when
        activation='adaptive_relu' (ignored for 'tanh'). Manav et al. 2024
        use m0=1 for a small 1D problem and m0=2-3 for their 2D problems.
    """
    if activation == 'tanh':
        act = None   # MLP defaults to nn.Tanh()
    elif activation == 'adaptive_relu':
        act = AdaptiveReLU(m0=m0)
    else:
        raise ValueError(f"Unknown activation: '{activation}' (expected 'tanh' or 'adaptive_relu')")

    u_in_dim = get_u_in_dim(coord_scheme)
    if arch == 'mixed':
        from experiments.mixed_pinn_variant import MixedDisplacementNet
        if activation != 'tanh':
            raise ValueError("activation='adaptive_relu' is not supported with arch='mixed' yet "
                              "(MixedDisplacementNet hardcodes nn.Tanh()).")
        u_net = MixedDisplacementNet(
            in_dim=u_in_dim, hidden_layers=hidden_layers, hidden_neurons=hidden_neurons
        ).to(device)
    elif arch in ('standard', 'y_sym', 'y_sym_hard'):
        u_net = DisplacementNet(
            in_dim=u_in_dim, out_dim=2,
            hidden_layers=hidden_layers, hidden_neurons=hidden_neurons, activation=act
        ).to(device)
    else:
        raise ValueError(f"Unknown arch: '{arch}'")

    phi_net = PhaseFieldNet(
        in_dim=2, out_dim=1,
        hidden_layers=hidden_layers, hidden_neurons=hidden_neurons, activation=act
    ).to(device)
    return u_net, phi_net


def run_elastic_diagnostic(cfg, coord_scheme='both', precision='fp32', arch='standard',
                            seed=0, steps=None, device=None, williams_r_floor=0.0,
                            use_dem=False, optimizer='adam', log_energy=False,
                            l2_reg=0.0, activation='tanh', m0=1.0):
    """
    Train the displacement network only (phi forced to 0 everywhere, i.e.
    g(phi)=1, fully elastic) and return the trained networks + evaluator +
    final loss values.

    use_dem : if True, train u_net with PFPhysicsEvaluator.compute_u_loss_dem
        (Deep Energy Method: minimizes total potential energy via first
        derivatives only) instead of compute_u_loss (strong-form equilibrium
        PDE residual via second derivatives). Orthogonal to arch/y_symmetric/
        hard_dirichlet -- composes with any of them. Not supported with
        arch='mixed' (compute_u_loss_mixed has no DEM variant yet). See
        docs/griffith-ablation-study.md §12.

    The following three arguments implement the "optimization aid" candidates
    from docs/griffith-drm-insights-experiment-plan.md §2.1 (Manav et al.
    2024, Deep Ritz Method for phase-field fracture). All three are
    orthogonal to use_dem/arch/coord_scheme and to each other -- they can be
    combined freely.

    optimizer : 'adam' (default) or 'rprop'. O7: the paper found Adam
        "considerably worse" than RPROP at converging sharp/localized fields
        in energy-based training; RPROP was used for all their reported
        results. torch.optim.Rprop's step size self-adapts multiplicatively
        from the given lr, so the same tc['u_lr'] value is reused as the
        initial step size for both choices.
    log_energy : if True and use_dem=True, use the log-energy loss transform
        (O9, see compute_u_loss_dem's docstring). Raises if True with
        use_dem=False (compute_u_loss has no energy term to transform).
    l2_reg : L2 weight-regularization coefficient applied to u_net's
        parameters (O8). Motivation: near a localized phase field the
        maximum expressible strain is theoretically unbounded (Manav et al.
        2024, Appendix D) -- unlike FEA, where element size implicitly caps
        it -- so an under-resolved quadrature/sampling grid can let the DEM
        loss converge to a spuriously low-energy, physically wrong solution.
        L2 regularization constrains the network's expressible max slope as
        a stand-in for that missing implicit bound. 0.0 (default) disables it.

    activation : 'tanh' (default) or 'adaptive_relu' (O10). Passed straight
        through to build_networks -- see its docstring for details. Also
        orthogonal to use_dem/optimizer/log_energy/l2_reg/arch/coord_scheme
        (arch='mixed' excepted, see build_networks).
    m0 : initial learnable-slope value for 'adaptive_relu' (ignored otherwise).

    This function never touches griffith/results/ (the production output
    directory); callers (e.g. experiments/run_diagnostic.py) are responsible
    for persisting whatever they need under experiments/results/.
    """
    if use_dem and arch == 'mixed':
        raise ValueError("use_dem=True is not supported with arch='mixed' yet.")
    if log_energy and not use_dem:
        raise ValueError("log_energy=True requires use_dem=True (compute_u_loss "
                          "has no energy term to log-transform).")
    if optimizer not in ('adam', 'rprop'):
        raise ValueError(f"Unknown optimizer: '{optimizer}' (expected 'adam' or 'rprop')")
    if activation not in ('tanh', 'adaptive_relu'):
        raise ValueError(f"Unknown activation: '{activation}' (expected 'tanh' or 'adaptive_relu')")
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
        coord_scheme, arch, mc['hidden_layers'], mc['hidden_neurons'], device,
        activation=activation, m0=m0)

    if arch == 'mixed':
        from experiments.mixed_pinn_variant import MixedPFEvaluator
        evaluator = MixedPFEvaluator(
            u_net, phi_net, E=E, nu=nu, G_c=G_c, l_0=l_0, a=a_phys, k=phys['k'],
            coord_scheme=coord_scheme, williams_r_floor=williams_r_floor
        ).to(device)
    elif arch == 'y_sym':
        evaluator = PFPhysicsEvaluator(
            u_net, phi_net, E=E, nu=nu, G_c=G_c, l_0=l_0, a=a_phys, k=phys['k'],
            coord_scheme=coord_scheme, williams_r_floor=williams_r_floor, y_symmetric=True
        ).to(device)
    elif arch == 'y_sym_hard':
        evaluator = PFPhysicsEvaluator(
            u_net, phi_net, E=E, nu=nu, G_c=G_c, l_0=l_0, a=a_phys, k=phys['k'],
            coord_scheme=coord_scheme, williams_r_floor=williams_r_floor, y_symmetric=True,
            H=H, hard_dirichlet=True
        ).to(device)
    else:
        evaluator = PFPhysicsEvaluator(
            u_net, phi_net, E=E, nu=nu, G_c=G_c, l_0=l_0, a=a_phys, k=phys['k'],
            coord_scheme=coord_scheme, williams_r_floor=williams_r_floor
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

    weights_nd = None
    if use_dem:
        p_mix = sampler.compute_mixture_density(pts_phys.cpu().numpy(), l_0=l_0)
        weights_nd = torch.tensor(1.0 / p_mix, dtype=torch.get_default_dtype(), device=device)

    wc = cfg['weighting']
    if optimizer == 'rprop':
        opt_u = torch.optim.Rprop(u_net.parameters(), lr=tc['u_lr'])
    else:
        opt_u = torch.optim.Adam(u_net.parameters(), lr=tc['u_lr'])

    n_steps = steps if steps is not None else tc.get('diag_steps', 4000)
    v_max_nd = tc['v_max'] / L0   # full load, no quasi-static ramp

    phi_net.eval()
    phi_net.forward = lambda x: torch.zeros(x.shape[0], 1, device=x.device)

    u_net.train()
    final_equil, final_bc = float('inf'), float('inf')
    tag = f"{coord_scheme}/{precision}/{arch}{'/dem' if use_dem else ''} seed={seed}"
    pbar = tqdm(range(n_steps), desc=f"[Diag {tag}]", leave=False)
    for _ in pbar:
        opt_u.zero_grad()
        if use_dem:
            loss_u, loss_equil, bc_u, cf_u, sym_u = evaluator.compute_u_loss_dem(
                pts_nd, weights_nd, bc_dict_nd, v_max_nd, wc['w_bc_u'],
                sym_axis_nd=sym_axis_nd, w_sym=wc['w_sym'], log_energy=log_energy
            )
        elif arch == 'mixed':
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
        if l2_reg > 0.0:
            reg_term = sum(p.pow(2).sum() for p in u_net.parameters())
            loss_u = loss_u + l2_reg * reg_term
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
