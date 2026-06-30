import torch
import numpy as np
import yaml
import os
import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from src.network import DisplacementNet, PhaseFieldNet
from src.geometry import GriffithGeometrySampler
from src.evaluator import PFPhysicsEvaluator


def main():
    # ------------------------------------------------------------------ #
    # Config
    # ------------------------------------------------------------------ #
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'default.yaml')
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    phys = cfg['physics']
    E, nu, G_c, l_0, a_phys = phys['E'], phys['nu'], phys['G_c'], phys['l_0'], phys['a']
    W, H = phys['W'], phys['H']

    # Reference scales  (see docs/griffith.md §六)
    L0 = a_phys          # reference length [mm]
    H0 = G_c / l_0       # reference energy density [MPa]

    # ------------------------------------------------------------------ #
    # Networks
    # ------------------------------------------------------------------ #
    mc = cfg['model']
    u_net = DisplacementNet(
        in_dim=mc['u_in_dim'], out_dim=mc['u_out_dim'],
        hidden_layers=mc['hidden_layers'], hidden_neurons=mc['hidden_neurons']
    ).to(device)

    phi_net = PhaseFieldNet(
        in_dim=mc['phi_in_dim'], out_dim=mc['phi_out_dim'],
        hidden_layers=mc['hidden_layers'], hidden_neurons=mc['hidden_neurons']
    ).to(device)

    # ------------------------------------------------------------------ #
    # Physics evaluator  (operates entirely in non-dimensional space)
    # ------------------------------------------------------------------ #
    evaluator = PFPhysicsEvaluator(
        u_net, phi_net,
        E=E, nu=nu, G_c=G_c, l_0=l_0, a=a_phys, k=phys['k']
    ).to(device)

    # ------------------------------------------------------------------ #
    # Geometry sampler  –  RIGHT HALF DOMAIN: x ∈ [0, W/2]
    # ------------------------------------------------------------------ #
    sampler = GriffithGeometrySampler(W=W, H=H, a=a_phys, device=device)

    tc = cfg['train']
    # Domain points in physical space (right half)
    pts_phys            = sampler.sample_fixed_domain(tc['domain_points'], l_0=l_0)
    bc_dict_phys        = sampler.sample_boundaries(tc['bc_points'])
    pts_crack_phys      = sampler.get_initial_crack_points(tc['crack_points'])
    pts_crack_face_phys = sampler.sample_crack_face(tc['crack_face_points'])
    pts_sym_axis_phys   = sampler.sample_symmetry_axis(tc['sym_axis_points'])

    # Append crack seed points to domain for history field
    pts_phys = torch.cat([pts_phys, pts_crack_phys], dim=0)
    N_total  = pts_phys.shape[0]
    N_crack  = tc['crack_points']

    # Convert to non-dimensional coords  (x* = x / L0)
    pts_nd           = pts_phys / L0
    bc_dict_nd       = {k: v / L0 for k, v in bc_dict_phys.items()}
    crack_face_nd    = pts_crack_face_phys / L0
    sym_axis_nd      = pts_sym_axis_phys / L0

    # ------------------------------------------------------------------ #
    # History field  H*  (non-dimensional, right half only)
    # ------------------------------------------------------------------ #
    history_H_nd = torch.zeros((N_total, 1), dtype=torch.float32, device=device)

    # Pre-condition crack with H* >> 1 to drive phi → 1 there
    large_H_nd = 500.0
    history_H_nd[-N_crack:, 0] = large_H_nd

    # ------------------------------------------------------------------ #
    # Optimizers
    # ------------------------------------------------------------------ #
    opt_u   = torch.optim.Adam(u_net.parameters(),   lr=tc['u_lr'])
    opt_phi = torch.optim.Adam(phi_net.parameters(), lr=tc['phi_lr'])

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(os.path.dirname(__file__), 'results', 'runs',
                           f'griffith_pf_{timestamp}')
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs: {log_dir}")

    wc = cfg['weighting']
    w_pde_u    = wc['w_pde_u']
    w_bc_u     = wc['w_bc_u']
    w_crack    = wc['w_crack_face']
    w_sym      = wc['w_sym']
    w_pde_phi  = wc['w_pde_phi']

    global_step = 0

    # ------------------------------------------------------------------ #
    # Quasi-static loading loop
    # ------------------------------------------------------------------ #
    for step in range(1, tc['num_steps'] + 1):
        v_max_phys = tc['v_max'] * (step / tc['num_steps'])
        v_max_nd   = v_max_phys / L0

        print(f"\n=== Load Step {step}/{tc['num_steps']} "
              f"| v_max = {v_max_phys:.6f} mm  ({v_max_nd:.4f} non-dim) ===")

        # --------------------------------------------------------------
        # Alternating Minimization
        # --------------------------------------------------------------
        for am in range(tc['am_iters']):
            print(f"  -- AM Iteration {am+1}/{tc['am_iters']} --")

            # ---- 1. Train DisplacementNet ----
            u_net.train()
            phi_net.eval()

            pbar_u = tqdm(range(tc['u_steps']), desc="Train U", leave=False)
            for _ in pbar_u:
                opt_u.zero_grad()
                loss_u, loss_equil, bc_u, cf_u, sym_u = evaluator.compute_u_loss(
                    pts_nd, bc_dict_nd, v_max_nd, w_pde_u, w_bc_u,
                    crack_face_nd=crack_face_nd, w_crack_face=w_crack,
                    sym_axis_nd=sym_axis_nd,    w_sym=w_sym
                )
                loss_u.backward()
                opt_u.step()

                writer.add_scalar('Loss/U_Total',     loss_u.item(), global_step)
                writer.add_scalar('Loss/U_Equil',     loss_equil,    global_step)
                writer.add_scalar('Loss/U_BC',        bc_u,          global_step)
                writer.add_scalar('Loss/U_CrackFace', cf_u,          global_step)
                writer.add_scalar('Loss/U_Sym',       sym_u,         global_step)
                pbar_u.set_postfix({'Eq': f'{loss_equil:.2e}', 'CF': f'{cf_u:.2e}'})
                global_step += 1

            # ---- 2. Update History Field H* ----
            u_net.eval()
            with torch.set_grad_enabled(True):
                pts_nd_g = pts_nd.clone().requires_grad_(True)
                eps_xx, eps_yy, eps_xy, _, _, _, _, _, _ = evaluator.compute_strains(pts_nd_g)
                phi_dummy = torch.zeros_like(eps_xx)
                psi_nd, _, _, _ = evaluator.compute_energy_and_stress(
                    eps_xx, eps_yy, eps_xy, phi_dummy
                )
                H_nd_new = psi_nd.detach() * (E / H0)

            history_H_nd = torch.maximum(history_H_nd, H_nd_new)
            history_H_nd[-N_crack:, 0] = large_H_nd   # restore crack pre-conditioning

            # ---- 3. Train PhaseFieldNet ----
            phi_net.train()

            pbar_phi = tqdm(range(tc['phi_steps']), desc="Train Phi", leave=False)
            for _ in pbar_phi:
                opt_phi.zero_grad()
                loss_phi, pde_phi = evaluator.compute_phi_loss(
                    pts_nd, history_H_nd, w_pde_phi
                )
                loss_phi.backward()
                opt_phi.step()

                writer.add_scalar('Loss/Phi_Total', loss_phi.item(), global_step)
                writer.add_scalar('Loss/Phi_PDE',   pde_phi,         global_step)
                pbar_phi.set_postfix({'PDE': f'{pde_phi:.2e}'})
                global_step += 1

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    print("\nTraining completed.")
    save_dir = os.path.join(os.path.dirname(__file__), 'results', 'weights')
    os.makedirs(save_dir, exist_ok=True)

    torch.save({
        'u_net':   u_net.state_dict(),
        'phi_net': phi_net.state_dict(),
        'scales':  {'L0': L0, 'E0': E, 'H0': H0},
        'domain':  'right_half',   # flag for post-processing
    }, os.path.join(save_dir, 'griffith_pf_model.pth'))

    print("Model saved successfully.")


if __name__ == '__main__':
    main()
