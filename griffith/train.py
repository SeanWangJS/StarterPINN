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
        in_dim=mc['in_dim'], out_dim=mc['u_out_dim'],
        hidden_layers=mc['hidden_layers'], hidden_neurons=mc['hidden_neurons']
    ).to(device)

    phi_net = PhaseFieldNet(
        in_dim=mc['in_dim'], out_dim=mc['phi_out_dim'],
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
    # Geometry sampler  –  generate points in PHYSICAL space,
    # then convert to non-dimensional before feeding the network.
    # ------------------------------------------------------------------ #
    sampler = GriffithGeometrySampler(W=W, H=H, a=a_phys, device=device)

    tc = cfg['train']
    pts_phys = sampler.sample_fixed_domain(tc['domain_points'], l_0=l_0)
    bc_dict_phys = sampler.sample_boundaries(tc['bc_points'])
    pts_crack_phys = sampler.get_initial_crack_points(tc['crack_points'])
    pts_crack_face_phys = sampler.sample_crack_face(tc['crack_face_points'])  # physical coords

    # Concatenate crack points into domain (needed for H0 pre-conditioning)
    pts_phys = torch.cat([pts_phys, pts_crack_phys], dim=0)
    N_total = pts_phys.shape[0]
    N_crack = tc['crack_points']

    # Convert everything to non-dimensional coords  (x* = x / L0)
    pts_nd = pts_phys / L0
    bc_dict_nd = {k: v / L0 for k, v in bc_dict_phys.items()}
    crack_face_nd = pts_crack_face_phys / L0

    # ------------------------------------------------------------------ #
    # History field  H*  (non-dimensional: stored as H / H0)
    # ------------------------------------------------------------------ #
    history_H_nd = torch.zeros((N_total, 1), dtype=torch.float32, device=device)

    # Pre-condition initial crack with H* → ∞ (use a large dimensionless value)
    large_H_nd = 1000.0   # H* >> 1  drives phi → 1 at crack points
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
    w_pde_u   = wc['w_pde_u']
    w_bc_u    = wc['w_bc_u']
    w_pde_phi = wc['w_pde_phi']

    global_step = 0

    # ------------------------------------------------------------------ #
    # Quasi-static loading loop
    # ------------------------------------------------------------------ #
    for step in range(1, tc['num_steps'] + 1):
        # Non-dimensional displacement load for this step
        v_max_phys = tc['v_max'] * (step / tc['num_steps'])
        v_max_nd   = v_max_phys / L0      # v* = v / L0   (O(1) for typical loads)

        print(f"\n=== Load Step {step}/{tc['num_steps']} "
              f"| v_max = {v_max_phys:.6f} mm  ({v_max_nd:.4f} non-dim) ===")

        # --------------------------------------------------------------
        # Alternating Minimization  (Staggered Training)
        # --------------------------------------------------------------
        for am in range(tc['am_iters']):
            print(f"  -- AM Iteration {am+1}/{tc['am_iters']} --")

            # ---- 1. Train DisplacementNet (freeze PhaseFieldNet) ------
            u_net.train()
            phi_net.eval()

            pbar_u = tqdm(range(tc['u_steps']), desc="Train U", leave=False)
            for _ in pbar_u:
                opt_u.zero_grad()
                loss_u, pde_u, bc_u, crack_face_u = evaluator.compute_u_loss(
                    pts_nd, bc_dict_nd, v_max_nd, w_pde_u, w_bc_u,
                    crack_face_nd=crack_face_nd,
                    w_crack_face=wc['w_crack_face']
                )
                loss_u.backward()
                opt_u.step()

                writer.add_scalar('Loss/U_Total',      loss_u.item(), global_step)
                writer.add_scalar('Loss/U_PDE',        pde_u,         global_step)
                writer.add_scalar('Loss/U_BC',         bc_u,          global_step)
                writer.add_scalar('Loss/U_CrackFace',  crack_face_u,  global_step)
                pbar_u.set_postfix({'PDE': f'{pde_u:.2e}', 'BC': f'{bc_u:.2e}', 'CF': f'{crack_face_u:.2e}'})
                global_step += 1

            # ---- 2. Update history field H*  -------------------------
            u_net.eval()
            with torch.set_grad_enabled(True):
                pts_nd_grad = pts_nd.clone().requires_grad_(True)
                eps_xx, eps_yy, eps_xy, _, _, _ = evaluator.compute_strains(pts_nd_grad)
                phi_dummy = torch.zeros_like(eps_xx)
                psi_nd, _, _, _ = evaluator.compute_energy_and_stress(
                    eps_xx, eps_yy, eps_xy, phi_dummy
                )
                # psi_nd is psi_e+ / E0; convert to H*:  H* = psi_nd * E0 / H0
                H_nd_new = psi_nd.detach() * (E / H0)

            # Irreversibility: H* = max(H*_old, H*_new)
            history_H_nd = torch.maximum(history_H_nd, H_nd_new)
            # Restore crack pre-conditioning
            history_H_nd[-N_crack:, 0] = large_H_nd

            # ---- 3. Train PhaseFieldNet (freeze DisplacementNet) ------
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
        # Also save the reference scales so post-processing scripts can reconstruct
        # physical quantities: u_phys = u_nd * L0, sig_phys = sig_nd * E0
        'scales': {'L0': L0, 'E0': E, 'H0': H0}
    }, os.path.join(save_dir, 'griffith_pf_model.pth'))

    print("Model saved successfully.")


if __name__ == '__main__':
    main()
