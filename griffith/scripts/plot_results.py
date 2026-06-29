import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.network import DisplacementNet, PhaseFieldNet
from src.evaluator import PFPhysicsEvaluator


def reflect_to_full_field(X_half, Y, U_half, V_half, PHI_half):
    """
    Mirror right-half predictions to the full domain.
    
    Symmetry rules for Mode I (left-right symmetric):
        u_full(x, y)   = +u_nn(|x|, y)  for x > 0
                       = -u_nn(|x|, y)  for x < 0   [antisymmetric in x]
        v_full(x, y)   =  v_nn(|x|, y)              [symmetric in x]
        phi_full(x, y) = phi_nn(|x|, y)             [symmetric in x]
    
    Inputs are on the grid x ∈ [0, W/2], so we mirror along x=0.
    """
    # X_half: (ny, nx_half), columns go 0 → W/2
    # Mirror: flip left-right, then negate u for the left side
    U_left   = -np.fliplr(U_half)    # antisymmetric
    V_left   =  np.fliplr(V_half)    # symmetric
    PHI_left =  np.fliplr(PHI_half)  # symmetric

    # Exclude the x=0 column when concatenating to avoid duplication
    U_full   = np.hstack([U_left[:, :-1], U_half])
    V_full   = np.hstack([V_left[:, :-1], V_half])
    PHI_full = np.hstack([PHI_left[:, :-1], PHI_half])

    # Build matching X grid
    X_left = -np.fliplr(X_half)
    X_full  = np.hstack([X_left[:, :-1], X_half])

    return X_full, Y, U_full, V_full, PHI_full


def plot_final_results():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default.yaml')
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    mc = cfg['model']
    u_net = DisplacementNet(
        in_dim=mc['in_dim'], out_dim=mc['u_out_dim'],
        hidden_layers=mc['hidden_layers'], hidden_neurons=mc['hidden_neurons']
    ).to(device)
    phi_net = PhaseFieldNet(
        in_dim=mc['in_dim'], out_dim=mc['phi_out_dim'],
        hidden_layers=mc['hidden_layers'], hidden_neurons=mc['hidden_neurons']
    ).to(device)

    model_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'weights', 'griffith_pf_model.pth')
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}!")
        return

    checkpoint = torch.load(model_path, map_location=device)
    u_net.load_state_dict(checkpoint['u_net'])
    phi_net.load_state_dict(checkpoint['phi_net'])
    u_net.eval()
    phi_net.eval()

    phys  = cfg['physics']
    W, H  = phys['W'], phys['H']
    a_phys = phys['a']
    L0 = a_phys
    E0 = phys['E']

    # ---- Generate right-half grid (x ∈ [0, W/2]) ----
    x_half = np.linspace(0.0, W/2, 150)
    y_vals = np.linspace(-H/2, H/2, 200)
    X_half, Y = np.meshgrid(x_half, y_vals)

    pts_phys = np.hstack([X_half.flatten()[:, None], Y.flatten()[:, None]])
    pts_nd   = torch.tensor(pts_phys / L0, dtype=torch.float32, device=device)
    pts_nd.requires_grad_(True)

    # ---- Evaluate on right half ----
    with torch.set_grad_enabled(True):
        evaluator = PFPhysicsEvaluator(
            u_net, phi_net,
            phys['E'], phys['nu'], phys['G_c'], phys['l_0'], a_phys, phys['k']
        ).to(device)

        eps_xx, eps_yy, eps_xy, _, u_nd, v_nd = evaluator.compute_strains(pts_nd)
        phi = phi_net(pts_nd)
        _, sig_xx_nd, sig_yy_nd, _ = evaluator.compute_energy_and_stress(
            eps_xx, eps_yy, eps_xy, phi)

    ny, nx_half = X_half.shape
    U_half   = (u_nd.detach().cpu().numpy().reshape(ny, nx_half)) * L0
    V_half   = (v_nd.detach().cpu().numpy().reshape(ny, nx_half)) * L0
    PHI_half = phi.detach().cpu().numpy().reshape(ny, nx_half)
    SIG_half = sig_yy_nd.detach().cpu().numpy().reshape(ny, nx_half) * E0

    # ---- Mirror to full field ----
    X_full, _, U_full, V_full, PHI_full = reflect_to_full_field(X_half, Y, U_half, V_half, PHI_half)
    _, _, _, _, SIG_full = reflect_to_full_field(X_half, Y, SIG_half, SIG_half, SIG_half)
    # For stress σ_yy: symmetric about x=0
    SIG_full = np.hstack([np.fliplr(SIG_half)[:, :-1], SIG_half])

    # ---- Analytical Westergaard comparison ----
    sys.path.insert(0, os.path.dirname(__file__))
    from plot_analytical import westergaard_field
    import yaml
    E_prime = phys['E'] / (1 - phys['nu']**2)
    v_max   = cfg['train']['v_max']
    sigma_0 = E_prime * (v_max / (H / 2.0))

    X_ana, Y_ana = np.meshgrid(np.linspace(-W/2, W/2, 300), y_vals)
    _, sig_yy_ana, _ = westergaard_field(X_ana, Y_ana, a_phys, sigma_0)

    # ---- Plot ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    im1 = axes[0, 0].pcolormesh(X_full, Y, V_full, cmap='jet', shading='auto')
    axes[0, 0].set_title('Predicted Vertical Displacement (v) [mm]')
    axes[0, 0].set_aspect('equal'); fig.colorbar(im1, ax=axes[0, 0])

    im2 = axes[0, 1].pcolormesh(X_full, Y, PHI_full, cmap='hot_r', shading='auto', vmin=0, vmax=1)
    axes[0, 1].set_title(r'Predicted Phase Field ($\phi$)')
    axes[0, 1].set_aspect('equal'); fig.colorbar(im2, ax=axes[0, 1])

    vmax_s = np.percentile(SIG_full, 95)
    vmin_s = np.min(SIG_full)
    im3 = axes[1, 0].pcolormesh(X_full, Y, SIG_full, cmap='jet', shading='auto', vmin=vmin_s, vmax=vmax_s)
    axes[1, 0].set_title(r'Predicted Stress ($\sigma_{yy}$) [MPa]')
    axes[1, 0].set_aspect('equal'); fig.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].pcolormesh(X_ana, Y_ana, sig_yy_ana, cmap='jet', shading='auto', vmin=vmin_s, vmax=vmax_s)
    axes[1, 1].set_title(r'Analytical Stress ($\sigma_{yy}$) [MPa]')
    axes[1, 1].set_aspect('equal'); fig.colorbar(im4, ax=axes[1, 1])

    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'final_prediction_with_stress.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved to {save_path}")


if __name__ == '__main__':
    plot_final_results()
