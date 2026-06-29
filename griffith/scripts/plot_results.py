import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
import sys

# Add parent directory to path to allow src imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.network import DisplacementNet, PhaseFieldNet

def plot_final_results():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default.yaml')
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    u_net = DisplacementNet(in_dim=2, out_dim=2, hidden_layers=4, hidden_neurons=64).to(device)
    phi_net = PhaseFieldNet(in_dim=2, out_dim=1, hidden_layers=4, hidden_neurons=64).to(device)
    
    model_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'weights', 'griffith_pf_model.pth')
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}!")
        return
        
    checkpoint = torch.load(model_path, map_location=device)
    u_net.load_state_dict(checkpoint['u_net'])
    phi_net.load_state_dict(checkpoint['phi_net'])
    u_net.eval()
    phi_net.eval()
    
    # Generate physical grid
    W, H = cfg['physics']['W'], cfg['physics']['H']
    a_phys = cfg['physics']['a']
    L0 = a_phys
    E0 = cfg['physics']['E']
    
    x = np.linspace(-W/2, W/2, 200)
    y = np.linspace(-H/2, H/2, 200)
    X, Y = np.meshgrid(x, y)
    pts_phys = torch.tensor(np.hstack([X.flatten()[:, None], Y.flatten()[:, None]]), dtype=torch.float32, device=device)
    
    # 1. Convert physical coords to non-dimensional coords for the network
    pts_nd = pts_phys / L0
    pts_nd.requires_grad_(True)
    
    with torch.set_grad_enabled(True):
        from src.evaluator import PFPhysicsEvaluator
        evaluator = PFPhysicsEvaluator(u_net, phi_net, cfg['physics']['E'], cfg['physics']['nu'], cfg['physics']['G_c'], cfg['physics']['l_0'], a_phys, cfg['physics']['k']).to(device)
        
        # 2. Evaluate using non-dimensional evaluator
        eps_xx, eps_yy, eps_xy, _, u_nd, v_nd = evaluator.compute_strains(pts_nd)
        phi = phi_net(pts_nd)
        _, sig_xx_nd, sig_yy_nd, tau_xy_nd = evaluator.compute_energy_and_stress(eps_xx, eps_yy, eps_xy, phi)
        
    # 3. Restore physical dimensions for plotting!
    V_phys = (v_nd.detach().cpu().numpy().reshape(200, 200)) * L0
    PHI = phi.detach().cpu().numpy().reshape(200, 200)
    SIG_YY_phys = (sig_yy_nd.detach().cpu().numpy().reshape(200, 200)) * E0
        
    # Also compute analytical Westergaard solution for comparison
    from plot_analytical import westergaard_field
    a = cfg['physics']['a']
    # Estimate equivalent far-field stress sigma_0 from the max displacement
    # Hooke's law: eps_yy = sigma_yy / E' -> v / (H/2) = sigma_0 / E'
    # For plane strain, E' = E / (1 - nu^2)
    E_prime = cfg['physics']['E'] / (1 - cfg['physics']['nu']**2)
    v_max = cfg['train']['v_max']
    sigma_0 = E_prime * (v_max / (cfg['physics']['H'] / 2.0))
    _, sig_yy_ana, _ = westergaard_field(X, Y, a, sigma_0)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    im1 = axes[0, 0].pcolormesh(X, Y, V_phys, cmap='jet', shading='auto')
    axes[0, 0].set_title('Predicted Vertical Displacement (v)')
    fig.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].pcolormesh(X, Y, PHI, cmap='hot_r', shading='auto', vmin=0, vmax=1)
    axes[0, 1].set_title('Predicted Phase Field ($\phi$)')
    fig.colorbar(im2, ax=axes[0, 1])
    
    im3 = axes[1, 0].pcolormesh(X, Y, SIG_YY_phys, cmap='jet', shading='auto')
    axes[1, 0].set_title('Predicted Stress ($\sigma_{yy}$)')
    fig.colorbar(im3, ax=axes[1, 0])
    
    # Use same color scale for analytical to compare
    vmax = np.max(SIG_YY_phys) if np.max(SIG_YY_phys) > 0 else sigma_0 * 2
    vmin = np.min(SIG_YY_phys) if np.min(SIG_YY_phys) < 0 else -sigma_0
    im4 = axes[1, 1].pcolormesh(X, Y, sig_yy_ana, cmap='jet', shading='auto', vmin=vmin, vmax=vmax)
    axes[1, 1].set_title('Analytical Stress ($\sigma_{yy}$)')
    fig.colorbar(im4, ax=axes[1, 1])
    
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'final_prediction_with_stress.png')
    plt.savefig(save_path, dpi=300)
    print(f"Saved plot to {save_path}")

if __name__ == '__main__':
    plot_final_results()
