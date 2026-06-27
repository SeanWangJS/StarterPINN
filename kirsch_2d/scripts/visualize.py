import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
import argparse
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import from core package
from src.network import MixedPINN_LogPolar
from src.geometry import LogPolarGeometrySampler

def get_model(cfg, device):
    model = MixedPINN_LogPolar(
        in_dim=cfg['model']['in_dim'], 
        out_dim=cfg['model']['out_dim'], 
        hidden_layers=cfg['model']['hidden_layers'], 
        hidden_neurons=cfg['model']['hidden_neurons'],
        use_ansatz=cfg['train'].get('use_ansatz', False)
    ).to(device)
    
    model_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'weights', 'kirsch_2d_model.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model file {model_path} not found! Showing untrained results.")
    
    model.eval()
    return model

def plot_fields(cfg, device, out_dir):
    model = get_model(cfg, device)
    W, H, a = cfg['physics']['W'], cfg['physics']['H'], cfg['physics']['a']
    E, sigma_0 = cfg['physics']['E'], cfg['physics']['sigma_0']

    grid_res = 150
    x_list = np.linspace(-W, W, grid_res)
    y_list = np.linspace(-H, H, grid_res)
    X, Y = np.meshgrid(x_list, y_list)
    
    mask = (X ** 2 + Y ** 2) >= a ** 2
    X_valid = X[mask]
    Y_valid = Y[mask]

    X_abs = np.abs(X_valid)
    Y_abs = np.abs(Y_valid)
    
    r = np.sqrt(X_abs**2 + Y_abs**2)
    s = np.log(r / a)
    theta = np.arctan2(Y_abs, X_abs)
    
    s_tensor = torch.tensor(s, dtype=torch.float32, device=device).view(-1, 1)
    t_tensor = torch.tensor(theta, dtype=torch.float32, device=device).view(-1, 1)

    with torch.no_grad():
        u_r_n, u_t_n, sig_rr_n, sig_tt_n, tau_rt_n = model(s_tensor, t_tensor)

    sig_rr = sig_rr_n.cpu().numpy().flatten() * sigma_0
    sig_tt = sig_tt_n.cpu().numpy().flatten() * sigma_0
    tau_rt = tau_rt_n.cpu().numpy().flatten() * sigma_0
    
    scale_u = sigma_0 * a / E
    u_r = u_r_n.cpu().numpy().flatten() * scale_u
    u_t = u_t_n.cpu().numpy().flatten() * scale_u

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    S_xx_1 = sig_rr * cos_t**2 + sig_tt * sin_t**2 - 2 * tau_rt * sin_t * cos_t
    S_yy_1 = sig_rr * sin_t**2 + sig_tt * cos_t**2 + 2 * tau_rt * sin_t * cos_t
    T_xy_1 = (sig_rr - sig_tt) * sin_t * cos_t + tau_rt * (cos_t**2 - sin_t**2)
    
    U_1 = u_r * cos_t - u_t * sin_t
    V_1 = u_r * sin_t + u_t * cos_t

    sign_x = np.sign(X_valid)
    sign_y = np.sign(Y_valid)
    sign_x[sign_x == 0] = 1.0
    sign_y[sign_y == 0] = 1.0

    S_xx_full = S_xx_1
    S_yy_full = S_yy_1
    T_xy_full = T_xy_1 * sign_x * sign_y
    U_full = U_1 * sign_x
    V_full = V_1 * sign_y

    def fill_grid(data):
        grid = np.full_like(X, np.nan)
        grid[mask] = data
        return grid

    S_xx = fill_grid(S_xx_full)
    S_yy = fill_grid(S_yy_full)
    T_xy = fill_grid(T_xy_full)
    U_grid = fill_grid(U_full)
    V_grid = fill_grid(V_full)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    fields = [S_xx, S_yy, T_xy, U_grid, V_grid]
    titles = [
        r'Stress $\sigma_{xx}$',
        r'Stress $\sigma_{yy}$',
        r'Shear Stress $\tau_{xy}$',
        r'Displacement $u$ (x-dir)',
        r'Displacement $v$ (y-dir)'
    ]
    cmaps = ['jet', 'jet', 'jet', 'coolwarm', 'coolwarm']

    theta_circle = np.linspace(0, 2.0 * np.pi, 200)
    hole_x = a * np.cos(theta_circle)
    hole_y = a * np.sin(theta_circle)

    for i in range(5):
        ax = axes[i]
        contour = ax.contourf(X, Y, fields[i], levels=60, cmap=cmaps[i])
        ax.plot(hole_x, hole_y, 'k-', linewidth=3)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(titles[i], fontsize=14)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(contour, ax=ax)

    axes[5].axis('off')

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'result_2d_kirsch_all_fields.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Fields chart saved to {out_path}")

def plot_hole_stress(cfg, device, out_dir, dimensionless=False):
    model = get_model(cfg, device)
    E, sigma_0, a = cfg['physics']['E'], cfg['physics']['sigma_0'], cfg['physics']['a']
    
    num_samples = 100
    theta_vals = np.linspace(0, np.pi / 2, num_samples)
    s_tensor = torch.zeros((num_samples, 1), dtype=torch.float32, device=device)
    theta_tensor = torch.tensor(theta_vals, dtype=torch.float32, device=device).view(-1, 1)
    
    with torch.no_grad():
        u_r_n, u_t_n, sig_rr_n, sig_tt_n, tau_rt_n = model(s_tensor, theta_tensor)
        
    pred_sig_rr = sig_rr_n.cpu().numpy().flatten() * sigma_0
    pred_sig_tt = sig_tt_n.cpu().numpy().flatten() * sigma_0
    pred_sig_rt = tau_rt_n.cpu().numpy().flatten() * sigma_0
    
    scale_u = sigma_0 * a / E
    pred_u_r = u_r_n.cpu().numpy().flatten() * scale_u
    pred_u_theta = u_t_n.cpu().numpy().flatten() * scale_u
    
    exact_sig_rr = np.zeros_like(theta_vals)
    exact_sig_rt = np.zeros_like(theta_vals)
    exact_sig_tt = sigma_0 * (1.0 - 2.0 * np.cos(2.0 * theta_vals))
    
    exact_u_r = (sigma_0 * a / E) * (1.0 + 2.0 * np.cos(2.0 * theta_vals))
    exact_u_theta = (sigma_0 * a / E) * (-2.0 * np.sin(2.0 * theta_vals))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()
    
    if dimensionless:
        plot_data = [
            (exact_sig_rr/sigma_0, pred_sig_rr/sigma_0, r'Dim. Radial Stress $\sigma_{rr}^*$', 'blue'),
            (exact_sig_tt/sigma_0, pred_sig_tt/sigma_0, r'Dim. Tangential Stress $\sigma_{\theta\theta}^*$', 'red'),
            (exact_sig_rt/sigma_0, pred_sig_rt/sigma_0, r'Dim. Shear Stress $\tau_{r\theta}^*$', 'purple'),
            (exact_u_r/scale_u, pred_u_r/scale_u, r'Dim. Radial Displacement $u_r^*$', 'green'),
            (exact_u_theta/scale_u, pred_u_theta/scale_u, r'Dim. Tangential Displacement $u_\theta^*$', 'orange')
        ]
        suffix = "_dimensionless"
    else:
        plot_data = [
            (exact_sig_rr, pred_sig_rr, r'Radial Stress $\sigma_{rr}$ (MPa)', 'blue'),
            (exact_sig_tt, pred_sig_tt, r'Tangential Stress $\sigma_{\theta\theta}$ (MPa)', 'red'),
            (exact_sig_rt, pred_sig_rt, r'Shear Stress $\tau_{r\theta}$ (MPa)', 'purple'),
            (exact_u_r, pred_u_r, r'Radial Displacement $u_r$ (mm)', 'green'),
            (exact_u_theta, pred_u_theta, r'Tangential Displacement $u_\theta$ (mm)', 'orange')
        ]
        suffix = ""
        
    for i, (exact, pred, title, color) in enumerate(plot_data):
        ax = axes[i]
        ax.plot(theta_vals, exact, '-', color=color, linewidth=2.5, label='Analytical (Exact)')
        ax.scatter(theta_vals, pred, color='black', facecolors='none', edgecolors='black', 
                   s=35, alpha=0.8, label='PINN Prediction')
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel(r'Angle $\theta$ (rad)', fontsize=11)
        ax.set_xlim(0, np.pi/2)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=10)
        ax.set_xticks([0, np.pi/4, np.pi/2])
        ax.set_xticklabels([r'$0$', r'$\pi/4$', r'$\pi/2$'])
        
    axes[5].axis('off')
    
    plt.suptitle(f"Comparison of {'Dimensionless ' if dimensionless else ''}Analytical vs. PINN Predicted Solutions at Hole Boundary", 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    save_path = os.path.join(out_dir, f'hole_stress_comparison{suffix}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Hole stress comparison plot saved to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize Mixed-PINN outputs")
    parser.add_argument('--config', type=str, default='../configs/default.yaml')
    parser.add_argument('--mode', type=str, choices=['fields', 'hole_stress', 'hole_stress_dimless', 'all'], default='all')
    args = parser.parse_args()

    with open(os.path.join(os.path.dirname(__file__), args.config), 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    
    if args.mode in ['fields', 'all']:
        plot_fields(cfg, device, out_dir)
    if args.mode in ['hole_stress', 'all']:
        plot_hole_stress(cfg, device, out_dir, dimensionless=False)
    if args.mode in ['hole_stress_dimless', 'all']:
        plot_hole_stress(cfg, device, out_dir, dimensionless=True)

if __name__ == '__main__':
    main()
