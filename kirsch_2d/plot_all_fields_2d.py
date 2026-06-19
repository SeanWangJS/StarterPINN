import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from network import MixedPINN_LogPolar

def plot_all_fields():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    W, H, a = 10.0, 10.0, 1.0
    E, nu, sigma_0 = 1000.0, 0.3, 10.0

    model = MixedPINN_LogPolar(
        in_dim=2, out_dim=5, hidden_layers=5, hidden_neurons=96
    ).to(device)
    
    save_dir = os.path.dirname(__file__)
    model_path = os.path.join(save_dir, 'kirsch_2d_model.pth')
    
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}. Please train the model first.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    grid_res = 150
    x_list = np.linspace(-W, W, grid_res)
    y_list = np.linspace(-H, H, grid_res)
    X, Y = np.meshgrid(x_list, y_list)
    
    mask = (X ** 2 + Y ** 2) >= a ** 2
    X_valid = X[mask]
    Y_valid = Y[mask]

    # Map to 1/4 domain
    X_abs = np.abs(X_valid)
    Y_abs = np.abs(Y_valid)
    
    r = np.sqrt(X_abs**2 + Y_abs**2)
    s = np.log(r / a)
    theta = np.arctan2(Y_abs, X_abs)
    
    s_tensor = torch.tensor(s, dtype=torch.float32, device=device).view(-1, 1)
    t_tensor = torch.tensor(theta, dtype=torch.float32, device=device).view(-1, 1)

    with torch.no_grad():
        u_r_n, u_t_n, sig_rr_n, sig_tt_n, tau_rt_n = model(s_tensor, t_tensor)

    # Denormalize
    sig_rr = sig_rr_n.cpu().numpy().flatten() * sigma_0
    sig_tt = sig_tt_n.cpu().numpy().flatten() * sigma_0
    tau_rt = tau_rt_n.cpu().numpy().flatten() * sigma_0
    
    scale_u = sigma_0 * a / E
    u_r = u_r_n.cpu().numpy().flatten() * scale_u
    u_t = u_t_n.cpu().numpy().flatten() * scale_u

    # Convert to Cartesian in 1st quadrant
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    S_xx_1 = sig_rr * cos_t**2 + sig_tt * sin_t**2 - 2 * tau_rt * sin_t * cos_t
    S_yy_1 = sig_rr * sin_t**2 + sig_tt * cos_t**2 + 2 * tau_rt * sin_t * cos_t
    T_xy_1 = (sig_rr - sig_tt) * sin_t * cos_t + tau_rt * (cos_t**2 - sin_t**2)
    
    U_1 = u_r * cos_t - u_t * sin_t
    V_1 = u_r * sin_t + u_t * cos_t

    # Apply symmetries
    sign_x = np.sign(X_valid)
    sign_y = np.sign(Y_valid)
    
    # Handle exactly zero
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
    out_path = os.path.join(save_dir, 'result_2d_kirsch_all_fields.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"All fields chart saved to {out_path}")

if __name__ == '__main__':
    plot_all_fields()
