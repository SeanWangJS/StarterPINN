import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from network import MixedPINN_LogPolar

def check_hole_stress_and_plot():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 物理和几何参数
    E, nu, sigma_0, a = 1000.0, 0.3, 10.0, 1.0
    
    # 初始化并加载模型
    model = MixedPINN_LogPolar(in_dim=2, out_dim=5, hidden_layers=5, hidden_neurons=96, use_ansatz=True).to(device)
    model_path = os.path.join(os.path.dirname(__file__), 'kirsch_2d_model.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Error: Model file {model_path} not found!")
        return
        
    model.eval()
    
    # 在孔口采样 (r = a, 即 s = 0)
    # theta 范围为 [0, pi/2]
    num_samples = 100
    theta_vals = np.linspace(0, np.pi / 2, num_samples)
    
    s_tensor = torch.zeros((num_samples, 1), dtype=torch.float32, device=device)
    theta_tensor = torch.tensor(theta_vals, dtype=torch.float32, device=device).view(-1, 1)
    
    # 模型预测
    with torch.no_grad():
        u_r_n, u_t_n, sig_rr_n, sig_tt_n, tau_rt_n = model(s_tensor, theta_tensor)
        
    # 反归一化
    pred_sig_rr = sig_rr_n.cpu().numpy().flatten() * sigma_0
    pred_sig_tt = sig_tt_n.cpu().numpy().flatten() * sigma_0
    pred_sig_rt = tau_rt_n.cpu().numpy().flatten() * sigma_0
    
    scale_u = sigma_0 * a / E
    pred_u_r = u_r_n.cpu().numpy().flatten() * scale_u
    pred_u_theta = u_t_n.cpu().numpy().flatten() * scale_u
    
    # 理论精确解计算
    # 1. 应力分量
    exact_sig_rr = np.zeros_like(theta_vals)
    exact_sig_rt = np.zeros_like(theta_vals)
    exact_sig_tt = sigma_0 * (1.0 - 2.0 * np.cos(2.0 * theta_vals))
    
    # 2. 位移分量
    G = E / (2.0 * (1.0 + nu))
    kappa = (3.0 - nu) / (1.0 + nu)
    # r = a
    exact_u_r = (sigma_0 * a / (8.0 * G)) * (
        (kappa - 1.0) + 2.0 + np.cos(2.0 * theta_vals) * (2.0 + (kappa + 1.0) - 1.0)
    )
    exact_u_theta = (sigma_0 * a / (8.0 * G)) * np.sin(2.0 * theta_vals) * (
        -2.0 - (kappa - 1.0) - 1.0
    )
    
    # 绘图设置
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()
    
    # 定义绘图的数据、标签和标题
    plot_data = [
        # (理论值, 预测值, 标签/标题, 颜色)
        (exact_sig_rr, pred_sig_rr, r'Radial Stress $\sigma_{rr}$', 'blue'),
        (exact_sig_tt, pred_sig_tt, r'Tangential Stress $\sigma_{\theta\theta}$', 'red'),
        (exact_sig_rt, pred_sig_rt, r'Shear Stress $\tau_{r\theta}$', 'purple'),
        (exact_u_r, pred_u_r, r'Radial Displacement $u_r$', 'green'),
        (exact_u_theta, pred_u_theta, r'Tangential Displacement $u_\theta$', 'orange')
    ]
    
    for i, (exact, pred, title, color) in enumerate(plot_data):
        ax = axes[i]
        # 理论解画为实线
        ax.plot(theta_vals, exact, '-', color=color, linewidth=2.5, label='Analytical (Exact)')
        # 预测解画为散点
        ax.scatter(theta_vals, pred, color='black', facecolors='none', edgecolors='black', 
                   s=35, alpha=0.8, label='PINN Prediction')
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel(r'Angle $\theta$ (rad)', fontsize=11)
        ax.set_xlim(0, np.pi/2)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=10)
        
        # 针对应力在 x 轴的刻度做特殊设置（0 到 pi/2）
        ax.set_xticks([0, np.pi/4, np.pi/2])
        ax.set_xticklabels([r'$0$', r'$\pi/4$', r'$\pi/2$'])
        
    # 关闭第6个子图
    axes[5].axis('off')
    
    plt.suptitle("Comparison of Analytical vs. PINN Predicted Solutions at the Hole Boundary ($r = a$)", 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    save_path = os.path.join(os.path.dirname(__file__), 'hole_stress_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Hole stress comparison plot successfully saved to: {save_path}")

if __name__ == '__main__':
    check_hole_stress_and_plot()
