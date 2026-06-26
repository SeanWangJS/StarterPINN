import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from network import MixedPINN_LogPolar

def check_hole_stress_and_plot_dimensionless():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 物理和几何参数
    nu = 0.3
    
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
    
    # 模型预测 (直接使用模型输出的无量纲值)
    with torch.no_grad():
        u_r_n, u_t_n, sig_rr_n, sig_tt_n, tau_rt_n = model(s_tensor, theta_tensor)
        
    pred_sig_rr = sig_rr_n.cpu().numpy().flatten()
    pred_sig_tt = sig_tt_n.cpu().numpy().flatten()
    pred_sig_rt = tau_rt_n.cpu().numpy().flatten()
    
    pred_u_r = u_r_n.cpu().numpy().flatten()
    pred_u_theta = u_t_n.cpu().numpy().flatten()
    
    # 理论精确解的无量纲表示
    # 1. 应力分量（无量纲，即除以 sigma_0）
    exact_sig_rr = np.zeros_like(theta_vals)
    exact_sig_rt = np.zeros_like(theta_vals)
    exact_sig_tt = 1.0 - 2.0 * np.cos(2.0 * theta_vals)
    
    # 2. 位移分量（无量纲，即除以 sigma_0 * a / E）
    # 在 r = a 处：
    # 根据严格的理论推导 (基于应变积分):
    # u_r_exact = \sigma_0 a / E * (1.0 + 2.0 * \cos(2\theta))
    # exact_u_theta = \sigma_0 a / E * (-2.0 * \sin(2\theta))
    exact_u_r = 1.0 + 2.0 * np.cos(2.0 * theta_vals)
    exact_u_theta = - 2.0 * np.sin(2.0 * theta_vals)
    
    # 绘图设置
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()
    
    # 定义绘图的数据、标签和标题
    plot_data = [
        # (理论值, 预测值, 标签/标题, Y轴单位/标注, 颜色)
        (exact_sig_rr, pred_sig_rr, r'Dimensionless Radial Stress $\sigma_{rr}^*$', r'$\sigma_{rr} / \sigma_0$', 'blue'),
        (exact_sig_tt, pred_sig_tt, r'Dimensionless Tangential Stress $\sigma_{\theta\theta}^*$', r'$\sigma_{\theta\theta} / \sigma_0$', 'red'),
        (exact_sig_rt, pred_sig_rt, r'Dimensionless Shear Stress $\tau_{r\theta}^*$', r'$\tau_{r\theta} / \sigma_0$', 'purple'),
        (exact_u_r, pred_u_r, r'Dimensionless Radial Displacement $u_r^*$', r'$u_r / (\sigma_0 a / E)$', 'green'),
        (exact_u_theta, pred_u_theta, r'Dimensionless Tangential Displacement $u_\theta^*$', r'$u_\theta / (\sigma_0 a / E)$', 'orange')
    ]
    
    for i, (exact, pred, title, ylabel, color) in enumerate(plot_data):
        ax = axes[i]
        # 理论解画为实线
        ax.plot(theta_vals, exact, '-', color=color, linewidth=2.5, label='Analytical (Dimensionless)')
        # 预测解画为散点
        ax.scatter(theta_vals, pred, color='black', facecolors='none', edgecolors='black', 
                   s=35, alpha=0.8, label='PINN Prediction')
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel(r'Angle $\theta$ (rad)', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xlim(0, np.pi/2)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=10)
        
        # 针对应力在 x 轴的刻度做特殊设置（0 到 pi/2）
        ax.set_xticks([0, np.pi/4, np.pi/2])
        ax.set_xticklabels([r'$0$', r'$\pi/4$', r'$\pi/2$'])
        
    # 关闭第6个子图
    axes[5].axis('off')
    
    plt.suptitle("Comparison of Dimensionless Analytical vs. PINN Predicted Solutions at the Hole Boundary ($r = a$)", 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    save_path = os.path.join(os.path.dirname(__file__), 'hole_stress_comparison_dimensionless.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Hole stress dimensionless comparison plot successfully saved to: {save_path}")

if __name__ == '__main__':
    check_hole_stress_and_plot_dimensionless()
