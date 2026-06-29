import numpy as np
import matplotlib.pyplot as plt
import os

def westergaard_field(X, Y, a, sigma_0):
    """
    计算无限大平板中心裂纹的 Westergaard 应力场 (Mode I)
    """
    Z_complex = X + 1j * Y
    
    # 避免奇点除零
    mask = np.abs(Z_complex - a) < 1e-6
    mask |= np.abs(Z_complex + a) < 1e-6
    
    # 初始化
    sig_xx = np.zeros_like(X)
    sig_yy = np.zeros_like(X)
    tau_xy = np.zeros_like(X)
    
    valid = ~mask
    z = Z_complex[valid]
    
    # 使用 np.sqrt(z - a) * np.sqrt(z + a) 避免 NumPy 在 x=0 处产生错误的分支切割 (Branch cut)
    # np.sqrt(w) 默认的分支切割在 w 的负实半轴上。
    # 当 w = z^2 - a^2 时，x=0 的整个 y 轴都会映射到 w < 0，导致出现一条虚假的垂直断层。
    # 改为分别求平方根后相乘，由于两次符号反转的抵消，分支切割将完美且仅仅落在真正的裂纹区间 [-a, a] 上。
    sqrt_term = np.sqrt(z - a + 1e-12j) * np.sqrt(z + a + 1e-12j)
    
    Z = sigma_0 * z / sqrt_term
    Z_prime = -sigma_0 * a**2 / (sqrt_term**3)
    
    sig_xx[valid] = np.real(Z) - Y[valid] * np.imag(Z_prime)
    sig_yy[valid] = np.real(Z) + Y[valid] * np.imag(Z_prime)
    tau_xy[valid] = -Y[valid] * np.real(Z_prime)
    
    # 处理裂纹面上的应力（理论上应该为 0，因为免力边界）
    crack_mask = (np.abs(Y) < 1e-6) & (np.abs(X) < a)
    sig_yy[crack_mask] = 0.0
    tau_xy[crack_mask] = 0.0
    
    return sig_xx, sig_yy, tau_xy

def irwin_field(X, Y, a, sigma_0, finite_correction=False, W=1.0):
    """
    计算右侧裂尖 (x=a, y=0) 的 Irwin 渐近应力场
    """
    # 裂尖极坐标 (相对于右侧裂尖)
    r = np.sqrt((X - a)**2 + Y**2)
    theta = np.arctan2(Y, X - a)
    
    # 避免奇点除零
    mask = r < 1e-6
    valid = ~mask
    
    r_val = r[valid]
    th_val = theta[valid]
    
    if finite_correction:
        Y_factor = np.sqrt(np.cos(np.pi * a / W)**-1) # sec = 1/cos
    else:
        Y_factor = 1.0
        
    K_I = Y_factor * sigma_0 * np.sqrt(np.pi * a)
    
    sig_xx = np.zeros_like(X)
    sig_yy = np.zeros_like(X)
    tau_xy = np.zeros_like(X)
    
    factor = K_I / np.sqrt(2 * np.pi * r_val)
    
    sig_xx[valid] = factor * np.cos(th_val/2) * (1 - np.sin(th_val/2) * np.sin(3*th_val/2))
    sig_yy[valid] = factor * np.cos(th_val/2) * (1 + np.sin(th_val/2) * np.sin(3*th_val/2))
    tau_xy[valid] = factor * np.cos(th_val/2) * np.sin(th_val/2) * np.cos(3*th_val/2)
    
    return sig_xx, sig_yy, tau_xy

def plot_analytical_solutions():
    # 从配置文件读取参数保持一致
    import yaml
    import os
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default.yaml')
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    W_half, H_half = cfg['physics']['W'] / 2, cfg['physics']['H'] / 2
    a = cfg['physics']['a']
    
    # 根据施加的位移载荷 v_max 估算远场等效拉应力 sigma_0
    # 平面应变假设: E' = E / (1 - nu^2)
    E_prime = cfg['physics']['E'] / (1 - cfg['physics']['nu']**2)
    v_max = cfg['train']['v_max']
    sigma_0 = E_prime * (v_max / H_half)
    
    # 网格生成
    x = np.linspace(-W_half, W_half, 400)
    y = np.linspace(-H_half, H_half, 400)
    X, Y = np.meshgrid(x, y)
    
    # 1. 计算 Westergaard 全局解
    w_sig_xx, w_sig_yy, w_tau_xy = westergaard_field(X, Y, a, sigma_0)
    
    # 2. 计算 Irwin 右侧裂尖局部解 (无限大平板)
    i_sig_xx, i_sig_yy, i_tau_xy = irwin_field(X, Y, a, sigma_0, finite_correction=False)
    
    # 画图比较 Sigma_yy
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Westergaard
    cmap = 'jet'
    vmax = sigma_0 * 5 # 截断应力以显示云图
    vmin = -sigma_0
    
    # Mask out the interior of the crack for better visualization
    crack_x = np.linspace(-a, a, 100)
    crack_y = np.zeros_like(crack_x)
    
    im1 = axes[0].pcolormesh(X, Y, w_sig_yy, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
    axes[0].plot(crack_x, crack_y, 'w-', linewidth=2)
    axes[0].set_title(r'Westergaard Global Field ($\sigma_{yy}$)', fontsize=14)
    axes[0].set_aspect('equal')
    fig.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].pcolormesh(X, Y, i_sig_yy, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
    axes[1].plot(crack_x, crack_y, 'w-', linewidth=2)
    axes[1].set_title(r'Irwin Near-Tip Field ($\sigma_{yy}$, Right Tip)', fontsize=14)
    axes[1].set_aspect('equal')
    
    # Irwin 的解只在右侧裂尖 (x=a, y=0) 附近有效，所以我们在图上标出一个有效范围虚线圈
    circle = plt.Circle((a, 0), 0.15, color='white', fill=False, linestyle='--', linewidth=1.5, alpha=0.7)
    axes[1].add_patch(circle)
    
    fig.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'analytical_sigma_yy_comparison.png')
    plt.savefig(out_path, dpi=300)
    print(f"Saved analytical comparison plot to {out_path}")
    
    # 额外画一条中心线 (y=0) 的应力分布对比
    plt.figure(figsize=(8, 6))
    x_line = np.linspace(a + 1e-4, W_half, 500) # 只画裂尖右侧
    y_line = np.zeros_like(x_line)
    
    _, w_line_yy, _ = westergaard_field(x_line, y_line, a, sigma_0)
    _, i_line_yy, _ = irwin_field(x_line, y_line, a, sigma_0, finite_correction=False)
    
    plt.plot(x_line, w_line_yy, 'b-', linewidth=2, label='Westergaard (Exact)')
    plt.plot(x_line, i_line_yy, 'r--', linewidth=2, label='Irwin (Asymptotic)')
    plt.axvline(x=a, color='gray', linestyle=':', label='Crack Tip')
    
    plt.title(r'Stress $\sigma_{yy}$ ahead of the right crack tip ($y=0$)', fontsize=14)
    plt.xlabel('x coordinate', fontsize=12)
    plt.ylabel(r'$\sigma_{yy}$', fontsize=12)
    plt.xlim(a, W_half)
    plt.ylim(0, sigma_0 * 8)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    line_out_path = os.path.join(out_dir, 'analytical_line_plot.png')
    plt.savefig(line_out_path, dpi=300)
    print(f"Saved line plot to {line_out_path}")

if __name__ == '__main__':
    plot_analytical_solutions()
