import numpy as np
import matplotlib.pyplot as plt
import os

def kirsch_analytical_stresses(x, y, a=1.0, sigma_0=10.0):
    """
    计算 Kirsch 问题理论精确解下的 Cartesian 应力分量 sigma_xx, sigma_yy, tau_xy
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    
    # 避免除以 0
    r = np.clip(r, 1e-6, None)
    
    # 极坐标下的精确应力分布
    sigma_rr = (sigma_0 / 2) * (1 - (a/r)**2) + (sigma_0 / 2) * (1 - 4*(a/r)**2 + 3*(a/r)**4) * np.cos(2*theta)
    sigma_tt = (sigma_0 / 2) * (1 + (a/r)**2) - (sigma_0 / 2) * (1 + 3*(a/r)**4) * np.cos(2*theta)
    tau_rt = - (sigma_0 / 2) * (1 + 2*(a/r)**2 - 3*(a/r)**4) * np.sin(2*theta)
    
    # 坐标变换：极坐标系 -> 笛卡尔坐标系
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    sigma_xx = sigma_rr * cos_t**2 + sigma_tt * sin_t**2 - 2 * tau_rt * sin_t * cos_t
    sigma_yy = sigma_rr * sin_t**2 + sigma_tt * cos_t**2 + 2 * tau_rt * sin_t * cos_t
    tau_xy = (sigma_rr - sigma_tt) * sin_t * cos_t + tau_rt * (cos_t**2 - sin_t**2)
    
    return sigma_xx, sigma_yy, tau_xy

def kirsch_analytical_displacements(x, y, a=1.0, E=1000.0, nu=0.3, sigma_0=10.0):
    """
    计算 Kirsch 问题理论精确解下的位移 u 和 v (基于平面应力假设)
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    r = np.clip(r, 1e-6, None)
    
    G = E / (2 * (1 + nu))
    kappa = (3 - nu) / (1 + nu)  # 平面应力条件下的 Kolosov 常数
    
    # 极坐标系下的正确理论位移解 (通过对Kirsch应力场应用胡克定律并严格积分得到)
    u_r = (sigma_0 * a / E) * (
        0.5 * (1 - nu) * (r / a) + 0.5 * (1 + nu) * (a / r) + 
        np.cos(2 * theta) * (0.5 * (1 + nu) * (r / a) + 2 * (a / r) - 0.5 * (1 + nu) * (a / r)**3)
    )
    
    u_theta = (sigma_0 * a / E) * np.sin(2 * theta) * (
        -0.5 * (1 + nu) * (r / a) + (nu - 1) * (a / r) - 0.5 * (1 + nu) * (a / r)**3
    )
    
    # 极坐标位移变换到笛卡尔坐标系
    u = u_r * np.cos(theta) - u_theta * np.sin(theta)
    v = u_r * np.sin(theta) + u_theta * np.cos(theta)
    
    return u, v

def plot_analytical():
    W, H, a = 10.0, 10.0, 1.0
    sigma_0 = 10.0
    E, nu = 1000.0, 0.3
    
    # 创建完整的正方形网格 (-W 到 W, -H 到 H)
    grid_res = 300
    x_list = np.linspace(-W, W, grid_res)
    y_list = np.linspace(-H, H, grid_res)
    X, Y = np.meshgrid(x_list, y_list)
    
    # 过滤孔洞区域
    mask = (X**2 + Y**2) >= a**2
    X_valid = X[mask]
    Y_valid = Y[mask]
    
    # 计算精确解
    sigma_xx_valid, sigma_yy_valid, tau_xy_valid = kirsch_analytical_stresses(X_valid, Y_valid, a=a, sigma_0=sigma_0)
    u_exact, v_exact = kirsch_analytical_displacements(X_valid, Y_valid, a=a, E=E, nu=nu, sigma_0=sigma_0)
    
    def fill_grid(data):
        grid = np.full_like(X, np.nan)
        grid[mask] = data
        return grid
        
    S_xx = fill_grid(sigma_xx_valid)
    S_yy = fill_grid(sigma_yy_valid)
    T_xy = fill_grid(tau_xy_valid)
    U_grid = fill_grid(u_exact)
    V_grid = fill_grid(v_exact)
    
    # 开始绘图 (2行3列)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    fields = [S_xx, S_yy, T_xy, U_grid, V_grid]
    titles = [
        r'Stress $\sigma_{xx}$', 
        r'Stress $\sigma_{yy}$', 
        r'Shear Stress $\tau_{xy}$',
        r'Displacement $u_x$ (x-dir)', 
        r'Displacement $u_y$ (y-dir)'
    ]
    cmaps = ['jet', 'jet', 'jet', 'coolwarm', 'coolwarm']
    
    # 画出完整的圆孔边界线
    theta_line = np.linspace(0, 2*np.pi, 200)
    hole_x = a * np.cos(theta_line)
    hole_y = a * np.sin(theta_line)
    
    for i in range(5):
        ax = axes[i]
        contour = ax.contourf(X, Y, fields[i], levels=60, cmap=cmaps[i])
        ax.plot(hole_x, hole_y, 'k-', linewidth=3)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f'Exact Analytical: {titles[i]}', fontsize=14, fontweight='bold')
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        plt.colorbar(contour, ax=ax)
        
    # 关闭第6个子图
    axes[5].axis('off')
    
    plt.suptitle("Kirsch 2D Exact Analytical Solution (Full Plate)", fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    save_path = os.path.join(os.path.dirname(__file__), 'result_2d_kirsch_exact_full_all.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Exact analytical chart saved to {save_path}")

if __name__ == '__main__':
    plot_analytical()
