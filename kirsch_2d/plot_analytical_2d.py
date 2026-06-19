import numpy as np
import matplotlib.pyplot as plt
import os

def kirsch_analytical_sigma_xx(x, y, a=1.0, sigma_0=10.0):
    """
    计算 Kirsch 问题理论精确解下的 sigma_xx
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
    sigma_xx = sigma_rr * np.cos(theta)**2 + sigma_tt * np.sin(theta)**2 - 2 * tau_rt * np.sin(theta) * np.cos(theta)
    
    return sigma_xx

def kirsch_analytical_displacements(x, y, a=1.0, E=1000.0, nu=0.3, sigma_0=10.0):
    """
    计算 Kirsch 问题理论精确解下的位移 u 和 v (基于平面应力假设)
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    r = np.clip(r, 1e-6, None)
    
    G = E / (2 * (1 + nu))
    kappa = (3 - nu) / (1 + nu)  # 平面应力条件下的 Kolosov 常数
    
    # 极坐标系下的位移
    u_r = (sigma_0 * a / (8 * G)) * (
        (r / a) * (kappa - 1) + 2 * (a / r) + np.cos(2 * theta) * (2 * (r / a) + (a / r) * (kappa + 1) - (a / r)**3)
    )
    
    u_theta = (sigma_0 * a / (8 * G)) * np.sin(2 * theta) * (
        -2 * (r / a) - (a / r) * (kappa - 1) - (a / r)**3
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
    sigma_xx_valid = kirsch_analytical_sigma_xx(X_valid, Y_valid, a=a, sigma_0=sigma_0)
    u_exact, v_exact = kirsch_analytical_displacements(X_valid, Y_valid, a=a, E=E, nu=nu, sigma_0=sigma_0)
    
    def fill_grid(data):
        grid = np.full_like(X, np.nan)
        grid[mask] = data
        return grid
        
    S_xx = fill_grid(sigma_xx_valid)
    U_grid = fill_grid(u_exact)
    V_grid = fill_grid(v_exact)
    
    # 开始绘图
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    
    fields = [S_xx, U_grid, V_grid]
    titles = [r'Stress $\sigma_{xx}$', r'Displacement $u$ (x-dir)', r'Displacement $v$ (y-dir)']
    cmaps = ['jet', 'coolwarm', 'coolwarm']
    
    # 画出完整的圆孔边界线
    theta_line = np.linspace(0, 2*np.pi, 200)
    hole_x = a * np.cos(theta_line)
    hole_y = a * np.sin(theta_line)
    
    for i in range(3):
        ax = axes[i]
        contour = ax.contourf(X, Y, fields[i], levels=60, cmap=cmaps[i])
        ax.plot(hole_x, hole_y, 'k-', linewidth=3)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f'Kirsch 2D Exact: {titles[i]} (Full Plate)', fontsize=14, fontweight='bold')
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        plt.colorbar(contour, ax=ax)
    
    save_path = os.path.join(os.path.dirname(__file__), 'result_2d_kirsch_exact_full_all.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Exact analytical chart saved to {save_path}")

if __name__ == '__main__':
    plot_analytical()
