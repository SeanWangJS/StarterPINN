"""
Kirsch 2D PINN - 对数极坐标与物理坐标空间采样分布可视化
展示计算域 (s, theta) 的均匀分布以及经过映射后在物理域 (x, y) 的自适应局部加密特征。
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from geometry import LogPolarGeometrySampler

def main():
    # ── 参数与采样 ──────────────────────────────────────────
    R_max = 15.0
    a = 1.0
    s_max = np.log(R_max / a)
    
    N_DOMAIN = 3000     # 域内点数
    N_BC = 150          # 边界点数

    sampler = LogPolarGeometrySampler(R_max=R_max, a=a, device='cpu')
    s_dom, t_dom = sampler.sample_domain(N_DOMAIN, requires_grad=False)
    bc_dict = sampler.sample_boundaries(N_BC, requires_grad=False)

    # 转换成 numpy 数组
    s_dom_np = s_dom.numpy().flatten()
    t_dom_np = t_dom.numpy().flatten()
    
    # 计算物理域坐标 (x, y)
    r_dom_np = a * np.exp(s_dom_np)
    x_dom_np = r_dom_np * np.cos(t_dom_np)
    y_dom_np = r_dom_np * np.sin(t_dom_np)

    # ── 绘图设置 ───────────────────────────────────────────────
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial'],
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    fig = plt.figure(figsize=(18, 5.5), dpi=200)
    fig.patch.set_facecolor('#F8F9FA')

    # 三列布局：左 = 计算域，中 = 物理域全局，右 = 物理域孔口放大
    axes = fig.subplots(1, 3)
    
    titles = [
        'Computational Domain (s, $\\theta$)\n[Uniform Flat Space]',
        'Physical Domain (x, y) [Full View]\n[Natural Exponential Growth]',
        'Physical Domain (x, y) [Near-Hole Zoom]\n[Dense Sampling at Stress Zone]',
    ]
    
    for ax, title in zip(axes, titles):
        ax.set_facecolor('#FFFFFF')
        ax.grid(True, linestyle='--', alpha=0.5, color='#BDC3C7', zorder=0)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=12, color='#2C3E50')

    # 颜色配置
    color_domain = '#7F8C8D'     # 域内点：灰色
    color_hole = '#E74C3C'       # 孔口：红色
    color_far = '#2980B9'        # 远场：蓝色
    color_sym_x = '#27AE60'      # 对称面x：绿色
    color_sym_y = '#8E44AD'      # 对称面y：紫色

    # ── 1. 绘制计算域 (s, theta) ─────────────────────────────
    ax0 = axes[0]
    ax0.scatter(s_dom_np, t_dom_np, s=1.0, c=color_domain, alpha=0.5, zorder=2)
    
    # 绘制计算域边界点
    s_hole, t_hole = bc_dict['hole']
    s_far, t_far = bc_dict['far_field']
    s_sym_x, t_sym_x = bc_dict['sym_x']
    s_sym_y, t_sym_y = bc_dict['sym_y']
    
    ax0.scatter(s_hole.numpy(), t_hole.numpy(), s=5, c=color_hole, zorder=3, label='Hole BC (s=0)')
    ax0.scatter(s_far.numpy(), t_far.numpy(), s=5, c=color_far, zorder=3, label=f'Far BC (s={s_max:.2f})')
    ax0.scatter(s_sym_x.numpy(), t_sym_x.numpy(), s=5, c=color_sym_x, zorder=3, label='Sym-X BC ($\\theta$=0)')
    ax0.scatter(s_sym_y.numpy(), t_sym_y.numpy(), s=5, c=color_sym_y, zorder=3, label='Sym-Y BC ($\\theta$={:.2f})'.format(np.pi/2))
    
    ax0.set_xlim(-0.1, s_max + 0.1)
    ax0.set_ylim(-0.1, np.pi/2 + 0.1)
    ax0.set_xlabel('s = ln(r/a)', fontsize=10, fontweight='bold')
    ax0.set_ylabel('$\\theta$ (rad)', fontsize=10, fontweight='bold')
    ax0.legend(loc='upper right', fontsize=8)

    # ── 2. 绘制物理域全局 (x, y) ─────────────────────────────
    ax1 = axes[1]
    ax1.scatter(x_dom_np, y_dom_np, s=1.0, c=color_domain, alpha=0.4, zorder=2)
    
    # 物理域边界映射
    def to_cartesian(s, t):
        r = a * np.exp(s.numpy())
        return r * np.cos(t.numpy()), r * np.sin(t.numpy())
    
    xh, yh = to_cartesian(s_hole, t_hole)
    xf, yf = to_cartesian(s_far, t_far)
    xsx, ysx = to_cartesian(s_sym_x, t_sym_x)
    xsy, ysy = to_cartesian(s_sym_y, t_sym_y)
    
    ax1.scatter(xh, yh, s=4, c=color_hole, zorder=3)
    ax1.scatter(xf, yf, s=4, c=color_far, zorder=3)
    ax1.scatter(xsx, ysx, s=4, c=color_sym_x, zorder=3)
    ax1.scatter(xsy, ysy, s=4, c=color_sym_y, zorder=3)
    
    # 画出边界轮廓线
    theta_arc = np.linspace(0, np.pi/2, 200)
    ax1.plot(a * np.cos(theta_arc), a * np.sin(theta_arc), 'k-', lw=1.5, zorder=4)
    ax1.plot(R_max * np.cos(theta_arc), R_max * np.sin(theta_arc), 'k--', lw=1.0, zorder=4)
    
    ax1.set_xlim(-0.5, R_max + 0.5)
    ax1.set_ylim(-0.5, R_max + 0.5)
    ax1.set_aspect('equal')
    ax1.set_xlabel('x', fontsize=10, fontweight='bold')
    ax1.set_ylabel('y', fontsize=10, fontweight='bold')

    # ── 3. 绘制物理域局部放大 (x, y) ─────────────────────────
    ax2 = axes[2]
    ax2.scatter(x_dom_np, y_dom_np, s=2.0, c=color_domain, alpha=0.6, zorder=2)
    ax2.scatter(xh, yh, s=8, c=color_hole, zorder=3)
    ax2.scatter(xsx, ysx, s=8, c=color_sym_x, zorder=3)
    ax2.scatter(xsy, ysy, s=8, c=color_sym_y, zorder=3)
    
    ax2.plot(a * np.cos(theta_arc), a * np.sin(theta_arc), 'k-', lw=2.0, zorder=4)
    
    # 绘制参考圆弧线
    for r_ref in [1.5 * a, 2.5 * a, 4.0 * a]:
        ax2.plot(r_ref * np.cos(theta_arc), r_ref * np.sin(theta_arc), ':', color='#7F8C8D', lw=0.8, zorder=1)
        ax2.text(r_ref * np.cos(np.pi/4), r_ref * np.sin(np.pi/4), f'r={r_ref:.1f}a', 
                 fontsize=7, color='#7F8C8D', ha='center', va='center', rotation=-45,
                 bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.7))

    ax2.set_xlim(-0.1, 4.0 * a)
    ax2.set_ylim(-0.1, 4.0 * a)
    ax2.set_aspect('equal')
    ax2.set_xlabel('x', fontsize=10, fontweight='bold')
    ax2.set_ylabel('y', fontsize=10, fontweight='bold')

    # ── 图例与标题 ───────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=color_domain, label='Domain Collocation Points'),
        mpatches.Patch(color=color_hole, label='Hole BC (Traction-Free)'),
        mpatches.Patch(color=color_far, label='Far-Field BC (Dirichlet)'),
        mpatches.Patch(color=color_sym_x, label='Symmetry X BC (u_theta=0, tau_rt=0)'),
        mpatches.Patch(color=color_sym_y, label='Symmetry Y BC (u_theta=0, tau_rt=0)'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=3,
               fontsize=9, frameon=True, edgecolor='#CCCCCC',
               bbox_to_anchor=(0.5, -0.06))

    plt.suptitle(
        'Log-Polar Coordinate Mapping & Adaptive Sampling Visualization',
        fontsize=13, fontweight='bold', color='#1A252F', y=1.01
    )

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), 'geometry_samples_logpolar.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Sampling visualization saved to: {save_path}")

if __name__ == '__main__':
    main()
