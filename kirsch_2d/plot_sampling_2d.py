"""
Kirsch 2D PINN - 径向偏置采样分布可视化
展示三个密度分区：高应力区 r < 1.3a / 过渡区 1.3-2.5a / 远端区 r > 2.5a
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import torch
from geometry import KirschGeometrySampler


def classify_points(x_np, y_np, a, thresholds=(1.3, 2.5)):
    """按径向距离将采样点分为三层"""
    r = np.sqrt(x_np**2 + y_np**2) / a
    inner  = r <= thresholds[0]
    middle = (r > thresholds[0]) & (r <= thresholds[1])
    outer  = r > thresholds[1]
    return inner, middle, outer


def main():
    # ── 参数与采样 ──────────────────────────────────────────
    W, H, a = 10.0, 10.0, 1.0
    k = 2.0           # 偏置指数，与 main.py 保持一致
    N_VIZ = 10000     # 可视化点数（适量避免过密）
    N_BC  = 400       # 每条边界点数

    sampler = KirschGeometrySampler(W=W, H=H, a=a, device='cpu')
    x_biased, y_biased = sampler.sample_radial_biased(N_VIZ, k=k, requires_grad=False)
    x_unif,   y_unif   = sampler.sample_domain(N_VIZ, requires_grad=False)
    bc_dict = sampler.sample_boundaries(N_BC, requires_grad=False)

    # ── NumPy 转换 ──────────────────────────────────────────
    xb = x_biased.numpy().flatten();  yb = y_biased.numpy().flatten()
    xu = x_unif.numpy().flatten();    yu = y_unif.numpy().flatten()

    # ── 分区颜色 ───────────────────────────────────────────
    ZONE_COLORS = {
        'inner':  '#E74C3C',   # 高应力区  r < 1.3a → 红
        'middle': '#9B59B6',   # 过渡区    1.3-2.5a → 紫
        'outer':  '#5DADE2',   # 远端区    r > 2.5a → 蓝
    }
    ZONE_ALPHA = {'inner': 0.8, 'middle': 0.6, 'outer': 0.25}

    # ── 绘图 ───────────────────────────────────────────────
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial'],
        'axes.spines.top': False, 'axes.spines.right': False,
    })

    fig = plt.figure(figsize=(18, 8), dpi=200)
    fig.patch.set_facecolor('#F7F9FC')

    # 三列布局：左=均匀，中=偏置全局，右=偏置缩放
    axes = fig.subplots(1, 3)
    titles = [
        'Uniform Sampling (k=0)\n[Before]',
        'Radial-Biased Sampling (k=2.0)\nFull Domain View',
        'Radial-Biased Sampling (k=2.0)\nZoomed Near-Hole (r < 4a)',
    ]
    for ax, title in zip(axes, titles):
        ax.set_facecolor('#FFFFFF')
        ax.grid(True, linestyle='--', alpha=0.4, color='#D5DBDB', zorder=0)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10, color='#2C3E50')

    # ── 辅助函数：绘制一幅采样图 ─────────────────────────────
    def draw_scatter(ax, x_np, y_np, xlim, ylim):
        inn, mid, out = classify_points(x_np, y_np, a)
        # 画点（从外到内，保证内圈在最上层）
        ax.scatter(x_np[out],  y_np[out],  s=0.8, c=ZONE_COLORS['outer'],
                   alpha=ZONE_ALPHA['outer'],  linewidths=0, zorder=2)
        ax.scatter(x_np[mid],  y_np[mid],  s=1.5, c=ZONE_COLORS['middle'],
                   alpha=ZONE_ALPHA['middle'], linewidths=0, zorder=3)
        ax.scatter(x_np[inn],  y_np[inn],  s=2.5, c=ZONE_COLORS['inner'],
                   alpha=ZONE_ALPHA['inner'],  linewidths=0, zorder=4)
        # 孔口边界
        theta = np.linspace(0, 2 * np.pi, 300)
        ax.fill(a * np.cos(theta), a * np.sin(theta),
                color='#ECF0F1', zorder=5)
        ax.plot(a * np.cos(theta), a * np.sin(theta),
                'k-', lw=1.5, zorder=6)
        # 分区圈虚线
        for r_circle, color in zip([1.3, 2.5], ['#E74C3C', '#9B59B6']):
            ax.plot(r_circle * np.cos(theta), r_circle * np.sin(theta),
                    '--', color=color, lw=0.8, alpha=0.7, zorder=7)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xlabel('x', fontsize=10); ax.set_ylabel('y', fontsize=10)

    # ── 左图：均匀采样 ─────────────────────────────────────
    draw_scatter(axes[0], xu, yu, (-W-0.5, W+0.5), (-H-0.5, H+0.5))

    # 右两图：径向偏置采样（全局 & 局部）
    draw_scatter(axes[1], xb, yb, (-W-0.5, W+0.5), (-H-0.5, H+0.5))
    draw_scatter(axes[2], xb, yb, (-4*a, 4*a), (-4*a, 4*a))

    # ── 边界配点（仅在全局视图上展示）─────────────────────
    for ax_idx, bc_colors in [
        (1, {'left': '#E74C3C', 'right': '#E74C3C',
             'top': '#2ECC71', 'bottom': '#2ECC71', 'hole': '#F39C12'}),
    ]:
        ax = axes[ax_idx]
        for name, (xc, yc) in bc_dict.items():
            ax.scatter(xc.numpy(), yc.numpy(), s=5,
                       c=bc_colors[name], zorder=8, linewidths=0)

    # ── 统计注释 ───────────────────────────────────────────
    for ax_idx, x_np, y_np, label in [
        (0, xu, yu, 'Uniform'), (1, xb, yb, 'Biased k=2')
    ]:
        inn, mid, out = classify_points(x_np, y_np, a)
        n_tot = len(x_np)
        pct_i = inn.sum() / n_tot * 100
        pct_m = mid.sum() / n_tot * 100
        pct_o = out.sum() / n_tot * 100
        axes[ax_idx].text(
            0.97, 0.03,
            f"r<1.3a: {pct_i:.1f}%\n1.3-2.5a: {pct_m:.1f}%\nr>2.5a: {pct_o:.1f}%",
            transform=axes[ax_idx].transAxes,
            ha='right', va='bottom', fontsize=8.5, color='#2C3E50',
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#CCCCCC', alpha=0.85)
        )

    # ── 图例 ───────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=ZONE_COLORS['inner'],  label='High-Stress Zone  r < 1.3a'),
        mpatches.Patch(color=ZONE_COLORS['middle'], label='Transition Zone  1.3a < r < 2.5a'),
        mpatches.Patch(color=ZONE_COLORS['outer'],  label='Far-Field Zone  r > 2.5a'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=3,
               fontsize=10, frameon=True, edgecolor='#CCCCCC',
               bbox_to_anchor=(0.5, -0.02))

    plt.suptitle(
        'Radial-Biased Sampling Strategy  |  p(r) ∝ (a/r)^k, k=2.0',
        fontsize=14, fontweight='bold', color='#1A252F', y=1.01
    )

    plt.tight_layout()
    save_path = 'geometry_samples_biased.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Biased sampling visualization saved to: {save_path}")

    # ── 密度统计对比 ────────────────────────────────────────
    print("\n--- Zone Distribution Comparison ---")
    for label, x_np, y_np in [("Uniform ", xu, yu), ("Biased  ", xb, yb)]:
        inn, mid, out = classify_points(x_np, y_np, a)
        n = len(x_np)
        print(f"  {label}| r<1.3a: {inn.sum()/n*100:5.1f}%  "
              f"| 1.3-2.5a: {mid.sum()/n*100:5.1f}%  "
              f"| r>2.5a: {out.sum()/n*100:5.1f}%")


if __name__ == '__main__':
    main()
