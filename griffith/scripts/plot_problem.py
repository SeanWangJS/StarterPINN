import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

def plot_griffith_problem():
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 几何尺寸
    W = 1.0
    H = 1.0
    a = 0.1
    
    # 绘制矩形平板
    plate = patches.Rectangle((-W/2, -H/2), W, H, linewidth=2, edgecolor='black', facecolor='#d9e1eb', zorder=1)
    ax.add_patch(plate)
    
    # 绘制中心裂纹
    ax.plot([-a, a], [0, 0], color='red', linewidth=4, zorder=2)
    
    # 添加裂纹标注
    ax.annotate('Initial Crack $2a$', xy=(0, 0.02), xytext=(0, 0.15),
                ha='center', va='bottom', fontsize=12,
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    # 绘制顶部边界条件 (位移控制 v = v_bar, u = 0)
    num_arrows = 7
    x_arrows = np.linspace(-W/2 + 0.05, W/2 - 0.05, num_arrows)
    for x in x_arrows:
        ax.arrow(x, H/2, 0, 0.1, head_width=0.02, head_length=0.03, fc='red', ec='red', lw=2)
    ax.text(0, H/2 + 0.18, r'Displacement Load: $v = \bar{v}, u = 0$', ha='center', va='center', fontsize=14, color='black')
    
    # 绘制底部边界条件 (位移控制 v = -v_bar, u = 0)
    for x in x_arrows:
        ax.arrow(x, -H/2, 0, -0.1, head_width=0.02, head_length=0.03, fc='red', ec='red', lw=2)
    ax.text(0, -H/2 - 0.18, r'Displacement Load: $v = -\bar{v}, u = 0$', ha='center', va='center', fontsize=14, color='black')
    
    # 侧边边界条件 (Traction Free)
    ax.text(-W/2 - 0.05, 0, r'Free Surface: $\sigma_{xx}=0, \tau_{xy}=0$', ha='right', va='center', rotation=90, fontsize=12)
    ax.text(W/2 + 0.05, 0, r'Free Surface: $\sigma_{xx}=0, \tau_{xy}=0$', ha='left', va='center', rotation=-90, fontsize=12)
    
    # 尺寸标注 - 宽度 W
    ax.annotate('', xy=(-W/2, -H/2 - 0.3), xytext=(W/2, -H/2 - 0.3),
                arrowprops=dict(arrowstyle='<->', lw=1.5))
    ax.text(0, -H/2 - 0.35, r'$W$', ha='center', va='top', fontsize=14)
    
    # 尺寸标注 - 高度 H
    ax.annotate('', xy=(W/2 + 0.3, -H/2), xytext=(W/2 + 0.3, H/2),
                arrowprops=dict(arrowstyle='<->', lw=1.5))
    ax.text(W/2 + 0.35, 0, r'$H$', ha='left', va='center', fontsize=14)
    
    # 中心坐标轴
    ax.annotate('', xy=(0, 0), xytext=(0.2, 0), arrowprops=dict(arrowstyle='->', lw=1))
    ax.annotate('', xy=(0, 0), xytext=(0, 0.2), arrowprops=dict(arrowstyle='->', lw=1))
    ax.text(0.22, 0, 'x', fontsize=12, va='center')
    ax.text(0, 0.22, 'y', fontsize=12, ha='center')
    ax.plot(0, 0, 'ko', markersize=4) # 原点
    
    # 设置显示范围和属性
    ax.set_xlim(-W/2 - 0.5, W/2 + 0.5)
    ax.set_ylim(-H/2 - 0.5, H/2 + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.title('Griffith Crack Problem: Physical Model and Boundary Conditions', fontsize=16, fontweight='bold', y=0.95)
    
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'griffith_problem_setup.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Problem setup schematic saved to: {save_path}")

if __name__ == '__main__':
    plot_griffith_problem()
