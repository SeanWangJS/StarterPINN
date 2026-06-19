import matplotlib.pyplot as plt
import numpy as np

def plot_kirsch_problem():
    fig, ax = plt.subplots(figsize=(8, 8))
    
    W, H = 10, 10
    a = 2.5 # 示意图中适当放大孔径以便于观察
    
    # 1. 绘制带有圆孔的 1/4 平板区域
    theta = np.linspace(0, np.pi/2, 100)
    x_hole = a * np.cos(theta)
    y_hole = a * np.sin(theta)
    
    # 拼接边界形成多边形
    x_bound = np.concatenate([[W], [W], [0], x_hole[::-1]])
    y_bound = np.concatenate([[0], [H], [H], y_hole[::-1]])
    
    # 填充浅蓝色作为计算域
    ax.fill(x_bound, y_bound, color='lightsteelblue', alpha=0.5, edgecolor='black', linewidth=2)
    
    # 2. 绘制对称轴的滚轴支撑 (Symmetry BCs)
    # 左边界 (x=0) 滚轴
    for y_pos in np.linspace(a + 0.5, H - 0.5, 6):
        ax.plot([0, -0.3, 0], [y_pos+0.3, y_pos, y_pos-0.3], 'k-', lw=1.5) # 三角形
        ax.plot([0, 0], [y_pos-0.4, y_pos+0.4], 'k-', lw=2) # 挡板
        
    # 下边界 (y=0) 滚轴
    for x_pos in np.linspace(a + 0.5, W - 0.5, 6):
        ax.plot([x_pos-0.3, x_pos, x_pos+0.3], [0, -0.3, 0], 'k-', lw=1.5)
        ax.plot([x_pos-0.4, x_pos+0.4], [0, 0], 'k-', lw=2)
        
    # 3. 绘制右边界的远场拉伸载荷 (Tension)
    for y_pos in np.linspace(1, H-1, 6):
        ax.arrow(W, y_pos, 1.5, 0, head_width=0.4, head_length=0.5, fc='red', ec='red', lw=2)
        
    # 标注拉应力 sigma_0
    ax.text(W + 2.5, H/2, r'$\sigma_0$', color='red', fontsize=18, va='center', ha='left')
    
    # 4. 标注各类边界条件
    # 上边界自由
    ax.text(W/2, H + 0.5, r'Free Surface: $\sigma_{yy}=0, \tau_{xy}=0$', ha='center', fontsize=12)
    
    # 右边界
    ax.text(W + 0.5, H/2 + 2, r'$\tau_{xy}=0$', ha='left', fontsize=12)
    
    # 孔边自由
    ax.text(a/2 - 0.2, a/2 - 0.2, r'Traction Free', ha='right', va='top', fontsize=12, rotation=45)
    # 箭头指向孔边
    ax.annotate('', xy=(a*np.cos(np.pi/4), a*np.sin(np.pi/4)), 
                xytext=((a-1)*np.cos(np.pi/4), (a-1)*np.sin(np.pi/4)),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
                
    # 对称边界标注
    ax.text(-0.8, (H+a)/2, r'Symmetry: $u=0, \tau_{xy}=0$', va='center', ha='right', rotation=90, fontsize=12)
    ax.text((W+a)/2, -0.8, r'Symmetry: $v=0, \tau_{xy}=0$', ha='center', va='top', fontsize=12)

    # 5. 格式化图表
    ax.set_aspect('equal')
    ax.set_xlim(-2, W + 4)
    ax.set_ylim(-2, H + 2)
    ax.axis('off') # 隐藏自带坐标轴
    
    plt.title('2D Kirsch Problem: 1/4 Plate Physical Model', fontsize=16, y=1.05, fontweight='bold')
    
    save_path = 'problem_2d_kirsch.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Problem schematic saved to {save_path}")

if __name__ == '__main__':
    plot_kirsch_problem()
