import matplotlib.pyplot as plt
from geometry import KirschGeometrySampler

def test_geometry():
    # 假设板宽 W=10, 高 H=10, 孔半径 a=1
    W, H, a = 10.0, 10.0, 1.0
    sampler = KirschGeometrySampler(W=W, H=H, a=a)
    
    # 在域内采样 2000 个配点，孔口附近密集采样 800 个点，每条边界采样 200 个点
    x_f, y_f = sampler.sample_domain(2000)
    x_dense, y_dense = sampler.sample_dense_near_hole(800, max_radius=3.0)
    bc_dict = sampler.sample_boundaries(200)
    
    # 开始绘图以验证
    plt.figure(figsize=(8, 8))
    
    # 画出内部配点
    plt.scatter(x_f.cpu().detach().numpy(), y_f.cpu().detach().numpy(), s=1, c='lightgray', label='Domain Points')
    
    # 画出孔口附近的密集配点
    plt.scatter(x_dense.cpu().detach().numpy(), y_dense.cpu().detach().numpy(), s=3, c='cyan', label='Dense Near Hole')
    
    # 画出五条边界的配点
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    labels = ['Left BC (x=0)', 'Bottom BC (y=0)', 'Right BC (x=W)', 'Top BC (y=H)', 'Hole BC (r=a)']
    
    for (name, (x_bc, y_bc)), color, label in zip(bc_dict.items(), colors, labels):
        plt.scatter(x_bc.cpu().detach().numpy(), y_bc.cpu().detach().numpy(), s=15, c=color, label=label)
        
    plt.xlim(-0.5, W + 0.5)
    plt.ylim(-0.5, H + 0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Kirsch Problem 2D Geometry Sampling (1/4 Plate)')
    plt.legend()
    plt.grid(True)
    
    save_path = 'geometry_samples.png'
    plt.savefig(save_path, dpi=150)
    print(f"Geometry sampled points saved to {save_path}")

if __name__ == '__main__':
    test_geometry()
