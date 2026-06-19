import torch
import matplotlib.pyplot as plt
import os

from sampler import GeometrySampler
from network import PINN_MLP
from evaluator import PhysicsEvaluator

def main():
    # 1. 初始化物理参数 (尝试设置 E 较大，以验证 loss 缩放的有效性)
    E = 1e5
    L = 1.0
    sigma_0 = 100.0
    N_f = 100
    
    # 2. 实例化各个核心模块
    print("Initializing PINN modules...")
    sampler = GeometrySampler(L=L, N_f=N_f)
    x_f, x_bc_left, x_bc_right = sampler.sample()
    
    model = PINN_MLP(in_dim=1, out_dim=1, hidden_layers=3, hidden_neurons=20)
    evaluator = PhysicsEvaluator(model=model, E=E, sigma_0=sigma_0)
    
    # 3. 第一阶段：Adam 训练 (全局搜索)
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=1e-3)
    max_steps_adam = 2000
    
    print("\nStarting Phase 1: Adam optimization...")
    for step in range(max_steps_adam):
        optimizer_adam.zero_grad()
        loss, _, _, _ = evaluator.compute_loss(x_f, x_bc_left, x_bc_right)
        loss.backward()
        optimizer_adam.step()
        
        if step % 500 == 0:
            print(f"Adam Step {step:4d} | Total Loss: {loss.item():.4e}")

    # 4. 第二阶段：L-BFGS 训练 (局部微调)
    print("\nStarting Phase 2: L-BFGS optimization...")
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(), 
        lr=1.0, 
        max_iter=1000, 
        tolerance_grad=1e-7, 
        tolerance_change=1e-9, 
        history_size=50
    )
    
    step_lbfgs = 0
    def closure():
        nonlocal step_lbfgs
        optimizer_lbfgs.zero_grad()
        loss, _, _, _ = evaluator.compute_loss(x_f, x_bc_left, x_bc_right)
        loss.backward()
        if step_lbfgs % 100 == 0:
            print(f"L-BFGS Step {step_lbfgs:4d} | Total Loss: {loss.item():.4e}")
        step_lbfgs += 1
        return loss

    optimizer_lbfgs.step(closure)
    
    # 5. 测试与可视化阶段
    print("\nTraining completed. Evaluating and plotting...")
    model.eval()
    with torch.no_grad():
        # 生成离散测试点
        x_test = torch.linspace(0, L, 200).view(-1, 1)
        
        # PINN 预测值
        u_pred = model(x_test)
        
        # 解析解： u(x) = (\sigma_0 / E) * x
        u_exact = (sigma_0 / E) * x_test
        
        # 计算相对误差
        error = torch.linalg.norm(u_pred - u_exact) / torch.linalg.norm(u_exact)
        print(f"Relative L2 Error: {error.item():.4e}")
        
        # 绘图: 包含原问题可视化和求解结果
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 2]})
        
        # --- 子图 1：原问题可视化 ---
        ax1.set_title('Original Problem Setup: 1D Elastic Rod')
        ax1.set_xlim(-0.2 * L, 1.3 * L)
        ax1.set_ylim(-0.6, 0.6)
        ax1.axis('off')
        
        # 绘制弹性杆
        import matplotlib.patches as patches
        rod_height = 0.2
        rod = patches.Rectangle((0, -rod_height/2), L, rod_height, linewidth=2, edgecolor='black', facecolor='lightgray')
        ax1.add_patch(rod)
        
        # 绘制左侧固定墙 (Dirichlet BC: u=0)
        ax1.plot([0, 0], [-0.4, 0.4], 'k-', linewidth=4)
        for i in range(7):
            y_pos = -0.3 + i * 0.1
            ax1.plot([-0.05, 0], [y_pos-0.05, y_pos], 'k-', linewidth=2)
        ax1.text(-0.08, 0, 'Fixed\n(u=0)', verticalalignment='center', horizontalalignment='right', fontsize=10, fontweight='bold')
            
        # 绘制右侧拉力箭头 (Neumann BC)
        arrow_len = 0.15 * L
        ax1.arrow(L, 0, arrow_len, 0, head_width=0.08, head_length=0.05*L, fc='red', ec='red', linewidth=2)
        ax1.text(L + arrow_len/2, 0.15, f'$\\sigma_0 = {sigma_0}$', color='red', horizontalalignment='center', fontsize=10, fontweight='bold')
        
        # 标注物理属性
        ax1.annotate('', xy=(0, -0.25), xytext=(L, -0.25), arrowprops=dict(arrowstyle='<->', lw=1.5))
        ax1.text(L/2, -0.45, f'Length $L = {L}$\nYoung\'s Modulus $E = {E:.1e}$', horizontalalignment='center', fontsize=10)
        
        # --- 子图 2：求解结果对比 ---
        ax2.plot(x_test.numpy(), u_exact.numpy(), 'b-', label='Exact Solution', linewidth=2)
        ax2.plot(x_test.numpy(), u_pred.numpy(), 'r--', label='PINN Prediction', linewidth=2)
        ax2.set_xlabel('Position x')
        ax2.set_ylabel('Displacement u(x)')
        ax2.set_title('Solution: PINN vs Exact')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        save_path = os.path.join(os.path.dirname(__file__), 'result_1d_static.png')
        plt.savefig(save_path)
        print(f"Result chart saved to {save_path}")

if __name__ == '__main__':
    main()
