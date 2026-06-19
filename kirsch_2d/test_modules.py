import torch
from network import PINN_2D_MLP
from evaluator import KirschPhysicsEvaluator
from geometry import KirschGeometrySampler

def test_modules():
    # 1. 几何采样
    sampler = KirschGeometrySampler(W=10, H=10, a=1)
    x_f, y_f = sampler.sample_domain(100)
    bc_dict = sampler.sample_boundaries(20)
    
    # 2. 实例化网络与评估器
    model = PINN_2D_MLP(in_dim=2, out_dim=2, hidden_layers=3, hidden_neurons=20)
    evaluator = KirschPhysicsEvaluator(model, E=1e5, nu=0.3, sigma_0=100, w_hole=50.0)
    
    # 3. 前向计算 Loss
    loss, raw_losses = evaluator.compute_total_loss(x_f, y_f, bc_dict, a=1.0, sync_metrics=True)
    print("Initial Total Loss:", loss.item())
    print("Raw Losses:", raw_losses)
    
    # 4. 反向传播测试
    loss.backward()
    print("Backward pass successful! Architecture and Autograd graph are verified.")

if __name__ == '__main__':
    test_modules()
