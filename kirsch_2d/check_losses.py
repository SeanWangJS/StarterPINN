import torch
import os
from network import MixedPINN_LogPolar
from geometry import LogPolarGeometrySampler
from evaluator import KirschPhysicsEvaluator

def print_losses():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 物理参数
    E, nu, sigma_0, a = 1000.0, 0.3, 10.0, 1.0
    R_max = 15.0
    
    model = MixedPINN_LogPolar(in_dim=2, out_dim=5, hidden_layers=5, hidden_neurons=96).to(device)
    model_path = os.path.join(os.path.dirname(__file__), 'kirsch_2d_model.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print("Model file not found!")
        return

    model.eval()
    
    # 采样点评估损失
    sampler = LogPolarGeometrySampler(R_max=R_max, a=a, device=device)
    s_f, t_f = sampler.sample_domain(40000, requires_grad=True)
    bc_dict = sampler.sample_boundaries(2000)
    
    evaluator = KirschPhysicsEvaluator(model, E=E, nu=nu, sigma_0=sigma_0, a=a).to(device)
    
    with torch.set_grad_enabled(True):
        raw = evaluator.compute_raw_losses(s_f, t_f, bc_dict)
        
    print("\n--- Current Loss Components (Dimensionless) ---")
    print(f"PDE Residual Loss: {raw['pde'].item():.6e}")
    print(f"Hole BC Loss (Traction-Free): {raw['hole'].item():.6e}")
    print(f"Symmetry BC Loss: {raw['sym'].item():.6e}")
    print(f"Far Field BC Loss: {raw['far'].item():.6e}")
    print(f"Total Unweighted Loss: {sum(v.item() for v in raw.values()):.6e}")

if __name__ == '__main__':
    print_losses()
