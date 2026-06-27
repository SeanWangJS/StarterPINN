import torch
import os
import yaml
import argparse
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import from core package
from src.network import MixedPINN_LogPolar
from src.geometry import LogPolarGeometrySampler
from src.evaluator import KirschPhysicsEvaluator
from src.weighting import ConstantWeighting

def main():
    parser = argparse.ArgumentParser(description="Evaluate Mixed-PINN losses")
    parser.add_argument('--config', type=str, default='../configs/default.yaml')
    args = parser.parse_args()

    with open(os.path.join(os.path.dirname(__file__), args.config), 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    phys_cfg = cfg['physics']
    E, nu, sigma_0, a = phys_cfg['E'], phys_cfg['nu'], phys_cfg['sigma_0'], phys_cfg['a']
    R_max = phys_cfg['R_max']
    
    model = MixedPINN_LogPolar(
        in_dim=cfg['model']['in_dim'], 
        out_dim=cfg['model']['out_dim'], 
        hidden_layers=cfg['model']['hidden_layers'], 
        hidden_neurons=cfg['model']['hidden_neurons'],
        use_ansatz=cfg['train'].get('use_ansatz', False)
    ).to(device)
    
    model_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'weights', 'kirsch_2d_model.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print("Model file not found! Evaluating on random initialization.")

    model.eval()
    
    sampler = LogPolarGeometrySampler(R_max=R_max, a=a, device=device)
    s_f, t_f = sampler.sample_domain(cfg['train']['domain_points'], requires_grad=True)
    bc_dict = sampler.sample_boundaries(cfg['train']['bc_points'])
    
    # Use ConstantWeighting by default for simple evaluation
    weighting_strategy = ConstantWeighting(w_pde=1.0, w_hole=1.0, w_sym=1.0, w_far=1.0).to(device)
    evaluator = KirschPhysicsEvaluator(
        model, E=E, nu=nu, sigma_0=sigma_0, a=a, weighting_strategy=weighting_strategy
    ).to(device)
    
    with torch.set_grad_enabled(True):
        raw = evaluator.compute_raw_losses(s_f, t_f, bc_dict)
        
    print("\n--- Current Loss Components (Dimensionless) ---")
    print(f"PDE Residual Loss: {raw.get('pde', torch.tensor(0.0)).item():.6e}")
    print(f"Hole BC Loss (Traction-Free): {raw.get('hole', torch.tensor(0.0)).item():.6e}")
    print(f"Symmetry BC Loss: {raw.get('sym', torch.tensor(0.0)).item():.6e}")
    print(f"Far Field BC Loss: {raw.get('far', torch.tensor(0.0)).item():.6e}")
    print(f"Total Unweighted Loss: {sum(v.item() for v in raw.values()):.6e}")

if __name__ == '__main__':
    main()
