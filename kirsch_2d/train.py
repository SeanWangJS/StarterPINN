import torch
import numpy as np
import os
import datetime
import yaml
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse

# Import from the refactored src package
from src.geometry import LogPolarGeometrySampler
from src.network import MixedPINN_LogPolar
from src.evaluator import KirschPhysicsEvaluator
from src.weighting import ConstantWeighting, LRAWeighting, SoftAdaptWeighting

def setup_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
    return device

def sample_mini_batch(s_pool, t_pool, batch_size):
    idx = torch.randint(0, s_pool.shape[0], (batch_size,), device=s_pool.device)
    s = s_pool[idx].clone().requires_grad_(True)
    t = t_pool[idx].clone().requires_grad_(True)
    return s, t

def parse_args():
    parser = argparse.ArgumentParser(description="Train Mixed-PINN for Kirsch problem")
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to configuration file')
    return parser.parse_args()

def main():
    args = parse_args()
    
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        
    train_cfg = cfg['train']
    model_cfg = cfg['model']
    phys_cfg = cfg['physics']
    w_cfg = cfg['weighting']

    device = setup_device()
    print(f"Using device: {device}")

    print("1. Initializing Mixed-PINN modules...")
    sampler = LogPolarGeometrySampler(R_max=phys_cfg['R_max'], a=phys_cfg['a'], device=device)
    s_pool, t_pool = sampler.sample_domain(train_cfg['domain_points'], requires_grad=False)
    bc_dict = sampler.sample_boundaries(train_cfg['bc_points'])

    model = MixedPINN_LogPolar(
        in_dim=model_cfg['in_dim'], 
        out_dim=model_cfg['out_dim'], 
        hidden_layers=model_cfg['hidden_layers'], 
        hidden_neurons=model_cfg['hidden_neurons'],
        use_ansatz=train_cfg.get('use_ansatz', False)
    ).to(device)

    use_ansatz = train_cfg.get('use_ansatz', False)
    target_keys = ['pde', 'far'] if use_ansatz else ['pde', 'hole', 'sym', 'far']
    
    strategy = w_cfg.get('strategy', 'constant')
    if strategy == 'constant':
        weighting_strategy = ConstantWeighting(
            w_pde=w_cfg['w_pde'], 
            w_hole=0.0 if use_ansatz else w_cfg['w_hole'], 
            w_far=w_cfg['w_far'], 
            w_sym=0.0 if use_ansatz else w_cfg['w_sym']
        ).to(device)
    elif strategy == 'lra':
        weighting_strategy = LRAWeighting(
            model=model, alpha=0.9, update_freq=10, target_keys=target_keys
        ).to(device)
    elif strategy == 'softadapt':
        weighting_strategy = SoftAdaptWeighting(
            beta=0.1, target_keys=target_keys
        ).to(device)
    else:
        raise ValueError(f"Unknown weighting strategy: {strategy}")

    evaluator = KirschPhysicsEvaluator(
        model, 
        E=phys_cfg['E'], 
        nu=phys_cfg['nu'], 
        sigma_0=phys_cfg['sigma_0'], 
        a=phys_cfg['a'],
        weighting_strategy=weighting_strategy
    ).to(device)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join('results', 'runs', f'kirsch_2d_mixed_pinn_{timestamp}')
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")
    
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=train_cfg['adam_lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_adam, T_max=train_cfg['adam_steps'], eta_min=1e-5
    )

    max_steps_adam = train_cfg['adam_steps']
    print("\n2. Starting Phase 1: Adam optimization (Mixed-PINN)...")
    pbar_adam = tqdm(range(max_steps_adam), desc="Adam Phase")
    for step in pbar_adam:
        if step > 0 and step % 500 == 0:
            bc_dict = sampler.sample_boundaries(train_cfg['bc_points'])

        s_f, t_f = sample_mini_batch(s_pool, t_pool, train_cfg['mini_batch_size'])

        optimizer_adam.zero_grad(set_to_none=True)
        loss, raw = evaluator.compute_total_loss(s_f, t_f, bc_dict, step=step, sync_metrics=False)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer_adam.step()
        scheduler.step()

        if step % 50 == 0:
            raw = {k: v.item() for k, v in raw.items()}
            desc = f"Total={raw['total']:.2e}, PDE={raw['pde']:.2e}, Hole={raw['hole']:.2e}, Sym={raw['sym']:.2e}, Far={raw['far']:.2e}"
            pbar_adam.set_postfix_str(desc)
            
            writer.add_scalar('Loss/Total', raw['total'], step)
            writer.add_scalar('Loss/PDE', raw['pde'], step)
            writer.add_scalar('Loss/Hole', raw['hole'], step)
            writer.add_scalar('Loss/Sym', raw['sym'], step)
            writer.add_scalar('Loss/Far', raw['far'], step)
            writer.add_scalar('Train/LR', scheduler.get_last_lr()[0], step)

    if train_cfg['use_lbfgs']:
        print("\n3. Starting Phase 2: L-BFGS optimization...")
        s_f, t_f = sample_mini_batch(s_pool, t_pool, min(train_cfg['mini_batch_size'] * 2, s_pool.shape[0]))
        max_iter_lbfgs = train_cfg['lbfgs_steps']
        optimizer_lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=max_iter_lbfgs,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            history_size=50,
            line_search_fn='strong_wolfe',
        )

        step_lbfgs = 0
        pbar_lbfgs = tqdm(total=max_iter_lbfgs, desc="L-BFGS Phase")

        def closure():
            nonlocal step_lbfgs
            optimizer_lbfgs.zero_grad()
            loss, raw = evaluator.compute_total_loss(s_f, t_f, bc_dict, step=max_steps_adam + step_lbfgs, sync_metrics=False)
            loss.backward()
            if step_lbfgs < max_iter_lbfgs and step_lbfgs % 20 == 0:
                metrics = {k: v.item() for k, v in raw.items()}
                pbar_lbfgs.set_postfix({'Total': f"{metrics['total']:.2e}", 'Hole': f"{metrics['hole']:.2e}"})
            if step_lbfgs < max_iter_lbfgs:
                pbar_lbfgs.update(1)
            step_lbfgs += 1
            return loss

        optimizer_lbfgs.step(closure)
        pbar_lbfgs.close()

    writer.close()

    print("\n4. Training completed.")
    save_dir = os.path.join(os.path.dirname(__file__), 'results', 'weights')
    os.makedirs(save_dir, exist_ok=True)
    model_save_path_timestamp = os.path.join(save_dir, f'kirsch_2d_model_{timestamp}.pth')
    model_save_path_latest = os.path.join(save_dir, 'kirsch_2d_model.pth')
    
    torch.save(model.state_dict(), model_save_path_timestamp)
    torch.save(model.state_dict(), model_save_path_latest)
    print(f"Model saved to {model_save_path_timestamp} and updated {model_save_path_latest}")

if __name__ == '__main__':
    main()
