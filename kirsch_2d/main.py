import torch
import numpy as np
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from geometry import LogPolarGeometrySampler
from network import MixedPINN_LogPolar
from evaluator import KirschPhysicsEvaluator

# --- 训练配置 ---
TRAIN_CONFIG = {
    'domain_points': 40000,
    'bc_points': 2000,
    'mini_batch_size': 8192,
    # Adam
    'adam_steps': 5000,
    'adam_lr': 1e-3,
    # L-BFGS
    'use_lbfgs': True,
    'lbfgs_steps': 1000,
}

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

def main():
    cfg = TRAIN_CONFIG
    device = setup_device()
    print(f"Using device: {device}")

    # R_max 设为 15，因为如果我们在 [-10, 10] 的正方形内绘制结果，最远距离是 10*sqrt(2) ≈ 14.14
    R_max, a = 15.0, 1.0
    E, nu, sigma_0 = 1000.0, 0.3, 10.0

    print("1. Initializing Mixed-PINN modules...")
    sampler = LogPolarGeometrySampler(R_max=R_max, a=a, device=device)
    s_pool, t_pool = sampler.sample_domain(cfg['domain_points'], requires_grad=False)
    bc_dict = sampler.sample_boundaries(cfg['bc_points'])

    model = MixedPINN_LogPolar(
        in_dim=2, out_dim=5, hidden_layers=5, hidden_neurons=96
    ).to(device)

    # 权重平衡：由于全是一阶微分，收敛比纯位移法容易很多。可以先采用全 1.0 权重。
    evaluator = KirschPhysicsEvaluator(
        model, E=E, nu=nu, sigma_0=sigma_0, a=a,
        w_pde=1.0,
        w_hole=2.0,   # 稍微强化孔边
        w_far=1.0,
        w_sym=2.0,
    ).to(device)

    writer = SummaryWriter(log_dir='runs/kirsch_2d_mixed_pinn')
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=cfg['adam_lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_adam, T_max=cfg['adam_steps'], eta_min=1e-5
    )

    max_steps_adam = cfg['adam_steps']
    print("\n2. Starting Phase 1: Adam optimization (Mixed-PINN)...")
    pbar_adam = tqdm(range(max_steps_adam), desc="Adam Phase")
    for step in pbar_adam:
        if step > 0 and step % 500 == 0:
            # 偶尔重采样边界
            bc_dict = sampler.sample_boundaries(cfg['bc_points'])

        s_f, t_f = sample_mini_batch(s_pool, t_pool, cfg['mini_batch_size'])

        optimizer_adam.zero_grad(set_to_none=True)
        loss, raw = evaluator.compute_total_loss(s_f, t_f, bc_dict, sync_metrics=False)
        loss.backward()
        
        # 截断梯度，防止由于 tanh 初期的过激导致爆炸
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

    if cfg['use_lbfgs']:
        print("\n3. Starting Phase 2: L-BFGS optimization...")
        # L-BFGS 使用更多点进行全量优化
        s_f, t_f = sample_mini_batch(s_pool, t_pool, min(cfg['mini_batch_size'] * 2, s_pool.shape[0]))
        max_iter_lbfgs = cfg['lbfgs_steps']
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
            loss, raw = evaluator.compute_total_loss(s_f, t_f, bc_dict, sync_metrics=False)
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
    save_dir = os.path.dirname(__file__)
    model_save_path = os.path.join(save_dir, 'kirsch_2d_model.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    print("\n5. Running plot generation...")
    # Import inside to avoid circular deps or polluting namespace
    from plot_all_fields_2d import plot_all_fields
    plot_all_fields()

if __name__ == '__main__':
    main()
