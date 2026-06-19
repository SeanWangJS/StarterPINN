"""
调试脚本：验证训练好的模型在各边界上的满足程度
重点检查：
1. 孔口边界牵引力 Tx, Ty 是否接近 0
2. 远场 sigma_xx 是否接近 sigma_0
3. 各边界的 sigma_xx/yy 沿边界的分布（找应力集中峰值）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import matplotlib.pyplot as plt
from network import PINN_2D_MLP
from evaluator import KirschPhysicsEvaluator

# ── 参数 ─────────────────────────────────────────────────────────────────────
W, H, a = 10.0, 10.0, 1.0
E, nu, sigma_0 = 1000.0, 0.3, 10.0
device = torch.device('cpu')

# ── 加载模型 ─────────────────────────────────────────────────────────────────
model = PINN_2D_MLP(in_dim=2, out_dim=2, hidden_layers=6, hidden_neurons=96, a=a, W=W).to(device)
model_path = os.path.join(os.path.dirname(__file__), 'kirsch_2d_model.pth')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"未找到权重文件: {model_path}")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
evaluator = KirschPhysicsEvaluator(model, E=E, nu=nu, sigma_0=sigma_0).to(device)
print(f"[OK] Model loaded: {model_path}")

# ═════════════════════════════════════════════════════════════════════════════
# 1. 孔口牵引力检查 (最关键)
#    理论值: Tx = Ty = 0 在所有孔边点上
# ═════════════════════════════════════════════════════════════════════════════
N_hole = 360
theta_hole = torch.linspace(0, 2 * np.pi, N_hole + 1)[:-1].view(-1, 1)
x_hole = (a * torch.cos(theta_hole)).requires_grad_(True)
y_hole = (a * torch.sin(theta_hole)).requires_grad_(True)

sxx, syy, txy, _, _ = evaluator.compute_stresses(x_hole, y_hole)
sxx_np = sxx.detach().numpy().flatten()
syy_np = syy.detach().numpy().flatten()
txy_np = txy.detach().numpy().flatten()

# 外法向（指向孔内，即材料侧的法向朝外）
nx = -torch.cos(theta_hole).detach().numpy().flatten()
ny = -torch.sin(theta_hole).detach().numpy().flatten()
Tx = sxx_np * nx + txy_np * ny
Ty = txy_np * nx + syy_np * ny

theta_np = theta_hole.numpy().flatten()
print("\n==== 1. Hole Boundary Traction ====")
print(f"   Tx: mean={Tx.mean():.4f}, max_abs={np.abs(Tx).max():.4f}  (theory=0)")
print(f"   Ty: mean={Ty.mean():.4f}, max_abs={np.abs(Ty).max():.4f}  (theory=0)")
print(f"   sigma_xx at hole: min={sxx_np.min():.3f}, max={sxx_np.max():.3f}  (theory: theta=0 -> -10, theta=90 -> 30)")
print(f"   sigma_xx at hole theta=0 (idx=0): {sxx_np[0]:.3f}  (theory=-10)")
print(f"   sigma_xx at hole theta=90 (idx={N_hole//4}): {sxx_np[N_hole//4]:.3f}  (theory=+30)")

# ═════════════════════════════════════════════════════════════════════════════
# 2. 远场边界检查
# ═════════════════════════════════════════════════════════════════════════════
N_far = 200

# 右侧边界 x=W
y_right = torch.linspace(-H, H, N_far).view(-1, 1).requires_grad_(True)
x_right = torch.full_like(y_right, W).requires_grad_(True)
sxx_r, syy_r, txy_r, _, _ = evaluator.compute_stresses(x_right, y_right)
sxx_r = sxx_r.detach().numpy().flatten()
txy_r = txy_r.detach().numpy().flatten()

# 上侧边界 y=H
x_top = torch.linspace(-W, W, N_far).view(-1, 1).requires_grad_(True)
y_top = torch.full_like(x_top, H).requires_grad_(True)
sxx_t, syy_t, txy_t, _, _ = evaluator.compute_stresses(x_top, y_top)
syy_t = syy_t.detach().numpy().flatten()
txy_t = txy_t.detach().numpy().flatten()

print("\n==== 2. Far-field Boundaries ====")
print(f"   Right (x={W}) sigma_xx: mean={sxx_r.mean():.3f}  (theory={sigma_0}), max_err={np.abs(sxx_r - sigma_0).max():.3f}")
print(f"   Right (x={W}) tau_xy: mean_abs={np.abs(txy_r).mean():.4f}  (theory=0)")
print(f"   Top (y={H}) sigma_yy: mean={syy_t.mean():.3f}  (theory=0), max_abs={np.abs(syy_t).max():.3f}")
print(f"   Top (y={H}) tau_xy: mean_abs={np.abs(txy_t).mean():.4f}  (theory=0)")

# ═════════════════════════════════════════════════════════════════════════════
# 3. 用精确解对比孔边 sigma_xx 的理论值
#    孔边 (a*cos θ, a*sin θ) 处:
#    σ_θθ = σ₀(1 - 2cos2θ)  →  σ_xx = σ_θθ·sin²θ + σ_rr·cos²θ - 2τ_rθ·sinθcosθ
#    在 r=a: σ_rr=0, τ_rθ=0  →  σ_xx_hole = σ_θθ·sin²θ = σ₀(1-2cos2θ)sin²θ
#    简洁地看关键点:
#    θ=0  → σ_θθ = -σ₀ = -10
#    θ=90 → σ_θθ = 3σ₀ = +30
# ═════════════════════════════════════════════════════════════════════════════
sigma_tt_theory = sigma_0 * (1 - 2 * np.cos(2 * theta_np))

# ═════════════════════════════════════════════════════════════════════════════
# 4. 绘图
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Kirsch PINN Boundary Diagnosis', fontsize=15, fontweight='bold')

deg = np.degrees(theta_np)

# (0,0) 孔边牵引力
ax = axes[0, 0]
ax.plot(deg, Tx, label='Tx (PINN)', color='royalblue')
ax.plot(deg, Ty, label='Ty (PINN)', color='tomato')
ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
ax.set_title('Hole Traction (should be = 0)')
ax.set_xlabel('θ (degrees)')
ax.legend(); ax.grid(True, alpha=0.3)

# (0,1) 孔边 σ_xx PINN vs 理论
ax = axes[0, 1]
ax.plot(deg, sxx_np, label='σ_xx PINN', color='royalblue')
ax.plot(deg, sigma_tt_theory, '--', label='σ_θθ Theory', color='k', linewidth=2)
ax.set_title('σ_xx on Hole Boundary')
ax.set_xlabel('θ (degrees)')
ax.legend(); ax.grid(True, alpha=0.3)

# (0,2) 孔边 σ_yy PINN vs 理论 σ_rr (=0 在孔边)
ax = axes[0, 2]
ax.plot(deg, syy_np, label='σ_yy PINN', color='royalblue')
ax.axhline(0, color='k', linestyle='--', linewidth=2, label='σ_rr Theory = 0')
ax.set_title('σ_yy on Hole Boundary (should match σ_rr=0)')
ax.set_xlabel('θ (degrees)')
ax.legend(); ax.grid(True, alpha=0.3)

# (1,0) 右边界 σ_xx
ax = axes[1, 0]
y_np = torch.linspace(-H, H, N_far).numpy()
ax.plot(y_np, sxx_r, label='σ_xx PINN')
ax.axhline(sigma_0, color='r', linestyle='--', label=f'Theory = {sigma_0}')
ax.set_title(f'Right Boundary (x={W}): σ_xx')
ax.set_xlabel('y'); ax.legend(); ax.grid(True, alpha=0.3)

# (1,1) 右边界 τ_xy
ax = axes[1, 1]
ax.plot(y_np, txy_r, label='τ_xy PINN')
ax.axhline(0, color='r', linestyle='--', label='Theory = 0')
ax.set_title(f'Right Boundary (x={W}): τ_xy')
ax.set_xlabel('y'); ax.legend(); ax.grid(True, alpha=0.3)

# (1,2) 上边界 σ_yy
ax = axes[1, 2]
x_np = torch.linspace(-W, W, N_far).numpy()
ax.plot(x_np, syy_t, label='σ_yy PINN')
ax.axhline(0, color='r', linestyle='--', label='Theory = 0')
ax.set_title(f'Top Boundary (y={H}): σ_yy')
ax.set_xlabel('x'); ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), 'debug_bc_diagnosis.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n[OK] Diagnostic figure saved: {out_path}")
