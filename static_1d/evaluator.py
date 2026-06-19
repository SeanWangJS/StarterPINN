import torch

class PhysicsEvaluator:
    """
    物理损失评估器模块
    负责计算导数并组装由 PDE 和 BCs 构成的全局 Loss
    """
    def __init__(self, model, E: float, sigma_0: float):
        self.model = model
        self.E = E
        self.sigma_0 = sigma_0

    def compute_loss(self, x_f, x_bc_left, x_bc_right):
        # === 1. PDE Loss (E * d2u_dx2 = 0) ===
        u_pred = self.model(x_f)
        
        # 计算一阶导数
        du_dx = torch.autograd.grad(
            u_pred, x_f, 
            grad_outputs=torch.ones_like(u_pred), 
            create_graph=True
        )[0]
        
        # 计算二阶导数
        d2u_dx2 = torch.autograd.grad(
            du_dx, x_f, 
            grad_outputs=torch.ones_like(du_dx), 
            create_graph=True
        )[0]
        
        loss_pde = torch.mean((self.E * d2u_dx2)**2)

        # === 2. Dirichlet BC Loss: u(0) = 0 ===
        u_left = self.model(x_bc_left)
        loss_bc_left = torch.mean((u_left - 0.0)**2)

        # === 3. Neumann BC Loss: E * u'(L) = sigma_0 ===
        u_right = self.model(x_bc_right)
        du_dx_right = torch.autograd.grad(
            u_right, x_bc_right, 
            grad_outputs=torch.ones_like(u_right), 
            create_graph=True
        )[0]
        loss_bc_right = torch.mean((self.E * du_dx_right - self.sigma_0)**2)

        # === 量纲平衡（无量纲化缩放）===
        # 因为 E 可能很大（比如 1e5 或更大），会导致 loss_pde 和 loss_bc_right 的数值远超 loss_bc_left
        # 这里进行简单的缩放处理，防止梯度爆炸或淹没位移边界条件
        scale_factor = 1.0 / (self.E ** 2)
        loss_pde_scaled = loss_pde * scale_factor
        loss_bc_right_scaled = loss_bc_right * scale_factor
        
        # 全局 Loss = PDE Loss + BCs Loss
        loss_total = loss_pde_scaled + loss_bc_left + loss_bc_right_scaled
        
        return loss_total, loss_pde.item(), loss_bc_left.item(), loss_bc_right.item()
