import torch

class GeometrySampler:
    """
    空间采样器模块
    负责在计算域和边界上生成训练所需的数据配点
    """
    def __init__(self, L: float, N_f: int):
        self.L = L
        self.N_f = N_f

    def sample(self):
        # 内部配点，使用均匀采样 (为了演示简单)
        # shape 必须是 [N_f, 1] 以避免广播错误
        x_f = torch.linspace(0, self.L, self.N_f).view(-1, 1)
        # 开启梯度追踪，因为我们需要对内部点求偏导
        x_f.requires_grad_(True)
        
        # 左边界点 x = 0
        x_bc_left = torch.tensor([[0.0]], requires_grad=True)
        
        # 右边界点 x = L
        x_bc_right = torch.tensor([[self.L]], requires_grad=True)
        
        return x_f, x_bc_left, x_bc_right
