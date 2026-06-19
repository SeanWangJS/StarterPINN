import torch
import torch.nn as nn

class PINN_MLP(nn.Module):
    """
    神经网络拟合器模块
    负责全局逼近位移场 u(x)
    """
    def __init__(self, in_dim=1, out_dim=1, hidden_layers=3, hidden_neurons=20):
        super().__init__()
        layers = []
        
        # 输入层
        layers.append(nn.Linear(in_dim, hidden_neurons))
        layers.append(nn.Tanh())  # 必须使用二阶可导激活函数
        
        # 隐藏层
        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            layers.append(nn.Tanh())
            
        # 输出层 (预测位移 u_pred)
        layers.append(nn.Linear(hidden_neurons, out_dim))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)
