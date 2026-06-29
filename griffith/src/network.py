import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_layers, hidden_neurons, activation=nn.Tanh()):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, hidden_neurons))
        layers.append(activation)
        
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            layers.append(activation)
            
        layers.append(nn.Linear(hidden_neurons, out_dim))
        self.net = nn.Sequential(*layers)
        
        # Initialize weights (Xavier for Tanh)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        return self.net(x)

class DisplacementNet(nn.Module):
    """
    Predicts u and v displacements given (x, y)
    """
    def __init__(self, in_dim=2, out_dim=2, hidden_layers=4, hidden_neurons=64):
        super().__init__()
        self.mlp = MLP(in_dim, out_dim, hidden_layers, hidden_neurons, activation=nn.Tanh())
        
    def forward(self, x_y_tensor):
        return self.mlp(x_y_tensor)

class PhaseFieldNet(nn.Module):
    """
    Predicts phase field phi given (x, y)
    Uses Sigmoid at the end to strictly enforce phi in [0, 1]
    """
    def __init__(self, in_dim=2, out_dim=1, hidden_layers=4, hidden_neurons=64):
        super().__init__()
        self.mlp = MLP(in_dim, out_dim, hidden_layers, hidden_neurons, activation=nn.Tanh())
        
    def forward(self, x_y_tensor):
        raw_phi = self.mlp(x_y_tensor)
        return torch.sigmoid(raw_phi)
