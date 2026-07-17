import copy

import torch
import torch.nn as nn


class AdaptiveReLU(nn.Module):
    """
    Learnable-slope ReLU activation (candidate O10, see
    docs/griffith-drm-insights-experiment-plan.md §2.1), from Manav et al.
    2024 (Deep Ritz Method for phase-field fracture), eq. (24):

        z_{k+1} = max(0, m_k * (W_k z_k + b_k))

    where m_k is a single learnable scalar per layer, initialized to a
    constant m0 (the paper uses m0=1 for a small 1D problem and m0=2/3 for
    2D problems; m_k is reported not to change much during training, but its
    initial value matters). The paper found tanh's smooth curvature causes
    Gibbs-like oscillations in autodiff-computed strain fields near
    localized phase fields (§3.2.2/§3.2.3), which plain ReLU alone does not
    fix; the learnable slope is what let their optimizer cross energy
    barriers during training.
    """
    def __init__(self, m0=1.0):
        super().__init__()
        self.m = nn.Parameter(torch.tensor(float(m0)))

    def forward(self, z):
        return torch.relu(self.m * z)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_layers, hidden_neurons, activation=None):
        super().__init__()
        if activation is None:
            activation = nn.Tanh()
        layers = []
        layers.append(nn.Linear(in_dim, hidden_neurons))
        layers.append(copy.deepcopy(activation))

        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            layers.append(copy.deepcopy(activation))

        layers.append(nn.Linear(hidden_neurons, out_dim))
        self.net = nn.Sequential(*layers)

        # Initialize weights (Xavier). Manav et al. 2024 use Xavier
        # initialization throughout, for both tanh and adaptive-ReLU
        # networks (no activation-specific switch to e.g. Kaiming init), so
        # we keep the same scheme regardless of `activation`.
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
    def __init__(self, in_dim=2, out_dim=2, hidden_layers=4, hidden_neurons=64, activation=None):
        super().__init__()
        self.mlp = MLP(in_dim, out_dim, hidden_layers, hidden_neurons, activation=activation)
        
    def forward(self, x_y_tensor):
        return self.mlp(x_y_tensor)

class PhaseFieldNet(nn.Module):
    """
    Predicts phase field phi given (x, y)
    Uses Sigmoid at the end to strictly enforce phi in [0, 1]
    """
    def __init__(self, in_dim=2, out_dim=1, hidden_layers=4, hidden_neurons=64, activation=None):
        super().__init__()
        self.mlp = MLP(in_dim, out_dim, hidden_layers, hidden_neurons, activation=activation)
        
    def forward(self, x_y_tensor):
        raw_phi = self.mlp(x_y_tensor)
        return torch.sigmoid(raw_phi)
