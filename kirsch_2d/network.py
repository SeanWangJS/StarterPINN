import torch
import torch.nn as nn

class MixedPINN_LogPolar(nn.Module):
    """
    Mixed-PINN in Log-Polar Coordinates.
    Inputs:
      s: ln(r/a), where a is the hole radius.
      theta: angle in [0, pi/2].
    Outputs:
      Normalized physical quantities:
      [u_r, u_theta, sigma_rr, sigma_theta_theta, tau_r_theta]
    """

    def __init__(
        self,
        in_dim=2,
        out_dim=5,
        hidden_layers=5,
        hidden_neurons=96,
    ):
        super().__init__()

        layers = []
        layers.append(nn.Linear(in_dim, hidden_neurons))
        layers.append(nn.Tanh())

        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_neurons, out_dim))
        self.net = nn.Sequential(*layers)

        # Initialize the last layer weights to be small to start from a near-zero state
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, s, theta):
        inputs = torch.cat([s, theta], dim=1)
        out = self.net(inputs)
        
        u_r_pred = out[:, 0:1]
        u_theta_pred = out[:, 1:2]
        sigma_rr_pred = out[:, 2:3]
        sigma_tt_pred = out[:, 3:4]
        tau_rt_pred = out[:, 4:5]

        return u_r_pred, u_theta_pred, sigma_rr_pred, sigma_tt_pred, tau_rt_pred
