import torch
import numpy as np
from network import MixedPINN_LogPolar
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MixedPINN_LogPolar(in_dim=2, out_dim=5, hidden_layers=5, hidden_neurons=96, use_ansatz=True).to(device)
model_path = os.path.join(os.path.dirname(__file__), 'kirsch_2d_model.pth')
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

num_samples = 100
theta_vals = np.linspace(0, np.pi / 2, num_samples)
s_tensor = torch.zeros((num_samples, 1), dtype=torch.float32, device=device)
theta_tensor = torch.tensor(theta_vals, dtype=torch.float32, device=device).view(-1, 1)

with torch.no_grad():
    u_r_n, u_t_n, sig_rr_n, sig_tt_n, tau_rt_n = model(s_tensor, theta_tensor)

pred_u_theta = u_t_n.cpu().numpy().flatten()
exact_u_theta = - 2.0 * np.sin(2.0 * theta_vals)

diff = pred_u_theta - exact_u_theta
print("Max diff in u_theta at r=a:", np.max(np.abs(diff)))
print("Min PINN u_theta:", np.min(pred_u_theta))
print("Min exact u_theta:", np.min(exact_u_theta))
