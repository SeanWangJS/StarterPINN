import torch
import numpy as np
from network import MixedPINN_LogPolar
from evaluator import KirschPhysicsEvaluator

class DummyModel(torch.nn.Module):
    def __init__(self, E, nu, sigma_0, a):
        super().__init__()
        self.E = E
        self.nu = nu
        self.sigma_0 = sigma_0
        self.a = a
        self.scale_u = sigma_0 * a / E

    def forward(self, s, theta):
        r = self.a * torch.exp(s)
        a2_r2 = (self.a / r)**2
        a4_r4 = a2_r2**2
        cos2t = torch.cos(2 * theta)
        sin2t = torch.sin(2 * theta)
        cost = torch.cos(theta)
        sint = torch.sin(theta)

        # Normalized exact stresses
        sig_rr = 0.5 * (1 - a2_r2) + 0.5 * (1 + 3 * a4_r4 - 4 * a2_r2) * cos2t
        sig_tt = 0.5 * (1 + a2_r2) - 0.5 * (1 + 3 * a4_r4) * cos2t
        tau_rt = -0.5 * (1 - 3 * a4_r4 + 2 * a2_r2) * sin2t

        # Exact displacements (polar)
        G = self.E / (2 * (1 + self.nu))
        # Plane stress kappa
        kappa = (3 - self.nu) / (1 + self.nu)

        # Exact polar displacements (not normalized yet)
        u_r_exact = (self.sigma_0 / (4 * G)) * ( r * ((kappa - 1) / 2 + cos2t) + (self.a**2 / r) * (1 + (kappa + 1) * cos2t) - (self.a**4 / r**3) * cos2t )
        u_t_exact = (self.sigma_0 / (4 * G)) * ( -r * sin2t - (self.a**2 / r) * (kappa - 1) * sin2t - (self.a**4 / r**3) * sin2t )

        # Normalize displacements
        u_r = u_r_exact / self.scale_u
        u_theta = u_t_exact / self.scale_u

        return u_r, u_theta, sig_rr, sig_tt, tau_rt

def test_pde():
    E, nu, sigma_0, a = 1000.0, 0.3, 10.0, 1.0
    model = DummyModel(E, nu, sigma_0, a)
    evaluator = KirschPhysicsEvaluator(model, E=E, nu=nu, sigma_0=sigma_0, a=a)
    
    s = torch.linspace(0.1, 2.0, 100, requires_grad=True).view(-1, 1)
    theta = torch.linspace(0.1, 1.5, 100, requires_grad=True).view(-1, 1)
    
    loss_pde = evaluator.pde_residual(s, theta)
    print(f"Exact Solution PDE Loss: {loss_pde.item():.2e}")
    if loss_pde.item() < 1e-4:
        print("PDE implementation is CORRECT!")
    else:
        print("PDE implementation has an ERROR!")

if __name__ == "__main__":
    test_pde()
