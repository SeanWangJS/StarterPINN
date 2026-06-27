import torch
import numpy as np

class LogPolarGeometrySampler:
    """
    Log-Polar Geometry Sampler for Kirsch Problem (1/4 symmetry domain).
    Domain: s in [0, s_max], theta in [0, pi/2]
    where s = ln(r/a).
    """

    def __init__(self, R_max: float, a: float, device='cpu'):
        self.R_max = R_max
        self.a = a
        self.s_max = np.log(R_max / a)
        self.device = device

    def sample_domain(self, n_f: int, requires_grad: bool = True):
        # Uniform sampling in s and theta
        s = torch.rand(n_f, 1, device=self.device) * self.s_max
        theta = torch.rand(n_f, 1, device=self.device) * (np.pi / 2.0)

        if requires_grad:
            s = s.requires_grad_(True)
            theta = theta.requires_grad_(True)
        return s, theta

    def sample_boundaries(self, n_bc: int, requires_grad: bool = True):
        # Hole: s = 0, theta in [0, pi/2]
        theta_hole = torch.rand(n_bc, 1, device=self.device) * (np.pi / 2.0)
        s_hole = torch.zeros_like(theta_hole)

        # Far-field: s = s_max, theta in [0, pi/2]
        theta_far = torch.rand(n_bc, 1, device=self.device) * (np.pi / 2.0)
        s_far = torch.full_like(theta_far, self.s_max)

        # Symmetry x-axis: theta = 0, s in [0, s_max]
        s_sym_x = torch.rand(n_bc, 1, device=self.device) * self.s_max
        theta_sym_x = torch.zeros_like(s_sym_x)

        # Symmetry y-axis: theta = pi/2, s in [0, s_max]
        s_sym_y = torch.rand(n_bc, 1, device=self.device) * self.s_max
        theta_sym_y = torch.full_like(s_sym_y, np.pi / 2.0)

        def maybe_grad(t):
            return t.requires_grad_(True) if requires_grad else t

        bc_dict = {
            'hole': (maybe_grad(s_hole), maybe_grad(theta_hole)),
            'far_field': (maybe_grad(s_far), maybe_grad(theta_far)),
            'sym_x': (maybe_grad(s_sym_x), maybe_grad(theta_sym_x)),
            'sym_y': (maybe_grad(s_sym_y), maybe_grad(theta_sym_y)),
        }
        return bc_dict
