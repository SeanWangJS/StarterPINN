import torch
import torch.nn as nn
import numpy as np

class KirschPhysicsEvaluator(nn.Module):
    """
    Mixed-PINN Evaluator for Kirsch Problem in Log-Polar Coordinates.
    """

    def __init__(
        self,
        model,
        E: float,
        nu: float,
        sigma_0: float,
        a: float = 1.0,
        weighting_strategy = None,
    ):
        super().__init__()
        self.model = model
        self.E = E
        self.nu = nu
        self.sigma_0 = sigma_0
        self.a = a
        self.weighting_strategy = weighting_strategy

    def pde_residual(self, s, theta):
        """Calculate the 5 first-order PDE residuals."""
        u_r, u_theta, sig_rr, sig_tt, tau_rt = self.model(s, theta)

        # Gradients wrt s
        du_r_ds = torch.autograd.grad(u_r, s, grad_outputs=torch.ones_like(u_r), create_graph=True)[0]
        du_theta_ds = torch.autograd.grad(u_theta, s, grad_outputs=torch.ones_like(u_theta), create_graph=True)[0]
        dsig_rr_ds = torch.autograd.grad(sig_rr, s, grad_outputs=torch.ones_like(sig_rr), create_graph=True)[0]
        dtau_rt_ds = torch.autograd.grad(tau_rt, s, grad_outputs=torch.ones_like(tau_rt), create_graph=True)[0]

        # Gradients wrt theta
        du_r_dt = torch.autograd.grad(u_r, theta, grad_outputs=torch.ones_like(u_r), create_graph=True)[0]
        du_theta_dt = torch.autograd.grad(u_theta, theta, grad_outputs=torch.ones_like(u_theta), create_graph=True)[0]
        dsig_tt_dt = torch.autograd.grad(sig_tt, theta, grad_outputs=torch.ones_like(sig_tt), create_graph=True)[0]
        dtau_rt_dt = torch.autograd.grad(tau_rt, theta, grad_outputs=torch.ones_like(tau_rt), create_graph=True)[0]

        # 1. Equilibrium in r
        eq_r = dsig_rr_ds + dtau_rt_dt + sig_rr - sig_tt
        
        # 2. Equilibrium in theta
        eq_t = dtau_rt_ds + dsig_tt_dt + 2 * tau_rt

        # Exponential factor for constitutive equations
        exp_s = torch.exp(s)

        # 3. Constitutive rr
        const_rr = exp_s * (1 - self.nu**2) * sig_rr - (du_r_ds + self.nu * (du_theta_dt + u_r))

        # 4. Constitutive theta-theta
        const_tt = exp_s * (1 - self.nu**2) * sig_tt - (du_theta_dt + u_r + self.nu * du_r_ds)

        # 5. Constitutive r-theta
        const_rt = exp_s * 2 * (1 + self.nu) * tau_rt - (du_r_dt + du_theta_ds - u_theta)

        # Total PDE loss
        loss_pde = torch.mean(eq_r**2 + eq_t**2 + const_rr**2 + const_tt**2 + const_rt**2)
        return loss_pde

    def hole_bc_loss(self, s_hole, theta_hole):
        """Traction-free boundary at the hole (s=0)."""
        _, _, sig_rr, _, tau_rt = self.model(s_hole, theta_hole)
        return torch.mean(sig_rr**2 + tau_rt**2)

    def sym_bc_loss(self, s_x, theta_x, s_y, theta_y):
        """Symmetry boundary conditions."""
        loss_sym = 0.0
        
        if s_x.numel() > 0:
            _, u_t_x, _, _, tau_rt_x = self.model(s_x, theta_x)
            loss_sym += torch.mean(u_t_x**2 + tau_rt_x**2)
            
        if s_y.numel() > 0:
            _, u_t_y, _, _, tau_rt_y = self.model(s_y, theta_y)
            loss_sym += torch.mean(u_t_y**2 + tau_rt_y**2)
            
        return loss_sym

    def far_field_bc_loss(self, s_far, theta_far):
        """Far-field Dirichlet boundary condition matching exact Kirsch solution."""
        _, _, sig_rr_pred, sig_tt_pred, tau_rt_pred = self.model(s_far, theta_far)

        r = self.a * torch.exp(s_far)
        a2_r2 = (self.a / r)**2
        a4_r4 = a2_r2**2
        cos2t = torch.cos(2 * theta_far)
        sin2t = torch.sin(2 * theta_far)

        sig_rr_exact = 0.5 * (1 - a2_r2) + 0.5 * (1 + 3 * a4_r4 - 4 * a2_r2) * cos2t
        sig_tt_exact = 0.5 * (1 + a2_r2) - 0.5 * (1 + 3 * a4_r4) * cos2t
        tau_rt_exact = -0.5 * (1 - 3 * a4_r4 + 2 * a2_r2) * sin2t

        loss_far = torch.mean((sig_rr_pred - sig_rr_exact)**2 + 
                              (sig_tt_pred - sig_tt_exact)**2 + 
                              (tau_rt_pred - tau_rt_exact)**2)
        return loss_far

    def compute_raw_losses(self, s_f, theta_f, bc_dict):
        loss_pde = self.pde_residual(s_f, theta_f)
        
        use_ansatz = getattr(self.model, 'use_ansatz', False)
        if use_ansatz:
            device = s_f.device
            loss_hole = torch.tensor(0.0, device=device)
            loss_sym = torch.tensor(0.0, device=device)
        else:
            loss_hole = self.hole_bc_loss(bc_dict['hole'][0], bc_dict['hole'][1])
            loss_sym = self.sym_bc_loss(bc_dict['sym_x'][0], bc_dict['sym_x'][1], bc_dict['sym_y'][0], bc_dict['sym_y'][1])
            
        loss_far = self.far_field_bc_loss(bc_dict['far_field'][0], bc_dict['far_field'][1])

        return {
            'pde': loss_pde,
            'hole': loss_hole,
            'sym': loss_sym,
            'far': loss_far,
        }

    def compute_total_loss(self, s_f, theta_f, bc_dict, step: int = 0, sync_metrics: bool = False):
        raw = self.compute_raw_losses(s_f, theta_f, bc_dict)
        
        if self.weighting_strategy is not None:
            total_loss, weights = self.weighting_strategy.compute_weighted_loss(raw, step=step)
        else:
            total_loss = raw['pde'] + raw['hole'] + raw['sym'] + raw['far']

        # Add total_loss to raw dict for logging
        raw['total'] = total_loss
        
        if sync_metrics:
            raw = {k: v.item() for k, v in raw.items()}
        return total_loss, raw
