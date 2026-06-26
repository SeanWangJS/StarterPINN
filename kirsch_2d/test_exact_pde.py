import torch
import numpy as np

def test_exact_pde():
    s_vals = torch.linspace(0, 2.7, 100, requires_grad=True).view(-1, 1)
    t_vals = torch.linspace(0, np.pi/2, 100, requires_grad=True).view(-1, 1)
    
    # Create a grid
    s, theta = torch.meshgrid(s_vals.squeeze(), t_vals.squeeze(), indexing='ij')
    s = s.reshape(-1, 1)
    theta = theta.reshape(-1, 1)
    
    a = 1.0
    nu = 0.3
    
    r = a * torch.exp(s)
    a_r = a / r
    
    # Exact stresses
    cos2t = torch.cos(2*theta)
    sin2t = torch.sin(2*theta)
    
    sig_rr = 0.5*(1 - a_r**2) + 0.5*(1 - 4*a_r**2 + 3*a_r**4)*cos2t
    sig_tt = 0.5*(1 + a_r**2) - 0.5*(1 + 3*a_r**4)*cos2t
    tau_rt = -0.5*(1 + 2*a_r**2 - 3*a_r**4)*sin2t
    
    # Exact displacements
    u_r = 0.25 * (1 + nu) * ( (r/a)*((3-nu)/(1+nu) - 1) + 2*a_r + cos2t * (2*(r/a) + a_r*((3-nu)/(1+nu) + 1) - a_r**3) )
    u_theta = 0.25 * (1 + nu) * sin2t * ( -2*(r/a) - a_r*((3-nu)/(1+nu) - 1) - a_r**3 )
    
    # Compute derivatives
    du_r_ds = torch.autograd.grad(u_r, s, torch.ones_like(u_r), create_graph=True)[0]
    du_theta_dt = torch.autograd.grad(u_theta, theta, torch.ones_like(u_theta), create_graph=True)[0]
    du_r_dt = torch.autograd.grad(u_r, theta, torch.ones_like(u_r), create_graph=True)[0]
    du_theta_ds = torch.autograd.grad(u_theta, s, torch.ones_like(u_theta), create_graph=True)[0]
    
    exp_s = torch.exp(s)
    
    const_rr = exp_s * (1 - nu**2) * sig_rr - (du_r_ds + nu * (du_theta_dt + u_r))
    const_tt = exp_s * (1 - nu**2) * sig_tt - (du_theta_dt + u_r + nu * du_r_ds)
    const_rt = exp_s * 2 * (1 + nu) * tau_rt - (du_r_dt + du_theta_ds - u_theta)
    
    print("const_rr max error:", torch.max(torch.abs(const_rr)).item())
    print("const_tt max error:", torch.max(torch.abs(const_tt)).item())
    print("const_rt max error:", torch.max(torch.abs(const_rt)).item())

if __name__ == '__main__':
    test_exact_pde()
