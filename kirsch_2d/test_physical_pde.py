import torch
import numpy as np

def test_physical():
    r_vals = torch.linspace(1.0, 15.0, 100, requires_grad=True).view(-1, 1)
    t_vals = torch.linspace(0, np.pi/2, 100, requires_grad=True).view(-1, 1)
    
    r, theta = torch.meshgrid(r_vals.squeeze(), t_vals.squeeze(), indexing='ij')
    r = r.reshape(-1, 1)
    theta = theta.reshape(-1, 1)
    
    a = 1.0
    nu = 0.3
    E = 1000.0
    sigma_0 = 10.0
    G = E / (2 * (1 + nu))
    kappa = (3 - nu) / (1 + nu)
    
    a_r = a / r
    cos2t = torch.cos(2*theta)
    sin2t = torch.sin(2*theta)
    
    # Exact physical stresses
    sig_rr = 0.5 * sigma_0 * (1 - a_r**2) + 0.5 * sigma_0 * (1 - 4*a_r**2 + 3*a_r**4) * cos2t
    sig_tt = 0.5 * sigma_0 * (1 + a_r**2) - 0.5 * sigma_0 * (1 + 3*a_r**4) * cos2t
    tau_rt = -0.5 * sigma_0 * (1 + 2*a_r**2 - 3*a_r**4) * sin2t
    
    # Exact physical displacements
    u_r = (sigma_0 * a / (8 * G)) * ( (r/a)*(kappa - 1) + 2*a_r + cos2t * (2*(r/a) + a_r*(kappa + 1) - a_r**3) )
    u_theta = (sigma_0 * a / (8 * G)) * sin2t * ( -2*(r/a) - a_r*(kappa - 1) - a_r**3 )
    
    # Strains
    du_r_dr = torch.autograd.grad(u_r, r, torch.ones_like(u_r), create_graph=True)[0]
    du_theta_dt = torch.autograd.grad(u_theta, theta, torch.ones_like(u_theta), create_graph=True)[0]
    du_r_dt = torch.autograd.grad(u_r, theta, torch.ones_like(u_r), create_graph=True)[0]
    du_theta_dr = torch.autograd.grad(u_theta, r, torch.ones_like(u_theta), create_graph=True)[0]
    
    eps_rr = du_r_dr
    eps_tt = (1/r) * du_theta_dt + u_r / r
    gamma_rt = (1/r) * du_r_dt + du_theta_dr - u_theta / r
    
    # Hooke's Law Plane Stress (E * eps_rr = sig_rr - nu*sig_tt)
    res_rr = E * eps_rr - (sig_rr - nu * sig_tt)
    res_tt = E * eps_tt - (sig_tt - nu * sig_rr)
    res_rt = G * gamma_rt - tau_rt
    
    print("res_rr max error:", torch.max(torch.abs(res_rr)).item())
    print("res_tt max error:", torch.max(torch.abs(res_tt)).item())
    print("res_rt max error:", torch.max(torch.abs(res_rt)).item())

if __name__ == '__main__':
    test_physical()
