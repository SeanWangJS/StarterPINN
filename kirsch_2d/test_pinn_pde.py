import torch
import numpy as np

def test_pinn_pde():
    s_vals = torch.linspace(0, 2.7, 100, requires_grad=True).view(-1, 1)
    t_vals = torch.linspace(0, np.pi/2, 100, requires_grad=True).view(-1, 1)
    
    s, theta = torch.meshgrid(s_vals.squeeze(), t_vals.squeeze(), indexing='ij')
    s = s.reshape(-1, 1)
    theta = theta.reshape(-1, 1)
    
    a = 1.0
    nu = 0.3
    E = 1000.0
    sigma_0 = 10.0
    
    r = a * torch.exp(s)
    a_r = a / r
    
    cos2t = torch.cos(2*theta)
    sin2t = torch.sin(2*theta)
    
    # Exact stresses
    sig_rr = 0.5 * sigma_0 * (1 - a_r**2) + 0.5 * sigma_0 * (1 - 4*a_r**2 + 3*a_r**4) * cos2t
    sig_tt = 0.5 * sigma_0 * (1 + a_r**2) - 0.5 * sigma_0 * (1 + 3*a_r**4) * cos2t
    tau_rt = -0.5 * sigma_0 * (1 + 2*a_r**2 - 3*a_r**4) * sin2t
    
    # THE PINN DISPLACEMENTS that the user says it predicted:
    # Based on my earlier Sympy, this is the solution with f(theta)=0, g(r)=0.
    # From derive_kirsch.py: 
    # u_r_integrated = sigma_0/E * (r * 0.65 + r * 1.3 * cos2t + a**2/r * 0.65 + a**2/r * 2 * cos2t - a**4/r**3 * 0.65 * cos2t)
    # Let's write the exact analytical expression from Sympy:
    u_r = (sigma_0 / E) * ( 0.5*(1+nu)*r + 0.5*(1+nu)*a**2/r + cos2t * ( (1+nu)*r + 2*a**2/r - 0.5*(1+nu)*a**4/r**3 ) )
    # The corresponding u_theta:
    # From derive_kirsch.py: 
    u_theta = (sigma_0 / E) * sin2t * ( - (1+nu)*r - 2*a**2/r - 0.5*(1+nu)*a**4/r**3 )
    
    # Check strains in physical coordinates
    du_r_dr = torch.autograd.grad(u_r, r, torch.ones_like(u_r), create_graph=True)[0]
    du_theta_dt = torch.autograd.grad(u_theta, theta, torch.ones_like(u_theta), create_graph=True)[0]
    du_r_dt = torch.autograd.grad(u_r, theta, torch.ones_like(u_r), create_graph=True)[0]
    du_theta_dr = torch.autograd.grad(u_theta, r, torch.ones_like(u_theta), create_graph=True)[0]
    
    eps_rr = du_r_dr
    eps_tt = (1/r) * du_theta_dt + u_r / r
    gamma_rt = (1/r) * du_r_dt + du_theta_dr - u_theta / r
    
    G = E / (2 * (1 + nu))
    
    res_rr = E * eps_rr - (sig_rr - nu * sig_tt)
    res_tt = E * eps_tt - (sig_tt - nu * sig_rr)
    res_rt = G * gamma_rt - tau_rt
    
    print("res_rr max error:", torch.max(torch.abs(res_rr)).item())
    print("res_tt max error:", torch.max(torch.abs(res_tt)).item())
    print("res_rt max error:", torch.max(torch.abs(res_rt)).item())

if __name__ == '__main__':
    test_pinn_pde()
