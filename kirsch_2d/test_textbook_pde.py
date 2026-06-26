import torch
import numpy as np

def test_textbook():
    s_vals = torch.linspace(0.0, 2.7, 100, requires_grad=True).view(-1, 1)
    t_vals = torch.linspace(0, np.pi/2, 100, requires_grad=True).view(-1, 1)
    
    s, theta = torch.meshgrid(s_vals.squeeze(), t_vals.squeeze(), indexing='ij')
    s = s.reshape(-1, 1)
    theta = theta.reshape(-1, 1)
    
    a = 1.0
    nu = 0.3
    E = 1000.0
    sigma_0 = 10.0
    G = E / (2 * (1 + nu))
    kappa = (3 - nu) / (1 + nu)
    
    r = a * torch.exp(s)
    a_r = a / r
    cos2t = torch.cos(2*theta)
    sin2t = torch.sin(2*theta)
    
    # Exact physical stresses
    sig_rr = 0.5 * sigma_0 * (1 - a_r**2) + 0.5 * sigma_0 * (1 - 4*a_r**2 + 3*a_r**4) * cos2t
    sig_tt = 0.5 * sigma_0 * (1 + a_r**2) - 0.5 * sigma_0 * (1 + 3*a_r**4) * cos2t
    tau_rt = -0.5 * sigma_0 * (1 + 2*a_r**2 - 3*a_r**4) * sin2t
    
    # Exact physical displacements (Textbook)
    u_r = (sigma_0 * a / (8 * G)) * ( (r/a)*(kappa - 1) + 2*a_r + cos2t * (2*(r/a) + a_r*(kappa + 1) - a_r**3) )
    u_theta = (sigma_0 * a / (8 * G)) * sin2t * ( -2*(r/a) - a_r*(kappa - 1) - a_r**3 )
    
    # Check strains in (s, theta)
    du_r_ds = torch.autograd.grad(u_r, s, torch.ones_like(u_r), create_graph=True)[0]
    du_theta_dt = torch.autograd.grad(u_theta, theta, torch.ones_like(u_theta), create_graph=True)[0]
    du_r_dt = torch.autograd.grad(u_r, theta, torch.ones_like(u_r), create_graph=True)[0]
    du_theta_ds = torch.autograd.grad(u_theta, s, torch.ones_like(u_theta), create_graph=True)[0]
    
    exp_s = torch.exp(s)
    
    const_rr = exp_s * (1 - nu**2) * sig_rr - (E/sigma_0/a)*(du_r_ds + nu * (du_theta_dt + u_r))
    const_tt = exp_s * (1 - nu**2) * sig_tt - (E/sigma_0/a)*(du_theta_dt + u_r + nu * du_r_ds)
    const_rt = exp_s * 2 * (1 + nu) * tau_rt - (E/sigma_0/a)*(du_r_dt + du_theta_ds - u_theta)
    
    # Let's check physical Hooke's law directly.
    eps_rr = (1 / (a * exp_s)) * du_r_ds
    eps_tt = (1 / (a * exp_s)) * (du_theta_dt + u_r)
    gam_rt = (1 / (a * exp_s)) * (du_r_dt + du_theta_ds - u_theta)
    
    res_rr = E * eps_rr - (sig_rr - nu * sig_tt)
    res_tt = E * eps_tt - (sig_tt - nu * sig_rr)
    res_rt = G * gam_rt - tau_rt
    
    print("Physical Hooke's Law Residuals:")
    print("res_rr max error:", torch.max(torch.abs(res_rr)).item())
    print("res_tt max error:", torch.max(torch.abs(res_tt)).item())
    print("res_rt max error:", torch.max(torch.abs(res_rt)).item())

if __name__ == '__main__':
    test_textbook()
