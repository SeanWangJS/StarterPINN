import sympy as sp

r, theta, a, sigma_0, E, nu = sp.symbols('r theta a sigma_0 E nu', positive=True)

cos2 = sp.cos(2*theta)
sin2 = sp.sin(2*theta)
a_r = a/r

sig_rr = 0.5*sigma_0*(1 - a_r**2) + 0.5*sigma_0*(1 - 4*a_r**2 + 3*a_r**4)*cos2
sig_tt = 0.5*sigma_0*(1 + a_r**2) - 0.5*sigma_0*(1 + 3*a_r**4)*cos2
tau_rt = -0.5*sigma_0*(1 + 2*a_r**2 - 3*a_r**4)*sin2

eps_rr = (sig_rr - nu*sig_tt) / E
eps_tt = (sig_tt - nu*sig_rr) / E
gam_rt = 2*(1+nu)*tau_rt / E

# 1. Integrate eps_rr to get u_r
u_r_int = sp.integrate(eps_rr, r)

# Add f(theta) 
# Wait, let's just use f(theta) as an unknown function
f = sp.Function('f')(theta)
u_r = u_r_int + f

# 2. Use eps_tt to get u_theta
du_theta_dt = r * eps_tt - u_r
u_theta_int = sp.integrate(du_theta_dt, theta)

# Add g(r)
g = sp.Function('g')(r)
u_theta = u_theta_int + g

# 3. Check gam_rt
gam_rt_derived = (1/r) * sp.diff(u_r, theta) + sp.diff(u_theta, r) - u_theta / r
gam_diff = sp.simplify(gam_rt - gam_rt_derived)

print("Difference in gamma_rt:")
sp.pprint(gam_diff)
