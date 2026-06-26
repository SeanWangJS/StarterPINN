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

u_r_int = sp.integrate(eps_rr, r)

du_theta_dt = r * eps_tt - u_r_int
u_theta_int = sp.integrate(du_theta_dt, theta)

u_theta_int = u_theta_int.subs(sp.sin(2.0*theta), sp.sin(2*theta))
u_theta_int = u_theta_int.subs(sp.cos(2.0*theta), sp.cos(2*theta))

u_theta_a = sp.simplify(u_theta_int.subs(r, a))

print("u_theta(a) =", sp.expand(u_theta_a))
