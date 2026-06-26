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

# Integrate eps_rr w.r.t r to get u_r
# Note: integration gives the term, but may have a function of theta.
u_r_integrated = sp.integrate(eps_rr, r)
u_r_integrated = sp.simplify(u_r_integrated)

# We know the true plane stress form from theory:
# kappa = (3-nu)/(1+nu)
# G = E / (2*(1+nu))
# u_r_exact = sigma_0 * a / (8*G) * ( ... )
# Let's derive it and check it directly!
print("Sympy integrated u_r (ignoring f(theta)):")
sp.pprint(u_r_integrated)

# Now let's calculate u_r at r=a directly from the integrated formula
u_r_at_a = u_r_integrated.subs(r, a)
u_r_at_a = sp.simplify(u_r_at_a)
print("\nSympy u_r at r=a:")
sp.pprint(u_r_at_a)

# Now for u_theta: eps_tt = 1/r * du_theta_dt + u_r/r
# du_theta_dt = r * eps_tt - u_r
# Integrate w.r.t theta
du_theta_dt = r * eps_tt - u_r_integrated
u_theta_integrated = sp.integrate(du_theta_dt, theta)
u_theta_integrated = sp.simplify(u_theta_integrated)

print("\nSympy integrated u_theta:")
sp.pprint(u_theta_integrated)

u_theta_at_a = u_theta_integrated.subs(r, a)
u_theta_at_a = sp.simplify(u_theta_at_a)
print("\nSympy u_theta at r=a:")
sp.pprint(u_theta_at_a)
