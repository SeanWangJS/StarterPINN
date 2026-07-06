"""
Method of Manufactured Solutions (MMS) verification for src/evaluator.py.

See docs/griffith-ablation-study.md \u00a7\u5341 for the rationale: every experiment
so far has been run directly on the hardest possible problem (the Griffith
crack-tip singularity), so a training/ablation failure can never be cleanly
attributed to "the code is wrong" vs "the problem is genuinely hard". This
script checks ONLY whether PFPhysicsEvaluator's
    compute_strains()  ->  compute_energy_and_stress()  ->  equilibrium
    residual assembly (the same autograd calls used inside compute_u_loss)
is internally correct -- completely independent of crack geometry, the
phase field, or any network training/optimization.

Method
------
1. Pick a smooth, non-physical manufactured displacement field (u*, v*)
   with no free parameters tied to the crack problem:

        u*(x, y) = a1*x + a2*sin(x)*cos(y)
        v*(x, y) = a1*y + a2*cos(x)*sin(y)

   with a1 >> a2 > 0 so that eps_vol = du*/dx + dv*/dy = 2*a1 + 2*a2*cos(x)cos(y)
   never changes sign on the test domain -- this keeps the Amor split's
   relu(eps_vol) branch fixed (no kink) so the whole constitutive law is
   C-infinity and can be differentiated exactly by hand.

2. Feed (u*, v*) through the evaluator's ACTUAL code by substituting u_net
   with a tiny wrapper module that returns this closed form instead of a
   real trained network (coord_scheme='cartesian' so _build_u_input is the
   identity). phi_net is monkey-patched to return exactly 0 everywhere
   (same trick used in experiments/diagnostics.py), so g(phi) = 1 + k is a
   constant, matching the hand derivation below.

3. Independently -- by hand calculus, NOT by re-calling any function from
   src/evaluator.py -- derive the exact closed forms for eps_xx, eps_yy,
   eps_xy, sig_xx, sig_yy, tau_xy and the equilibrium residual
   res_x = d(sig_xx)/dx + d(tau_xy)/dy,  res_y = d(tau_xy)/dx + d(sig_yy)/dy.

   Since eps_xx(x,y) = eps_yy(x,y) = a1 + a2*cos(x)*cos(y) by construction,
   the deviatoric parts simplify to e_xx = e_yy = eps_xx/3, giving

        sig_xx = sig_yy = g*(2K + 2*mu/3)*(a1 + a2*cos(x)*cos(y))
        tau_xy = -2*g*mu*a2*sin(x)*sin(y)

   and differentiating once more:

        res_x = -g*a2*(2K + 8*mu/3) * sin(x)*cos(y)
        res_y = -g*a2*(2K + 8*mu/3) * cos(x)*sin(y)

   where g = 1 + k (constant), K = K_nd, mu = mu_nd (see PFPhysicsEvaluator).

4. Compare the evaluator-code-computed eps/sig/res fields against these
   exact closed forms at a grid of test points. Any mismatch beyond normal
   fp64 floating point noise (~1e-10) indicates a genuine bug in the
   strain/stress/residual code, not an artifact of the crack singularity.

Stage 4 below repeats the same idea for the OTHER half of the physics --
PFPhysicsEvaluator.compute_phi_loss(), i.e. the phase-field PDE
    phi/l0*  -  l0* * Laplacian(phi)  =  2*(1-phi)*H*
Here phi_net is replaced by a manufactured closed-form phi*(x,y), and the
(otherwise externally-supplied) history field H* is likewise replaced by a
manufactured closed form -- both freely chosen, smooth, and independent of
any crack/training data. The exact Laplacian and residual are hand-derived
and compared against the same autograd pattern compute_phi_loss uses
internally (phi_x, phi_y, phi_xx, phi_yy via torch.autograd.grad).
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.evaluator import PFPhysicsEvaluator  # noqa: E402

torch.set_default_dtype(torch.float64)

A1 = 1.0     # dominant affine dilation, chosen >> A2 so eps_vol never changes sign
A2 = 0.05    # amplitude of the smooth trig perturbation


class ManufacturedDisplacementNet(torch.nn.Module):
    """Returns the closed-form manufactured field (u*, v*) instead of a real
    trained network's output. Used only for this MMS verification."""

    def __init__(self, a1=A1, a2=A2):
        super().__init__()
        self.a1 = a1
        self.a2 = a2

    def forward(self, pts):
        x = pts[:, 0:1]
        y = pts[:, 1:2]
        u = self.a1 * x + self.a2 * torch.sin(x) * torch.cos(y)
        v = self.a1 * y + self.a2 * torch.cos(x) * torch.sin(y)
        return torch.cat([u, v], dim=1)


def exact_reference(x, y, K, mu, k, a1=A1, a2=A2):
    """Hand-derived closed forms (numpy), independent of src/evaluator.py."""
    g = 1.0 + k

    eps_xx = a1 + a2 * np.cos(x) * np.cos(y)
    eps_yy = a1 + a2 * np.cos(x) * np.cos(y)     # == eps_xx by construction
    eps_xy = -a2 * np.sin(x) * np.sin(y)

    sig_xx = g * (2 * K + 2 * mu / 3.0) * (a1 + a2 * np.cos(x) * np.cos(y))
    sig_yy = sig_xx.copy()
    tau_xy = -2.0 * g * mu * a2 * np.sin(x) * np.sin(y)

    res_x = -g * a2 * (2 * K + 8 * mu / 3.0) * np.sin(x) * np.cos(y)
    res_y = -g * a2 * (2 * K + 8 * mu / 3.0) * np.cos(x) * np.sin(y)

    return {
        'eps_xx': eps_xx, 'eps_yy': eps_yy, 'eps_xy': eps_xy,
        'sig_xx': sig_xx, 'sig_yy': sig_yy, 'tau_xy': tau_xy,
        'res_x': res_x, 'res_y': res_y,
    }


def build_evaluator(nu=0.3, k=1e-5):
    u_net = ManufacturedDisplacementNet()
    phi_net = torch.nn.Module()
    phi_net.forward = lambda x: torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)

    evaluator = PFPhysicsEvaluator(
        u_net, phi_net, E=1.0, nu=nu, G_c=1.0, l_0=0.1, a=1.0, k=k,
        coord_scheme='cartesian',
    )
    return evaluator


# ----------------------------------------------------------------------
# Stage 4 fixtures: phase-field PDE (compute_phi_loss)
# ----------------------------------------------------------------------
B0 = 0.5     # manufactured phi* baseline (kept well inside [0, 1])
B1 = 0.1     # amplitude of the smooth trig perturbation on phi*
C0 = 1.0     # manufactured history field H* baseline
C1 = 0.3     # amplitude of the smooth trig perturbation on H*


class ManufacturedPhiNet(torch.nn.Module):
    """Returns the closed-form manufactured field phi*(x,y) instead of a
    real trained PhaseFieldNet's output. Used only for this MMS verification."""

    def __init__(self, b0=B0, b1=B1):
        super().__init__()
        self.b0 = b0
        self.b1 = b1

    def forward(self, pts):
        x = pts[:, 0:1]
        y = pts[:, 1:2]
        return self.b0 + self.b1 * torch.sin(x) * torch.cos(y)


def history_field_exact(x, y, c0=C0, c1=C1):
    """Manufactured history field H*(x,y) -- freely chosen since H* is an
    externally-supplied (detached) input to compute_phi_loss, not something
    derived from phi itself."""
    return c0 + c1 * np.cos(x) * np.sin(y)


def phase_field_exact_residual(x, y, l0_nd, b0=B0, b1=B1, c0=C0, c1=C1):
    """Hand-derived closed form (numpy) for the phase-field PDE residual
        phi/l0*  -  l0* * Laplacian(phi)  -  2*(1-phi)*H*
    given phi* = b0 + b1*sin(x)*cos(y), so
        Laplacian(phi*) = -2*b1*sin(x)*cos(y)
    and H* = c0 + c1*cos(x)*sin(y) (see history_field_exact)."""
    phi = b0 + b1 * np.sin(x) * np.cos(y)
    laplacian_phi = -2.0 * b1 * np.sin(x) * np.cos(y)
    H = history_field_exact(x, y, c0, c1)
    return (phi / l0_nd - l0_nd * laplacian_phi) - 2.0 * (1.0 - phi) * H


def build_phi_evaluator(nu=0.3, k=1e-5, l_0=0.1, a=1.0):
    u_net = torch.nn.Linear(2, 2)  # dummy: never called by compute_phi_loss
    phi_net = ManufacturedPhiNet()
    evaluator = PFPhysicsEvaluator(
        u_net, phi_net, E=1.0, nu=nu, G_c=1.0, l_0=l_0, a=a, k=k,
        coord_scheme='cartesian',
    )
    return evaluator


def make_grid(n=15, lo=-0.8, hi=0.8):
    xs = np.linspace(lo, hi, n)
    ys = np.linspace(lo, hi, n)
    X, Y = np.meshgrid(xs, ys)
    return X.flatten(), Y.flatten()


def rel_err(a, b):
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-14))


def main():
    nu, k = 0.3, 1e-5
    evaluator = build_evaluator(nu=nu, k=k)
    x_np, y_np = make_grid()
    ref = exact_reference(x_np, y_np, evaluator.K_nd, evaluator.mu_nd, k)

    pts = torch.tensor(np.stack([x_np, y_np], axis=1))

    # ---- Stage 1: compute_strains() ----
    eps_xx, eps_yy, eps_xy, x_y_nd, u_nd, v_nd, *_ = evaluator.compute_strains(pts)

    print("=" * 70)
    print("Stage 1: compute_strains()  vs  hand-derived exact strains")
    print("=" * 70)
    for name, computed, exact in [
        ('eps_xx', eps_xx, ref['eps_xx']),
        ('eps_yy', eps_yy, ref['eps_yy']),
        ('eps_xy', eps_xy, ref['eps_xy']),
    ]:
        e = rel_err(computed.detach().numpy(), exact)
        print(f"  {name:8s}  max relative error = {e:.3e}")

    # ---- Stage 2: compute_energy_and_stress() ----
    phi_zero = torch.zeros_like(eps_xx)
    psi_nd, sig_xx, sig_yy, tau_xy = evaluator.compute_energy_and_stress(
        eps_xx, eps_yy, eps_xy, phi_zero)

    print("\n" + "=" * 70)
    print("Stage 2: compute_energy_and_stress()  vs  hand-derived exact stresses")
    print("=" * 70)
    for name, computed, exact in [
        ('sig_xx', sig_xx, ref['sig_xx']),
        ('sig_yy', sig_yy, ref['sig_yy']),
        ('tau_xy', tau_xy, ref['tau_xy']),
    ]:
        e = rel_err(computed.detach().numpy(), exact)
        print(f"  {name:8s}  max relative error = {e:.3e}")

    # ---- Stage 3: equilibrium residual (same autograd pattern as compute_u_loss) ----
    dsig_xx_dx = torch.autograd.grad(sig_xx.sum(), x_y_nd, create_graph=True)[0][:, 0:1]
    dtau_xy_dy = torch.autograd.grad(tau_xy.sum(), x_y_nd, create_graph=True)[0][:, 1:2]
    dtau_xy_dx = torch.autograd.grad(tau_xy.sum(), x_y_nd, create_graph=True)[0][:, 0:1]
    dsig_yy_dy = torch.autograd.grad(sig_yy.sum(), x_y_nd, create_graph=True)[0][:, 1:2]

    res_x = (dsig_xx_dx + dtau_xy_dy).detach().numpy()
    res_y = (dtau_xy_dx + dsig_yy_dy).detach().numpy()

    print("\n" + "=" * 70)
    print("Stage 3: equilibrium residual  vs  hand-derived exact residual")
    print("=" * 70)
    e_res_x = rel_err(res_x, ref['res_x'])
    e_res_y = rel_err(res_y, ref['res_y'])
    print(f"  res_x     max relative error = {e_res_x:.3e}")
    print(f"  res_y     max relative error = {e_res_y:.3e}")

    # ---- Stage 4: compute_phi_loss() phase-field PDE residual ----
    phi_evaluator = build_phi_evaluator(nu=nu, k=k)
    l0_nd = phi_evaluator.l0_nd

    x4_np, y4_np = make_grid()
    pts4 = torch.tensor(np.stack([x4_np, y4_np], axis=1))
    history_H_nd = torch.tensor(history_field_exact(x4_np, y4_np)).reshape(-1, 1)

    # Replicate the exact autograd pattern used inside compute_phi_loss so we
    # can compare the per-point residual, not just the aggregated MSE loss.
    pts4_grad = pts4.clone().requires_grad_(True)
    phi4 = phi_evaluator.phi_net(pts4_grad)
    phi_x = torch.autograd.grad(phi4.sum(), pts4_grad, create_graph=True)[0][:, 0:1]
    phi_y = torch.autograd.grad(phi4.sum(), pts4_grad, create_graph=True)[0][:, 1:2]
    phi_xx = torch.autograd.grad(phi_x.sum(), pts4_grad, create_graph=True)[0][:, 0:1]
    phi_yy = torch.autograd.grad(phi_y.sum(), pts4_grad, create_graph=True)[0][:, 1:2]
    laplacian_phi = phi_xx + phi_yy
    residual_computed = (phi4 / l0_nd - l0_nd * laplacian_phi) - 2.0 * (1.0 - phi4) * history_H_nd
    residual_computed_np = residual_computed.detach().numpy()

    residual_exact_np = phase_field_exact_residual(x4_np, y4_np, l0_nd)

    print("\n" + "=" * 70)
    print("Stage 4: compute_phi_loss() residual  vs  hand-derived exact residual")
    print("=" * 70)
    e_phi_residual = rel_err(residual_computed_np, residual_exact_np)
    print(f"  phi_residual  max relative error = {e_phi_residual:.3e}")

    # Cross-check: the actual public compute_phi_loss() scalar loss should
    # also match mean(residual_exact**2) (sanity check on the public API,
    # in addition to the internal per-point residual check above).
    loss_phi_actual = phi_evaluator.compute_phi_loss(pts4, history_H_nd, w_pde=1.0)[1]
    loss_phi_exact = float(np.mean(residual_exact_np ** 2))
    e_phi_loss = abs(loss_phi_actual - loss_phi_exact) / (abs(loss_phi_exact) + 1e-14)
    print(f"  compute_phi_loss() scalar   : {loss_phi_actual:.6e}")
    print(f"  mean(residual_exact**2)     : {loss_phi_exact:.6e}")
    print(f"  scalar loss relative error  = {e_phi_loss:.3e}")

    tol = 1e-8
    all_errs = []
    for name, computed, exact in [
        ('eps_xx', eps_xx.detach().numpy(), ref['eps_xx']),
        ('eps_yy', eps_yy.detach().numpy(), ref['eps_yy']),
        ('eps_xy', eps_xy.detach().numpy(), ref['eps_xy']),
        ('sig_xx', sig_xx.detach().numpy(), ref['sig_xx']),
        ('sig_yy', sig_yy.detach().numpy(), ref['sig_yy']),
        ('tau_xy', tau_xy.detach().numpy(), ref['tau_xy']),
        ('res_x', res_x, ref['res_x']),
        ('res_y', res_y, ref['res_y']),
    ]:
        all_errs.append(rel_err(computed, exact))
    all_errs.append(e_phi_residual)
    all_errs.append(e_phi_loss)

    ok = all(e < tol for e in all_errs)
    print("\n" + "=" * 70)
    verdict = f"PASS -- all fields (elastic + phase-field) match exact MMS reference to < {tol:.0e}" if ok \
        else f"FAIL -- at least one field exceeds the {tol:.0e} tolerance (see above)"
    print(f"Verdict: {verdict}")
    print("=" * 70)
    return ok


if __name__ == '__main__':
    main()
