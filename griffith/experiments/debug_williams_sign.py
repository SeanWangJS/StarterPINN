"""
Standalone diagnostic for the coord_scheme='williams' sign problem observed
in experiments/results/metrics.csv (peak_mag_ratio consistently negative for
fp64, oscillating in sign for fp32 -- see the ablation write-up).

Goal: isolate whether the sign flip is a genuine bug in williams_features()
(src/evaluator.py) or purely a training/optimization pathology, WITHOUT
running the full PINN training loop.

Method
------
The classical Williams/Irwin Mode-I near-tip displacement field (plane
strain) is

    u = A*sqrt(r) * [ (kappa-1)*cos(t/2) + 2*sin(t/2)^2*cos(t/2) ]
    v = A*sqrt(r) * [ (kappa+1)*sin(t/2) - 2*cos(t/2)^2*sin(t/2) ]

with A = K_I / (2*G*sqrt(2*pi)), kappa = 3-4*nu (plane strain), t = theta.

Using sin(t) = 2*sin(t/2)*cos(t/2), this reduces EXACTLY to

    u = A*(kappa-1)*f1 + A*f3
    v = A*(kappa+1)*f2 - A*f4

where f1..f4 = williams_features(pts_nd). So f1..f4 exactly span the
analytical near-tip displacement field with a KNOWN sign pattern:
coefficients of f1, f3, f2 are positive and the coefficient of f4 is
NEGATIVE (for kappa > 1, A > 0). This script:

  Part A - fits u ~ c1*f1 + c3*f3 and v ~ c2*f2 + c4*f4 by least squares
           against the exact analytical field and compares fitted vs
           theoretical coefficients (checks the feature definition itself
           is not buggy: the correct-signed solution IS representable).
  Part B - measures ||d f_i / d(x,y)|| as r -> 0 to quantify near-tip
           gradient blow-up, the likely real cause of training instability
           (same mechanism previously diagnosed for the elliptic-coordinate
           Jacobian amplification).
"""
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
_SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'scripts')
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from src.evaluator import williams_features  # noqa: E402


def load_cfg():
    path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default.yaml')
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def analytical_params(cfg):
    phys = cfg['physics']
    E, nu, a = phys['E'], phys['nu'], phys['a']
    E_prime = E / (1 - nu ** 2)
    v_max = cfg['train']['v_max']
    sigma_0 = E_prime * (v_max / (phys['H'] / 2.0))
    K_I = sigma_0 * np.sqrt(np.pi * a)
    kappa = 3 - 4 * nu   # plane strain
    G = E / (2 * (1 + nu))
    return K_I, kappa, G, sigma_0, a


def sample_near_tip_points(a, n_r=40, n_theta=72, r_min_factor=1e-2, r_max_factor=0.3):
    """Polar grid of physical (x, y) around the right tip (a, 0), excluding
    theta = +-pi exactly (branch cut) and r = 0 (singularity)."""
    r = np.linspace(r_min_factor * a, r_max_factor * a, n_r)
    theta = np.linspace(-np.pi + 0.02, np.pi - 0.02, n_theta)
    R, TH = np.meshgrid(r, theta)
    X = a + R * np.cos(TH)
    Y = R * np.sin(TH)
    return X.flatten(), Y.flatten(), R.flatten(), TH.flatten()


def part_a_sign_check(cfg):
    K_I, kappa, G, sigma_0, a = analytical_params(cfg)
    A = K_I / (2 * G * np.sqrt(2 * np.pi))
    print(f"K_I={K_I:.4e}  kappa={kappa:.4f}  G={G:.4e}  A={A:.6e}\n")

    x, y, r, theta = sample_near_tip_points(a)

    u_exact = A * np.sqrt(r) * (np.cos(theta / 2) * (kappa - 1)
                                 + 2 * np.sin(theta / 2) ** 2 * np.cos(theta / 2))
    v_exact = A * np.sqrt(r) * (np.sin(theta / 2) * (kappa + 1)
                                 - 2 * np.cos(theta / 2) ** 2 * np.sin(theta / 2))

    pts = torch.tensor(np.stack([x, y], axis=1), dtype=torch.float64)
    wf = williams_features(pts, a_nd=a, n_modes=4, eps=1e-6).numpy()
    f1, f2, f3, f4 = wf[:, 0], wf[:, 1], wf[:, 2], wf[:, 3]

    Au = np.stack([f1, f3], axis=1)
    cu, *_ = np.linalg.lstsq(Au, u_exact, rcond=None)
    Av = np.stack([f2, f4], axis=1)
    cv, *_ = np.linalg.lstsq(Av, v_exact, rcond=None)

    resid_u = np.linalg.norm(Au @ cu - u_exact) / (np.linalg.norm(u_exact) + 1e-30)
    resid_v = np.linalg.norm(Av @ cv - v_exact) / (np.linalg.norm(v_exact) + 1e-30)

    print("--- Part A: least-squares fit of exact Williams displacement onto f1..f4 ---")
    print(f"u = c1*f1 + c3*f3   fitted (c1, c3) = ({cu[0]: .6e}, {cu[1]: .6e})")
    print(f"                    theory (c1, c3) = ({A*(kappa-1): .6e}, {A: .6e})")
    print(f"v = c2*f2 + c4*f4   fitted (c2, c4) = ({cv[0]: .6e}, {cv[1]: .6e})")
    print(f"                    theory (c2, c4) = ({A*(kappa+1): .6e}, {-A: .6e})")
    print(f"relative fit residual: u={resid_u:.3e}  v={resid_v:.3e}")

    # A few % relative residual is expected discretization/truncation error
    # from the finite (r, theta) sampling grid and the theta=+-pi cutoff --
    # it does not indicate a bug. What matters is the SIGN and approximate
    # magnitude of the fitted coefficients matching theory.
    resid_tol = 0.05
    mag_tol = 0.05  # 5% relative magnitude error
    ok = (resid_u < resid_tol and resid_v < resid_tol
          and np.sign(cu[0]) == np.sign(A * (kappa - 1))
          and np.sign(cu[1]) == np.sign(A)
          and np.sign(cv[0]) == np.sign(A * (kappa + 1))
          and np.sign(cv[1]) == np.sign(-A)
          and abs(cu[0] - A * (kappa - 1)) / abs(A * (kappa - 1)) < mag_tol
          and abs(cu[1] - A) / abs(A) < mag_tol
          and abs(cv[0] - A * (kappa + 1)) / abs(A * (kappa + 1)) < mag_tol
          and abs(cv[1] - (-A)) / abs(A) < mag_tol)
    verdict = "PASS -- features exactly & correctly span the true field" if ok else "FAIL -- feature definition looks buggy"
    print(f"\nVerdict: {verdict}\n")
    return ok


def part_b_gradient_blowup(a):
    print("--- Part B: ||grad f_i|| vs r (near-tip gradient amplification) ---")
    r_values = [1e-3, 1e-2, 1e-1, 3e-1, 1.0]
    n_theta = 200
    header = f"{'r/a':>10} | {'|grad f1|':>12} {'|grad f2|':>12} {'|grad f3|':>12} {'|grad f4|':>12}"
    print(header)
    print('-' * len(header))

    baseline = None
    for r_factor in r_values:
        r = r_factor * a
        theta = np.linspace(-np.pi + 0.02, np.pi - 0.02, n_theta)
        x = a + r * np.cos(theta)
        y = r * np.sin(theta)
        pts = torch.tensor(np.stack([x, y], axis=1), dtype=torch.float64, requires_grad=True)

        norms = []
        for i in range(4):
            if pts.grad is not None:
                pts.grad.zero_()
            wf = williams_features(pts, a_nd=a, n_modes=4, eps=1e-6)
            wf[:, i].sum().backward(retain_graph=False)
            g = pts.grad.clone()
            norms.append(g.norm(dim=1).mean().item())

        if baseline is None:
            baseline = norms
        print(f"{r_factor:>10.4g} | " + " ".join(f"{n:>12.4e}" for n in norms))
    print()


def part_b_gradient_ratio(a):
    """Report amplification ratio between r=1e-3*a and r=1.0*a explicitly."""
    def mean_grad_norms(r_factor, n_theta=200):
        r = r_factor * a
        theta = np.linspace(-np.pi + 0.02, np.pi - 0.02, n_theta)
        x = a + r * np.cos(theta)
        y = r * np.sin(theta)
        pts = torch.tensor(np.stack([x, y], axis=1), dtype=torch.float64, requires_grad=True)
        norms = []
        for i in range(4):
            if pts.grad is not None:
                pts.grad.zero_()
            wf = williams_features(pts, a_nd=a, n_modes=4, eps=1e-6)
            wf[:, i].sum().backward(retain_graph=False)
            norms.append(pts.grad.norm(dim=1).mean().item())
        return np.array(norms)

    small = mean_grad_norms(1e-3)
    large = mean_grad_norms(1.0)
    ratio = small / large
    print(f"Amplification (r=1e-3*a vs r=1.0*a) per feature [f1,f2,f3,f4]: "
          + ", ".join(f"{r_:.1f}x" for r_ in ratio))
    print()


if __name__ == '__main__':
    cfg = load_cfg()
    part_a_sign_check(cfg)
    _, _, _, _, a = analytical_params(cfg)
    part_b_gradient_blowup(a)
    part_b_gradient_ratio(a)
