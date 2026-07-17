import torch
import numpy as np


class GriffithGeometrySampler:
    """
    Collocation point sampler for the Griffith center-crack plate problem.

    The problem is solved on the RIGHT HALF domain: x ∈ [0, W/2], y ∈ [-H/2, H/2].
    The full field is recovered by left-right reflection:
        u_full(x, y)   =  u_nn(|x|, y)   * sign(x)    [antisymmetric in x]
        v_full(x, y)   =  v_nn(|x|, y)                [symmetric    in x]
        phi_full(x, y) = phi_nn(|x|, y)               [symmetric    in x]

    Symmetry BC on x=0:  u = 0,  τ_xy = 0

    Crack is treated as fully internal to the domain and handled by the
    phase-field H0 pre-conditioning (no explicit crack-face BC needed on x=0).

    Sampling density (physical coords):
      - Global uniform background              (~30%)
      - Crack-band refinement  |y| < 3*l_0    (~25%)
      - Crack-tip enrichment   r < 4*l_0 at (a, 0)  (~25%)   [only right tip in half domain]
      - Fine band along x=0 symmetry axis      (~20%)
    """

    def __init__(self, W, H, a, device='cpu'):
        self.W = W
        self.H = H
        self.a = a
        self.device = device

    # ------------------------------------------------------------------
    # Domain interior  (RIGHT HALF: x ∈ [0, W/2])
    # ------------------------------------------------------------------
    def sample_fixed_domain(self, num_points, l_0: float = 0.01):
        """
        Mixed-density sampling on the right half domain x ∈ [0, W/2].
        """
        n_uniform  = int(num_points * 0.30)
        n_band     = int(num_points * 0.25)
        n_tip      = int(num_points * 0.25)   # right crack tip only
        n_sym_axis = int(num_points * 0.20)
        # remainder absorbed into uniform
        n_uniform += num_points - n_uniform - n_band - n_tip - n_sym_axis

        parts = []

        # ---- 1. Global background (x ≥ 0) ----
        x_uni = np.random.uniform(0.0, self.W/2, (n_uniform, 1))
        y_uni = np.random.uniform(-self.H/2, self.H/2, (n_uniform, 1))
        parts.append(np.hstack([x_uni, y_uni]))

        # ---- 2. Crack-band refinement  |y| < 3*l_0 ----
        band_w = 3.0 * l_0
        x_band = np.random.uniform(0.0, self.W/2, (n_band, 1))
        y_band = np.random.uniform(-band_w, band_w, (n_band, 1))
        parts.append(np.hstack([x_band, y_band]))

        # ---- 3. Crack-tip enrichment around RIGHT tip (a, 0) ----
        tip_r = 4.0 * l_0
        r   = tip_r * np.sqrt(np.random.uniform(0, 1, (n_tip, 1)))
        ang = np.random.uniform(-np.pi, np.pi, (n_tip, 1))
        x_t = self.a + r * np.cos(ang)
        y_t = r * np.sin(ang)
        x_t = np.clip(x_t, 0.0, self.W/2)
        y_t = np.clip(y_t, -self.H/2, self.H/2)
        parts.append(np.hstack([x_t, y_t]))

        # ---- 4. Fine band near x=0 symmetry axis ----
        x_sym = np.abs(np.random.normal(0, l_0, (n_sym_axis, 1)))
        x_sym = np.clip(x_sym, 0.0, self.W/2)
        y_sym = np.random.uniform(-self.H/2, self.H/2, (n_sym_axis, 1))
        parts.append(np.hstack([x_sym, y_sym]))

        pts = np.vstack(parts)
        idx = np.random.permutation(pts.shape[0])
        pts = pts[idx]

        return torch.tensor(pts, dtype=torch.get_default_dtype(), device=self.device)

    # ------------------------------------------------------------------
    # Mixture sampling density p(x, y) of sample_fixed_domain(), for
    # importance-weighted Monte Carlo integration (Deep Energy Method --
    # see docs/griffith-ablation-study.md §12 and
    # PFPhysicsEvaluator.compute_u_loss_dem()).
    # ------------------------------------------------------------------
    def compute_mixture_density(self, pts_phys, l_0: float = 0.01):
        """
        Evaluate the closed-form mixture density p(x, y) that
        sample_fixed_domain() draws from, at arbitrary query points.

        sample_fixed_domain() is a 30/25/25/20 mixture of four components
        (global uniform / crack-band / crack-tip disk / symmetry-axis band).
        Each component's density has a simple closed form; weighting a
        network output by 1/p_mixture(x,y) turns torch.mean(f * weights)
        into an unbiased Monte Carlo estimate of the domain-average of f,
        despite the non-uniform sampling. Ignores the negligible edge-clipping
        applied in sample_fixed_domain() (the tip disk and symmetry band are
        both far narrower than the domain, so clipping essentially never
        triggers for this problem's geometry).

        pts_phys : (N, 2) numpy array of physical (x, y) coordinates.
        Returns  : (N, 1) numpy array of p(x, y) > 0.
        """
        x = pts_phys[:, 0:1]
        y = pts_phys[:, 1:2]
        band_w = 3.0 * l_0
        tip_r = 4.0 * l_0

        p_uniform = np.full_like(x, 1.0 / (self.W / 2.0 * self.H))
        p_band = np.where(np.abs(y) < band_w, 1.0 / (self.W / 2.0 * 2.0 * band_w), 0.0)
        r2 = (x - self.a) ** 2 + y ** 2
        p_tip = np.where(r2 < tip_r ** 2, 1.0 / (np.pi * tip_r ** 2), 0.0)
        p_symband = (2.0 / (l_0 * np.sqrt(2.0 * np.pi))) * np.exp(-x ** 2 / (2.0 * l_0 ** 2)) * (1.0 / self.H)

        p_mix = 0.30 * p_uniform + 0.25 * p_band + 0.25 * p_tip + 0.20 * p_symband
        return p_mix

    # ------------------------------------------------------------------
    # C6: deterministic, tip-graded quadrature grid for the DEM energy
    # integral -- see docs/griffith-drm-insights-experiment-plan.md §2.2
    # (candidate C6) and PFPhysicsEvaluator.compute_u_loss_dem().
    # ------------------------------------------------------------------
    def sample_dem_quadrature(self, n_bg_x: int = 100, n_bg_y: int = 100,
                               n_tip_r: int = 30, n_tip_theta: int = 60,
                               l_0: float = 0.01, r_tip_factor: float = 4.0,
                               y_grade_power: float = 3.0):
        """
        Deterministic replacement for sample_fixed_domain() +
        compute_mixture_density()'s random importance-weighted Monte Carlo
        scheme (candidate C6). u_net/phi_net are still evaluated pointwise
        via plain forward passes + autograd -- no shape functions, no mesh,
        no field interpolation -- only the *integration rule* changes, from
        random MC sampling with 1/p importance weights to a fixed structured
        grid with explicit area quadrature weights.

        Motivation (see plan doc for the full writeup): under random
        importance-weighted MC, a point drawn from the densely-sampled
        crack-tip region gets a small 1/p weight so the estimator stays
        unbiased in expectation, but at any *finite* batch size this adds
        sampling noise on top of the (already small, since the tip disk is
        a tiny fraction of the domain's area) tip contribution to the total
        energy -- diluting the training signal from exactly the region whose
        accuracy matters most for the near-tip diagnostics. A deterministic
        grid removes the MC noise entirely (same points every step, no
        variance to average out), which is the actual root cause fix,
        without reintroducing FEM shape functions/elements/mesh-based field
        representation (Manav et al. 2024's own approach, which this project
        deliberately does NOT adopt -- see plan doc's C5 entry).

        Construction (right-half domain x ∈ [0, W/2], y ∈ [-H/2, H/2]):
          - Background: a Cartesian tensor grid covering the full rectangle,
            uniform in x, graded (power-law, finer near y=0) in y so the
            crack band -- which runs the full x-range, not just near the tip
            -- still gets extra resolution. Grid points that fall inside the
            crack-tip disk (see below) are dropped to avoid double-counting;
            each kept point's quadrature weight is its Cartesian cell area
            dx * dy_j (midpoint/box rule).
          - Crack tip: a polar grid centered at (a, 0), radius up to
            r_tip_factor * l_0, with geometrically graded radial edges
            (finer near r=0) and uniform angular spacing. Each node's weight
            is the exact area of its annular-sector cell,
            0.5*(r_out^2 - r_in^2)*dtheta -- exact, not an approximation,
            regardless of the radial grading.

        Returns
        -------
        pts_phys : (N, 2) torch tensor, physical (x, y) quadrature nodes.
        weights_phys : (N, 1) torch tensor, area quadrature weights (units
            of physical area). sum(weights_phys) approximates the right-half
            domain area (W/2)*H, up to the small stair-step error where the
            uniform/graded background grid meets the tip disk boundary.

        NOTE on plugging into compute_u_loss_dem(): that function expects
        weights_nd such that torch.mean(weights_nd * psi) approximates the
        raw energy integral ∫ psi dΩ_phys (see its docstring) -- i.e. the
        same convention as weights_nd = 1/p_mixture used for the random MC
        scheme. Since torch.mean divides by N, callers must rescale:
            weights_nd = weights_phys * N   (N = pts_phys.shape[0])
        so that torch.mean(weights_nd * psi) == sum(weights_phys * psi).
        """
        r_tip = r_tip_factor * l_0

        # ---- background: uniform in x, graded (finer near y=0) in y ----
        x_edges = np.linspace(0.0, self.W / 2.0, n_bg_x + 1)
        x_mid = 0.5 * (x_edges[:-1] + x_edges[1:])
        dx = np.diff(x_edges)   # (n_bg_x,) -- uniform, but kept general

        half = self.H / 2.0
        n_half = max(1, n_bg_y // 2)
        s = np.linspace(0.0, 1.0, n_half + 1) ** y_grade_power
        half_edges = half * s   # 0 .. H/2, finer near 0
        y_edges = np.concatenate([-half_edges[::-1], half_edges[1:]])
        y_mid = 0.5 * (y_edges[:-1] + y_edges[1:])
        dy = np.diff(y_edges)

        Xg, Yg = np.meshgrid(x_mid, y_mid, indexing='ij')
        Dxg, Dyg = np.meshgrid(dx, dy, indexing='ij')
        bg_pts = np.hstack([Xg.reshape(-1, 1), Yg.reshape(-1, 1)])
        bg_w = (Dxg * Dyg).reshape(-1, 1)

        inside_tip = (bg_pts[:, 0:1] - self.a) ** 2 + bg_pts[:, 1:2] ** 2 < r_tip ** 2
        keep = ~inside_tip.flatten()
        bg_pts = bg_pts[keep]
        bg_w = bg_w[keep]

        # ---- crack-tip disk: polar grid, geometrically graded radius ----
        p = 2.0   # grading power, >1 concentrates radial edges near r=0
        k = np.arange(0, n_tip_r + 1)
        r_edges = r_tip * (k / n_tip_r) ** p
        r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])

        theta_edges = np.linspace(-np.pi, np.pi, n_tip_theta + 1)
        theta_mid = 0.5 * (theta_edges[:-1] + theta_edges[1:])
        dtheta = theta_edges[1] - theta_edges[0]

        Rm, Tm = np.meshgrid(r_mid, theta_mid, indexing='ij')
        r_out2 = np.broadcast_to((r_edges[1:] ** 2)[:, None], Rm.shape)
        r_in2 = np.broadcast_to((r_edges[:-1] ** 2)[:, None], Rm.shape)
        tip_x = self.a + Rm * np.cos(Tm)
        tip_y = Rm * np.sin(Tm)
        tip_w = 0.5 * (r_out2 - r_in2) * dtheta   # exact annular-sector area

        tip_pts = np.hstack([tip_x.reshape(-1, 1), tip_y.reshape(-1, 1)])
        tip_w = tip_w.reshape(-1, 1)

        inside_domain = (tip_pts[:, 0:1] >= 0.0) & (tip_pts[:, 0:1] <= self.W / 2.0) & \
                        (np.abs(tip_pts[:, 1:2]) <= self.H / 2.0)
        inside_domain = inside_domain.flatten()
        tip_pts = tip_pts[inside_domain]
        tip_w = tip_w[inside_domain]

        pts = np.vstack([bg_pts, tip_pts])
        weights = np.vstack([bg_w, tip_w])

        pts_t = torch.tensor(pts, dtype=torch.get_default_dtype(), device=self.device)
        w_t = torch.tensor(weights, dtype=torch.get_default_dtype(), device=self.device)
        return pts_t, w_t

    # ------------------------------------------------------------------
    # Initial crack seed points  (right half: x ∈ [0, a])
    # ------------------------------------------------------------------
    def get_initial_crack_points(self, num_points):
        """
        Points on y=0, x ∈ [0, a]  (right half of crack line).
        Pre-conditioning these with large H0 drives phi → 1 on the crack.
        """
        x = np.random.uniform(0.0, self.a, (num_points, 1))
        y = np.zeros((num_points, 1))
        pts = torch.tensor(np.hstack([x, y]), dtype=torch.get_default_dtype(), device=self.device)
        return pts

    # ------------------------------------------------------------------
    # Crack-face boundary points  (still useful for explicit BC enforcement)
    # ------------------------------------------------------------------
    def sample_crack_face(self, num_points: int, offset: float = 1e-5):
        """
        Points on y = ±offset, x ∈ [0, a] (upper and lower crack faces, right half).
        Used to explicitly enforce σ_yy = τ_xy = 0 on the crack faces.
        """
        n_half = num_points // 2
        x = np.random.uniform(0.0, self.a, (n_half, 1))
        pts_up = np.hstack([x, np.full((n_half, 1), +offset)])
        pts_dn = np.hstack([x, np.full((n_half, 1), -offset)])
        pts = np.vstack([pts_up, pts_dn])
        return torch.tensor(pts, dtype=torch.get_default_dtype(), device=self.device)

    # ------------------------------------------------------------------
    # Symmetry axis points  x = 0,  y ∈ [-H/2, H/2]
    # ------------------------------------------------------------------
    def sample_symmetry_axis(self, num_points: int):
        """
        Points on x = 0 for enforcing the symmetry BC:
            u = 0  (no horizontal displacement across symmetry plane)
            τ_xy = 0  (no shear on symmetry plane)
        """
        x = np.zeros((num_points, 1))
        y = np.random.uniform(-self.H/2, self.H/2, (num_points, 1))
        pts = torch.tensor(np.hstack([x, y]), dtype=torch.get_default_dtype(), device=self.device)
        return pts

    # ------------------------------------------------------------------
    # External boundaries  (right half domain)
    # ------------------------------------------------------------------
    def sample_boundaries(self, num_points_per_edge):
        """
        Top  : y = +H/2,  x ∈ [0, W/2]   → v = +v_max, u = 0
        Bottom: y = -H/2,  x ∈ [0, W/2]   → v = -v_max, u = 0
        Right : x = +W/2,  y ∈ [-H/2, H/2] → σ_xx = τ_xy = 0  (free surface)
        Center: (0, 0)                       → extra fix for rigid body (precautionary)
        """
        n = num_points_per_edge

        def _t(x_arr, y_arr):
            return torch.tensor(np.hstack([x_arr, y_arr]),
                                dtype=torch.get_default_dtype(), device=self.device)

        top_pts   = _t(np.random.uniform(0.0, self.W/2, (n, 1)), np.full((n, 1), +self.H/2))
        bot_pts   = _t(np.random.uniform(0.0, self.W/2, (n, 1)), np.full((n, 1), -self.H/2))
        right_pts = _t(np.full((n, 1), self.W/2),  np.random.uniform(-self.H/2, self.H/2, (n, 1)))
        center_pt = torch.tensor([[0.0, 0.0]], dtype=torch.get_default_dtype(), device=self.device)

        return {
            'top':    top_pts,
            'bottom': bot_pts,
            'right':  right_pts,
            'center': center_pt,
        }
