import torch
import torch.nn as nn

def cartesian_to_elliptic(pts_nd, a_nd):
    """
    pts_nd: (N, 2) tensor of non-dimensional coordinates (x*, y*)
    a_nd: non-dimensional crack half-length (which is 1.0 since L0 = a)
    Returns: mu, nu tensors of shape (N, 1)
    """
    x = pts_nd[:, 0:1]
    y = pts_nd[:, 1:2]
    
    # 物理启发：将坐标系在一个假想的 z 轴上偏移极小量 sqrt(eps)
    # 这从几何上保证了点永远不会"正好"落在奇异线上，从而让
    # acosh(1.0) 和 acos(+-1.0) 对应的无穷大导数被完美截断（不再出现 NaN）
    eps = 1e-4  
    
    L1 = torch.sqrt((x + a_nd)**2 + y**2 + eps)
    L2 = torch.sqrt((x - a_nd)**2 + y**2 + eps)
    
    # 因为存在 eps 补偿，这里几何上保证了 val_mu 严格 > 1.0，val_nu 严格在 (-1, 1) 内
    val_mu = (L1 + L2) / (2.0 * a_nd)
    val_nu = (L1 - L2) / (2.0 * a_nd)
    
    mu = torch.acosh(val_mu)
    
    # 恢复 Branch Cut (分支切割)！
    # 如果仅使用 acos，nu 的范围始终在 [0, pi]，这意味着网络在 y>0 和 y<0 看到的输入完全一样，
    # 导致网络强制输出在 y 轴对称的连续位移，裂纹永远无法“张开”！
    # 我们必须使用 atan2(sin_nu, cos_nu) 来让 nu 具备方向性 [-pi, pi]。
    y_sign = torch.sign(y)
    # 对于恰好落在 y=0 的边界点，默认将其归入上裂纹面 (y_sign = 1)
    y_sign = torch.where(y == 0.0, torch.ones_like(y_sign), y_sign)
    
    # sin(nu) 的大小为 sqrt(1 - cos^2(nu))，符号与 y_sign 相同
    sin_nu = y_sign * torch.sqrt(torch.clamp(1.0 - val_nu**2, min=1e-8))
    
    nu = torch.atan2(sin_nu, val_nu)
    
    return mu, nu


def williams_features(pts_nd, a_nd=1.0, n_modes=4, eps=1e-6, r_floor=0.0):
    """
    Williams eigenfunction expansion features around the right crack tip (a_nd, 0).

    Provides the network with the *shape* of the leading-order asymptotic
    displacement modes without hard-coding their amplitude (K_I is still
    inferred by the network from the PDE residual). See
    docs/griffith-next-steps.md §步骤2 for the derivation.

    pts_nd   : (N, 2) non-dimensional coordinates (x*, y*)
    a_nd     : non-dimensional crack half-length (1.0 since L0 = a)
    n_modes  : number of leading modes to return (<= 4)
    r_floor  : additional radial floor (same non-dimensional length units as
               pts_nd). Points with true radius r_raw < r_floor are smoothly
               pushed outward to r_floor (their angle theta is preserved
               exactly), which caps d(f_i)/d(x,y) at O(1/sqrt(r_floor))
               instead of diverging as r_raw -> 0. r_floor=0 (default)
               recovers the original, mathematically-exact features.
               See experiments/debug_williams_sign.py, which measured a
               ~64-86x gradient amplification near the tip with r_floor=0
               and confirmed the feature *definition* itself is not buggy
               (it exactly spans the analytical field with correct signs) --
               the instability comes purely from this gradient blow-up.
    Returns  : (N, n_modes) tensor
    """
    x = pts_nd[:, 0:1] - a_nd
    y = pts_nd[:, 1:2]
    r_raw = torch.sqrt(x**2 + y**2 + eps)

    if r_floor > 0:
        r = r_raw + r_floor
        scale = r / r_raw
        x = x * scale
        y = y * scale
    else:
        r = r_raw

    theta = torch.atan2(y, x)

    sqrt_r = torch.sqrt(r)
    f1 = sqrt_r * torch.cos(theta / 2)
    f2 = sqrt_r * torch.sin(theta / 2)
    f3 = sqrt_r * torch.sin(theta / 2) * torch.sin(theta)
    f4 = sqrt_r * torch.cos(theta / 2) * torch.sin(theta)

    feats = [f1, f2, f3, f4]
    return torch.cat(feats[:n_modes], dim=1)


def get_u_in_dim(coord_scheme):
    """Input feature dimension of DisplacementNet for a given coord_scheme."""
    dims = {'cartesian': 2, 'elliptic': 2, 'both': 4, 'williams': 6}
    if coord_scheme not in dims:
        raise ValueError(f"Unknown coord_scheme: '{coord_scheme}'")
    return dims[coord_scheme]


class PFPhysicsEvaluator(nn.Module):
    """
    Phase-Field PINN Evaluator for Griffith crack problem.

    All internal computations are performed in NON-DIMENSIONAL form:
        - Reference length:           L0 = a  (crack half-length)
        - Reference stress/modulus:   E0 = E  (Young's modulus)
        - Reference energy density:   H0 = G_c / l_0

    Non-dimensional variables used internally:
        x*  = x / L0,   u*  = u / L0,   eps* = eps (unchanged)
        sig* = sig / E0, H*  = H / H0,   l0* = l0 / L0

    This collapses all physical quantities to O(1) and removes the
    ~10^10 scale mismatch that causes the network to degenerate.
    """

    def __init__(self, u_net, phi_net, E, nu, G_c, l_0, a, k=1e-5, coord_scheme='both',
                 williams_r_floor=0.0, y_symmetric=False, H=None, hard_dirichlet=False):
        super().__init__()
        self.u_net = u_net
        self.phi_net = phi_net

        # Feature scheme fed to DisplacementNet: 'cartesian' | 'elliptic' | 'both' | 'williams'
        # See docs/griffith-ablation-study.md §三 factor A for definitions.
        self.coord_scheme = coord_scheme

        # Near-tip gradient damping for coord_scheme='williams' (non-dimensional
        # length units, i.e. a fraction of the crack half-length a). See
        # williams_features()'s r_floor docstring for rationale.
        self.williams_r_floor = williams_r_floor

        # Architecturally enforce sigma_yy(x,y) == sigma_yy(x,-y) (u even in y,
        # v odd in y) via _eval_uv_raw() below, instead of relying on the
        # network to learn this symmetry unconstrained. See
        # docs/griffith-ablation-study.md §9 (L4 tier-2 factors): an n=8-seed
        # comparison at coord_scheme='elliptic' showed this significantly
        # improves l2_rel_err (0.503->0.353, |t|~3.35) and trivially drives
        # sym_err to exactly 0.0, with no statistically significant change to
        # peak_loc_err or peak_mag_ratio. Originally prototyped as a separate
        # subclass in experiments/y_symmetric_variant.py.
        self.y_symmetric = y_symmetric

        # Physical parameters (stored for reference / post-processing)
        self.E = E
        self.nu = nu
        self.G_c = G_c
        self.l_0 = l_0
        self.a = a
        self.k = k

        # Reference scales derived from (a, E, G_c, l_0)
        self.L0 = a                 # reference length [mm]
        self.E0 = E                 # reference stress [MPa]
        self.H0 = G_c / l_0        # reference energy density [MPa]

        # Architecturally enforce v*(x, +-H*/2) == +-v_max* (top/bottom
        # Dirichlet displacement BC) via _eval_uv_raw() below, instead of the
        # soft-penalty loss_top/loss_bot terms in compute_u_loss(). Uses the
        # clamped-envelope trial function
        #     v*(x,y) = (y*/halfH*) * v_max* + (1 - (y*/halfH*)^2)^2 * v_net*(x,y)
        # which equals +-v_max* exactly at y* = +-halfH* for ANY raw network
        # output v_net*, for all coord_scheme/arch choices. The envelope is
        # SQUARED so that its y-derivative also vanishes at y*=+-halfH* (not
        # just its value) -- otherwise eps_yy=dv*/dy leaks an unregularized
        # v_net*(x,+-halfH*) term right at the boundary, which was found to
        # produce a spurious sigma_yy peak at the domain corners (see
        # docs/griffith-ablation-study.md §9). This is odd in y*
        # whenever v_net* is odd in y* (true when y_symmetric=True), so the
        # two flags compose correctly. v_max* is dynamic (changes every
        # quasi-static load step) and is refreshed by compute_u_loss() via
        # self.v_max_nd every call; it is "sticky" afterwards so that
        # post-training evaluation code (e.g. metrics.py's evaluate_on_grid,
        # which never calls compute_u_loss) sees the last load applied during
        # training. See docs/griffith-ablation-study.md §9 for background.
        self.hard_dirichlet = hard_dirichlet
        self.half_H_nd = (H / 2.0) / self.L0 if H is not None else None
        if hard_dirichlet and self.half_H_nd is None:
            raise ValueError("hard_dirichlet=True requires H (plate height) to be passed")
        self.v_max_nd = 0.0

        # Non-dimensional moduli  (both O(0.1 ~ 0.5))
        self.K_nd = 1.0 / (3.0 * (1.0 - 2.0 * nu))   # K* = K/E
        self.mu_nd = 1.0 / (2.0 * (1.0 + nu))         # mu* = mu/E

        # Non-dimensional length scale  (l0* = l0/a)
        self.l0_nd = l_0 / a

    # ------------------------------------------------------------------
    # Helper: build the DisplacementNet input tensor according to coord_scheme
    # ------------------------------------------------------------------
    def _build_u_input(self, pts_nd):
        scheme = self.coord_scheme
        if scheme == 'cartesian':
            return pts_nd
        mu, nu = cartesian_to_elliptic(pts_nd, 1.0)
        if scheme == 'elliptic':
            return torch.cat([mu, nu], dim=1)
        if scheme == 'both':
            return torch.cat([pts_nd, mu, nu], dim=1)
        if scheme == 'williams':
            wf = williams_features(pts_nd, a_nd=1.0, r_floor=self.williams_r_floor)
            return torch.cat([pts_nd, wf], dim=1)
        raise ValueError(f"Unknown coord_scheme: '{scheme}'")

    # ------------------------------------------------------------------
    # Helper: raw (u*, v*) network evaluation, factored out of
    # compute_strains() and compute_u_loss()'s Dirichlet/rigid-body terms so
    # that self.y_symmetric can apply consistently everywhere u_net is
    # queried without duplicating compute_u_loss. When y_symmetric=True, the
    # network is evaluated at (x,y) and its raw-coordinate mirror (x,-y) (both
    # differentiable functions of the same pts_nd, so autograd threads
    # gradients through both branches correctly) and combined so that u is
    # forced even in y and v is forced odd in y -- this holds for any
    # coord_scheme since features are rebuilt from scratch for the mirrored
    # point via _build_u_input(), with no per-scheme parity bookkeeping
    # required. Base (y_symmetric=False) behavior: identical to calling
    # self.u_net(self._build_u_input(pts)) directly.
    # ------------------------------------------------------------------
    def _eval_uv_raw(self, pts_nd):
        if not self.y_symmetric:
            out = self.u_net(self._build_u_input(pts_nd))
        else:
            x = pts_nd[:, 0:1]
            y = pts_nd[:, 1:2]
            pts_mirror = torch.cat([x, -y], dim=1)

            out_pos = self.u_net(self._build_u_input(pts_nd))
            out_neg = self.u_net(self._build_u_input(pts_mirror))

            u_nd = 0.5 * (out_pos[:, 0:1] + out_neg[:, 0:1])
            v_nd = 0.5 * (out_pos[:, 1:2] - out_neg[:, 1:2])
            out = torch.cat([u_nd, v_nd], dim=1)

        if not self.hard_dirichlet:
            return out

        # Clamp v* to exactly +-v_max* at y* = +-half_H_nd (see __init__
        # docstring). u* is left untouched -- only v is hard-constrained.
        #
        # The envelope B(y) = (1 - y_ratio**2)**2 is SQUARED (rather than the
        # bare (1 - y_ratio**2)) so that BOTH B(+-half_H_nd)=0 *and*
        # B'(+-half_H_nd)=0. With the un-squared envelope, eps_yy = dv*/dy
        # picks up a residual B'(y)*v_net term that does NOT vanish at the
        # clamped edge (since B'(+-1) != 0 there even though B(+-1)=0), and
        # v_net is otherwise completely unconstrained at y=+-half_H_nd (no
        # loss term touches it there once loss_top/loss_bot are removed) --
        # this let stray/unregularized v_net values leak into sigma_yy right
        # at the domain corners, producing a spurious global argmax there
        # that corrupted the peak_loc_err metric (traced to an exact
        # sqrt(0.61) match against the far corner (x=-W/2, y=+-H/2) in an
        # n=6-seed y_sym_hard run -- see docs/griffith-ablation-study.md §9).
        # Squaring makes B'(+-1)=0 too, so v_net's value AND its y-gradient
        # are both suppressed at the exact boundary, eliminating this failure
        # mode while still satisfying the Dirichlet BC exactly.
        u_nd = out[:, 0:1]
        v_net = out[:, 1:2]
        y = pts_nd[:, 1:2]
        y_ratio = y / self.half_H_nd
        envelope = (1.0 - y_ratio**2)**2
        v_hard = y_ratio * self.v_max_nd + envelope * v_net
        return torch.cat([u_nd, v_hard], dim=1)

    # ------------------------------------------------------------------
    # Helper: compute engineering strains from network output (u, v)
    # Everything is already in non-dimensional coordinates because the
    # network is fed x* = x/L0 and outputs u* = u/L0.
    # ------------------------------------------------------------------
    def compute_strains(self, pts_nd):
        """
        pts_nd : (N, 2) tensor of NON-DIMENSIONAL coordinates (x*, y*)
        Returns eps_xx, eps_yy, eps_xy (all dimensionless),
                plus pts_nd (with grad), u*, v*, and NN predicted stresses (None for Standard PINN)
        """
        pts_nd = pts_nd.clone().requires_grad_(True)
        
        # Input to DisplacementNet depends on self.coord_scheme (see _build_u_input)
        out_nd = self._eval_uv_raw(pts_nd)  # network outputs u*, v*
        u_nd = out_nd[:, 0:1]
        v_nd = out_nd[:, 1:2]
        
        # Standard PINN has no stress output
        sig_xx_nn = None
        sig_yy_nn = None
        tau_xy_nn = None

        grad_u = torch.autograd.grad(
            u_nd.sum(), pts_nd, create_graph=True)[0]
        grad_v = torch.autograd.grad(
            v_nd.sum(), pts_nd, create_graph=True)[0]

        eps_xx = grad_u[:, 0:1]
        eps_yy = grad_v[:, 1:2]
        eps_xy = 0.5 * (grad_u[:, 1:2] + grad_v[:, 0:1])

        return eps_xx, eps_yy, eps_xy, pts_nd, u_nd, v_nd, sig_xx_nn, sig_yy_nn, tau_xy_nn

    # ------------------------------------------------------------------
    # Helper: Amor volumetric-deviatoric energy split + degraded stress
    # Uses non-dimensional moduli K*, mu*; returns sig* = sig/E0
    # ------------------------------------------------------------------
    def compute_energy_and_stress(self, eps_xx, eps_yy, eps_xy, phi):
        """
        Returns:
            psi_nd  : non-dimensional active strain energy H* = psi_e+ / H0
            sig_xx* : non-dimensional stress
            sig_yy* : non-dimensional stress
            tau_xy* : non-dimensional stress
        """
        K_nd = self.K_nd
        mu_nd = self.mu_nd

        eps_vol = eps_xx + eps_yy
        eps_vol_pos = torch.relu(eps_vol)
        eps_vol_neg = -torch.relu(-eps_vol)

        # Deviatoric part (plane-strain: e_zz = -eps_vol / 3 in 3D sense)
        e_xx = eps_xx - (1.0 / 3.0) * eps_vol
        e_yy = eps_yy - (1.0 / 3.0) * eps_vol
        e_zz = -(1.0 / 3.0) * eps_vol          # out-of-plane deviatoric (plane strain)
        tr_e2 = e_xx**2 + e_yy**2 + e_zz**2 + 2.0 * eps_xy**2

        # Active strain energy density psi_e+ (in units of E0):
        psi_nd = 0.5 * K_nd * eps_vol_pos**2 + mu_nd * tr_e2

        g_phi = (1.0 - phi)**2 + self.k

        # Degraded stress components (Amor split), in units of E0:
        sig_xx = g_phi * (K_nd * eps_vol_pos + 2.0 * mu_nd * e_xx) + K_nd * eps_vol_neg
        sig_yy = g_phi * (K_nd * eps_vol_pos + 2.0 * mu_nd * e_yy) + K_nd * eps_vol_neg
        tau_xy = g_phi * (2.0 * mu_nd * eps_xy)

        return psi_nd, sig_xx, sig_yy, tau_xy

    # ------------------------------------------------------------------
    # Loss 1: DisplacementNet loss (PhaseFieldNet is frozen / detached)
    # ------------------------------------------------------------------
    def compute_u_loss(self, pts_nd, bc_dict_nd, v_max_nd, w_pde, w_bc,
                       crack_face_nd=None, w_crack_face=10.0,
                       sym_axis_nd=None, w_sym=50.0):
        """
        pts_nd         : (N, 2) domain collocation points in non-dim coords (right half: x≥0)
        bc_dict_nd     : dict of non-dim BC point tensors
        v_max_nd       : v_bar* = v_bar / L0
        crack_face_nd  : crack-face points for σ_yy = τ_xy = 0
        sym_axis_nd    : (M, 2) points on x=0 for symmetry BC: u=0, τ_xy=0
        w_sym          : weight for symmetry axis penalty (high: hard constraint)
        """
        # Refresh the dynamic load for _eval_uv_raw()'s hard_dirichlet clamp
        # (no-op when hard_dirichlet=False). "Sticky" afterwards -- see
        # __init__ docstring.
        self.v_max_nd = v_max_nd

        eps_xx, eps_yy, eps_xy, x_y_nd, _, _, _, _, _ = self.compute_strains(pts_nd)
        phi = self.phi_net(x_y_nd).detach()

        _, sig_xx, sig_yy, tau_xy = self.compute_energy_and_stress(
            eps_xx, eps_yy, eps_xy, phi)

        # 1. Equilibrium Equation (2nd-order derivatives on NN displacement)
        dsig_xx_dx = torch.autograd.grad(
            sig_xx.sum(), x_y_nd, create_graph=True)[0][:, 0:1]
        dtau_xy_dy = torch.autograd.grad(
            tau_xy.sum(), x_y_nd, create_graph=True)[0][:, 1:2]
        dtau_xy_dx = torch.autograd.grad(
            tau_xy.sum(), x_y_nd, create_graph=True)[0][:, 0:1]
        dsig_yy_dy = torch.autograd.grad(
            sig_yy.sum(), x_y_nd, create_graph=True)[0][:, 1:2]

        res_x = dsig_xx_dx + dtau_xy_dy
        res_y = dtau_xy_dx + dsig_yy_dy
        loss_equil = torch.mean(res_x**2 + res_y**2)

        loss_pde = w_pde * loss_equil
        
        def eval_u(pts):
            return self._eval_uv_raw(pts)

        # ---- Dirichlet: top / bottom displacement ----
        if self.hard_dirichlet:
            # v*(x, +-half_H_nd) = +-v_max* is satisfied exactly by
            # construction in _eval_uv_raw(); no soft-penalty term needed.
            loss_top = torch.tensor(0.0, device=pts_nd.device)
            loss_bot = torch.tensor(0.0, device=pts_nd.device)
        else:
            pts_top = bc_dict_nd['top'].clone().requires_grad_(True)
            uv_top  = eval_u(pts_top)
            loss_top = torch.mean((uv_top[:, 1:2] - v_max_nd)**2)

            pts_bot = bc_dict_nd['bottom'].clone().requires_grad_(True)
            uv_bot  = eval_u(pts_bot)
            loss_bot = torch.mean((uv_bot[:, 1:2] + v_max_nd)**2)

        # ---- Neumann: right free surface σ_xx = τ_xy = 0 ----
        pts_right = bc_dict_nd['right'].clone().requires_grad_(True)
        eps_xx_r, eps_yy_r, eps_xy_r, _, _, _, _, _, _ = self.compute_strains(pts_right)
        phi_r = self.phi_net(pts_right).detach()
        _, sxx_r, _, txy_r = self.compute_energy_and_stress(eps_xx_r, eps_yy_r, eps_xy_r, phi_r)
        loss_right = torch.mean(sxx_r**2 + txy_r**2)

        # ---- Rigid-body fix: u*(0, 0) = 0 ----
        pts_c = bc_dict_nd['center']
        uv_c  = eval_u(pts_c)
        loss_center = torch.mean(uv_c[:, 0:1]**2)

        # ---- Symmetry axis x=0: u=0, τ_xy=0 ----
        loss_sym = torch.tensor(0.0, device=pts_nd.device)
        if sym_axis_nd is not None and sym_axis_nd.numel() > 0:
            pts_sa = sym_axis_nd.clone().requires_grad_(True)
            # Need τ_xy, so we must compute strains
            eps_xx_sa, eps_yy_sa, eps_xy_sa, _, u_sa, _, _, _, _ = self.compute_strains(pts_sa)
            phi_sa = self.phi_net(pts_sa).detach()
            _, _, _, txy_sa = self.compute_energy_and_stress(eps_xx_sa, eps_yy_sa, eps_xy_sa, phi_sa)
            loss_sym = torch.mean(u_sa**2 + txy_sa**2)

        # ---- Crack-face traction-free: σ_yy = τ_xy = 0 ----
        loss_crack_face = torch.tensor(0.0, device=pts_nd.device)
        if crack_face_nd is not None and crack_face_nd.numel() > 0:
            pts_cf = crack_face_nd.clone().requires_grad_(True)
            eps_xx_cf, eps_yy_cf, eps_xy_cf, _, _, _, _, _, _ = self.compute_strains(pts_cf)
            phi_cf = self.phi_net(pts_cf).detach()
            _, _, syy_cf, txy_cf = self.compute_energy_and_stress(eps_xx_cf, eps_yy_cf, eps_xy_cf, phi_cf)
            loss_crack_face = torch.mean(syy_cf**2 + txy_cf**2)

        loss_bc = (loss_top + loss_bot + loss_center
                   + 0.1 * loss_right
                   + w_sym * loss_sym
                   + w_crack_face * loss_crack_face)

        total_loss = loss_pde + w_bc * loss_bc
        return total_loss, loss_equil.item(), loss_bc.item(), loss_crack_face.item(), loss_sym.item()

    # ------------------------------------------------------------------
    # Loss 1b: DisplacementNet loss, Deep Energy Method (DEM) variant.
    #
    # Minimizes the total potential energy functional directly instead of
    # the strong-form equilibrium PDE residual, so only FIRST spatial
    # derivatives of (u, v) are needed (no create_graph=True second
    # differentiation) -- see docs/griffith-ablation-study.md §12 for the
    # variational derivation and rationale (this is tier-3 candidate C3,
    # "Deep Energy Method", promoted out of the candidate list because the
    # strong-form second-derivative cost was identified as the main
    # bottleneck for scaling up the full coupled model).
    #
    # Energy density (matches compute_energy_and_stress()'s Amor split
    # exactly, confirmed variationally consistent -- i.e. stress = d(psi)/d(eps)
    # -- by construction, so this is NOT the "hybrid model" variational-
    # inconsistency trap some DEM literature warns about):
    #     psi = g(phi) * psi_plus + psi_minus
    #     psi_plus  = 0.5*K*eps_vol_pos^2 + mu*tr(e_dev^2)   (degradable)
    #     psi_minus = 0.5*K*eps_vol_neg^2                    (never degraded)
    #
    # Traction-free Neumann BCs (right free edge, crack faces) are NATURAL
    # boundary conditions of this functional and are satisfied automatically
    # by energy minimization -- no penalty terms are needed for them at all
    # (this is the other headline DEM benefit besides the derivative-order
    # reduction). Only genuine Dirichlet BCs (top/bottom v, symmetry-axis u)
    # still need explicit enforcement, same as compute_u_loss().
    # ------------------------------------------------------------------
    def compute_u_loss_dem(self, pts_nd, weights_nd, bc_dict_nd, v_max_nd, w_bc,
                            sym_axis_nd=None, w_sym=50.0):
        """
        weights_nd : (N, 1) importance weights = 1 / p_mixture(x_i, y_i),
            where p_mixture is GriffithGeometrySampler.compute_mixture_density()
            evaluated at the same points as pts_nd. Required because
            sample_fixed_domain() draws from a non-uniform mixture (crack-band
            / tip / symmetry-axis refinement), so a plain torch.mean(psi)
            would be a biased estimate of the domain integral -- weighting by
            1/p turns it back into an unbiased Monte Carlo estimator of
            (1/|Omega|) * integral(psi) dOmega (up to the same constant
            |Omega| normalization compute_u_loss's PDE-residual mean() also
            implicitly uses, so w_bc/w_pde-style tuning transfers unchanged).

        Returns the same 5-tuple shape as compute_u_loss() for drop-in
        compatibility with the training loop / result-logging code:
        (total_loss, energy.item(), bc_u.item(), 0.0, sym_u.item()).
        NOTE: the second element is the total potential ENERGY value, not a
        PDE residual -- unlike compute_u_loss()'s loss_equil, it is not
        expected to shrink to ~0 near convergence, so 'equil_final' in
        metrics/CSVs means something different (and is not numerically
        comparable) for DEM runs vs strong-form runs.
        """
        self.v_max_nd = v_max_nd

        eps_xx, eps_yy, eps_xy, x_y_nd, _, _, _, _, _ = self.compute_strains(pts_nd)
        phi = self.phi_net(x_y_nd).detach()

        K_nd, mu_nd = self.K_nd, self.mu_nd
        eps_vol = eps_xx + eps_yy
        eps_vol_pos = torch.relu(eps_vol)
        eps_vol_neg = -torch.relu(-eps_vol)
        e_xx = eps_xx - (1.0 / 3.0) * eps_vol
        e_yy = eps_yy - (1.0 / 3.0) * eps_vol
        e_zz = -(1.0 / 3.0) * eps_vol
        tr_e2 = e_xx**2 + e_yy**2 + e_zz**2 + 2.0 * eps_xy**2

        psi_plus = 0.5 * K_nd * eps_vol_pos**2 + mu_nd * tr_e2
        psi_minus = 0.5 * K_nd * eps_vol_neg**2
        g_phi = (1.0 - phi)**2 + self.k
        psi_total = g_phi * psi_plus + psi_minus

        energy = torch.mean(weights_nd * psi_total)

        def eval_u(pts):
            return self._eval_uv_raw(pts)

        # ---- Dirichlet: top / bottom displacement ----
        if self.hard_dirichlet:
            loss_top = torch.tensor(0.0, device=pts_nd.device)
            loss_bot = torch.tensor(0.0, device=pts_nd.device)
        else:
            pts_top = bc_dict_nd['top'].clone().requires_grad_(True)
            uv_top = eval_u(pts_top)
            loss_top = torch.mean((uv_top[:, 1:2] - v_max_nd)**2)

            pts_bot = bc_dict_nd['bottom'].clone().requires_grad_(True)
            uv_bot = eval_u(pts_bot)
            loss_bot = torch.mean((uv_bot[:, 1:2] + v_max_nd)**2)

        # ---- Rigid-body fix: u*(0, 0) = 0 ----
        pts_c = bc_dict_nd['center']
        uv_c = eval_u(pts_c)
        loss_center = torch.mean(uv_c[:, 0:1]**2)

        # ---- Symmetry axis x=0: u=0, τ_xy=0 ----
        # (u=0 is a genuine Dirichlet condition; τ_xy=0 here is also kept as
        # a soft penalty for simplicity, even though on a true symmetry plane
        # it too would be natural under a symmetrized functional -- not worth
        # the extra bookkeeping for this first DEM implementation.)
        loss_sym = torch.tensor(0.0, device=pts_nd.device)
        if sym_axis_nd is not None and sym_axis_nd.numel() > 0:
            pts_sa = sym_axis_nd.clone().requires_grad_(True)
            eps_xx_sa, eps_yy_sa, eps_xy_sa, _, u_sa, _, _, _, _ = self.compute_strains(pts_sa)
            phi_sa = self.phi_net(pts_sa).detach()
            _, _, _, txy_sa = self.compute_energy_and_stress(eps_xx_sa, eps_yy_sa, eps_xy_sa, phi_sa)
            loss_sym = torch.mean(u_sa**2 + txy_sa**2)

        loss_bc = loss_top + loss_bot + loss_center + w_sym * loss_sym

        total_loss = energy + w_bc * loss_bc
        cf_placeholder = torch.tensor(0.0, device=pts_nd.device)
        return total_loss, energy.item(), loss_bc.item(), cf_placeholder.item(), loss_sym.item()

    # ------------------------------------------------------------------
    # Loss 2: PhaseFieldNet loss (DisplacementNet is frozen)
    # ------------------------------------------------------------------
    def compute_phi_loss(self, pts_nd, history_H_nd, w_pde):
        """
        pts_nd       : domain collocation points (non-dim)
        history_H_nd : H* = H / H0  (non-dimensional history field, detached)

        Non-dimensional phase-field PDE:
            phi / l0*  -  l0* * Laplacian(phi)  =  2 * (1 - phi) * H*

        (G_c cancels out completely in the non-dimensional formulation)
        """
        pts_nd = pts_nd.clone().requires_grad_(True)
        phi = self.phi_net(pts_nd)

        phi_x = torch.autograd.grad(
            phi.sum(), pts_nd, create_graph=True)[0][:, 0:1]
        phi_y = torch.autograd.grad(
            phi.sum(), pts_nd, create_graph=True)[0][:, 1:2]
        phi_xx = torch.autograd.grad(
            phi_x.sum(), pts_nd, create_graph=True)[0][:, 0:1]
        phi_yy = torch.autograd.grad(
            phi_y.sum(), pts_nd, create_graph=True)[0][:, 1:2]
        laplacian_phi = phi_xx + phi_yy

        l0_nd = self.l0_nd
        # PDE residual (all terms are O(1) after non-dimensionalization)
        residual = (phi / l0_nd - l0_nd * laplacian_phi) - 2.0 * (1.0 - phi) * history_H_nd

        loss_phi = torch.mean(residual**2)
        return w_pde * loss_phi, loss_phi.item()
