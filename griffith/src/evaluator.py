import torch
import torch.nn as nn


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

    def __init__(self, u_net, phi_net, E, nu, G_c, l_0, a, k=1e-5):
        super().__init__()
        self.u_net = u_net
        self.phi_net = phi_net

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

        # Non-dimensional moduli  (both O(0.1 ~ 0.5))
        self.K_nd = 1.0 / (3.0 * (1.0 - 2.0 * nu))   # K* = K/E
        self.mu_nd = 1.0 / (2.0 * (1.0 + nu))         # mu* = mu/E

        # Non-dimensional length scale  (l0* = l0/a)
        self.l0_nd = l_0 / a

    # ------------------------------------------------------------------
    # Helper: compute engineering strains from network output (u, v)
    # Everything is already in non-dimensional coordinates because the
    # network is fed x* = x/L0 and outputs u* = u/L0.
    # ------------------------------------------------------------------
    def compute_strains(self, pts_nd):
        """
        pts_nd : (N, 2) tensor of NON-DIMENSIONAL coordinates (x*, y*)
        Returns eps_xx, eps_yy, eps_xy (all dimensionless),
                plus pts_nd (with grad), u*, v*
        """
        pts_nd = pts_nd.clone().requires_grad_(True)
        uv_nd = self.u_net(pts_nd)          # network outputs u*, v*
        u_nd = uv_nd[:, 0:1]
        v_nd = uv_nd[:, 1:2]

        grad_u = torch.autograd.grad(
            u_nd.sum(), pts_nd, create_graph=True)[0]
        grad_v = torch.autograd.grad(
            v_nd.sum(), pts_nd, create_graph=True)[0]

        eps_xx = grad_u[:, 0:1]
        eps_yy = grad_v[:, 1:2]
        eps_xy = 0.5 * (grad_u[:, 1:2] + grad_v[:, 0:1])

        return eps_xx, eps_yy, eps_xy, pts_nd, u_nd, v_nd

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
                       crack_face_nd=None, w_crack_face=10.0):
        """
        pts_nd         : (N, 2) domain collocation points in non-dim coords
        bc_dict_nd     : dict of non-dim BC point tensors
        v_max_nd       : v_bar* = v_bar / L0  (non-dimensional target displacement)
        crack_face_nd  : (M, 2) non-dim coords of crack-face points (y ≈ ±ε, |x|<a)
                         If provided, enforces σ_yy = τ_xy = 0 explicitly.
        w_crack_face   : weight for the crack-face traction-free penalty
        """
        eps_xx, eps_yy, eps_xy, x_y_nd, _, _ = self.compute_strains(pts_nd)
        phi = self.phi_net(x_y_nd).detach()   # freeze phi network

        _, sig_xx, sig_yy, tau_xy = self.compute_energy_and_stress(
            eps_xx, eps_yy, eps_xy, phi)

        # Equilibrium PDE residuals  div(sigma*) = 0  (non-dimensional)
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

        loss_pde = torch.mean(res_x**2 + res_y**2)

        # ---- Dirichlet BC: top / bottom displacement ----
        pts_top = bc_dict_nd['top'].clone().requires_grad_(True)
        uv_top = self.u_net(pts_top)
        loss_top = torch.mean((uv_top[:, 1:2] - v_max_nd)**2)

        pts_bot = bc_dict_nd['bottom'].clone().requires_grad_(True)
        uv_bot = self.u_net(pts_bot)
        loss_bot = torch.mean((uv_bot[:, 1:2] + v_max_nd)**2)

        # ---- Traction-free: left / right sides  (sig_xx = tau_xy = 0) ----
        pts_left = bc_dict_nd['left'].clone().requires_grad_(True)
        ex_l, ey_l, exy_l, _, _, _ = self.compute_strains(pts_left)
        phi_l = self.phi_net(pts_left).detach()
        _, sxx_l, _, txy_l = self.compute_energy_and_stress(ex_l, ey_l, exy_l, phi_l)
        loss_sides = torch.mean(sxx_l**2 + txy_l**2)

        pts_right = bc_dict_nd['right'].clone().requires_grad_(True)
        ex_r, ey_r, exy_r, _, _, _ = self.compute_strains(pts_right)
        phi_r = self.phi_net(pts_right).detach()
        _, sxx_r, _, txy_r = self.compute_energy_and_stress(ex_r, ey_r, exy_r, phi_r)
        loss_sides = loss_sides + torch.mean(sxx_r**2 + txy_r**2)

        # ---- Rigid-body fix: u*(0, 0) = 0 ----
        pts_c = bc_dict_nd['center']
        uv_c = self.u_net(pts_c)
        loss_center = torch.mean(uv_c[:, 0:1]**2)

        # ---- Crack-face traction-free: σ_yy = τ_xy = 0 on y=0, |x|<a ----
        loss_crack_face = torch.tensor(0.0, device=pts_nd.device)
        if crack_face_nd is not None and crack_face_nd.numel() > 0:
            pts_cf = crack_face_nd.clone().requires_grad_(True)
            ex_cf, ey_cf, exy_cf, _, _, _ = self.compute_strains(pts_cf)
            phi_cf = self.phi_net(pts_cf).detach()
            _, _, syy_cf, txy_cf = self.compute_energy_and_stress(
                ex_cf, ey_cf, exy_cf, phi_cf)
            loss_crack_face = torch.mean(syy_cf**2 + txy_cf**2)

        loss_bc = (loss_top + loss_bot + loss_center
                   + 0.1 * loss_sides
                   + w_crack_face * loss_crack_face)

        total_loss = w_pde * loss_pde + w_bc * loss_bc

        return total_loss, loss_pde.item(), loss_bc.item(), loss_crack_face.item()

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
