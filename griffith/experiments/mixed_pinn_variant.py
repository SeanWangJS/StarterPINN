"""
Mixed-PINN variant (ablation factor C2): DisplacementNet directly predicts
stress components [u, v, sigma_xx*, sigma_yy*, tau_xy*] instead of only
[u, v]. Kept isolated from src/evaluator.py because docs/griffith.md already
argued once for abandoning this architecture in favor of the elliptic-
coordinate autodiff trick; this module lets us re-test that decision
empirically without risking the stability of the production evaluator.

If the ablation study concludes Mixed-PINN is worth adopting, promote the
relevant pieces back into src/network.py and src/evaluator.py.
"""
import torch
import torch.nn as nn

from src.network import MLP
from src.evaluator import PFPhysicsEvaluator, cartesian_to_elliptic  # noqa: F401


class MixedDisplacementNet(nn.Module):
    """Outputs [u*, v*, sigma_xx*, sigma_yy*, tau_xy*] directly."""

    def __init__(self, in_dim, hidden_layers=4, hidden_neurons=64):
        super().__init__()
        self.mlp = MLP(in_dim, 5, hidden_layers, hidden_neurons, activation=nn.Tanh())

    def forward(self, x):
        return self.mlp(x)


class MixedPFEvaluator(PFPhysicsEvaluator):
    """
    Mixed-PINN evaluator: equilibrium residual is computed from first-order
    derivatives of the *network-predicted* stress, while a consistency loss
    ties that predicted stress back to the Amor-split constitutive law
    evaluated from strains derived from (u, v).
    """

    def compute_strains_and_stress(self, pts_nd):
        """
        Returns eps_xx, eps_yy, eps_xy, pts_nd(with grad), u*, v*,
        and the network-predicted sig_xx*, sig_yy*, tau_xy* (not None,
        unlike the base class).
        """
        pts_nd = pts_nd.clone().requires_grad_(True)
        u_in = self._build_u_input(pts_nd)
        out_nd = self.u_net(u_in)

        u_nd = out_nd[:, 0:1]
        v_nd = out_nd[:, 1:2]
        sig_xx_nn = out_nd[:, 2:3]
        sig_yy_nn = out_nd[:, 3:4]
        tau_xy_nn = out_nd[:, 4:5]

        grad_u = torch.autograd.grad(u_nd.sum(), pts_nd, create_graph=True)[0]
        grad_v = torch.autograd.grad(v_nd.sum(), pts_nd, create_graph=True)[0]

        eps_xx = grad_u[:, 0:1]
        eps_yy = grad_v[:, 1:2]
        eps_xy = 0.5 * (grad_u[:, 1:2] + grad_v[:, 0:1])

        return eps_xx, eps_yy, eps_xy, pts_nd, u_nd, v_nd, sig_xx_nn, sig_yy_nn, tau_xy_nn

    def compute_u_loss_mixed(self, pts_nd, bc_dict_nd, v_max_nd, w_pde, w_bc,
                              crack_face_nd=None, w_crack_face=10.0,
                              sym_axis_nd=None, w_sym=50.0, w_consistency=1.0):
        eps_xx, eps_yy, eps_xy, x_y_nd, _, _, sig_xx_nn, sig_yy_nn, tau_xy_nn = \
            self.compute_strains_and_stress(pts_nd)
        phi = self.phi_net(x_y_nd).detach()

        # Constitutive (Amor) stress computed from strains: consistency target
        _, sig_xx_c, sig_yy_c, tau_xy_c = self.compute_energy_and_stress(
            eps_xx, eps_yy, eps_xy, phi)
        loss_consistency = torch.mean(
            (sig_xx_nn - sig_xx_c) ** 2
            + (sig_yy_nn - sig_yy_c) ** 2
            + (tau_xy_nn - tau_xy_c) ** 2
        )

        # Equilibrium residual: first-order derivatives of predicted stress
        dsig_xx_dx = torch.autograd.grad(
            sig_xx_nn.sum(), x_y_nd, create_graph=True)[0][:, 0:1]
        dtau_xy_dy = torch.autograd.grad(
            tau_xy_nn.sum(), x_y_nd, create_graph=True)[0][:, 1:2]
        dtau_xy_dx = torch.autograd.grad(
            tau_xy_nn.sum(), x_y_nd, create_graph=True)[0][:, 0:1]
        dsig_yy_dy = torch.autograd.grad(
            sig_yy_nn.sum(), x_y_nd, create_graph=True)[0][:, 1:2]

        res_x = dsig_xx_dx + dtau_xy_dy
        res_y = dtau_xy_dx + dsig_yy_dy
        loss_equil = torch.mean(res_x ** 2 + res_y ** 2)

        loss_pde = w_pde * loss_equil + w_consistency * loss_consistency

        def eval_uv(pts):
            out = self.u_net(self._build_u_input(pts))
            return out[:, 0:2]

        pts_top = bc_dict_nd['top'].clone().requires_grad_(True)
        loss_top = torch.mean((eval_uv(pts_top)[:, 1:2] - v_max_nd) ** 2)

        pts_bot = bc_dict_nd['bottom'].clone().requires_grad_(True)
        loss_bot = torch.mean((eval_uv(pts_bot)[:, 1:2] + v_max_nd) ** 2)

        pts_right = bc_dict_nd['right'].clone().requires_grad_(True)
        _, _, _, _, _, _, sxx_r, _, txy_r = self.compute_strains_and_stress(pts_right)
        loss_right = torch.mean(sxx_r ** 2 + txy_r ** 2)

        pts_c = bc_dict_nd['center']
        loss_center = torch.mean(eval_uv(pts_c)[:, 0:1] ** 2)

        loss_sym = torch.tensor(0.0, device=pts_nd.device)
        if sym_axis_nd is not None and sym_axis_nd.numel() > 0:
            pts_sa = sym_axis_nd.clone().requires_grad_(True)
            _, _, _, _, u_sa, _, _, _, txy_sa = self.compute_strains_and_stress(pts_sa)
            loss_sym = torch.mean(u_sa ** 2 + txy_sa ** 2)

        loss_crack_face = torch.tensor(0.0, device=pts_nd.device)
        if crack_face_nd is not None and crack_face_nd.numel() > 0:
            pts_cf = crack_face_nd.clone().requires_grad_(True)
            _, _, _, _, _, _, _, syy_cf, txy_cf = self.compute_strains_and_stress(pts_cf)
            loss_crack_face = torch.mean(syy_cf ** 2 + txy_cf ** 2)

        loss_bc = (loss_top + loss_bot + loss_center
                   + 0.1 * loss_right
                   + w_sym * loss_sym
                   + w_crack_face * loss_crack_face)

        total_loss = loss_pde + w_bc * loss_bc
        return (total_loss, loss_equil.item(), loss_bc.item(),
                loss_crack_face.item(), loss_sym.item(), loss_consistency.item())
