import torch
import numpy as np


class GriffithGeometrySampler:
    """
    Collocation point sampler for the Griffith center-crack plate problem.

    Sampling strategy (all in physical coordinates):
      - Global uniform background             (~35%)
      - Crack-band refinement  |y| < band_w  (~30%)
      - Crack-tip enrichment   r < tip_r     (~20% around each tip)
      - Crack-face points  y=0, |x|<a        (explicit traction-free BC)
    """

    def __init__(self, W, H, a, device='cpu'):
        self.W = W
        self.H = H
        self.a = a
        self.device = device

    # ------------------------------------------------------------------
    # Domain interior
    # ------------------------------------------------------------------
    def sample_fixed_domain(self, num_points, l_0: float = 0.01):
        """
        Mixed-density domain sampling.

        Composition (approximate):
          35 % uniform over entire plate
          30 % refined in the crack band   |y| < 3*l_0
          17.5 % refined around left  crack tip  (x=-a, y=0)
          17.5 % refined around right crack tip  (x=+a, y=0)
        """
        n_uniform  = int(num_points * 0.35)
        n_band     = int(num_points * 0.30)
        n_tip_each = int(num_points * 0.175)      # each tip
        n_tip      = n_tip_each * 2
        # remainder goes to uniform
        n_uniform += (num_points - n_uniform - n_band - n_tip)

        parts = []

        # ---- 1. Global background ----
        x_uni = np.random.uniform(-self.W/2, self.W/2, (n_uniform, 1))
        y_uni = np.random.uniform(-self.H/2, self.H/2, (n_uniform, 1))
        parts.append(np.hstack([x_uni, y_uni]))

        # ---- 2. Crack-band refinement  |y| < 3*l_0 ----
        band_w = 3.0 * l_0
        x_band = np.random.uniform(-self.W/2, self.W/2, (n_band, 1))
        y_band = np.random.uniform(-band_w, band_w, (n_band, 1))
        parts.append(np.hstack([x_band, y_band]))

        # ---- 3. Crack-tip enrichment  (circular patches, radius = 3*l_0) ----
        tip_r = 3.0 * l_0
        for cx in [-self.a, +self.a]:
            r   = tip_r * np.sqrt(np.random.uniform(0, 1, (n_tip_each, 1)))
            ang = np.random.uniform(0, 2 * np.pi, (n_tip_each, 1))
            x_t = cx + r * np.cos(ang)
            y_t = r * np.sin(ang)
            # clip to domain
            x_t = np.clip(x_t, -self.W/2, self.W/2)
            y_t = np.clip(y_t, -self.H/2, self.H/2)
            parts.append(np.hstack([x_t, y_t]))

        pts = np.vstack(parts)

        # Shuffle
        idx = np.random.permutation(pts.shape[0])
        pts = pts[idx]

        return torch.tensor(pts, dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------
    # Initial crack seed points  (used to preset H0 → ∞)
    # ------------------------------------------------------------------
    def get_initial_crack_points(self, num_points):
        """
        Points exactly on y=0, x ∈ [-a, a].
        Dense enough to fully cover the 1-D crack segment.
        """
        x = np.random.uniform(-self.a, self.a, (num_points, 1))
        y = np.zeros((num_points, 1))
        pts = torch.tensor(np.hstack([x, y]), dtype=torch.float32, device=self.device)
        return pts

    # ------------------------------------------------------------------
    # Crack-face boundary points  (explicit traction-free BC: σ_yy = τ_xy = 0)
    # ------------------------------------------------------------------
    def sample_crack_face(self, num_points: int, offset: float = 1e-5):
        """
        Points on the upper (+offset) and lower (-offset) faces of the crack,
        i.e. y = ±ε, x ∈ [-a, a].

        We avoid exactly y=0 so that autograd can compute gradients cleanly.
        Both upper and lower faces are returned as a single concatenated tensor.
        """
        n_half = num_points // 2
        x = np.random.uniform(-self.a, self.a, (n_half, 1))

        # Upper face
        y_up  = np.full((n_half, 1), +offset)
        pts_up = np.hstack([x, y_up])

        # Lower face (same x, mirrored y)
        y_dn   = np.full((n_half, 1), -offset)
        pts_dn = np.hstack([x, y_dn])

        pts = np.vstack([pts_up, pts_dn])
        return torch.tensor(pts, dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------
    # External boundaries
    # ------------------------------------------------------------------
    def sample_boundaries(self, num_points_per_edge):
        """
        Top  : y = +H/2   (Dirichlet: v = +v_max, u = 0)
        Bottom: y = -H/2  (Dirichlet: v = -v_max, u = 0)
        Left : x = -W/2   (Neumann: σ_xx = τ_xy = 0)
        Right: x = +W/2   (Neumann: σ_xx = τ_xy = 0)
        Center: (0, 0)    (fix: u = 0 to remove rigid-body translation)
        """
        def edge_pts(x_vals, y_vals):
            return torch.tensor(
                np.hstack([x_vals, y_vals]), dtype=torch.float32, device=self.device)

        n = num_points_per_edge

        top_pts   = edge_pts(np.random.uniform(-self.W/2, self.W/2, (n, 1)),
                             np.full((n, 1), +self.H/2))
        bot_pts   = edge_pts(np.random.uniform(-self.W/2, self.W/2, (n, 1)),
                             np.full((n, 1), -self.H/2))
        left_pts  = edge_pts(np.full((n, 1), -self.W/2),
                             np.random.uniform(-self.H/2, self.H/2, (n, 1)))
        right_pts = edge_pts(np.full((n, 1), +self.W/2),
                             np.random.uniform(-self.H/2, self.H/2, (n, 1)))

        center_pt = torch.tensor([[0.0, 0.0]], dtype=torch.float32, device=self.device)

        return {
            'top':    top_pts,
            'bottom': bot_pts,
            'left':   left_pts,
            'right':  right_pts,
            'center': center_pt,
        }
