"""
Y-mirror-symmetric variant of the elastic-diagnostic evaluator.

Motivation (docs/griffith-ablation-study.md \u00a79, L4 tier-2 sweep):
the Griffith mode-I setup (v = +v_max on top, v = -v_max on bottom, symmetric
geometry) implies the true displacement field is even in y for u and odd in
y for v, hence sigma_xx/sigma_yy are even in y and tau_xy is odd in y -- i.e.
sigma_yy(x, y) == sigma_yy(x, -y). The elastic-diagnostic ablation matrix
(experiments/results/metrics.csv) showed this symmetry badly violated
(sym_err ~ 1e2-1e3, comparable to the full stress scale) across every
coord_scheme/precision/arch combination tried so far. A follow-up sweep over
w_sym in {50, 10, 0} (experiments/l4_wsym_sweep.py,
experiments/results/l4_wsym_sweep_metrics.csv) falsified the hypothesis that
w_sym (the *x=0* symmetry-axis BC weight -- a different symmetry entirely)
was responsible: sym_err did not improve when w_sym was reduced or removed,
and w_sym=0 additionally caused a systematic sign flip of peak_mag_ratio
across all 8 seeds tested, evidence that w_sym should if anything be kept
at its production value, not blamed for this failure mode.

Per docs/griffith-ablation-study.md \u00a78's three-way classification, y-mirror
symmetry here is a *definitional* constraint (implied by the problem's own
geometry/BCs, not a tunable "optimization aid"), so per that section's own
guidance it should be enforced *exactly by construction* rather than added
as yet another soft loss weight to tune. This module does exactly that:

    u(x, y) = 1/2 [ u_net(x, y) + u_net(x, -y) ]   (forced even in y)
    v(x, y) = 1/2 [ v_net(x, y) - v_net(x, -y) ]   (forced odd in y)

implemented as a single override of PFPhysicsEvaluator._eval_uv_raw(), the
hook factored out of src/evaluator.py's compute_strains() and
compute_u_loss() specifically so this kind of variant does not need to
duplicate any loss-assembly code (contrast with MixedPFEvaluator in
mixed_pinn_variant.py, which changes the network's *output shape* and so
does need its own compute_u_loss_mixed). The mirror is computed at the raw
(x*, y*) coordinate level and features are rebuilt from scratch for the
mirrored point via self._build_u_input(), so this works unchanged for any
coord_scheme (cartesian / elliptic / both / williams) with no per-scheme
parity bookkeeping required.

If this ablation factor turns out to help (lower sym_err / peak_loc_err
without regressing l2_rel_err or bc_err), promote _eval_uv_raw's override
back into src/evaluator.py as a constructor flag (e.g. y_symmetric=True)
rather than leaving it only here.

**UPDATE**: an n=8-seed comparison at coord_scheme='elliptic' (see
docs/griffith-ablation-study.md \u00a79) showed y-mirror symmetry significantly
improves l2_rel_err (0.503 -> 0.353, |t|~3.35) with no statistically
significant change to peak_loc_err or peak_mag_ratio, so this has now been
promoted into src/evaluator.py's PFPhysicsEvaluator as the y_symmetric=True
constructor flag. This module is kept only as a thin backward-compatible
alias -- new code should pass y_symmetric=True to PFPhysicsEvaluator (or
experiments/diagnostics.py's arch='y_sym') directly instead of importing
YSymmetricPFEvaluator.
"""
from src.evaluator import PFPhysicsEvaluator  # noqa: F401


class YSymmetricPFEvaluator(PFPhysicsEvaluator):
    """Deprecated alias for PFPhysicsEvaluator(..., y_symmetric=True).

    Kept only for backward compatibility with any code that still imports
    this class directly; prefer passing y_symmetric=True to
    PFPhysicsEvaluator instead.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('y_symmetric', True)
        super().__init__(*args, **kwargs)

