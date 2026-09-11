"""Phase-field degradation law selected for the first Griffith 2D baseline."""

import math

from torch import Tensor


def _validate_residual_stiffness(residual_stiffness: float) -> float:
    value = float(residual_stiffness)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("residual_stiffness must be finite and satisfy 0 < k < 1")
    return value


def quadratic_degradation(phase_field: Tensor, residual_stiffness: float) -> Tensor:
    """Return ``g(phi) = (1 - phi)^2 + k`` without clamping ``phi``.

    ``phase_field`` is dimensionless. Callers are responsible for constructing a
    field in the physical interval ``[0, 1]``; silently clamping it would alter
    derivatives and conceal constraint violations.
    """

    if not isinstance(phase_field, Tensor):
        raise TypeError("phase_field must be a torch.Tensor")
    if not phase_field.is_floating_point():
        raise TypeError("phase_field must have a floating-point dtype")
    residual_stiffness = _validate_residual_stiffness(residual_stiffness)
    return (1.0 - phase_field).square() + residual_stiffness


def quadratic_degradation_derivative(phase_field: Tensor, residual_stiffness: float) -> Tensor:
    """Return ``g'(phi) = -2 (1 - phi)`` for the selected degradation law."""

    _validate_residual_stiffness(residual_stiffness)
    if not isinstance(phase_field, Tensor):
        raise TypeError("phase_field must be a torch.Tensor")
    if not phase_field.is_floating_point():
        raise TypeError("phase_field must have a floating-point dtype")
    return -2.0 * (1.0 - phase_field)
