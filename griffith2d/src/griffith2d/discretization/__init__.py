"""Numerical representations and integration rules for Griffith 2D."""

from .autodiff import displacement_gradient, phase_field_gradient, phase_field_laplacian
from .networks import DisplacementMLP, PhaseFieldMLP
from .quadrature import (
    QuadratureRule,
    gauss_legendre,
    interval_quadrature,
    line_quadrature,
    rectangle_quadrature,
)
from .trial_solutions import PhaseFieldTrialSolution, SymmetryTrialSolution

__all__ = [
    "DisplacementMLP",
    "PhaseFieldMLP",
    "PhaseFieldTrialSolution",
    "QuadratureRule",
    "SymmetryTrialSolution",
    "displacement_gradient",
    "gauss_legendre",
    "interval_quadrature",
    "line_quadrature",
    "phase_field_gradient",
    "phase_field_laplacian",
    "rectangle_quadrature",
]
