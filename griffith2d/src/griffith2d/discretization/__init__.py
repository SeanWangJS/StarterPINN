"""Numerical representations and integration rules for Griffith 2D."""

from .autodiff import displacement_gradient, phase_field_gradient, phase_field_laplacian
from .center_crack import CenterCrackPlate, PlateBoundaryRules
from .networks import DisplacementMLP, PhaseFieldMLP
from .quadrature import (
    QuadratureRule,
    gauss_legendre,
    interval_quadrature,
    line_quadrature,
    partitioned_rectangle_quadrature,
    rectangle_quadrature,
)
from .trial_solutions import PhaseFieldTrialSolution, PlateDisplacementTrialSolution, SymmetryTrialSolution

__all__ = [
    "CenterCrackPlate",
    "DisplacementMLP",
    "PhaseFieldMLP",
    "PhaseFieldTrialSolution",
    "PlateBoundaryRules",
    "PlateDisplacementTrialSolution",
    "QuadratureRule",
    "SymmetryTrialSolution",
    "displacement_gradient",
    "gauss_legendre",
    "interval_quadrature",
    "line_quadrature",
    "phase_field_gradient",
    "phase_field_laplacian",
    "partitioned_rectangle_quadrature",
    "rectangle_quadrature",
]
