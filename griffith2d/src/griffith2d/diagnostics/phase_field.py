"""Analytic references and posterior diagnostics for independent AT2 problems."""

from dataclasses import dataclass
import math
from numbers import Real

import torch
from torch import Tensor

from griffith2d.discretization import QuadratureRule, phase_field_gradient, phase_field_laplacian
from griffith2d.physics import PhaseFieldMaterial, frozen_history_energy_density


def _positive(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


@dataclass(frozen=True, slots=True)
class PrecrackProfileBenchmark:
    """Finite half-interval AT2 profile with a hard crack-center value."""

    half_width: float
    material: PhaseFieldMaterial

    def __post_init__(self) -> None:
        object.__setattr__(self, "half_width", _positive(self.half_width, "half_width"))

    def phase(self, coordinates: Tensor) -> Tensor:
        """Return ``cosh((S-s)/l0) / cosh(S/l0)`` at ``(..., 1)`` points."""

        self._validate_coordinates(coordinates)
        length_scale = self.material.length_scale
        return torch.cosh((self.half_width - coordinates) / length_scale) / math.cosh(
            self.half_width / length_scale
        )

    def gradient(self, coordinates: Tensor) -> Tensor:
        """Return the exact derivative with respect to the half-interval coordinate."""

        self._validate_coordinates(coordinates)
        length_scale = self.material.length_scale
        return -torch.sinh((self.half_width - coordinates) / length_scale) / (
            length_scale * math.cosh(self.half_width / length_scale)
        )

    def laplacian(self, coordinates: Tensor) -> Tensor:
        """Return the exact one-dimensional Laplacian."""

        return self.phase(coordinates) / self.material.length_scale**2

    @property
    def exact_fracture_energy(self) -> float:
        """Return the regularized fracture energy on the modeled half interval."""

        ratio = self.half_width / self.material.length_scale
        return 0.5 * self.material.fracture_toughness * math.tanh(ratio)

    @staticmethod
    def _validate_coordinates(coordinates: Tensor) -> None:
        if not isinstance(coordinates, Tensor):
            raise TypeError("coordinates must be a torch.Tensor")
        if not coordinates.is_floating_point():
            raise TypeError("coordinates must have a floating-point dtype")
        if coordinates.ndim == 0 or coordinates.shape[-1] != 1:
            raise ValueError("coordinates must have shape (..., 1)")


def uniform_history_phase(
    history: Tensor,
    material: PhaseFieldMaterial,
    *,
    validate: bool = True,
) -> Tensor:
    """Return the exact constant solution ``2 H / (Gc/l0 + 2 H)``."""

    if not isinstance(history, Tensor):
        raise TypeError("history must be a torch.Tensor")
    if not history.is_floating_point():
        raise TypeError("history must have a floating-point dtype")
    if validate and bool(torch.any(history < 0.0)):
        raise ValueError("history must be nonnegative")
    return 2.0 * history / (
        material.fracture_toughness / material.length_scale + 2.0 * history
    )


def phase_field_residual(
    phase_field: Tensor,
    phase_laplacian: Tensor,
    history_field: Tensor,
    material: PhaseFieldMaterial,
) -> Tensor:
    """Return the dimensional frozen-history Euler-Lagrange residual."""

    if phase_field.shape != phase_laplacian.shape or phase_field.shape != history_field.shape:
        raise ValueError("phase_field, phase_laplacian, and history_field must have the same shape")
    return (
        material.fracture_toughness * phase_field / material.length_scale
        - material.fracture_toughness * material.length_scale * phase_laplacian
        - 2.0 * (1.0 - phase_field) * history_field
    )


def natural_phase_flux(phase_gradient: Tensor, outward_normal: Tensor) -> Tensor:
    """Return ``grad(phi) dot n`` for matching pointwise vectors."""

    if phase_gradient.shape != outward_normal.shape:
        raise ValueError("phase_gradient and outward_normal must have the same shape")
    return (phase_gradient * outward_normal).sum(dim=-1, keepdim=True)


@dataclass(frozen=True, slots=True)
class PhaseFieldConstraintViolations:
    """Maximum phase bounds and load-step monotonicity violations."""

    below_zero: Tensor
    above_one: Tensor
    decrease: Tensor


def phase_field_constraint_violations(
    phase_field: Tensor,
    previous_phase_field: Tensor | None = None,
) -> PhaseFieldConstraintViolations:
    """Measure violations without clamping or otherwise changing the supplied field."""

    below_zero = torch.clamp_min(-phase_field, 0.0).max()
    above_one = torch.clamp_min(phase_field - 1.0, 0.0).max()
    decrease = phase_field.new_zeros(())
    if previous_phase_field is not None:
        if previous_phase_field.shape != phase_field.shape:
            raise ValueError("previous_phase_field and phase_field must have the same shape")
        decrease = torch.clamp_min(previous_phase_field - phase_field, 0.0).max()
    return PhaseFieldConstraintViolations(below_zero, above_one, decrease)


@dataclass(frozen=True, slots=True)
class PhaseFieldBenchmarkMetrics:
    """Dimensionless posterior errors for an independent phase-field benchmark."""

    phase_relative_l2: float
    phase_max_abs: float
    phase_min: float
    phase_max: float
    energy_relative_error: float
    residual_relative_rms: float
    boundary_flux_max_scaled: float
    hard_boundary_max_abs: float
    below_zero: float
    above_one: float
    decrease: float


def evaluate_phase_field_benchmark(
    phase_field,
    exact_phase,
    exact_energy: float,
    domain_rule: QuadratureRule,
    history_field: Tensor,
    material: PhaseFieldMaterial,
    *,
    hard_center: bool,
    previous_phase_field: Tensor | None = None,
) -> PhaseFieldBenchmarkMetrics:
    """Evaluate field, energy, residual, natural-boundary, and constraint errors."""

    points = domain_rule.points
    phase = phase_field(points)
    reference = exact_phase(points)
    gradient = phase_field_gradient(phase_field, points)
    density = frozen_history_energy_density(phase, gradient, history_field, material)
    numerical_energy = domain_rule.integrate(density.total[:, 0])
    error_l2 = torch.sqrt(
        domain_rule.integrate((phase[:, 0] - reference[:, 0]).square())
        / domain_rule.integrate(reference[:, 0].square())
    )

    laplacian = phase_field_laplacian(phase_field, points)
    residual = phase_field_residual(phase, laplacian, history_field, material)
    residual_scale = material.fracture_toughness / material.length_scale
    residual_rms = torch.sqrt(
        domain_rule.integrate(residual[:, 0].square()) / domain_rule.weights.sum()
    ) / residual_scale

    half_width = float(domain_rule.weights.sum().detach().cpu())
    endpoints = points.new_tensor([[0.0], [half_width]])
    endpoint_gradient = phase_field_gradient(phase_field, endpoints)
    natural_gradient = endpoint_gradient[-1:] if hard_center else endpoint_gradient
    boundary_flux = material.length_scale * natural_gradient.abs().max()
    hard_error = phase.new_zeros(())
    if hard_center:
        hard_error = torch.abs(phase_field(endpoints[:1]) - exact_phase(endpoints[:1])).max()

    violations = phase_field_constraint_violations(phase, previous_phase_field)
    return PhaseFieldBenchmarkMetrics(
        phase_relative_l2=float(error_l2.detach().cpu()),
        phase_max_abs=float((phase - reference).abs().max().detach().cpu()),
        phase_min=float(phase.min().detach().cpu()),
        phase_max=float(phase.max().detach().cpu()),
        energy_relative_error=float(
            (torch.abs(numerical_energy - exact_energy) / abs(exact_energy)).detach().cpu()
        ),
        residual_relative_rms=float(residual_rms.detach().cpu()),
        boundary_flux_max_scaled=float(boundary_flux.detach().cpu()),
        hard_boundary_max_abs=float(hard_error.detach().cpu()),
        below_zero=float(violations.below_zero.detach().cpu()),
        above_one=float(violations.above_one.detach().cpu()),
        decrease=float(violations.decrease.detach().cpu()),
    )