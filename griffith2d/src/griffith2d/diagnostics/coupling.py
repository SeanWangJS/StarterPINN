"""Analytic references for uniform degraded and coupled tension states."""

from dataclasses import dataclass
import math
from numbers import Integral

import torch
from torch import Tensor

from griffith2d.discretization import QuadratureRule, displacement_gradient
from griffith2d.physics import (
    degraded_stress,
    quadratic_degradation,
    small_strain,
)

from .elasticity import PlaneStrainTensionBenchmark


def plane_strain_tension_modulus(benchmark: PlaneStrainTensionBenchmark) -> float:
    """Return the effective modulus ``T / epsilon_xx`` of the tension benchmark."""

    return benchmark.traction / benchmark.strain_xx


def dimensionless_traction(
    traction: float,
    effective_modulus: float,
    fracture_toughness: float,
    length_scale: float,
) -> float:
    """Return ``tau = T sqrt(l0 / (E_eff Gc))``."""

    values = {
        "traction": traction,
        "effective_modulus": effective_modulus,
        "fracture_toughness": fracture_toughness,
        "length_scale": length_scale,
    }
    for name, value in values.items():
        if not math.isfinite(value) or value < 0.0 or (name != "traction" and value == 0.0):
            raise ValueError(f"{name} must be finite and {'nonnegative' if name == 'traction' else 'positive'}")
    return traction * math.sqrt(length_scale / (effective_modulus * fracture_toughness))


def coupled_phase_residual(phase: float, tau: float, residual_stiffness: float) -> float:
    degradation = (1.0 - phase) ** 2 + residual_stiffness
    return phase * (degradation**2 + tau**2) - tau**2


@dataclass(frozen=True, slots=True)
class CoupledPhaseRoot:
    phase: float
    residual: float
    residual_derivative: float
    fixed_point_derivative: float

    @property
    def fixed_point_stable(self) -> bool:
        return abs(self.fixed_point_derivative) < 1.0


def coupled_phase_roots(
    tau: float,
    residual_stiffness: float,
    *,
    subdivisions: int = 4096,
    tolerance: float = 1e-14,
) -> tuple[CoupledPhaseRoot, ...]:
    """Enumerate bracketed roots of the uniform coupled equation on ``[0, 1]``."""

    if not math.isfinite(tau) or tau < 0.0:
        raise ValueError("tau must be finite and nonnegative")
    if not math.isfinite(residual_stiffness) or not 0.0 < residual_stiffness < 1.0:
        raise ValueError("residual_stiffness must satisfy 0 < k < 1")
    if isinstance(subdivisions, bool) or not isinstance(subdivisions, Integral):
        raise TypeError("subdivisions must be an integer")
    if subdivisions < 2:
        raise ValueError("subdivisions must be at least 2")

    brackets: list[tuple[float, float]] = []
    left = 0.0
    left_value = coupled_phase_residual(left, tau, residual_stiffness)
    if left_value == 0.0:
        brackets.append((left, left))
    for index in range(1, subdivisions + 1):
        right = index / subdivisions
        right_value = coupled_phase_residual(right, tau, residual_stiffness)
        if left_value * right_value < 0.0:
            brackets.append((left, right))
        elif right_value == 0.0:
            brackets.append((right, right))
        left, left_value = right, right_value

    roots: list[CoupledPhaseRoot] = []
    for lower, upper in brackets:
        if lower != upper:
            lower_value = coupled_phase_residual(lower, tau, residual_stiffness)
            while upper - lower > tolerance:
                midpoint = 0.5 * (lower + upper)
                midpoint_value = coupled_phase_residual(midpoint, tau, residual_stiffness)
                if lower_value * midpoint_value <= 0.0:
                    upper = midpoint
                else:
                    lower = midpoint
                    lower_value = midpoint_value
        phase = 0.5 * (lower + upper)
        degradation = (1.0 - phase) ** 2 + residual_stiffness
        degradation_derivative = -2.0 * (1.0 - phase)
        denominator = degradation**2 + tau**2
        residual_derivative = denominator + 2.0 * phase * degradation * degradation_derivative
        fixed_point_derivative = (
            4.0 * tau**2 * degradation * (1.0 - phase) / denominator**2
            if denominator > 0.0
            else 0.0
        )
        roots.append(
            CoupledPhaseRoot(
                phase=phase,
                residual=coupled_phase_residual(phase, tau, residual_stiffness),
                residual_derivative=residual_derivative,
                fixed_point_derivative=fixed_point_derivative,
            )
        )
    return tuple(roots)


def continue_low_damage_root(
    tau: float,
    residual_stiffness: float,
    previous_phase: float,
) -> tuple[CoupledPhaseRoot, tuple[CoupledPhaseRoot, ...]]:
    """Select the stable root continuous from the previous low-damage state."""

    roots = coupled_phase_roots(tau, residual_stiffness)
    stable_roots = tuple(root for root in roots if root.fixed_point_stable)
    if not stable_roots:
        raise ValueError("no fixed-point-stable coupled phase root was found")
    selected = min(stable_roots, key=lambda root: abs(root.phase - previous_phase))
    return selected, roots


@dataclass(frozen=True, slots=True)
class UniformDegradedTensionBenchmark:
    """Uniform plane-strain tension with a prescribed constant phase field."""

    elastic: PlaneStrainTensionBenchmark
    phase: float
    residual_stiffness: float

    @property
    def degradation(self) -> float:
        if not math.isfinite(self.phase):
            raise ValueError("phase must be finite")
        value = quadratic_degradation(
            torch.tensor(self.phase, dtype=torch.float64), self.residual_stiffness
        )
        return float(value)

    @property
    def compliance_ratio(self) -> float:
        return 1.0 / self.degradation

    def displacement(self, coordinates: Tensor) -> Tensor:
        return self.elastic.displacement(coordinates) * self.compliance_ratio

    def strain(self, coordinates: Tensor) -> Tensor:
        return self.elastic.strain(coordinates) * self.compliance_ratio

    def stress(self, coordinates: Tensor) -> Tensor:
        return self.elastic.stress(coordinates)

    @property
    def exact_internal_energy(self) -> float:
        return self.elastic.exact_internal_energy * self.compliance_ratio

    @property
    def exact_traction_work(self) -> float:
        return self.elastic.exact_traction_work * self.compliance_ratio

    @property
    def exact_total_potential(self) -> float:
        return self.elastic.exact_total_potential * self.compliance_ratio


@dataclass(frozen=True, slots=True)
class DegradedElasticBenchmarkMetrics:
    displacement_relative_l2: float
    strain_relative_l2: float
    stress_relative_l2: float
    reaction_relative_error: float
    right_traction_relative_rms: float
    top_traction_relative_rms: float
    equilibrium_relative_rms: float
    hard_boundary_max_abs: float


def degraded_equilibrium_residual(
    displacement_field,
    coordinates: Tensor,
    phase: float,
    benchmark: UniformDegradedTensionBenchmark,
) -> Tensor:
    """Return ``div(sigma)`` for a prescribed uniform phase field."""

    def stress_at(point: Tensor) -> Tensor:
        strain = small_strain(torch.func.jacrev(displacement_field)(point))
        return degraded_stress(
            strain,
            point.new_tensor(phase),
            benchmark.elastic.material,
            benchmark.residual_stiffness,
        )

    stress_gradient = torch.vmap(torch.func.jacrev(stress_at))(coordinates)
    return torch.stack(
        (
            stress_gradient[:, 0, 0, 0] + stress_gradient[:, 0, 1, 1],
            stress_gradient[:, 1, 0, 0] + stress_gradient[:, 1, 1, 1],
        ),
        dim=-1,
    )


def degraded_boundary_reaction(
    displacement_field,
    phase: float,
    benchmark: UniformDegradedTensionBenchmark,
    boundary: QuadratureRule,
    outward_normal: Tensor,
) -> Tensor:
    """Integrate the degraded traction vector on one boundary."""

    if outward_normal.shape != (2,):
        raise ValueError("outward_normal must have shape (2,)")
    if outward_normal.dtype != boundary.points.dtype or outward_normal.device != boundary.points.device:
        raise ValueError("outward_normal and boundary must share dtype and device")
    strain = small_strain(displacement_gradient(displacement_field, boundary.points))
    phase_field = boundary.points.new_full((boundary.points.shape[0],), phase)
    stress = degraded_stress(
        strain,
        phase_field,
        benchmark.elastic.material,
        benchmark.residual_stiffness,
    )
    traction = torch.einsum("nij,j->ni", stress, outward_normal)
    return boundary.integrate(traction)


def evaluate_degraded_elastic_benchmark(
    displacement_field,
    benchmark: UniformDegradedTensionBenchmark,
    domain: QuadratureRule,
    right: QuadratureRule,
    top: QuadratureRule,
    left: QuadratureRule,
    bottom: QuadratureRule,
) -> DegradedElasticBenchmarkMetrics:
    """Evaluate field, traction, reaction, equilibrium, and hard-boundary errors."""

    points = domain.points
    displacement = displacement_field(points)
    exact_displacement = benchmark.displacement(points)
    strain = small_strain(displacement_gradient(displacement_field, points))
    exact_strain = benchmark.strain(points)
    phase = points.new_full((points.shape[0],), benchmark.phase)
    stress = degraded_stress(
        strain, phase, benchmark.elastic.material, benchmark.residual_stiffness
    )
    exact_stress = benchmark.stress(points)

    def relative_l2(error: Tensor, reference: Tensor) -> Tensor:
        error_squared = error.reshape(points.shape[0], -1).square().sum(dim=-1)
        reference_squared = reference.reshape(points.shape[0], -1).square().sum(dim=-1)
        return torch.sqrt(domain.integrate(error_squared) / domain.integrate(reference_squared))

    def traction_on(rule: QuadratureRule, normal: Tensor) -> Tensor:
        boundary_strain = small_strain(displacement_gradient(displacement_field, rule.points))
        boundary_phase = rule.points.new_full((rule.points.shape[0],), benchmark.phase)
        boundary_stress = degraded_stress(
            boundary_strain,
            boundary_phase,
            benchmark.elastic.material,
            benchmark.residual_stiffness,
        )
        return torch.einsum("nij,j->ni", boundary_stress, normal)

    traction_scale = benchmark.elastic.traction
    right_traction = traction_on(right, points.new_tensor([1.0, 0.0]))
    right_target = points.new_tensor([traction_scale, 0.0]).expand_as(right_traction)
    right_error = torch.sqrt(
        right.integrate((right_traction - right_target).square().sum(dim=-1))
        / right.weights.sum()
    ) / traction_scale
    top_traction = traction_on(top, points.new_tensor([0.0, 1.0]))
    top_error = torch.sqrt(
        top.integrate(top_traction.square().sum(dim=-1)) / top.weights.sum()
    ) / traction_scale
    left_traction = traction_on(left, points.new_tensor([-1.0, 0.0]))
    reaction = left.integrate(left_traction)[0]
    expected_reaction = -traction_scale * benchmark.elastic.height
    reaction_error = torch.abs(reaction - expected_reaction) / abs(expected_reaction)
    residual = degraded_equilibrium_residual(displacement_field, points, benchmark.phase, benchmark)
    residual_error = torch.sqrt(
        domain.integrate(residual.square().sum(dim=-1)) / domain.weights.sum()
    ) / (traction_scale / benchmark.elastic.length)
    hard_error = torch.maximum(
        displacement_field(left.points)[:, 0].abs().max(),
        displacement_field(bottom.points)[:, 1].abs().max(),
    )
    return DegradedElasticBenchmarkMetrics(
        displacement_relative_l2=float(relative_l2(displacement - exact_displacement, exact_displacement).detach().cpu()),
        strain_relative_l2=float(relative_l2(strain - exact_strain, exact_strain).detach().cpu()),
        stress_relative_l2=float(relative_l2(stress - exact_stress, exact_stress).detach().cpu()),
        reaction_relative_error=float(reaction_error.detach().cpu()),
        right_traction_relative_rms=float(right_error.detach().cpu()),
        top_traction_relative_rms=float(top_error.detach().cpu()),
        equilibrium_relative_rms=float(residual_error.detach().cpu()),
        hard_boundary_max_abs=float(hard_error.detach().cpu()),
    )