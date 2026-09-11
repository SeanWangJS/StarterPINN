"""Analytic reference for the uniform plane-strain tension benchmark."""

from dataclasses import dataclass
from numbers import Real

import torch
from torch import Tensor

from griffith2d.discretization import QuadratureRule, displacement_gradient
from griffith2d.physics import IsotropicMaterial, small_strain, undamaged_stress


def _positive(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    value = float(value)
    if not torch.isfinite(torch.tensor(value)) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


@dataclass(frozen=True, slots=True)
class PlaneStrainTensionBenchmark:
    """Uniform horizontal traction on a quarter-symmetry rectangle."""

    length: float
    height: float
    traction: float
    material: IsotropicMaterial

    def __post_init__(self) -> None:
        object.__setattr__(self, "length", _positive(self.length, "length"))
        object.__setattr__(self, "height", _positive(self.height, "height"))
        object.__setattr__(self, "traction", _positive(self.traction, "traction"))

    @property
    def lame_lambda(self) -> float:
        return self.material.bulk_modulus - 2.0 * self.material.shear_modulus / 3.0

    @property
    def strain_xx(self) -> float:
        lame_lambda = self.lame_lambda
        shear_modulus = self.material.shear_modulus
        return (
            (lame_lambda + 2.0 * shear_modulus)
            * self.traction
            / (4.0 * shear_modulus * (lame_lambda + shear_modulus))
        )

    @property
    def strain_yy(self) -> float:
        lame_lambda = self.lame_lambda
        shear_modulus = self.material.shear_modulus
        return (
            -lame_lambda
            * self.traction
            / (4.0 * shear_modulus * (lame_lambda + shear_modulus))
        )

    @property
    def displacement_scale(self) -> float:
        return self.traction * self.length / self.material.young_modulus

    def displacement(self, coordinates: Tensor) -> Tensor:
        """Return the exact affine displacement at ``(..., 2)`` points."""

        if coordinates.ndim < 1 or coordinates.shape[-1] != 2:
            raise ValueError("coordinates must have shape (..., 2)")
        factors = coordinates.new_tensor([self.strain_xx, self.strain_yy])
        return coordinates * factors

    def strain(self, coordinates: Tensor) -> Tensor:
        """Return the exact constant in-plane strain at the supplied points."""

        result = coordinates.new_zeros((*coordinates.shape[:-1], 2, 2))
        result[..., 0, 0] = self.strain_xx
        result[..., 1, 1] = self.strain_yy
        return result

    def stress(self, coordinates: Tensor) -> Tensor:
        """Return the exact constant in-plane stress at the supplied points."""

        result = coordinates.new_zeros((*coordinates.shape[:-1], 2, 2))
        result[..., 0, 0] = self.traction
        return result

    @property
    def exact_internal_energy(self) -> float:
        return 0.5 * self.traction * self.strain_xx * self.length * self.height

    @property
    def exact_traction_work(self) -> float:
        return self.traction * self.strain_xx * self.length * self.height

    @property
    def exact_total_potential(self) -> float:
        return self.exact_internal_energy - self.exact_traction_work


@dataclass(frozen=True, slots=True)
class ElasticBenchmarkMetrics:
    """Dimensionless posterior errors for the uniform tension benchmark."""

    displacement_relative_l2: float
    strain_relative_l2: float
    stress_relative_l2: float
    reaction_relative_error: float
    right_traction_relative_rms: float
    top_traction_relative_rms: float
    equilibrium_relative_rms: float
    hard_boundary_max_abs: float


def _relative_l2(rule: QuadratureRule, error: Tensor, reference: Tensor) -> Tensor:
    error_squared = error.square().sum(dim=-1)
    reference_squared = reference.square().sum(dim=-1)
    return torch.sqrt(rule.integrate(error_squared) / rule.integrate(reference_squared))


def equilibrium_residual(
    displacement_field,
    coordinates: Tensor,
    material: IsotropicMaterial,
) -> Tensor:
    """Return ``div(sigma)`` using nested pointwise automatic differentiation."""

    def stress_at(point: Tensor) -> Tensor:
        gradient = torch.func.jacrev(displacement_field)(point)
        return undamaged_stress(small_strain(gradient), material)

    stress_gradient = torch.vmap(torch.func.jacrev(stress_at))(coordinates)
    residual_x = stress_gradient[:, 0, 0, 0] + stress_gradient[:, 0, 1, 1]
    residual_y = stress_gradient[:, 1, 0, 0] + stress_gradient[:, 1, 1, 1]
    return torch.stack((residual_x, residual_y), dim=-1)


def evaluate_elastic_benchmark(
    displacement_field,
    benchmark: PlaneStrainTensionBenchmark,
    domain_rule: QuadratureRule,
    right_rule: QuadratureRule,
    top_rule: QuadratureRule,
    left_rule: QuadratureRule,
    bottom_rule: QuadratureRule,
) -> ElasticBenchmarkMetrics:
    """Evaluate field, equilibrium, traction, reaction, and hard-boundary errors."""

    points = domain_rule.points
    displacement = displacement_field(points)
    exact_displacement = benchmark.displacement(points)
    gradient = displacement_gradient(displacement_field, points)
    strain = small_strain(gradient)
    exact_strain = benchmark.strain(points)
    stress = undamaged_stress(strain, benchmark.material)
    exact_stress = benchmark.stress(points)

    displacement_error = _relative_l2(
        domain_rule, displacement - exact_displacement, exact_displacement
    )
    strain_error = _relative_l2(
        domain_rule,
        (strain - exact_strain).reshape(points.shape[0], -1),
        exact_strain.reshape(points.shape[0], -1),
    )
    stress_error = _relative_l2(
        domain_rule,
        (stress - exact_stress).reshape(points.shape[0], -1),
        exact_stress.reshape(points.shape[0], -1),
    )

    def traction_on(rule: QuadratureRule, normal: Tensor) -> Tensor:
        boundary_gradient = displacement_gradient(displacement_field, rule.points)
        boundary_stress = undamaged_stress(small_strain(boundary_gradient), benchmark.material)
        return torch.einsum("nij,j->ni", boundary_stress, normal)

    traction_scale = benchmark.traction
    right_traction = traction_on(right_rule, points.new_tensor([1.0, 0.0]))
    right_target = points.new_tensor([benchmark.traction, 0.0]).expand_as(right_traction)
    right_error = torch.sqrt(
        right_rule.integrate((right_traction - right_target).square().sum(dim=-1))
        / right_rule.weights.sum()
    ) / traction_scale

    top_traction = traction_on(top_rule, points.new_tensor([0.0, 1.0]))
    top_error = torch.sqrt(
        top_rule.integrate(top_traction.square().sum(dim=-1)) / top_rule.weights.sum()
    ) / traction_scale

    left_traction = traction_on(left_rule, points.new_tensor([-1.0, 0.0]))
    reaction = left_rule.integrate(left_traction)[0]
    expected_reaction = -benchmark.traction * benchmark.height
    reaction_error = torch.abs(reaction - expected_reaction) / abs(expected_reaction)

    residual = equilibrium_residual(displacement_field, points, benchmark.material)
    residual_scale = benchmark.traction / benchmark.length
    equilibrium_error = torch.sqrt(
        domain_rule.integrate(residual.square().sum(dim=-1)) / domain_rule.weights.sum()
    ) / residual_scale

    left_displacement = displacement_field(left_rule.points)[:, 0]
    bottom_displacement = displacement_field(bottom_rule.points)[:, 1]
    hard_boundary_error = torch.maximum(
        left_displacement.abs().max(), bottom_displacement.abs().max()
    )

    return ElasticBenchmarkMetrics(
        displacement_relative_l2=float(displacement_error.detach().cpu()),
        strain_relative_l2=float(strain_error.detach().cpu()),
        stress_relative_l2=float(stress_error.detach().cpu()),
        reaction_relative_error=float(reaction_error.detach().cpu()),
        right_traction_relative_rms=float(right_error.detach().cpu()),
        top_traction_relative_rms=float(top_error.detach().cpu()),
        equilibrium_relative_rms=float(equilibrium_error.detach().cpu()),
        hard_boundary_max_abs=float(hard_boundary_error.detach().cpu()),
    )
