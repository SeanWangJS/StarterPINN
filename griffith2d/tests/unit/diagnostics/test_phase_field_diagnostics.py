import math

import pytest
import torch

from griffith2d.diagnostics import (
    PrecrackProfileBenchmark,
    natural_phase_flux,
    phase_field_constraint_violations,
    phase_field_residual,
    uniform_history_phase,
)
from griffith2d.discretization import interval_quadrature
from griffith2d.physics import PhaseFieldMaterial, frozen_history_energy_density


DTYPE = torch.float64
MATERIAL = PhaseFieldMaterial(fracture_toughness=2.4, length_scale=0.2)


def test_precrack_profile_satisfies_equation_and_boundaries() -> None:
    benchmark = PrecrackProfileBenchmark(half_width=1.0, material=MATERIAL)
    coordinates = torch.linspace(0.0, 1.0, 21, dtype=DTYPE)[:, None]
    phase = benchmark.phase(coordinates)

    residual = phase_field_residual(
        phase,
        benchmark.laplacian(coordinates),
        torch.zeros_like(phase),
        MATERIAL,
    )

    torch.testing.assert_close(phase[0], torch.ones(1, dtype=DTYPE))
    torch.testing.assert_close(benchmark.gradient(coordinates)[-1], torch.zeros(1, dtype=DTYPE))
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1e-13, rtol=0.0)


@pytest.mark.parametrize("order", [8, 16, 32])
def test_precrack_quadrature_matches_exact_finite_interval_energy(order: int) -> None:
    benchmark = PrecrackProfileBenchmark(half_width=1.0, material=MATERIAL)
    rule = interval_quadrature(0.0, benchmark.half_width, order=order)
    phase = benchmark.phase(rule.points)
    gradient = benchmark.gradient(rule.points)
    energy = frozen_history_energy_density(phase, gradient, torch.zeros_like(phase), MATERIAL)

    numerical = rule.integrate(energy.total[:, 0])

    tolerance = {8: 2e-5, 16: 1e-11, 32: 1e-13}[order]
    torch.testing.assert_close(
        numerical,
        torch.tensor(benchmark.exact_fracture_energy, dtype=DTYPE),
        atol=tolerance,
        rtol=0.0,
    )


def test_precrack_finite_interval_converges_to_half_crack_limit() -> None:
    errors = []
    for ratio in (1.0, 2.0, 4.0, 8.0):
        benchmark = PrecrackProfileBenchmark(ratio * MATERIAL.length_scale, MATERIAL)
        errors.append(abs(benchmark.exact_fracture_energy - 0.5 * MATERIAL.fracture_toughness))

    assert all(next_error < error for error, next_error in zip(errors, errors[1:]))
    assert errors[-1] < 1e-6


def test_uniform_history_solution_satisfies_residual_and_is_monotone() -> None:
    history = torch.tensor([[0.0], [0.5], [2.0], [10.0]], dtype=DTYPE)
    phase = uniform_history_phase(history, MATERIAL)

    residual = phase_field_residual(phase, torch.zeros_like(phase), history, MATERIAL)

    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1e-14, rtol=0.0)
    assert bool(torch.all(phase >= 0.0))
    assert bool(torch.all(phase < 1.0))
    assert bool(torch.all(phase[1:] > phase[:-1]))


def test_uniform_history_solution_rejects_negative_history_by_default() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        uniform_history_phase(torch.tensor([[-1.0]], dtype=DTYPE), MATERIAL)


def test_natural_flux_and_constraint_violations_are_explicit() -> None:
    gradient = torch.tensor([[2.0, -1.0]], dtype=DTYPE)
    normal = torch.tensor([[0.0, 1.0]], dtype=DTYPE)
    phase = torch.tensor([[-0.2], [0.8], [1.3]], dtype=DTYPE)
    previous = torch.tensor([[0.0], [0.9], [1.0]], dtype=DTYPE)

    flux = natural_phase_flux(gradient, normal)
    violations = phase_field_constraint_violations(phase, previous)

    torch.testing.assert_close(flux, torch.tensor([[-1.0]], dtype=DTYPE))
    assert math.isclose(violations.below_zero.item(), 0.2)
    assert math.isclose(violations.above_one.item(), 0.3)
    assert math.isclose(violations.decrease.item(), 0.2)