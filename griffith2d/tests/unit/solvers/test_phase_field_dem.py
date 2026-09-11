import torch

from griffith2d.diagnostics import PrecrackProfileBenchmark
from griffith2d.discretization import interval_quadrature
from griffith2d.physics import PhaseFieldMaterial
from griffith2d.solvers import phase_field_energy


DTYPE = torch.float64


def test_phase_field_energy_matches_exact_precrack_components() -> None:
    material = PhaseFieldMaterial(fracture_toughness=2.4, length_scale=0.2)
    benchmark = PrecrackProfileBenchmark(half_width=1.0, material=material)
    rule = interval_quadrature(0.0, 1.0, order=32)
    history = torch.zeros_like(rule.points)

    energy = phase_field_energy(benchmark.phase, rule, history, material)

    torch.testing.assert_close(energy.history, torch.tensor(0.0, dtype=DTYPE))
    expected_component_difference = (
        material.fracture_toughness
        * benchmark.half_width
        / (2.0 * material.length_scale * torch.cosh(torch.tensor(5.0, dtype=DTYPE)).square())
    )
    torch.testing.assert_close(
        energy.local_fracture - energy.gradient_fracture,
        expected_component_difference,
        atol=1e-13,
        rtol=0.0,
    )
    torch.testing.assert_close(
        energy.total,
        torch.tensor(benchmark.exact_fracture_energy, dtype=DTYPE),
        atol=1e-13,
        rtol=0.0,
    )


def test_phase_field_energy_has_no_hidden_bounds_projection() -> None:
    material = PhaseFieldMaterial(fracture_toughness=1.0, length_scale=0.5)
    rule = interval_quadrature(0.0, 1.0, order=4)

    def out_of_range(coordinates: torch.Tensor) -> torch.Tensor:
        return 1.2 + 0.0 * coordinates

    energy = phase_field_energy(out_of_range, rule, torch.zeros_like(rule.points), material)

    torch.testing.assert_close(energy.total, torch.tensor(1.44, dtype=DTYPE))