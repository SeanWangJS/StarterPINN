import torch

from griffith2d.diagnostics import PlaneStrainTensionBenchmark
from griffith2d.discretization import line_quadrature, rectangle_quadrature
from griffith2d.physics import IsotropicMaterial, undamaged_stress
from griffith2d.solvers import elastic_potential_energy


DTYPE = torch.float64
MATERIAL = IsotropicMaterial(young_modulus=1.0, poisson_ratio=0.3)
BENCHMARK = PlaneStrainTensionBenchmark(1.0, 1.0, 0.1, MATERIAL)


def _rules(order: int):
    domain = rectangle_quadrature((0.0, 1.0), (0.0, 1.0), order=order)
    boundary = line_quadrature(
        torch.tensor([1.0, 0.0], dtype=DTYPE),
        torch.tensor([1.0, 1.0], dtype=DTYPE),
        order=order,
    )
    return domain, boundary


def test_plane_strain_benchmark_recovers_prescribed_stress() -> None:
    points = torch.tensor([[0.2, 0.4], [0.7, 0.9]], dtype=DTYPE)

    computed = undamaged_stress(BENCHMARK.strain(points), MATERIAL)

    torch.testing.assert_close(computed, BENCHMARK.stress(points), atol=1e-14, rtol=0.0)


def test_exact_affine_field_has_analytic_energy_components() -> None:
    domain, boundary = _rules(order=3)

    components = elastic_potential_energy(
        BENCHMARK.displacement,
        domain,
        boundary,
        torch.tensor([BENCHMARK.traction, 0.0], dtype=DTYPE),
        MATERIAL,
    )

    torch.testing.assert_close(
        components.internal_energy,
        torch.tensor(BENCHMARK.exact_internal_energy, dtype=DTYPE),
        atol=1e-14,
        rtol=0.0,
    )
    torch.testing.assert_close(
        components.traction_work,
        torch.tensor(BENCHMARK.exact_traction_work, dtype=DTYPE),
        atol=1e-14,
        rtol=0.0,
    )
    torch.testing.assert_close(
        components.total_potential,
        torch.tensor(BENCHMARK.exact_total_potential, dtype=DTYPE),
        atol=1e-14,
        rtol=0.0,
    )
    assert components.total_potential < 0.0


def test_constant_strain_energy_is_independent_of_quadrature_order() -> None:
    totals = []
    for order in (1, 2, 3, 5):
        domain, boundary = _rules(order)
        components = elastic_potential_energy(
            BENCHMARK.displacement,
            domain,
            boundary,
            torch.tensor([BENCHMARK.traction, 0.0], dtype=DTYPE),
            MATERIAL,
        )
        totals.append(components.total_potential)

    expected = torch.tensor(BENCHMARK.exact_total_potential, dtype=DTYPE)
    for total in totals:
        torch.testing.assert_close(total, expected, atol=1e-14, rtol=0.0)


def test_potential_along_exact_mode_is_stationary_at_unit_amplitude() -> None:
    domain, boundary = _rules(order=3)
    amplitudes = torch.tensor([0.9, 1.0, 1.1], dtype=DTYPE)
    potentials = []

    for amplitude in amplitudes:
        def scaled_displacement(coordinates: torch.Tensor) -> torch.Tensor:
            return amplitude * BENCHMARK.displacement(coordinates)

        potentials.append(
            elastic_potential_energy(
                scaled_displacement,
                domain,
                boundary,
                torch.tensor([BENCHMARK.traction, 0.0], dtype=DTYPE),
                MATERIAL,
            ).total_potential
        )

    assert potentials[1] < potentials[0]
    assert potentials[1] < potentials[2]