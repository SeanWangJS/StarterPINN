import torch

from griffith2d.diagnostics import PlaneStrainTensionBenchmark, evaluate_elastic_benchmark
from griffith2d.discretization import line_quadrature, rectangle_quadrature
from griffith2d.physics import IsotropicMaterial
from griffith2d.solvers import ElasticDEMTrainingConfig, train_elastic_dem


DTYPE = torch.float64
BENCHMARK = PlaneStrainTensionBenchmark(
    length=1.0,
    height=1.0,
    traction=0.1,
    material=IsotropicMaterial(young_modulus=1.0, poisson_ratio=0.3),
)
CONFIG = ElasticDEMTrainingConfig()


def _evaluation_rules():
    domain = rectangle_quadrature((0.0, 1.0), (0.0, 1.0), order=8)
    right = line_quadrature(
        torch.tensor([1.0, 0.0], dtype=DTYPE), torch.tensor([1.0, 1.0], dtype=DTYPE), 8
    )
    top = line_quadrature(
        torch.tensor([0.0, 1.0], dtype=DTYPE), torch.tensor([1.0, 1.0], dtype=DTYPE), 8
    )
    left = line_quadrature(
        torch.tensor([0.0, 0.0], dtype=DTYPE), torch.tensor([0.0, 1.0], dtype=DTYPE), 8
    )
    bottom = line_quadrature(
        torch.tensor([0.0, 0.0], dtype=DTYPE), torch.tensor([1.0, 0.0], dtype=DTYPE), 8
    )
    return domain, right, top, left, bottom


def test_elastic_dem_recovers_uniform_plane_strain_tension() -> None:
    first = train_elastic_dem(BENCHMARK, CONFIG)
    second = train_elastic_dem(BENCHMARK, CONFIG)
    rules = _evaluation_rules()
    metrics = evaluate_elastic_benchmark(first.model, BENCHMARK, *rules)

    assert metrics.displacement_relative_l2 <= 2e-3
    assert metrics.strain_relative_l2 <= 5e-3
    assert metrics.stress_relative_l2 <= 5e-3
    assert metrics.reaction_relative_error <= 5e-3
    assert metrics.right_traction_relative_rms <= 2e-2
    assert metrics.top_traction_relative_rms <= 2e-2
    assert metrics.equilibrium_relative_rms <= 2e-2
    assert metrics.hard_boundary_max_abs <= 1e-14
    assert first.final_potential < first.initial_potential
    assert first.final_potential == second.final_potential
    assert first.stop_reason == second.stop_reason
    for first_parameter, second_parameter in zip(
        first.model.parameters(), second.model.parameters(), strict=True
    ):
        torch.testing.assert_close(first_parameter, second_parameter, rtol=0.0, atol=0.0)