import pytest
import torch

from griffith2d.diagnostics import PlaneStrainTensionBenchmark, UniformDegradedTensionBenchmark
from griffith2d.discretization import line_quadrature, rectangle_quadrature
from griffith2d.physics import IsotropicMaterial, degraded_stress
from griffith2d.solvers import degraded_elastic_potential_energy
from griffith2d.solvers import DegradedElasticDEMTrainingConfig, train_degraded_elastic_dem


DTYPE = torch.float64
MATERIAL = IsotropicMaterial(young_modulus=1.0, poisson_ratio=0.3)
ELASTIC = PlaneStrainTensionBenchmark(1.0, 1.0, 0.1, MATERIAL)
RESIDUAL_STIFFNESS = 1e-3


def _rules():
    domain = rectangle_quadrature((0.0, 1.0), (0.0, 1.0), order=3)
    right = line_quadrature(
        torch.tensor([1.0, 0.0], dtype=DTYPE),
        torch.tensor([1.0, 1.0], dtype=DTYPE),
        order=3,
    )
    return domain, right


@pytest.mark.parametrize("phase", [0.0, 0.4, 0.8])
def test_exact_degraded_state_has_expected_compliance_and_energy(phase: float) -> None:
    benchmark = UniformDegradedTensionBenchmark(ELASTIC, phase, RESIDUAL_STIFFNESS)
    domain, right = _rules()
    phase_values = torch.full((domain.points.shape[0],), phase, dtype=DTYPE)

    energy = degraded_elastic_potential_energy(
        benchmark.displacement,
        phase_values,
        domain,
        right,
        torch.tensor([ELASTIC.traction, 0.0], dtype=DTYPE),
        MATERIAL,
        RESIDUAL_STIFFNESS,
    )
    stress = degraded_stress(
        benchmark.strain(domain.points), phase_values, MATERIAL, RESIDUAL_STIFFNESS
    )

    expected_degradation = (1.0 - phase) ** 2 + RESIDUAL_STIFFNESS
    assert benchmark.compliance_ratio == pytest.approx(1.0 / expected_degradation)
    torch.testing.assert_close(stress, benchmark.stress(domain.points), atol=1e-13, rtol=0.0)
    torch.testing.assert_close(
        energy.internal_energy,
        torch.tensor(benchmark.exact_internal_energy, dtype=DTYPE),
        atol=1e-13,
        rtol=0.0,
    )
    torch.testing.assert_close(
        energy.traction_work,
        torch.tensor(benchmark.exact_traction_work, dtype=DTYPE),
        atol=1e-13,
        rtol=0.0,
    )
    torch.testing.assert_close(
        energy.total_potential,
        torch.tensor(benchmark.exact_total_potential, dtype=DTYPE),
        atol=1e-13,
        rtol=0.0,
    )


def test_phase_values_must_explicitly_match_domain_points() -> None:
    domain, right = _rules()

    with pytest.raises(ValueError, match="number_of_domain_points"):
        degraded_elastic_potential_energy(
            ELASTIC.displacement,
            torch.tensor(0.0, dtype=DTYPE),
            domain,
            right,
            torch.tensor([ELASTIC.traction, 0.0], dtype=DTYPE),
            MATERIAL,
            RESIDUAL_STIFFNESS,
        )


def test_warm_start_uses_explicit_reduced_adam_budget() -> None:
    benchmark = UniformDegradedTensionBenchmark(ELASTIC, 0.4, RESIDUAL_STIFFNESS)
    config = DegradedElasticDEMTrainingConfig(
        adam_steps=2,
        warm_start_adam_steps=0,
        lbfgs_max_iter=1,
    )
    cold = train_degraded_elastic_dem(benchmark, config)
    state = {name: value.detach().clone() for name, value in cold.model.network.state_dict().items()}

    warm = train_degraded_elastic_dem(benchmark, config, initial_network_state=state)

    assert len(cold.adam_potential_history) == 2
    assert warm.adam_potential_history == ()


def test_warm_start_preserves_solution_and_reduces_optimizer_work() -> None:
    benchmark = UniformDegradedTensionBenchmark(ELASTIC, 0.4, RESIDUAL_STIFFNESS)
    config = DegradedElasticDEMTrainingConfig()
    cold = train_degraded_elastic_dem(benchmark, config)
    state = {
        name: value.detach().clone()
        for name, value in cold.model.network.state_dict().items()
    }

    warm = train_degraded_elastic_dem(benchmark, config, initial_network_state=state)

    assert warm.final_potential == cold.final_potential
    assert warm.lbfgs_closure_evaluations <= cold.lbfgs_closure_evaluations
    assert warm.stop_reason == "lbfgs_converged"