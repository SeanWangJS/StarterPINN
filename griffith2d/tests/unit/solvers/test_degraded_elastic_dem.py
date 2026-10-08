import pytest
import torch

from griffith2d.diagnostics import PlaneStrainTensionBenchmark, UniformDegradedTensionBenchmark
from griffith2d.discretization import CenterCrackPlate, DisplacementMLP, line_quadrature, rectangle_quadrature
from griffith2d.physics import IsotropicMaterial, degraded_stress
from griffith2d.solvers import degraded_elastic_potential_energy
from griffith2d.solvers import DegradedElasticDEMTrainingConfig, train_degraded_elastic_dem
from griffith2d.solvers import (
    FrozenPlateElasticDEMTrainingConfig,
    frozen_plate_elastic_energy,
    train_frozen_plate_elastic_dem,
)


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


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_frozen_plate_energy_and_gradient_match_nonuniform_affine_formula(device: str) -> None:
    domain = rectangle_quadrature((-2.0, 2.0), (-1.0, 1.0), order=4, device=device)
    phase = (0.4 + 0.05 * domain.points[:, 0]).requires_grad_()
    snapshot = phase.detach().clone()
    vertical_strain = torch.nn.Parameter(torch.tensor(0.01, dtype=DTYPE, device=device))
    horizontal_strain = -MATERIAL.poisson_ratio / (1.0 - MATERIAL.poisson_ratio) * 0.01

    def displacement(point):
        return torch.stack((horizontal_strain * point[..., 0], vertical_strain * point[..., 1]), dim=-1)

    energy = frozen_plate_elastic_energy(displacement, phase, domain, MATERIAL, RESIDUAL_STIFFNESS)
    degraded_area = domain.integrate((1.0 - phase.detach()).square() + RESIDUAL_STIFFNESS)
    effective_modulus = MATERIAL.young_modulus / (1.0 - MATERIAL.poisson_ratio**2)
    torch.testing.assert_close(energy, 0.5 * effective_modulus * 0.01**2 * degraded_area)
    energy.backward()
    torch.testing.assert_close(vertical_strain.grad, effective_modulus * 0.01 * degraded_area)
    assert phase.grad is None
    torch.testing.assert_close(phase.detach(), snapshot, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_frozen_plate_training_smoke_preserves_phase_and_reports_actual_stationarity(device: str) -> None:
    plate = CenterCrackPlate(2.0, 1.0, 0.5)
    domain = plate.domain_rule(0.2, order=2, device=device)
    phase = plate.initial_phase(domain.points, 0.2).squeeze(-1).requires_grad_()
    snapshot = phase.detach().clone()
    config = FrozenPlateElasticDEMTrainingConfig(
        hidden_width=8, adam_steps=1, lbfgs_max_iter=2, parameter_gradient_tolerance=1e-30,
    )
    result = train_frozen_plate_elastic_dem(plate, 0.01, 0.01, phase, domain, MATERIAL, RESIDUAL_STIFFNESS, config)
    assert result.final_energy < result.initial_energy
    assert len(result.adam_energy_history) == 1
    assert result.lbfgs_closure_evaluations > 0
    assert result.final_parameter_gradient_inf > config.parameter_gradient_tolerance
    assert result.stop_reason == "optimizer_stopped_without_stationarity"
    assert phase.grad is None
    torch.testing.assert_close(phase.detach(), snapshot, rtol=0.0, atol=0.0)
    boundaries = plate.boundary_rules(2, device=device)
    torch.testing.assert_close(result.model(boundaries.top.points)[:, 1], phase.new_full((2,), 0.01), rtol=0.0, atol=0.0)
    torch.testing.assert_close(result.model(boundaries.bottom.points)[:, 1], phase.new_full((2,), -0.01), rtol=0.0, atol=0.0)
    actual_energy = frozen_plate_elastic_energy(result.model, phase, domain, MATERIAL, RESIDUAL_STIFFNESS)
    assert result.final_energy == pytest.approx(float(actual_energy.detach().cpu()), rel=1e-12)
    gradients = torch.autograd.grad(actual_energy, tuple(result.model.parameters()))
    gradient_inf = max(float(gradient.detach().abs().max().cpu()) for gradient in gradients)
    assert result.final_parameter_gradient_inf == pytest.approx(gradient_inf, rel=1e-12)


def test_frozen_plate_zero_load_zero_network_is_stationary_without_changing_initial_state() -> None:
    plate = CenterCrackPlate(2.0, 1.0, 0.5)
    domain = plate.domain_rule(0.2, order=2)
    phase = plate.initial_phase(domain.points, 0.2).squeeze(-1)
    config = FrozenPlateElasticDEMTrainingConfig(hidden_width=8, lbfgs_max_iter=1)
    network = DisplacementMLP(hidden_width=8)
    state = {name: torch.zeros_like(value) for name, value in network.state_dict().items()}
    snapshot = {name: value.clone() for name, value in state.items()}
    result = train_frozen_plate_elastic_dem(
        plate, 0.0, 0.01, phase, domain, MATERIAL, RESIDUAL_STIFFNESS, config, initial_network_state=state,
    )
    assert result.initial_energy == result.final_energy == 0.0
    assert result.final_parameter_gradient_inf == 0.0
    assert result.stop_reason == "parameter_gradient_tolerance"
    for name, value in state.items():
        torch.testing.assert_close(value, snapshot[name], rtol=0.0, atol=0.0)


def test_frozen_plate_compressive_volume_is_not_degraded() -> None:
    domain = rectangle_quadrature((-2.0, 2.0), (-1.0, 1.0), order=3)
    phase = 0.4 + 0.05 * domain.points[:, 0]
    gradient = torch.tensor([[0.01, 0.003], [-0.004, -0.02]], dtype=DTYPE)

    def displacement(point):
        return point @ gradient.T

    strain = 0.5 * (gradient + gradient.T)
    trace = strain.trace()
    active = MATERIAL.shear_modulus * (strain.square().sum() - trace.square() / 3.0)
    inactive = 0.5 * MATERIAL.bulk_modulus * trace.square()
    expected = domain.integrate(((1.0 - phase).square() + RESIDUAL_STIFFNESS) * active + inactive)
    actual = frozen_plate_elastic_energy(displacement, phase, domain, MATERIAL, RESIDUAL_STIFFNESS)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("fault", ["shape", "dtype", "negative", "above_one", "nan", "not_tensor"])
def test_frozen_plate_energy_rejects_invalid_phase(fault) -> None:
    domain = rectangle_quadrature((-2.0, 2.0), (-1.0, 1.0), order=2)
    phase = domain.points.new_full((len(domain.points),), 0.4)
    error = ValueError
    if fault == "shape":
        phase = phase[:, None]
    elif fault == "dtype":
        phase = phase.to(torch.float32)
    elif fault == "negative":
        phase[0] = -0.01
    elif fault == "above_one":
        phase[0] = 1.01
    elif fault == "nan":
        phase[0] = float("nan")
    else:
        phase = [0.4]
        error = TypeError
    with pytest.raises(error, match="phase_field"):
        frozen_plate_elastic_energy(ELASTIC.displacement, phase, domain, MATERIAL, RESIDUAL_STIFFNESS)


@pytest.mark.parametrize("name,value,error", [
    ("hidden_width", 0, ValueError),
    ("seed", True, TypeError),
    ("adam_steps", -1, ValueError),
    ("lbfgs_max_iter", 0, ValueError),
    ("adam_learning_rate", float("nan"), ValueError),
    ("parameter_gradient_tolerance", 0.0, ValueError),
    ("lbfgs_tolerance_change", float("inf"), ValueError),
])
def test_frozen_plate_training_config_rejects_invalid_budget_and_tolerances(name, value, error) -> None:
    with pytest.raises(error, match=name):
        FrozenPlateElasticDEMTrainingConfig(**{name: value})


@pytest.mark.parametrize("fault", ["optimizer", "nonfinite_energy"])
def test_frozen_plate_training_failure_preserves_phase_and_initial_state(fault, monkeypatch) -> None:
    plate = CenterCrackPlate(2.0, 1.0, 0.5)
    domain = plate.domain_rule(0.2, order=2)
    phase = plate.initial_phase(domain.points, 0.2).squeeze(-1)
    snapshot = phase.clone()
    state = DisplacementMLP(hidden_width=8).state_dict()
    state_snapshot = {name: value.clone() for name, value in state.items()}
    if fault == "optimizer":
        def fail_step(*args, **kwargs):
            raise RuntimeError("injected optimizer failure")

        monkeypatch.setattr(torch.optim.LBFGS, "step", fail_step)
        message = "injected optimizer failure"
    else:
        import griffith2d.solvers.degraded_elastic_dem as module

        monkeypatch.setattr(module, "frozen_plate_elastic_energy", lambda *args: phase.new_tensor(float("nan")))
        message = "energy is nonfinite"
    with pytest.raises(RuntimeError, match=message):
        train_frozen_plate_elastic_dem(
            plate, 0.01, 0.01, phase, domain, MATERIAL, RESIDUAL_STIFFNESS,
            FrozenPlateElasticDEMTrainingConfig(hidden_width=8, lbfgs_max_iter=1), initial_network_state=state,
        )
    torch.testing.assert_close(phase, snapshot, rtol=0.0, atol=0.0)
    for name, value in state.items():
        torch.testing.assert_close(value, state_snapshot[name], rtol=0.0, atol=0.0)


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