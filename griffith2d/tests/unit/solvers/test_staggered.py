from types import SimpleNamespace

import pytest
import torch
from torch import nn

from griffith2d.discretization import rectangle_quadrature
from griffith2d.physics import IsotropicMaterial, PhaseFieldMaterial
from griffith2d.solvers import (
    StaggeredCouplingConfig,
    active_energy_at_points,
    expand_uniform_phase,
    initial_staggered_state,
    run_staggered_load_path,
    solve_staggered_load_step,
    solve_uniform_phase,
)


DTYPE = torch.float64
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class _AffineDisplacement(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Linear(1, 1, bias=False, dtype=DTYPE)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return torch.stack((0.1 * points[..., 0], -0.03 * points[..., 1]), dim=-1)


def _deterministic_displacement_train(*args, **kwargs):
    return SimpleNamespace(
        model=_AffineDisplacement(),
        final_potential=-0.005,
        stop_reason="lbfgs_converged",
    )


def test_uniform_phase_is_explicit_and_matches_closed_form_minimizer() -> None:
    rule = rectangle_quadrature((0.0, 2.0), (0.0, 3.0), order=3)
    material = PhaseFieldMaterial(fracture_toughness=2.4, length_scale=0.2)
    history = torch.full((rule.points.shape[0],), 3.0, dtype=DTYPE)

    result = solve_uniform_phase(history, rule, material)
    values = expand_uniform_phase(result.phase, rule)

    assert result.phase == pytest.approx(1.0 / 3.0)
    assert abs(result.stationarity) < 1e-13
    torch.testing.assert_close(values, torch.full_like(values, 1.0 / 3.0))


def test_active_energy_extraction_matches_hand_calculated_affine_strain() -> None:
    rule = rectangle_quadrature((0.0, 1.0), (0.0, 1.0), order=3)
    material = IsotropicMaterial(1.0, 0.3)
    field = _AffineDisplacement()
    points_before = rule.points.clone()
    parameters_before = tuple(parameter.detach().clone() for parameter in field.parameters())

    active = active_energy_at_points(field, rule.points, material)

    volumetric_strain = 0.1 - 0.03
    deviatoric_norm_squared = sum(
        value**2
        for value in (
            0.1 - volumetric_strain / 3.0,
            -0.03 - volumetric_strain / 3.0,
            -volumetric_strain / 3.0,
        )
    )
    expected = (
        0.5 * material.bulk_modulus * volumetric_strain**2
        + material.shear_modulus * deviatoric_norm_squared
    )
    torch.testing.assert_close(active, torch.full_like(active, expected))
    torch.testing.assert_close(rule.points, points_before)
    for parameter, parameter_before in zip(field.parameters(), parameters_before, strict=True):
        torch.testing.assert_close(parameter, parameter_before)


@pytest.mark.parametrize("device", DEVICES)
def test_uniform_phase_transfer_preserves_coupling_dtype_and_device(device: str) -> None:
    rule = rectangle_quadrature(
        (0.0, 1.0), (0.0, 1.0), order=3, dtype=DTYPE, device=device
    )
    material = PhaseFieldMaterial(fracture_toughness=2.4, length_scale=0.2)
    history = torch.full((rule.points.shape[0],), 3.0, dtype=DTYPE, device=device)

    result = solve_uniform_phase(history, rule, material)
    phase = expand_uniform_phase(result.phase, rule)

    assert phase.dtype == DTYPE
    assert phase.device.type == device
    torch.testing.assert_close(phase, torch.full_like(phase, 1.0 / 3.0))


def test_failed_step_discards_computed_candidate_state(monkeypatch) -> None:
    monkeypatch.setattr(
        "griffith2d.solvers.staggered.train_degraded_elastic_dem",
        _deterministic_displacement_train,
    )
    config = StaggeredCouplingConfig(max_iterations=1)
    previous = initial_staggered_state(1.0, 1.0, config)

    result = solve_staggered_load_step(
        previous,
        traction=0.1,
        length=1.0,
        height=1.0,
        elastic_material=IsotropicMaterial(1.0, 0.3),
        phase_material=PhaseFieldMaterial(2.4, 0.2),
        residual_stiffness=1e-3,
        config=config,
    )

    assert not result.committed
    assert result.state is previous
    assert len(result.iterations) == 1
    assert result.iterations[0].phase_after > 0.0
    assert result.iterations[0].history_max > 0.0
    assert result.stop_reason == "iteration_budget"


def test_converged_step_commits_candidate_only_after_second_fixed_point_iteration(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "griffith2d.solvers.staggered.train_degraded_elastic_dem",
        _deterministic_displacement_train,
    )
    config = StaggeredCouplingConfig(max_iterations=2)
    previous = initial_staggered_state(1.0, 1.0, config)

    result = solve_staggered_load_step(
        previous,
        traction=0.1,
        length=1.0,
        height=1.0,
        elastic_material=IsotropicMaterial(1.0, 0.3),
        phase_material=PhaseFieldMaterial(2.4, 0.2),
        residual_stiffness=1e-3,
        config=config,
    )

    assert result.committed
    assert result.state is not previous
    assert result.state.load_index == 0
    assert result.state.phase > previous.phase
    assert len(result.iterations) == 2
    assert result.iterations[-1].phase_change == pytest.approx(0.0)
    torch.testing.assert_close(result.state.history, torch.full_like(result.state.history, result.state.history[0]))


def test_load_path_stops_at_failed_step_and_keeps_last_commit(monkeypatch) -> None:
    monkeypatch.setattr(
        "griffith2d.solvers.staggered.train_degraded_elastic_dem",
        _deterministic_displacement_train,
    )
    config = StaggeredCouplingConfig(max_iterations=1)

    result = run_staggered_load_path(
        (0.0, 0.1, 0.2),
        length=1.0,
        height=1.0,
        elastic_material=IsotropicMaterial(1.0, 0.3),
        phase_material=PhaseFieldMaterial(2.4, 0.2),
        residual_stiffness=1e-3,
        config=config,
    )

    assert result.stop_reason == "load_step_failed"
    assert len(result.steps) == 2
    assert result.steps[0].committed
    assert not result.steps[1].committed
    assert result.final_state is result.steps[0].state


def test_zero_load_commits_exact_mechanical_state_without_changing_history() -> None:
    config = StaggeredCouplingConfig()
    previous = initial_staggered_state(1.0, 1.0, config)

    result = solve_staggered_load_step(
        previous,
        traction=0.0,
        length=1.0,
        height=1.0,
        elastic_material=IsotropicMaterial(1.0, 0.3),
        phase_material=PhaseFieldMaterial(2.4, 0.2),
        residual_stiffness=1e-3,
        config=config,
    )

    assert result.committed
    assert result.state.load_index == 0
    assert result.state.phase == 0.0
    torch.testing.assert_close(result.state.history, previous.history)
    torch.testing.assert_close(result.state.displacement_at_points, torch.zeros_like(previous.displacement_at_points))