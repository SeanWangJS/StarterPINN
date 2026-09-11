from pathlib import Path
import tomllib

import torch

from griffith2d.benchmarks.phase_field import run
from griffith2d.diagnostics import uniform_history_phase
from griffith2d.physics import PhaseFieldMaterial, update_history
from griffith2d.solvers import train_phase_field_dem


ROOT = Path(__file__).resolve().parents[2]
PRECRACK_CONFIG = ROOT / "configs" / "baselines" / "phase_field_precrack.toml"
UNIFORM_CONFIG = ROOT / "configs" / "baselines" / "phase_field_uniform_history.toml"


def _assert_acceptance(config_path: Path) -> None:
    with config_path.open("rb") as stream:
        acceptance = tomllib.load(stream)["acceptance"]
    metrics = run(config_path).metrics

    assert metrics.phase_relative_l2 <= acceptance["phase_relative_l2"]
    assert metrics.phase_max_abs <= acceptance["phase_max_abs"]
    assert metrics.energy_relative_error <= acceptance["energy_relative_error"]
    assert metrics.residual_relative_rms <= acceptance["residual_relative_rms"]
    assert metrics.boundary_flux_max_scaled <= acceptance["boundary_flux_max_scaled"]
    assert metrics.hard_boundary_max_abs <= acceptance["hard_boundary_max_abs"]
    assert metrics.below_zero <= acceptance["bounds_violation"]
    assert metrics.above_one <= acceptance["bounds_violation"]


def test_stable_phase_field_configurations_meet_frozen_acceptance() -> None:
    _assert_acceptance(PRECRACK_CONFIG)
    _assert_acceptance(UNIFORM_CONFIG)


def test_precrack_training_is_parameter_deterministic() -> None:
    material = PhaseFieldMaterial(fracture_toughness=2.4, length_scale=0.2)

    def zero_history(coordinates: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(coordinates)

    first = train_phase_field_dem(1.0, material, zero_history, precrack=True)
    second = train_phase_field_dem(1.0, material, zero_history, precrack=True)

    assert first.final_energy == second.final_energy
    assert first.stop_reason == second.stop_reason
    for first_parameter, second_parameter in zip(
        first.model.parameters(), second.model.parameters(), strict=True
    ):
        torch.testing.assert_close(first_parameter, second_parameter, rtol=0.0, atol=0.0)


def test_uniform_loading_unloading_sequence_is_monotone() -> None:
    material = PhaseFieldMaterial(fracture_toughness=2.4, length_scale=0.2)
    candidates = (0.0, 3.0, 1.0, 6.0, 2.0)
    history = torch.zeros((1, 1), dtype=torch.float64)
    phases = []

    for candidate in candidates:
        history = update_history(history, torch.full_like(history, candidate))
        phases.append(uniform_history_phase(history, material))

    history_values = [0.0, 3.0, 3.0, 6.0, 6.0]
    for phase, expected_history in zip(phases, history_values, strict=True):
        expected = uniform_history_phase(
            torch.tensor([[expected_history]], dtype=torch.float64), material
        )
        torch.testing.assert_close(phase, expected)
    assert all(bool(next_phase >= phase) for phase, next_phase in zip(phases, phases[1:]))