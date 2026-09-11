from dataclasses import asdict
from pathlib import Path
import tomllib

from griffith2d.benchmarks.coupling import StaggeredCouplingRun, run


CONFIG_PATH = (
    Path(__file__).parents[2]
    / "configs"
    / "baselines"
    / "staggered_uniform_tension.toml"
)


def test_staggered_path_matches_reference_and_is_deterministic() -> None:
    with CONFIG_PATH.open("rb") as stream:
        config = tomllib.load(stream)
    first = run(CONFIG_PATH)
    second = run(CONFIG_PATH)

    assert isinstance(first, StaggeredCouplingRun)
    assert asdict(first) == asdict(second)
    assert first.stop_reason == "completed"
    assert len(first.steps) == len(config["loading"]["tractions"])
    assert all(step.committed for step in first.steps)
    assert max(step.displacement_relative_error for step in first.steps) <= config["acceptance"]["displacement_relative_error"]
    assert max(step.phase_absolute_error for step in first.steps) <= config["acceptance"]["phase_absolute_error"]
    assert max(step.reaction_relative_error for step in first.steps) <= config["acceptance"]["reaction_relative_error"]
    assert max(step.scalar_residual for step in first.steps) <= config["acceptance"]["scalar_residual_absolute"]
    assert any(step.root_count > 1 for step in first.steps)
    assert first.steps[0].network_state_sha256 is None
    assert all(step.network_state_sha256 is not None for step in first.steps[1:])

    loading_peak = first.steps[2]
    unloaded = first.steps[3]
    reloaded_below_peak = first.steps[4]
    reloaded_above_peak = first.steps[5]
    hold_tolerance = config["acceptance"]["history_hold_absolute"]
    assert abs(unloaded.history_max - loading_peak.history_max) <= hold_tolerance
    assert abs(reloaded_below_peak.history_max - loading_peak.history_max) <= hold_tolerance
    assert unloaded.phase_max == loading_peak.phase_max
    assert reloaded_below_peak.phase_max == loading_peak.phase_max
    assert reloaded_above_peak.history_max > loading_peak.history_max
    assert reloaded_above_peak.phase_max > loading_peak.phase_max

    coupling = config["coupling"]
    for step in first.steps[1:]:
        final_iteration = step.iterations[-1]
        assert step.stop_reason == "converged"
        assert final_iteration.displacement_stop_reason == "lbfgs_converged"
        assert final_iteration.displacement_relative_change <= coupling["displacement_relative_tolerance"]
        assert final_iteration.phase_change <= coupling["phase_absolute_tolerance"]
        assert final_iteration.history_relative_change <= coupling["history_relative_tolerance"]
        assert final_iteration.energy_relative_change <= coupling["energy_relative_tolerance"]