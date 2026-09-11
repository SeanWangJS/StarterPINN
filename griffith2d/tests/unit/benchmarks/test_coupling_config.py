from dataclasses import asdict
from pathlib import Path
import tomllib

from griffith2d.benchmarks.coupling import StaggeredCouplingRun


BASELINES = Path(__file__).parents[3] / "configs" / "baselines"


def test_stage_four_baselines_freeze_complete_optimizer_and_path_decisions() -> None:
    paths = (
        BASELINES / "degraded_elastic_tension.toml",
        BASELINES / "staggered_uniform_tension.toml",
    )
    configs = []
    for path in paths:
        with path.open("rb") as stream:
            configs.append(tomllib.load(stream))

    assert {config["benchmark"]["kind"] for config in configs} == {
        "degraded_elastic_tension",
        "staggered_uniform_tension",
    }
    assert all(config["network"]["dtype"] == "float64" for config in configs)
    assert all(config["optimizer"]["warm_start_adam_steps"] == 0 for config in configs)
    assert configs[1]["loading"]["tractions"] == [0.0, 0.02, 0.04, 0.02, 0.03, 0.05]
    assert set(configs[1]["coupling"]) == {
        "max_iterations",
        "displacement_relative_tolerance",
        "phase_absolute_tolerance",
        "history_relative_tolerance",
        "energy_relative_tolerance",
    }


def test_staggered_run_has_named_json_ready_fields() -> None:
    serialized = asdict(StaggeredCouplingRun("staggered_uniform_tension", "completed", ()))

    assert serialized == {
        "benchmark": "staggered_uniform_tension",
        "stop_reason": "completed",
        "steps": (),
    }