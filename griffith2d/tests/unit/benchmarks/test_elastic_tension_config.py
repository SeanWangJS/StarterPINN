from dataclasses import asdict
from pathlib import Path
import tomllib

from griffith2d.diagnostics import ElasticBenchmarkMetrics


CONFIG_PATH = Path(__file__).parents[3] / "configs" / "baselines" / "elastic_tension.toml"


def test_elastic_tension_baseline_records_all_frozen_decisions() -> None:
    with CONFIG_PATH.open("rb") as stream:
        config = tomllib.load(stream)

    assert set(config) == {
        "geometry",
        "material",
        "loading",
        "quadrature",
        "network",
        "optimizer",
        "acceptance",
    }
    assert config["network"] == {
        "hidden_width": 16,
        "seed": 1729,
        "dtype": "float64",
    }
    assert config["optimizer"]["adam_steps"] == 1000
    assert config["optimizer"]["lbfgs_max_iter"] == 200
    assert config["acceptance"]["equilibrium_relative_rms"] == 2e-2


def test_benchmark_metrics_serialize_with_named_fields() -> None:
    metrics = ElasticBenchmarkMetrics(*(0.0 for _ in range(8)))

    serialized = asdict(metrics)

    assert set(serialized) == {
        "displacement_relative_l2",
        "strain_relative_l2",
        "stress_relative_l2",
        "reaction_relative_error",
        "right_traction_relative_rms",
        "top_traction_relative_rms",
        "equilibrium_relative_rms",
        "hard_boundary_max_abs",
    }