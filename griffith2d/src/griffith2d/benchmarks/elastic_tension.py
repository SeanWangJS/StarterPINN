"""Run the deterministic uniform plane-strain tension DEM benchmark."""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import tomllib

import torch

from griffith2d.diagnostics import (
    ElasticBenchmarkMetrics,
    PlaneStrainTensionBenchmark,
    evaluate_elastic_benchmark,
)
from griffith2d.discretization import line_quadrature, rectangle_quadrature
from griffith2d.physics import IsotropicMaterial
from griffith2d.solvers import ElasticDEMTrainingConfig, train_elastic_dem


@dataclass(frozen=True, slots=True)
class ElasticTensionRun:
    """Serializable result of the stable elastic tension benchmark."""

    initial_potential: float
    final_potential: float
    exact_total_potential: float
    lbfgs_closure_evaluations: int
    stop_reason: str
    metrics: ElasticBenchmarkMetrics


def _line(start: tuple[float, float], end: tuple[float, float], order: int):
    return line_quadrature(
        torch.tensor(start, dtype=torch.float64),
        torch.tensor(end, dtype=torch.float64),
        order,
    )


def run(config_path: Path) -> ElasticTensionRun:
    """Run the benchmark described by a complete TOML configuration."""

    with config_path.open("rb") as stream:
        config = tomllib.load(stream)

    geometry = config["geometry"]
    material_config = config["material"]
    loading = config["loading"]
    quadrature = config["quadrature"]
    network = config["network"]
    optimizer = config["optimizer"]
    if network["dtype"] != "float64":
        raise ValueError("stage-002 baseline requires network.dtype = 'float64'")
    if optimizer["line_search"] != "strong_wolfe":
        raise ValueError("stage-002 baseline requires optimizer.line_search = 'strong_wolfe'")

    material = IsotropicMaterial(
        material_config["young_modulus"], material_config["poisson_ratio"]
    )
    benchmark = PlaneStrainTensionBenchmark(
        geometry["length"], geometry["height"], loading["traction"], material
    )
    training_config = ElasticDEMTrainingConfig(
        quadrature_order=quadrature["training_order"],
        hidden_width=network["hidden_width"],
        seed=network["seed"],
        adam_steps=optimizer["adam_steps"],
        adam_learning_rate=optimizer["adam_learning_rate"],
        lbfgs_max_iter=optimizer["lbfgs_max_iter"],
        lbfgs_tolerance_grad=optimizer["lbfgs_tolerance_grad"],
        lbfgs_tolerance_change=optimizer["lbfgs_tolerance_change"],
    )
    training = train_elastic_dem(benchmark, training_config)

    evaluation_order = quadrature["evaluation_order"]
    length = benchmark.length
    height = benchmark.height
    domain = rectangle_quadrature((0.0, length), (0.0, height), evaluation_order)
    right = _line((length, 0.0), (length, height), evaluation_order)
    top = _line((0.0, height), (length, height), evaluation_order)
    left = _line((0.0, 0.0), (0.0, height), evaluation_order)
    bottom = _line((0.0, 0.0), (length, 0.0), evaluation_order)
    metrics = evaluate_elastic_benchmark(
        training.model, benchmark, domain, right, top, left, bottom
    )
    return ElasticTensionRun(
        initial_potential=training.initial_potential,
        final_potential=training.final_potential,
        exact_total_potential=benchmark.exact_total_potential,
        lbfgs_closure_evaluations=training.lbfgs_closure_evaluations,
        stop_reason=training.stop_reason,
        metrics=metrics,
    )


def main() -> None:
    """CLI entry point for the stable elastic tension benchmark."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baselines/elastic_tension.toml"),
        help="Path to the complete TOML benchmark configuration",
    )
    arguments = parser.parse_args()
    print(json.dumps(asdict(run(arguments.config)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
