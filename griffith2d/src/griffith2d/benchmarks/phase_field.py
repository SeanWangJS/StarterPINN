"""Run deterministic independent AT2 phase-field benchmarks."""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import tomllib

import torch

from griffith2d.diagnostics import (
    PhaseFieldBenchmarkMetrics,
    PrecrackProfileBenchmark,
    evaluate_phase_field_benchmark,
    uniform_history_phase,
)
from griffith2d.discretization import interval_quadrature
from griffith2d.physics import PhaseFieldMaterial
from griffith2d.solvers import PhaseFieldDEMTrainingConfig, phase_field_energy, train_phase_field_dem


@dataclass(frozen=True, slots=True)
class PhaseFieldRun:
    """Serializable result of a stable independent phase-field benchmark."""

    kind: str
    initial_energy: float
    final_energy: float
    exact_energy: float
    lbfgs_closure_evaluations: int
    stop_reason: str
    metrics: PhaseFieldBenchmarkMetrics


def run(config_path: Path) -> PhaseFieldRun:
    """Run the phase-field benchmark described by a complete TOML configuration."""

    with config_path.open("rb") as stream:
        config = tomllib.load(stream)

    benchmark_config = config["benchmark"]
    material_config = config["material"]
    quadrature = config["quadrature"]
    network = config["network"]
    optimizer = config["optimizer"]
    if network["dtype"] != "float64":
        raise ValueError("stage-003 baseline requires network.dtype = 'float64'")
    if optimizer["line_search"] != "strong_wolfe":
        raise ValueError("stage-003 baseline requires optimizer.line_search = 'strong_wolfe'")

    material = PhaseFieldMaterial(
        material_config["fracture_toughness"], material_config["length_scale"]
    )
    kind = benchmark_config["kind"]
    half_width = benchmark_config["half_width"]
    history_value = benchmark_config["history"]
    if kind not in {"precrack", "uniform_history"}:
        raise ValueError("benchmark.kind must be 'precrack' or 'uniform_history'")
    if kind == "precrack" and history_value != 0.0:
        raise ValueError("precrack benchmark requires zero history")

    def history_function(coordinates: torch.Tensor) -> torch.Tensor:
        return torch.full_like(coordinates, history_value)

    training_config = PhaseFieldDEMTrainingConfig(
        quadrature_order=quadrature["training_order"],
        hidden_width=network["hidden_width"],
        seed=network["seed"],
        zero_initialization=network["zero_initialization"],
        adam_steps=optimizer["adam_steps"],
        adam_learning_rate=optimizer["adam_learning_rate"],
        lbfgs_max_iter=optimizer["lbfgs_max_iter"],
        lbfgs_tolerance_grad=optimizer["lbfgs_tolerance_grad"],
        lbfgs_tolerance_change=optimizer["lbfgs_tolerance_change"],
    )
    training = train_phase_field_dem(
        half_width,
        material,
        history_function,
        precrack=kind == "precrack",
        config=training_config,
    )
    evaluation_rule = interval_quadrature(0.0, half_width, quadrature["evaluation_order"])
    history = history_function(evaluation_rule.points)
    if kind == "precrack":
        reference = PrecrackProfileBenchmark(half_width, material)
        exact_phase = reference.phase
        exact_energy = reference.exact_fracture_energy
    else:
        exact_phase = lambda coordinates: uniform_history_phase(
            history_function(coordinates), material, validate=False
        )
        exact_energy = float(
            phase_field_energy(exact_phase, evaluation_rule, history, material).total
        )
    metrics = evaluate_phase_field_benchmark(
        training.model,
        exact_phase,
        exact_energy,
        evaluation_rule,
        history,
        material,
        hard_center=kind == "precrack",
    )
    return PhaseFieldRun(
        kind=kind,
        initial_energy=training.initial_energy,
        final_energy=training.final_energy,
        exact_energy=exact_energy,
        lbfgs_closure_evaluations=training.lbfgs_closure_evaluations,
        stop_reason=training.stop_reason,
        metrics=metrics,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(asdict(run(arguments.config)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()