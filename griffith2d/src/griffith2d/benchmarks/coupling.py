"""Run stable degraded-elastic and uniform staggered-coupling benchmarks."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import tomllib
from typing import Any

import torch

from griffith2d.diagnostics import (
    DegradedElasticBenchmarkMetrics,
    PlaneStrainTensionBenchmark,
    UniformDegradedTensionBenchmark,
    continue_low_damage_root,
    coupled_phase_residual,
    dimensionless_traction,
    evaluate_degraded_elastic_benchmark,
    plane_strain_tension_modulus,
)
from griffith2d.discretization import line_quadrature, rectangle_quadrature
from griffith2d.physics import IsotropicMaterial, PhaseFieldMaterial
from griffith2d.solvers import (
    DegradedElasticDEMTrainingConfig,
    StaggeredCouplingConfig,
    StaggeredIterationRecord,
    run_staggered_load_path,
    train_degraded_elastic_dem,
)


@dataclass(frozen=True, slots=True)
class DegradedElasticRun:
    benchmark: str
    phase: float
    degradation: float
    initial_potential: float
    final_potential: float
    exact_total_potential: float
    lbfgs_closure_evaluations: int
    stop_reason: str
    metrics: DegradedElasticBenchmarkMetrics


@dataclass(frozen=True, slots=True)
class StaggeredStepRun:
    load_index: int
    traction: float
    committed: bool
    stop_reason: str
    phase_min: float
    phase_max: float
    displacement_relative_error: float
    reference_phase: float
    phase_absolute_error: float
    scalar_residual: float
    history_max: float
    reaction_x: float
    reaction_relative_error: float
    displacement_potential: float
    phase_energy: float
    total_energy: float
    network_state_sha256: str | None
    root_count: int
    iterations: tuple[StaggeredIterationRecord, ...]


@dataclass(frozen=True, slots=True)
class StaggeredCouplingRun:
    benchmark: str
    stop_reason: str
    steps: tuple[StaggeredStepRun, ...]


def _training_config(config: dict[str, Any]) -> DegradedElasticDEMTrainingConfig:
    quadrature = config["quadrature"]
    network = config["network"]
    optimizer = config["optimizer"]
    if network["dtype"] != "float64":
        raise ValueError("coupling baselines require network.dtype = 'float64'")
    if optimizer["line_search"] != "strong_wolfe":
        raise ValueError("coupling baselines require optimizer.line_search = 'strong_wolfe'")
    return DegradedElasticDEMTrainingConfig(
        quadrature_order=quadrature.get("training_order", quadrature.get("coupling_order")),
        hidden_width=network["hidden_width"],
        seed=network["seed"],
        adam_steps=optimizer["adam_steps"],
        warm_start_adam_steps=optimizer["warm_start_adam_steps"],
        adam_learning_rate=optimizer["adam_learning_rate"],
        lbfgs_max_iter=optimizer["lbfgs_max_iter"],
        lbfgs_tolerance_grad=optimizer["lbfgs_tolerance_grad"],
        lbfgs_tolerance_change=optimizer["lbfgs_tolerance_change"],
    )


def _line(start: tuple[float, float], end: tuple[float, float], order: int):
    return line_quadrature(
        torch.tensor(start, dtype=torch.float64),
        torch.tensor(end, dtype=torch.float64),
        order,
    )


def _network_state_sha256(state: dict[str, torch.Tensor] | None) -> str | None:
    if state is None:
        return None
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _run_degraded(config: dict[str, Any]) -> DegradedElasticRun:
    geometry = config["geometry"]
    material_config = config["elastic_material"]
    phase_config = config["phase"]
    material = IsotropicMaterial(
        material_config["young_modulus"], material_config["poisson_ratio"]
    )
    elastic = PlaneStrainTensionBenchmark(
        geometry["length"], geometry["height"], config["loading"]["traction"], material
    )
    benchmark = UniformDegradedTensionBenchmark(
        elastic, phase_config["prescribed"], phase_config["residual_stiffness"]
    )
    training = train_degraded_elastic_dem(benchmark, _training_config(config))
    order = config["quadrature"]["evaluation_order"]
    length = elastic.length
    height = elastic.height
    metrics = evaluate_degraded_elastic_benchmark(
        training.model,
        benchmark,
        rectangle_quadrature((0.0, length), (0.0, height), order),
        _line((length, 0.0), (length, height), order),
        _line((0.0, height), (length, height), order),
        _line((0.0, 0.0), (0.0, height), order),
        _line((0.0, 0.0), (length, 0.0), order),
    )
    return DegradedElasticRun(
        benchmark="degraded_elastic_tension",
        phase=benchmark.phase,
        degradation=benchmark.degradation,
        initial_potential=training.initial_potential,
        final_potential=training.final_potential,
        exact_total_potential=benchmark.exact_total_potential,
        lbfgs_closure_evaluations=training.lbfgs_closure_evaluations,
        stop_reason=training.stop_reason,
        metrics=metrics,
    )


def _run_staggered(config: dict[str, Any]) -> StaggeredCouplingRun:
    geometry = config["geometry"]
    elastic_config = config["elastic_material"]
    phase_config = config["phase_material"]
    elastic_material = IsotropicMaterial(
        elastic_config["young_modulus"], elastic_config["poisson_ratio"]
    )
    phase_material = PhaseFieldMaterial(
        phase_config["fracture_toughness"], phase_config["length_scale"]
    )
    coupling = config["coupling"]
    solver_config = StaggeredCouplingConfig(
        quadrature_order=config["quadrature"]["coupling_order"],
        max_iterations=coupling["max_iterations"],
        displacement_relative_tolerance=coupling["displacement_relative_tolerance"],
        phase_absolute_tolerance=coupling["phase_absolute_tolerance"],
        history_relative_tolerance=coupling["history_relative_tolerance"],
        energy_relative_tolerance=coupling["energy_relative_tolerance"],
        displacement=_training_config(config),
    )
    tractions = tuple(config["loading"]["tractions"])
    path = run_staggered_load_path(
        tractions,
        geometry["length"],
        geometry["height"],
        elastic_material,
        phase_material,
        phase_config["residual_stiffness"],
        solver_config,
    )
    domain = rectangle_quadrature(
        (0.0, geometry["length"]),
        (0.0, geometry["height"]),
        config["quadrature"]["coupling_order"],
    )
    unit_benchmark = PlaneStrainTensionBenchmark(
        geometry["length"], geometry["height"], 1.0, elastic_material
    )
    effective_modulus = plane_strain_tension_modulus(unit_benchmark)
    peak_traction = 0.0
    reference_phase = 0.0
    steps: list[StaggeredStepRun] = []
    for traction, result in zip(tractions, path.steps, strict=True):
        peak_traction = max(peak_traction, traction)
        tau = dimensionless_traction(
            peak_traction,
            effective_modulus,
            phase_material.fracture_toughness,
            phase_material.length_scale,
        )
        root, roots = continue_low_damage_root(
            tau, phase_config["residual_stiffness"], reference_phase
        )
        reference_phase = root.phase
        state = result.state
        if traction == 0.0:
            displacement_error = float(
                torch.sqrt(
                    domain.integrate(state.displacement_at_points.square().sum(dim=-1))
                ).detach().cpu()
            )
        else:
            current_elastic = PlaneStrainTensionBenchmark(
                geometry["length"], geometry["height"], traction, elastic_material
            )
            current_benchmark = UniformDegradedTensionBenchmark(
                current_elastic, state.phase, phase_config["residual_stiffness"]
            )
            exact_displacement = current_benchmark.displacement(domain.points)
            displacement_error_norm = torch.sqrt(
                domain.integrate(
                    (state.displacement_at_points - exact_displacement).square().sum(dim=-1)
                )
            )
            displacement_reference_norm = torch.sqrt(
                domain.integrate(exact_displacement.square().sum(dim=-1))
            )
            displacement_error = float(
                (displacement_error_norm / displacement_reference_norm).detach().cpu()
            )
        expected_reaction = -traction * geometry["height"]
        reaction_error = (
            abs(state.reaction_x - expected_reaction) / abs(expected_reaction)
            if expected_reaction != 0.0
            else abs(state.reaction_x)
        )
        steps.append(
            StaggeredStepRun(
                load_index=state.load_index,
                traction=traction,
                committed=result.committed,
                stop_reason=result.stop_reason,
                phase_min=state.phase,
                phase_max=state.phase,
                displacement_relative_error=displacement_error,
                reference_phase=root.phase,
                phase_absolute_error=abs(state.phase - root.phase),
                scalar_residual=abs(
                    coupled_phase_residual(
                        state.phase, tau, phase_config["residual_stiffness"]
                    )
                ),
                history_max=float(state.history.max().detach().cpu()),
                reaction_x=state.reaction_x,
                reaction_relative_error=reaction_error,
                displacement_potential=state.displacement_potential,
                phase_energy=state.phase_energy,
                total_energy=state.total_energy,
                network_state_sha256=_network_state_sha256(state.network_state),
                root_count=len(roots),
                iterations=result.iterations,
            )
        )
    return StaggeredCouplingRun(
        benchmark="staggered_uniform_tension",
        stop_reason=path.stop_reason,
        steps=tuple(steps),
    )


def run(config_path: Path) -> DegradedElasticRun | StaggeredCouplingRun:
    """Run the stable benchmark selected by a complete TOML configuration."""

    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    kind = config["benchmark"]["kind"]
    if kind == "degraded_elastic_tension":
        return _run_degraded(config)
    if kind == "staggered_uniform_tension":
        return _run_staggered(config)
    raise ValueError(f"unsupported coupling benchmark kind: {kind!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baselines/staggered_uniform_tension.toml"),
        help="Path to a complete degraded or staggered TOML configuration",
    )
    arguments = parser.parse_args()
    print(json.dumps(asdict(run(arguments.config)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()