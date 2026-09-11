"""Uniform staggered history-field coupling and transactional load steps."""

from dataclasses import dataclass, field
import math

import torch
from torch import Tensor

from griffith2d.diagnostics import (
    PlaneStrainTensionBenchmark,
    UniformDegradedTensionBenchmark,
    degraded_boundary_reaction,
)
from griffith2d.discretization import (
    QuadratureRule,
    displacement_gradient,
    line_quadrature,
    rectangle_quadrature,
)
from griffith2d.physics import (
    IsotropicMaterial,
    PhaseFieldMaterial,
    amor_energy_split,
    small_strain,
    update_history,
)

from .degraded_elastic_dem import (
    DegradedElasticDEMTrainingConfig,
    expand_uniform_phase,
    train_degraded_elastic_dem,
)


def active_energy_at_points(
    displacement_field,
    points: Tensor,
    material: IsotropicMaterial,
) -> Tensor:
    """Return Amor active energy at explicit displacement quadrature points."""

    strain = small_strain(displacement_gradient(displacement_field, points))
    return amor_energy_split(strain, material).active


class UniformPhaseSolveResult(tuple):
    """Exact minimizer and diagnostics of the uniform scalar phase subproblem."""

    __slots__ = ()

    def __new__(cls, phase: float, energy: float, stationarity: float):
        return super().__new__(cls, (phase, energy, stationarity))

    phase = property(lambda self: self[0])
    energy = property(lambda self: self[1])
    stationarity = property(lambda self: self[2])


def solve_uniform_phase(
    history: Tensor,
    rule: QuadratureRule,
    material: PhaseFieldMaterial,
) -> UniformPhaseSolveResult:
    """Minimize the frozen-history AT2 energy over one uniform scalar degree of freedom."""

    if history.shape != (rule.points.shape[0],):
        raise ValueError("history must have shape (number_of_points,)")
    if history.dtype != rule.points.dtype or history.device != rule.points.device:
        raise ValueError("history and rule must share dtype and device")
    area = rule.weights.sum()
    history_integral = rule.integrate(history)
    coefficient = material.fracture_toughness * area / material.length_scale
    phase = 2.0 * history_integral / (coefficient + 2.0 * history_integral)
    energy = (1.0 - phase).square() * history_integral + 0.5 * coefficient * phase.square()
    stationarity = -2.0 * (1.0 - phase) * history_integral + coefficient * phase
    return UniformPhaseSolveResult(
        float(phase.detach().cpu()),
        float(energy.detach().cpu()),
        float(stationarity.detach().cpu()),
    )


@dataclass(frozen=True, slots=True)
class StaggeredCouplingConfig:
    quadrature_order: int = 4
    max_iterations: int = 20
    displacement_relative_tolerance: float = 2e-4
    phase_absolute_tolerance: float = 1e-7
    history_relative_tolerance: float = 1e-7
    energy_relative_tolerance: float = 1e-7
    displacement: DegradedElasticDEMTrainingConfig = field(
        default_factory=DegradedElasticDEMTrainingConfig
    )


@dataclass(frozen=True, slots=True)
class StaggeredIterationRecord:
    iteration: int
    phase_before: float
    phase_after: float
    phase_change: float
    displacement_relative_change: float
    history_relative_change: float
    energy_relative_change: float
    history_max: float
    displacement_potential: float
    phase_energy: float
    phase_stationarity: float
    displacement_stop_reason: str


@dataclass(frozen=True, slots=True)
class StaggeredState:
    load_index: int
    traction: float
    phase: float
    history: Tensor
    network_state: dict[str, Tensor] | None
    displacement_at_points: Tensor
    reaction_x: float
    displacement_potential: float
    phase_energy: float
    total_energy: float


@dataclass(frozen=True, slots=True)
class LoadStepResult:
    committed: bool
    state: StaggeredState
    iterations: tuple[StaggeredIterationRecord, ...]
    stop_reason: str


@dataclass(frozen=True, slots=True)
class LoadPathResult:
    initial_state: StaggeredState
    steps: tuple[LoadStepResult, ...]
    final_state: StaggeredState
    stop_reason: str


def _domain_rule(
    length: float,
    height: float,
    config: StaggeredCouplingConfig,
    device: torch.device | str,
) -> QuadratureRule:
    return rectangle_quadrature(
        (0.0, length),
        (0.0, height),
        config.quadrature_order,
        dtype=torch.float64,
        device=device,
    )


def initial_staggered_state(
    length: float,
    height: float,
    config: StaggeredCouplingConfig = StaggeredCouplingConfig(),
    *,
    device: torch.device | str = "cpu",
) -> StaggeredState:
    """Return the exact zero-load state on the configured coupling rule."""

    rule = _domain_rule(length, height, config, device)
    return StaggeredState(
        load_index=-1,
        traction=0.0,
        phase=0.0,
        history=rule.points.new_zeros(rule.points.shape[0]),
        network_state=None,
        displacement_at_points=rule.points.new_zeros(rule.points.shape),
        reaction_x=0.0,
        displacement_potential=0.0,
        phase_energy=0.0,
        total_energy=0.0,
    )


def _relative_change(current: Tensor, previous: Tensor, scale: float) -> float:
    numerator = torch.linalg.vector_norm(current - previous)
    denominator = torch.maximum(
        torch.linalg.vector_norm(current), current.new_tensor(scale)
    )
    return float((numerator / denominator).detach().cpu())


def _network_snapshot(model) -> dict[str, Tensor]:
    return {
        name: value.detach().clone()
        for name, value in model.network.state_dict().items()
    }


def solve_staggered_load_step(
    previous: StaggeredState,
    traction: float,
    length: float,
    height: float,
    elastic_material: IsotropicMaterial,
    phase_material: PhaseFieldMaterial,
    residual_stiffness: float,
    config: StaggeredCouplingConfig = StaggeredCouplingConfig(),
    *,
    device: torch.device | str = "cpu",
) -> LoadStepResult:
    """Solve one load step and commit only after all fixed-point criteria pass."""

    if not math.isfinite(traction) or traction < 0.0:
        raise ValueError("traction must be finite and nonnegative")
    rule = _domain_rule(length, height, config, device)
    if previous.history.shape != (rule.points.shape[0],):
        raise ValueError("previous history does not match the coupling quadrature")
    if config.max_iterations < 1:
        return LoadStepResult(False, previous, (), "iteration_budget")

    if traction == 0.0:
        phase_result = solve_uniform_phase(previous.history, rule, phase_material)
        state = StaggeredState(
            load_index=previous.load_index + 1,
            traction=0.0,
            phase=phase_result.phase,
            history=previous.history.detach().clone(),
            network_state=(
                None
                if previous.network_state is None
                else {name: value.detach().clone() for name, value in previous.network_state.items()}
            ),
            displacement_at_points=torch.zeros_like(previous.displacement_at_points),
            reaction_x=0.0,
            displacement_potential=0.0,
            phase_energy=phase_result.energy,
            total_energy=phase_result.energy,
        )
        return LoadStepResult(True, state, (), "zero_load_exact")

    elastic = PlaneStrainTensionBenchmark(length, height, traction, elastic_material)
    phase = previous.phase
    candidate_history = previous.history.detach().clone()
    network_state = previous.network_state
    previous_displacement = previous.displacement_at_points
    previous_energy = previous.total_energy
    records: list[StaggeredIterationRecord] = []

    for iteration in range(1, config.max_iterations + 1):
        benchmark = UniformDegradedTensionBenchmark(elastic, phase, residual_stiffness)
        displacement_result = train_degraded_elastic_dem(
            benchmark,
            config.displacement,
            initial_network_state=network_state,
            device=device,
        )
        displacement = displacement_result.model(rule.points)
        active_energy = active_energy_at_points(
            displacement_result.model, rule.points, elastic_material
        ).detach()
        updated_history = update_history(candidate_history, active_energy)
        updated_history = update_history(previous.history, updated_history)
        phase_result = solve_uniform_phase(updated_history, rule, phase_material)
        aggregate_energy = displacement_result.final_potential + phase_result.energy

        displacement_change = _relative_change(
            displacement,
            previous_displacement,
            traction * length / elastic_material.young_modulus,
        )
        history_change = _relative_change(
            updated_history,
            candidate_history,
            phase_material.fracture_toughness / phase_material.length_scale,
        )
        phase_change = abs(phase_result.phase - phase)
        energy_change = abs(aggregate_energy - previous_energy) / max(
            abs(aggregate_energy), phase_material.fracture_toughness * length, 1e-15
        )
        records.append(
            StaggeredIterationRecord(
                iteration=iteration,
                phase_before=phase,
                phase_after=phase_result.phase,
                phase_change=phase_change,
                displacement_relative_change=displacement_change,
                history_relative_change=history_change,
                energy_relative_change=energy_change,
                history_max=float(updated_history.max().detach().cpu()),
                displacement_potential=displacement_result.final_potential,
                phase_energy=phase_result.energy,
                phase_stationarity=abs(phase_result.stationarity),
                displacement_stop_reason=displacement_result.stop_reason,
            )
        )
        network_state = _network_snapshot(displacement_result.model)

        converged = (
            iteration >= 2
            and displacement_result.stop_reason == "lbfgs_converged"
            and displacement_change <= config.displacement_relative_tolerance
            and phase_change <= config.phase_absolute_tolerance
            and history_change <= config.history_relative_tolerance
            and energy_change <= config.energy_relative_tolerance
            and abs(phase_result.stationarity) <= 1e-12
        )
        if converged:
            left = line_quadrature(
                rule.points.new_tensor([0.0, 0.0]),
                rule.points.new_tensor([0.0, height]),
                config.quadrature_order,
            )
            reaction = degraded_boundary_reaction(
                displacement_result.model,
                phase_result.phase,
                UniformDegradedTensionBenchmark(
                    elastic, phase_result.phase, residual_stiffness
                ),
                left,
                rule.points.new_tensor([-1.0, 0.0]),
            )
            state = StaggeredState(
                load_index=previous.load_index + 1,
                traction=traction,
                phase=phase_result.phase,
                history=updated_history.detach().clone(),
                network_state=network_state,
                displacement_at_points=displacement.detach().clone(),
                reaction_x=float(reaction[0].detach().cpu()),
                displacement_potential=displacement_result.final_potential,
                phase_energy=phase_result.energy,
                total_energy=aggregate_energy,
            )
            return LoadStepResult(True, state, tuple(records), "converged")

        phase = phase_result.phase
        candidate_history = updated_history
        previous_displacement = displacement.detach()
        previous_energy = aggregate_energy

    return LoadStepResult(False, previous, tuple(records), "iteration_budget")


def run_staggered_load_path(
    tractions: tuple[float, ...],
    length: float,
    height: float,
    elastic_material: IsotropicMaterial,
    phase_material: PhaseFieldMaterial,
    residual_stiffness: float,
    config: StaggeredCouplingConfig = StaggeredCouplingConfig(),
    *,
    device: torch.device | str = "cpu",
) -> LoadPathResult:
    """Run a fixed load schedule, stopping transactionally at the first failed step."""

    initial = initial_staggered_state(length, height, config, device=device)
    state = initial
    steps: list[LoadStepResult] = []
    for traction in tractions:
        step = solve_staggered_load_step(
            state,
            traction,
            length,
            height,
            elastic_material,
            phase_material,
            residual_stiffness,
            config,
            device=device,
        )
        steps.append(step)
        if not step.committed:
            return LoadPathResult(initial, tuple(steps), state, "load_step_failed")
        state = step.state
    return LoadPathResult(initial, tuple(steps), state, "completed")