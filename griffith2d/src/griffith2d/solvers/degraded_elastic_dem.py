"""Potential-energy assembly for elasticity with a frozen phase field."""

from collections.abc import Callable
from dataclasses import dataclass
import math
from typing import NamedTuple

import torch
from torch import Tensor

from griffith2d.diagnostics.coupling import UniformDegradedTensionBenchmark
from griffith2d.discretization import (
    DisplacementMLP,
    QuadratureRule,
    SymmetryTrialSolution,
    displacement_gradient,
    line_quadrature,
    rectangle_quadrature,
)
from griffith2d.physics import IsotropicMaterial, degraded_elastic_energy, small_strain


class DegradedElasticEnergyComponents(NamedTuple):
    """Degraded internal energy, traction work, and total potential."""

    internal_energy: Tensor
    traction_work: Tensor
    total_potential: Tensor


@dataclass(frozen=True, slots=True)
class DegradedElasticDEMTrainingConfig:
    quadrature_order: int = 4
    hidden_width: int = 16
    seed: int = 1729
    adam_steps: int = 1000
    warm_start_adam_steps: int = 0
    adam_learning_rate: float = 1e-2
    lbfgs_max_iter: int = 200
    lbfgs_tolerance_grad: float = 1e-10
    lbfgs_tolerance_change: float = 1e-12


class DegradedElasticDEMTrainingResult(NamedTuple):
    model: SymmetryTrialSolution
    initial_potential: float
    final_potential: float
    adam_potential_history: tuple[float, ...]
    lbfgs_closure_evaluations: int
    stop_reason: str


def expand_uniform_phase(phase: float, rule: QuadratureRule) -> Tensor:
    """Materialize a finite scalar phase value at every quadrature point."""

    if not math.isfinite(phase):
        raise ValueError("phase must be finite")
    return rule.points.new_full((rule.points.shape[0],), phase)


def degraded_elastic_potential_energy(
    displacement_field: Callable[[Tensor], Tensor],
    phase_field: Tensor,
    domain_rule: QuadratureRule,
    traction_rule: QuadratureRule,
    traction: Tensor,
    material: IsotropicMaterial,
    residual_stiffness: float,
) -> DegradedElasticEnergyComponents:
    """Assemble the frozen-phase potential without residual penalties."""

    if phase_field.shape != (domain_rule.points.shape[0],):
        raise ValueError("phase_field must have shape (number_of_domain_points,)")
    if phase_field.dtype != domain_rule.points.dtype or phase_field.device != domain_rule.points.device:
        raise ValueError("phase_field and domain_rule must share dtype and device")
    if traction.shape != (2,):
        raise ValueError("traction must have shape (2,)")
    if traction.dtype != domain_rule.points.dtype or traction.device != domain_rule.points.device:
        raise ValueError("traction and domain_rule must share dtype and device")
    if traction_rule.points.dtype != domain_rule.points.dtype or traction_rule.points.device != domain_rule.points.device:
        raise ValueError("domain and traction rules must share dtype and device")

    strain = small_strain(displacement_gradient(displacement_field, domain_rule.points))
    density = degraded_elastic_energy(strain, phase_field, material, residual_stiffness)
    internal_energy = domain_rule.integrate(density)
    boundary_displacement = displacement_field(traction_rule.points)
    traction_work = traction_rule.integrate((boundary_displacement * traction).sum(dim=-1))
    return DegradedElasticEnergyComponents(
        internal_energy=internal_energy,
        traction_work=traction_work,
        total_potential=internal_energy - traction_work,
    )


def train_degraded_elastic_dem(
    benchmark: UniformDegradedTensionBenchmark,
    config: DegradedElasticDEMTrainingConfig = DegradedElasticDEMTrainingConfig(),
    *,
    initial_network_state: dict[str, Tensor] | None = None,
    device: torch.device | str = "cpu",
) -> DegradedElasticDEMTrainingResult:
    """Train a frozen uniform-phase displacement field with optional warm start."""

    resolved_device = torch.device(device)
    dtype = torch.float64
    elastic = benchmark.elastic
    network = DisplacementMLP(
        hidden_width=config.hidden_width,
        seed=config.seed,
        dtype=dtype,
        device=resolved_device,
    )
    if initial_network_state is not None:
        network.load_state_dict(initial_network_state, strict=True)
    model = SymmetryTrialSolution(
        network,
        length=elastic.length,
        height=elastic.height,
        displacement_scale=elastic.displacement_scale,
    )
    domain = rectangle_quadrature(
        (0.0, elastic.length),
        (0.0, elastic.height),
        config.quadrature_order,
        dtype=dtype,
        device=resolved_device,
    )
    right = line_quadrature(
        torch.tensor([elastic.length, 0.0], dtype=dtype, device=resolved_device),
        torch.tensor([elastic.length, elastic.height], dtype=dtype, device=resolved_device),
        config.quadrature_order,
    )
    phase = expand_uniform_phase(benchmark.phase, domain)
    traction = torch.tensor([elastic.traction, 0.0], dtype=dtype, device=resolved_device)

    def potential() -> Tensor:
        return degraded_elastic_potential_energy(
            model,
            phase,
            domain,
            right,
            traction,
            elastic.material,
            benchmark.residual_stiffness,
        ).total_potential

    initial_potential = float(potential().detach().cpu())
    optimizer = torch.optim.Adam(model.parameters(), lr=config.adam_learning_rate)
    adam_history: list[float] = []
    adam_steps = config.adam_steps if initial_network_state is None else config.warm_start_adam_steps
    for _ in range(adam_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = potential()
        loss.backward()
        optimizer.step()
        adam_history.append(float(loss.detach().cpu()))

    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        max_iter=config.lbfgs_max_iter,
        tolerance_grad=config.lbfgs_tolerance_grad,
        tolerance_change=config.lbfgs_tolerance_change,
        line_search_fn="strong_wolfe",
    )
    closure_evaluations = 0

    def closure() -> Tensor:
        nonlocal closure_evaluations
        optimizer_lbfgs.zero_grad(set_to_none=True)
        loss = potential()
        loss.backward()
        closure_evaluations += 1
        return loss

    optimizer_lbfgs.step(closure)
    final_potential = float(potential().detach().cpu())
    return DegradedElasticDEMTrainingResult(
        model=model,
        initial_potential=initial_potential,
        final_potential=final_potential,
        adam_potential_history=tuple(adam_history),
        lbfgs_closure_evaluations=closure_evaluations,
        stop_reason=(
            "lbfgs_iteration_budget"
            if closure_evaluations >= config.lbfgs_max_iter
            else "lbfgs_converged"
        ),
    )