"""Energy assembly and deterministic training for independent AT2 problems."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor

from griffith2d.discretization import (
    PhaseFieldMLP,
    PhaseFieldTrialSolution,
    QuadratureRule,
    interval_quadrature,
    phase_field_gradient,
)
from griffith2d.physics import PhaseFieldMaterial, frozen_history_energy_density


class PhaseFieldEnergyComponents(NamedTuple):
    """Integrated history, local-fracture, gradient-fracture, and total energies."""

    history: Tensor
    local_fracture: Tensor
    gradient_fracture: Tensor
    total: Tensor


@dataclass(frozen=True, slots=True)
class PhaseFieldDEMTrainingConfig:
    """Frozen optimization and representation settings for stage 003."""

    quadrature_order: int = 16
    hidden_width: int = 16
    seed: int = 1729
    zero_initialization: bool = False
    adam_steps: int = 1000
    adam_learning_rate: float = 1e-2
    lbfgs_max_iter: int = 200
    lbfgs_tolerance_grad: float = 1e-10
    lbfgs_tolerance_change: float = 1e-12


class PhaseFieldDEMTrainingResult(NamedTuple):
    """Trained scalar trial field and deterministic optimization record."""

    model: PhaseFieldTrialSolution
    initial_energy: float
    final_energy: float
    adam_energy_history: tuple[float, ...]
    lbfgs_closure_evaluations: int
    stop_reason: str


def phase_field_energy(
    phase_field: Callable[[Tensor], Tensor],
    domain_rule: QuadratureRule,
    history_field: Tensor,
    material: PhaseFieldMaterial,
) -> PhaseFieldEnergyComponents:
    """Integrate the frozen-history AT2 energy without residual penalties."""

    phase = phase_field(domain_rule.points)
    if phase.shape != domain_rule.points.shape:
        raise ValueError("phase_field must return shape (number_of_points, 1)")
    if history_field.shape != phase.shape:
        raise ValueError("history_field must have shape (number_of_points, 1)")
    gradient = phase_field_gradient(phase_field, domain_rule.points)
    density = frozen_history_energy_density(phase, gradient, history_field, material)
    history = domain_rule.integrate(density.history[:, 0])
    local_fracture = domain_rule.integrate(density.local_fracture[:, 0])
    gradient_fracture = domain_rule.integrate(density.gradient_fracture[:, 0])
    return PhaseFieldEnergyComponents(
        history=history,
        local_fracture=local_fracture,
        gradient_fracture=gradient_fracture,
        total=history + local_fracture + gradient_fracture,
    )


def train_phase_field_dem(
    half_width: float,
    material: PhaseFieldMaterial,
    history_function: Callable[[Tensor], Tensor],
    *,
    precrack: bool,
    config: PhaseFieldDEMTrainingConfig = PhaseFieldDEMTrainingConfig(),
    device: torch.device | str = "cpu",
) -> PhaseFieldDEMTrainingResult:
    """Train a frozen-history phase field with Adam followed by L-BFGS."""

    resolved_device = torch.device(device)
    dtype = torch.float64
    network = PhaseFieldMLP(
        hidden_width=config.hidden_width,
        seed=config.seed,
        zero_initialization=config.zero_initialization,
        dtype=dtype,
        device=resolved_device,
    )
    model = PhaseFieldTrialSolution(
        network,
        length_scale=material.length_scale,
        precrack=precrack,
    )
    domain_rule = interval_quadrature(
        0.0,
        half_width,
        config.quadrature_order,
        dtype=dtype,
        device=resolved_device,
    )
    history = history_function(domain_rule.points)
    if not isinstance(history, Tensor) or history.shape != domain_rule.points.shape:
        raise ValueError("history_function must return shape (number_of_points, 1)")
    history = history.detach()

    def energy() -> Tensor:
        return phase_field_energy(model, domain_rule, history, material).total

    initial_energy = float(energy().detach().cpu())
    optimizer = torch.optim.Adam(model.parameters(), lr=config.adam_learning_rate)
    adam_history: list[float] = []
    for _ in range(config.adam_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = energy()
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
        loss = energy()
        loss.backward()
        closure_evaluations += 1
        return loss

    optimizer_lbfgs.step(closure)
    final_energy = float(energy().detach().cpu())
    stop_reason = (
        "lbfgs_iteration_budget"
        if closure_evaluations >= config.lbfgs_max_iter
        else "lbfgs_converged"
    )
    return PhaseFieldDEMTrainingResult(
        model=model,
        initial_energy=initial_energy,
        final_energy=final_energy,
        adam_energy_history=tuple(adam_history),
        lbfgs_closure_evaluations=closure_evaluations,
        stop_reason=stop_reason,
    )