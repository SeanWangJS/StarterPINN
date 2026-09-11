"""Total potential energy assembly for the undamaged elastic DEM problem."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor

from griffith2d.discretization import (
    DisplacementMLP,
    QuadratureRule,
    SymmetryTrialSolution,
    displacement_gradient,
    line_quadrature,
    rectangle_quadrature,
)
from griffith2d.diagnostics.elasticity import PlaneStrainTensionBenchmark
from griffith2d.physics import IsotropicMaterial, small_strain, undamaged_elastic_energy


class ElasticEnergyComponents(NamedTuple):
    """Internal energy, traction work, and total potential energy."""

    internal_energy: Tensor
    traction_work: Tensor
    total_potential: Tensor


@dataclass(frozen=True, slots=True)
class ElasticDEMTrainingConfig:
    """Frozen stage-002 optimization and discretization settings."""

    quadrature_order: int = 4
    hidden_width: int = 16
    seed: int = 1729
    adam_steps: int = 1000
    adam_learning_rate: float = 1e-2
    lbfgs_max_iter: int = 200
    lbfgs_tolerance_grad: float = 1e-10
    lbfgs_tolerance_change: float = 1e-12


class ElasticDEMTrainingResult(NamedTuple):
    """Trained trial field and deterministic optimization record."""

    model: SymmetryTrialSolution
    initial_potential: float
    final_potential: float
    adam_potential_history: tuple[float, ...]
    lbfgs_closure_evaluations: int
    stop_reason: str


def elastic_potential_energy(
    displacement_field: Callable[[Tensor], Tensor],
    domain_rule: QuadratureRule,
    traction_rule: QuadratureRule,
    traction: Tensor,
    material: IsotropicMaterial,
) -> ElasticEnergyComponents:
    """Assemble ``Pi = integral_Omega psi_0 - integral_Gamma t.u``.

    Body forces are zero in the stage-002 benchmark. ``traction`` is a constant
    vector with the same dtype and device as the quadrature rules.
    """

    if not isinstance(traction, Tensor) or traction.shape != (2,):
        raise ValueError("traction must be a tensor with shape (2,)")
    if traction.dtype != domain_rule.points.dtype or traction.device != domain_rule.points.device:
        raise ValueError("traction and quadrature rules must have the same dtype and device")
    if (
        traction_rule.points.dtype != domain_rule.points.dtype
        or traction_rule.points.device != domain_rule.points.device
    ):
        raise ValueError("domain and traction rules must have the same dtype and device")

    gradient = displacement_gradient(displacement_field, domain_rule.points)
    strain = small_strain(gradient)
    energy_density = undamaged_elastic_energy(strain, material)
    internal_energy = domain_rule.integrate(energy_density)

    boundary_displacement = displacement_field(traction_rule.points)
    if boundary_displacement.shape != traction_rule.points.shape:
        raise ValueError("displacement field must return shape (number_of_points, 2)")
    traction_density = (boundary_displacement * traction).sum(dim=-1)
    traction_work = traction_rule.integrate(traction_density)
    return ElasticEnergyComponents(
        internal_energy=internal_energy,
        traction_work=traction_work,
        total_potential=internal_energy - traction_work,
    )


def train_elastic_dem(
    benchmark: PlaneStrainTensionBenchmark,
    config: ElasticDEMTrainingConfig = ElasticDEMTrainingConfig(),
    *,
    device: torch.device | str = "cpu",
) -> ElasticDEMTrainingResult:
    """Train the fixed stage-002 DEM baseline with Adam followed by L-BFGS."""

    resolved_device = torch.device(device)
    dtype = torch.float64
    network = DisplacementMLP(
        hidden_width=config.hidden_width,
        seed=config.seed,
        dtype=dtype,
        device=resolved_device,
    )
    model = SymmetryTrialSolution(
        network,
        length=benchmark.length,
        height=benchmark.height,
        displacement_scale=benchmark.displacement_scale,
    )
    domain_rule = rectangle_quadrature(
        (0.0, benchmark.length),
        (0.0, benchmark.height),
        config.quadrature_order,
        dtype=dtype,
        device=resolved_device,
    )
    right_rule = line_quadrature(
        torch.tensor([benchmark.length, 0.0], dtype=dtype, device=resolved_device),
        torch.tensor([benchmark.length, benchmark.height], dtype=dtype, device=resolved_device),
        config.quadrature_order,
    )
    traction = torch.tensor([benchmark.traction, 0.0], dtype=dtype, device=resolved_device)

    def potential() -> Tensor:
        return elastic_potential_energy(
            model, domain_rule, right_rule, traction, benchmark.material
        ).total_potential

    initial_potential = float(potential().detach().cpu())
    optimizer = torch.optim.Adam(model.parameters(), lr=config.adam_learning_rate)
    adam_history: list[float] = []
    for _ in range(config.adam_steps):
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
    stop_reason = (
        "lbfgs_iteration_budget"
        if closure_evaluations >= config.lbfgs_max_iter
        else "lbfgs_converged"
    )
    return ElasticDEMTrainingResult(
        model=model,
        initial_potential=initial_potential,
        final_potential=final_potential,
        adam_potential_history=tuple(adam_history),
        lbfgs_closure_evaluations=closure_evaluations,
        stop_reason=stop_reason,
    )
