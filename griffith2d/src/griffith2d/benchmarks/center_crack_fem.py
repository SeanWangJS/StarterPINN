"""Independent finite-domain plane-strain FEM reference for a displacement-loaded plate."""

import argparse
import base64
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import platform
import subprocess
import tomllib
import sys
import traceback
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, minimize
from skfem import BilinearForm, Basis, ElementTriP1, ElementVector, LinearForm, MeshTri, asm, condense, solve
from skfem.helpers import ddot, div, dot, grad, sym_grad
from skfem.models.elasticity import linear_elasticity

from griffith2d.discretization import CenterCrackPlate
from griffith2d.physics import IsotropicMaterial


@dataclass(frozen=True, slots=True)
class PlateElasticFEMResult:
    """P1 node positions and displacements, positive upper reaction and stored energy."""

    nodes: NDArray[np.float64]
    displacement: NDArray[np.float64]
    top_reaction: float
    strain_energy: float


@dataclass(frozen=True, slots=True)
class FrozenPrecrackFEMResult:
    nodes: NDArray[np.float64]
    displacement: NDArray[np.float64]
    top_reaction: float
    strain_energy: float
    iterations: int
    free_residual_relative: float


@dataclass(frozen=True, slots=True)
class FrozenPhaseFEMResult:
    nodes: NDArray[np.float64]
    phase: NDArray[np.float64]
    projected_gradient: float


@dataclass(frozen=True, slots=True)
class PlateStaggeredFEMRecord:
    iteration: int
    phase_change: float
    displacement_relative_change: float
    history_relative_change: float
    energy_relative_change: float
    phase_projected_gradient: float


@dataclass(frozen=True, slots=True)
class PlateStaggeredFEMResult:
    displacement: FrozenPrecrackFEMResult
    phase: FrozenPhaseFEMResult
    history: NDArray[np.float64]
    records: tuple[PlateStaggeredFEMRecord, ...]


@dataclass(frozen=True, slots=True)
class PlateFEMPathStep:
    load_index: int
    imposed_displacement: float
    state: PlateStaggeredFEMResult
    fracture_energy: float
    cumulative_work: float
    phase_violation: float
    history_violation: float


class PlateFEMPathError(RuntimeError):
    """A failed 1-based load step with only the successfully committed prefix."""

    def __init__(
        self, load_index: int, imposed_displacement: float,
        committed_steps: tuple[PlateFEMPathStep, ...],
    ) -> None:
        self.load_index = load_index
        self.imposed_displacement = imposed_displacement
        self.committed_steps = committed_steps
        super().__init__(f"FEM path failed at load step {load_index}, displacement={imposed_displacement:g}")


def _plate_constraints(
    basis: Basis,
    plate: CenterCrackPlate,
    imposed_displacement: float,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    top = basis.get_dofs(nodes=lambda coordinates: np.isclose(coordinates[1], plate.half_height))
    bottom = basis.get_dofs(nodes=lambda coordinates: np.isclose(coordinates[1], -plate.half_height))
    center = basis.get_dofs(
        nodes=lambda coordinates: np.isclose(coordinates[0], 0.0)
        & np.isclose(coordinates[1], 0.0)
    )
    top_vertical = top.nodal["u^2"]
    bottom_vertical = bottom.nodal["u^2"]
    center_horizontal = center.nodal["u^1"]
    constraints = np.concatenate((top_vertical, bottom_vertical, center_horizontal))
    prescribed = np.zeros(basis.N)
    prescribed[top_vertical] = imposed_displacement
    prescribed[bottom_vertical] = -imposed_displacement
    return prescribed, constraints, top_vertical


def _axis_with_landmarks(
    lower: float, upper: float, cells: int, landmarks: tuple[float, ...]
) -> NDArray[np.float64]:
    edges = np.linspace(lower, upper, cells + 1)
    tolerance = 16.0 * np.finfo(float).eps * max(abs(lower), abs(upper))
    for landmark in landmarks:
        closest = int(np.argmin(np.abs(edges - landmark)))
        if abs(edges[closest] - landmark) <= tolerance:
            edges[closest] = landmark
        else:
            edges = np.append(edges, landmark)
    return np.sort(edges)


def solve_uncracked_plate_fem(
    plate: CenterCrackPlate,
    material: IsotropicMaterial,
    imposed_displacement: float,
    cells_x: int,
    cells_y: int,
) -> PlateElasticFEMResult:
    """Solve an uncracked plate, pinning only horizontal translation at its center."""

    for name, count in (("cells_x", cells_x), ("cells_y", cells_y)):
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 2 or count % 2:
            raise ValueError(f"{name} must be an even integer >= 2")
    if isinstance(imposed_displacement, bool) or not isinstance(imposed_displacement, Real):
        raise TypeError("imposed_displacement must be a real scalar")
    imposed_displacement = float(imposed_displacement)
    if not math.isfinite(imposed_displacement) or imposed_displacement < 0.0:
        raise ValueError("imposed_displacement must be finite and nonnegative")

    mesh = MeshTri.init_tensor(
        np.linspace(-plate.half_width, plate.half_width, cells_x + 1),
        np.linspace(-plate.half_height, plate.half_height, cells_y + 1),
    )
    basis = Basis(mesh, ElementVector(ElementTriP1()))
    shear = material.shear_modulus
    lame = material.young_modulus * material.poisson_ratio / (
        (1.0 + material.poisson_ratio) * (1.0 - 2.0 * material.poisson_ratio)
    )
    stiffness = asm(linear_elasticity(Lambda=lame, Mu=shear), basis)

    prescribed, constraints, top_vertical = _plate_constraints(
        basis, plate, imposed_displacement
    )
    displacement_dofs = solve(
        *condense(stiffness, np.zeros(basis.N), x=prescribed, D=constraints)
    )
    boundary_forces = stiffness @ displacement_dofs
    return PlateElasticFEMResult(
        nodes=mesh.p.T.copy(),
        displacement=displacement_dofs[basis.nodal_dofs].T.copy(),
        top_reaction=float(boundary_forces[top_vertical].sum()),
        strain_energy=float(0.5 * displacement_dofs @ boundary_forces),
    )


@BilinearForm
def _amor_tangent(trial, test, weights):
    trial_volume = div(trial)
    test_volume = div(test)
    deviatoric = ddot(sym_grad(trial), sym_grad(test)) - trial_volume * test_volume / 3.0
    return (
        weights.bulk_modulus * weights.volume_degradation * trial_volume * test_volume
        + 2.0 * weights.shear_modulus * weights.degradation * deviatoric
    )


@BilinearForm
def _at2_tangent(trial, test, weights):
    return (
        (weights.fracture_toughness / weights.length_scale + 2.0 * weights.history)
        * trial * test
        + weights.fracture_toughness * weights.length_scale * dot(grad(trial), grad(test))
    )


@LinearForm
def _at2_load(test, weights):
    return 2.0 * weights.history * test


def _phase_residual_and_fracture_energy(
    phase: FrozenPhaseFEMResult,
    history: NDArray[np.float64],
    lower: NDArray[np.float64],
    fracture_toughness: float,
    length_scale: float,
) -> tuple[float, float]:
    mesh = MeshTri.init_tensor(np.unique(phase.nodes[:, 0]), np.unique(phase.nodes[:, 1]))
    basis = Basis(mesh, ElementTriP1(), intorder=4)
    tangent = asm(
        _at2_tangent, basis, fracture_toughness=fracture_toughness,
        length_scale=length_scale, history=basis.interpolate(history),
    )
    load = asm(_at2_load, basis, history=basis.interpolate(history))
    gradient = tangent @ phase.phase - load
    residual = np.max(np.abs(phase.phase - np.clip(phase.phase - gradient, lower, 1.0)))
    field = basis.interpolate(phase.phase)
    density = fracture_toughness * (
        field**2 / (2.0 * length_scale)
        + 0.5 * length_scale * np.sum(field.grad**2, axis=0)
    )
    return float(residual), float(np.sum(density * basis.dx))


@LinearForm
def _active_energy_load(test, weights):
    return weights.active_energy * test


def transfer_active_energy_to_history_fem(
    nodes: NDArray[np.float64],
    displacement: NDArray[np.float64],
    material: IsotropicMaterial,
    *,
    previous_history: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Mass-lump P1 element Amor active energy, then update mesh-ordered history.

    Nodes must follow MeshTri.init_tensor order and connectivity. Recovery is
    nonnegative and integral-preserving before the pointwise history maximum;
    it is not a pointwise upper bound on the discontinuous element energy.
    """

    nodes = np.asarray(nodes)
    displacement = np.asarray(displacement)
    for name, values in (("nodes", nodes), ("displacement", displacement)):
        if values.ndim != 2 or values.shape[1] != 2 or not np.issubdtype(values.dtype, np.floating):
            raise ValueError(f"{name} must be a floating-point array with shape (N, 2)")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be finite")
    if displacement.shape != nodes.shape:
        raise ValueError("displacement must match nodes without broadcasting")
    if len(np.unique(nodes[:, 0])) < 2 or len(np.unique(nodes[:, 1])) < 2:
        raise ValueError("nodes must span a two-dimensional tensor mesh")
    mesh = MeshTri.init_tensor(np.unique(nodes[:, 0]), np.unique(nodes[:, 1]))
    if mesh.p.T.shape != nodes.shape or not np.array_equal(mesh.p.T, nodes):
        raise ValueError("nodes must follow MeshTri.init_tensor node order")
    vector_basis = Basis(mesh, ElementVector(ElementTriP1()), intorder=2)
    scalar_basis = Basis(mesh, ElementTriP1(), intorder=2)
    dofs = np.zeros(vector_basis.N)
    dofs[vector_basis.nodal_dofs] = displacement.T
    gradient = vector_basis.interpolate(dofs).grad
    strain = 0.5 * (gradient + np.swapaxes(gradient, 0, 1))
    trace = np.trace(strain, axis1=0, axis2=1)
    active = (
        0.5 * material.bulk_modulus * np.maximum(trace, 0.0) ** 2
        + material.shear_modulus * (np.sum(strain**2, axis=(0, 1)) - trace**2 / 3.0)
    )
    load = asm(_active_energy_load, scalar_basis, active_energy=active)
    lumped_mass = asm(_active_energy_load, scalar_basis, active_energy=1.0)
    recovered = load / lumped_mass
    if previous_history is not None:
        previous = np.asarray(previous_history)
        if previous.shape != recovered.shape or not np.issubdtype(previous.dtype, np.floating):
            raise ValueError("previous_history must be a floating-point array in mesh node order")
        if not np.all(np.isfinite(previous)) or np.any(previous < 0.0):
            raise ValueError("previous_history must be finite and nonnegative")
        recovered = np.maximum(previous, recovered)
    return recovered


def solve_frozen_phase_fem(
    plate: CenterCrackPlate,
    fracture_toughness: float,
    length_scale: float,
    history: float | NDArray[np.float64],
    cells_x: int,
    cells_y: int,
    *,
    initial_precrack: bool = True,
    previous_phase: NDArray[np.float64] | None = None,
) -> FrozenPhaseFEMResult:
    """Minimize AT2 energy with scalar or mesh-ordered nodal history and bounds."""

    for name, count in (("cells_x", cells_x), ("cells_y", cells_y)):
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 2 or count % 2:
            raise ValueError(f"{name} must be an even integer >= 2")
    for name, value in (("fracture_toughness", fracture_toughness),
                        ("length_scale", length_scale)):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not 0.0 < length_scale < plate.half_height:
        raise ValueError("length_scale must be less than half_height")

    mesh = MeshTri.init_tensor(
        _axis_with_landmarks(-plate.half_width, plate.half_width, cells_x,
                             (-plate.crack_half_length, 0.0, plate.crack_half_length)),
        _axis_with_landmarks(-plate.half_height, plate.half_height, cells_y,
                             (-length_scale, 0.0, length_scale)),
    )
    basis = Basis(mesh, ElementTriP1(), intorder=4)
    if isinstance(history, Real) and not isinstance(history, bool):
        history_nodes = np.full(basis.N, history, dtype=float)
    else:
        history_nodes = np.asarray(history)
        if history_nodes.shape != (basis.N,) or not np.issubdtype(history_nodes.dtype, np.floating):
            raise ValueError("history must be a real scalar or floating-point array in mesh node order")
    if not np.all(np.isfinite(history_nodes)) or np.any(history_nodes < 0.0):
        raise ValueError("history must be finite and nonnegative")
    history_quadrature = basis.interpolate(history_nodes)
    longitudinal = np.maximum(np.abs(mesh.p[0]) - plate.crack_half_length, 0.0)
    distance = np.hypot(longitudinal, mesh.p[1])
    lower = np.exp(-distance / length_scale) if initial_precrack else np.zeros(basis.N)
    if previous_phase is not None:
        previous = np.asarray(previous_phase)
        if previous.shape != (basis.N,) or not np.issubdtype(previous.dtype, np.floating):
            raise ValueError("previous_phase must be a floating-point array in mesh node order")
        if not np.all(np.isfinite(previous)) or np.any((previous < 0.0) | (previous > 1.0)):
            raise ValueError("previous_phase must be finite and within [0, 1]")
        lower = np.maximum(lower, previous)
    tangent = asm(
        _at2_tangent, basis, fracture_toughness=fracture_toughness,
        length_scale=length_scale, history=history_quadrature,
    )
    load = asm(_at2_load, basis, history=history_quadrature)

    def objective(phase):
        gradient = tangent @ phase - load
        return 0.5 * phase @ tangent @ phase - load @ phase, gradient

    result = minimize(
        objective, lower, jac=True, method="L-BFGS-B",
        bounds=Bounds(lower, np.ones(basis.N)),
        options={"gtol": 1e-11, "ftol": 1e-15, "maxiter": 2000},
    )
    gradient = tangent @ result.x - load
    projected = np.where(
        result.x <= lower + 1e-10, np.minimum(gradient, 0.0),
        np.where(result.x >= 1.0 - 1e-10, np.maximum(gradient, 0.0), gradient),
    )
    violation = float(np.max(np.abs(projected)))
    if not result.success or violation > 1e-7:
        raise RuntimeError(f"frozen phase FEM did not satisfy box KKT conditions: {result.message}, residual={violation:.3e}")
    return FrozenPhaseFEMResult(mesh.p.T.copy(), result.x.copy(), violation)


def solve_frozen_precrack_fem(
    plate: CenterCrackPlate,
    material: IsotropicMaterial,
    imposed_displacement: float,
    length_scale: float,
    residual_stiffness: float,
    cells_x: int,
    cells_y: int,
    *,
    max_iterations: int = 30,
    residual_tolerance: float = 1e-9,
    quadrature_order: int = 2,
    frozen_phase: FrozenPhaseFEMResult | None = None,
) -> FrozenPrecrackFEMResult:
    """Solve Amor equilibrium with an exponential precrack or same-mesh P1 phase."""

    for name, count in (("cells_x", cells_x), ("cells_y", cells_y)):
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 2 or count % 2:
            raise ValueError(f"{name} must be an even integer >= 2")
    if not math.isfinite(imposed_displacement) or imposed_displacement < 0.0:
        raise ValueError("imposed_displacement must be finite and nonnegative")
    if not math.isfinite(length_scale) or not 0.0 < length_scale < plate.half_height:
        raise ValueError("length_scale must be strictly between 0 and half_height")
    if not math.isfinite(residual_stiffness) or not 0.0 < residual_stiffness < 1.0:
        raise ValueError("residual_stiffness must satisfy 0 < k < 1")
    if max_iterations < 1 or residual_tolerance <= 0.0:
        raise ValueError("iteration budget and residual tolerance must be positive")
    if isinstance(quadrature_order, bool) or not isinstance(quadrature_order, Integral) or quadrature_order < 1:
        raise ValueError("quadrature_order must be a positive integer")

    mesh = MeshTri.init_tensor(
        _axis_with_landmarks(
            -plate.half_width, plate.half_width, cells_x,
            (-plate.crack_half_length, 0.0, plate.crack_half_length),
        ),
        _axis_with_landmarks(
            -plate.half_height, plate.half_height, cells_y,
            (-length_scale, 0.0, length_scale),
        ),
    )
    basis = Basis(mesh, ElementVector(ElementTriP1()), intorder=quadrature_order)
    prescribed, constraints, top_vertical = _plate_constraints(
        basis, plate, imposed_displacement
    )
    free_dofs = np.setdiff1d(np.arange(basis.N), constraints)
    if frozen_phase is None:
        quadrature_points = basis.global_coordinates()
        distance = np.hypot(
            np.maximum(np.abs(quadrature_points[0]) - plate.crack_half_length, 0.0),
            quadrature_points[1],
        )
        phase_quadrature = np.exp(-distance / length_scale)
    else:
        if not np.array_equal(frozen_phase.nodes, mesh.p.T):
            raise ValueError("frozen_phase nodes must match the displacement mesh node order")
        phase = np.asarray(frozen_phase.phase)
        if phase.shape != (mesh.nvertices,) or not np.issubdtype(phase.dtype, np.floating):
            raise ValueError("frozen_phase must contain a floating-point phase array in mesh node order")
        if not np.all(np.isfinite(phase)) or np.any((phase < 0.0) | (phase > 1.0)):
            raise ValueError("frozen_phase must be finite and within [0, 1]")
        scalar_basis = Basis(mesh, ElementTriP1(), quadrature=basis.quadrature)
        phase_quadrature = scalar_basis.interpolate(phase)
    degradation = (1.0 - phase_quadrature) ** 2 + residual_stiffness
    shear = material.shear_modulus
    bulk = material.bulk_modulus

    lame = material.young_modulus * material.poisson_ratio / (
        (1.0 + material.poisson_ratio) * (1.0 - 2.0 * material.poisson_ratio)
    )
    vertical_strain = imposed_displacement / plate.half_height
    horizontal_strain = -lame * vertical_strain / (lame + 2.0 * shear)
    current = np.zeros(basis.N)
    current[basis.nodal_dofs] = np.stack(
        (horizontal_strain * mesh.p[0], vertical_strain * mesh.p[1])
    )

    def assemble_tangent(displacement_dofs):
        trace = np.trace(basis.interpolate(displacement_dofs).grad, axis1=0, axis2=1)
        return asm(
            _amor_tangent, basis, bulk_modulus=bulk, shear_modulus=shear,
            degradation=degradation,
            volume_degradation=np.where(trace >= 0.0, degradation, 1.0),
        )

    for iteration in range(1, max_iterations + 1):
        tangent = assemble_tangent(current)
        updated = solve(
            *condense(tangent, np.zeros(basis.N), x=prescribed, D=constraints)
        )
        direction = updated - current
        current_forces = tangent @ current
        current_energy = 0.5 * current @ current_forces
        directional_derivative = current_forces[free_dofs] @ direction[free_dofs]
        step = 1.0
        for _ in range(24):
            candidate = current + step * direction
            candidate_tangent = assemble_tangent(candidate)
            forces = candidate_tangent @ candidate
            candidate_energy = 0.5 * candidate @ forces
            if candidate_energy <= current_energy + 1e-4 * step * directional_derivative:
                break
            step *= 0.5
        else:
            raise RuntimeError(
                f"frozen precrack FEM line search failed at iteration {iteration}: "
                f"directional_derivative={directional_derivative:.3e}, "
                f"free_residual={np.linalg.norm(current_forces[free_dofs]):.3e}"
            )
        relative_residual = np.linalg.norm(forces[free_dofs]) / max(
            np.linalg.norm(forces[top_vertical]), 1e-15
        )
        if relative_residual <= residual_tolerance:
            return FrozenPrecrackFEMResult(
                nodes=mesh.p.T.copy(),
                displacement=candidate[basis.nodal_dofs].T.copy(),
                top_reaction=float(forces[top_vertical].sum()),
                strain_energy=float(candidate_energy),
                iterations=iteration,
                free_residual_relative=float(relative_residual),
            )
        current = candidate
    raise RuntimeError(
        f"frozen precrack FEM did not satisfy free-DOF equilibrium: "
        f"residual={relative_residual:.3e} after {max_iterations} iterations"
    )


def solve_staggered_plate_fem(
    plate: CenterCrackPlate,
    material: IsotropicMaterial,
    imposed_displacement: float,
    fracture_toughness: float,
    length_scale: float,
    residual_stiffness: float,
    cells_x: int,
    cells_y: int,
    *,
    previous_phase: FrozenPhaseFEMResult | None = None,
    previous_history: NDArray[np.float64] | None = None,
    max_iterations: int = 30,
    phase_tolerance: float = 1e-7,
    relative_tolerance: float = 1e-7,
) -> PlateStaggeredFEMResult:
    """Solve one fixed-load nodal-history proxy step without mutating prior state.

    The phase lower bound and committed history stay fixed throughout iteration.
    Unlike the uniform 004 solver, intermediate history maxima are not retained.
    Failure raises instead of returning a partially converged load step.
    """

    if isinstance(max_iterations, bool) or not isinstance(max_iterations, Integral) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    for name, value in (("phase_tolerance", phase_tolerance),
                        ("relative_tolerance", relative_tolerance)):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    seed = solve_frozen_phase_fem(plate, fracture_toughness, length_scale, 0.0, cells_x, cells_y)
    distance = np.hypot(
        np.maximum(np.abs(seed.nodes[:, 0]) - plate.crack_half_length, 0.0), seed.nodes[:, 1]
    )
    lower = np.exp(-distance / length_scale)
    if previous_phase is not None:
        if not np.array_equal(previous_phase.nodes, seed.nodes):
            raise ValueError("previous_phase nodes must match the mesh node order")
        previous = np.asarray(previous_phase.phase)
        if previous.shape != lower.shape or not np.issubdtype(previous.dtype, np.floating):
            raise ValueError("previous_phase must be a floating-point array in mesh node order")
        if not np.all(np.isfinite(previous)) or np.any((previous < 0.0) | (previous > 1.0)):
            raise ValueError("previous_phase must be finite and within [0, 1]")
        lower = np.maximum(lower, previous)
    committed_history = np.zeros(len(seed.nodes))
    if previous_history is not None:
        previous = np.asarray(previous_history)
        if previous.shape != committed_history.shape or not np.issubdtype(previous.dtype, np.floating):
            raise ValueError("previous_history must be a floating-point array in mesh node order")
        if not np.all(np.isfinite(previous)) or np.any(previous < 0.0):
            raise ValueError("previous_history must be finite and nonnegative")
        committed_history = previous.copy()
    phase = solve_frozen_phase_fem(
        plate, fracture_toughness, length_scale, committed_history, cells_x, cells_y,
        previous_phase=lower,
    )
    displacement = solve_frozen_precrack_fem(
        plate, material, imposed_displacement, length_scale, residual_stiffness,
        cells_x, cells_y, frozen_phase=phase, quadrature_order=4,
    )
    history = transfer_active_energy_to_history_fem(
        displacement.nodes, displacement.displacement, material, previous_history=committed_history
    )
    _, fracture_energy = _phase_residual_and_fracture_energy(
        phase, history, lower, fracture_toughness, length_scale
    )
    energy = displacement.strain_energy + fracture_energy
    records = []
    for iteration in range(1, max_iterations + 1):
        candidate_phase = solve_frozen_phase_fem(
            plate, fracture_toughness, length_scale, history, cells_x, cells_y,
            previous_phase=lower,
        )
        candidate_displacement = solve_frozen_precrack_fem(
            plate, material, imposed_displacement, length_scale, residual_stiffness,
            cells_x, cells_y, frozen_phase=candidate_phase, quadrature_order=4,
        )
        candidate_history = transfer_active_energy_to_history_fem(
            candidate_displacement.nodes, candidate_displacement.displacement, material,
            previous_history=committed_history,
        )
        residual, fracture_energy = _phase_residual_and_fracture_energy(
            candidate_phase, candidate_history, lower, fracture_toughness, length_scale
        )
        candidate_energy = candidate_displacement.strain_energy + fracture_energy
        record = PlateStaggeredFEMRecord(
            iteration=iteration,
            phase_change=float(np.max(np.abs(candidate_phase.phase - phase.phase))),
            displacement_relative_change=float(np.linalg.norm(
                candidate_displacement.displacement - displacement.displacement
            ) / max(np.linalg.norm(candidate_displacement.displacement), imposed_displacement, 1e-15)),
            history_relative_change=float(np.max(np.abs(candidate_history - history)) / max(
                np.max(candidate_history), fracture_toughness / length_scale, 1e-15
            )),
            energy_relative_change=abs(candidate_energy - energy) / max(
                abs(candidate_energy), fracture_toughness * 2.0 * plate.crack_half_length, 1e-15
            ),
            phase_projected_gradient=residual,
        )
        records.append(record)
        if (
            iteration >= 2 and record.phase_change <= phase_tolerance
            and record.displacement_relative_change <= relative_tolerance
            and record.history_relative_change <= relative_tolerance
            and record.energy_relative_change <= relative_tolerance
            and residual <= 1e-7
        ):
            return PlateStaggeredFEMResult(
                candidate_displacement,
                FrozenPhaseFEMResult(candidate_phase.nodes, candidate_phase.phase, residual),
                candidate_history, tuple(records),
            )
        phase, displacement, history, energy = (
            candidate_phase, candidate_displacement, candidate_history, candidate_energy
        )
    raise RuntimeError(
        f"staggered FEM iteration budget exhausted after {max_iterations} iterations: "
        f"phase_change={record.phase_change:.3e}, phase_KKT={record.phase_projected_gradient:.3e}"
    )


def solve_plate_fem_path(
    plate: CenterCrackPlate,
    material: IsotropicMaterial,
    imposed_displacements: tuple[float, ...],
    fracture_toughness: float,
    length_scale: float,
    residual_stiffness: float,
    cells_x: int,
    cells_y: int,
    *,
    max_iterations: int = 30,
    phase_tolerance: float = 1e-7,
    relative_tolerance: float = 1e-7,
) -> tuple[PlateFEMPathStep, ...]:
    """Commit a prescribed zero-start path, stopping at the first failed step.

    Work is the signed trapezoidal integral of 2 * top_reaction against the
    top displacement, since the bottom moves oppositely. This history proxy
    does not guarantee exact physical energy balance.
    """

    loads = tuple(imposed_displacements)
    if not loads or loads[0] != 0.0:
        raise ValueError("imposed_displacements must be nonempty and start at zero")
    for value in loads:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0.0:
            raise ValueError("imposed_displacements must be finite nonnegative real scalars")
    committed = []
    work = 0.0
    for load_index, load in enumerate(loads, start=1):
        previous = committed[-1] if committed else None
        try:
            state = solve_staggered_plate_fem(
                plate, material, load, fracture_toughness, length_scale,
                residual_stiffness, cells_x, cells_y,
                previous_phase=previous.state.phase if previous is not None else None,
                previous_history=previous.state.history if previous is not None else None,
                max_iterations=max_iterations, phase_tolerance=phase_tolerance,
                relative_tolerance=relative_tolerance,
            )
            distance = np.hypot(
                np.maximum(np.abs(state.phase.nodes[:, 0]) - plate.crack_half_length, 0.0),
                state.phase.nodes[:, 1],
            )
            lower = previous.state.phase.phase if previous is not None else np.exp(-distance / length_scale)
            history_lower = previous.state.history if previous is not None else np.zeros_like(state.history)
            phase_violation = max(0.0, float(np.max(lower - state.phase.phase)))
            history_violation = max(0.0, float(np.max(history_lower - state.history)))
            if phase_violation > 1e-12 or history_violation > 1e-12:
                raise RuntimeError("FEM path candidate violates nodal irreversibility")
            _, fracture_energy = _phase_residual_and_fracture_energy(
                state.phase, state.history, lower, fracture_toughness, length_scale
            )
        except RuntimeError as error:
            raise PlateFEMPathError(load_index, float(load), tuple(committed)) from error
        if previous is not None:
            work += (previous.state.displacement.top_reaction + state.displacement.top_reaction) * (
                load - previous.imposed_displacement
            )
        committed.append(PlateFEMPathStep(
            load_index, float(load), state, fracture_energy, work, phase_violation, history_violation,
        ))
    return tuple(committed)


def _reference_provenance(config_path: Path) -> dict:
    package_root = Path(__file__).resolve().parents[1]
    project_root = package_root.parent.parent
    sources = {}
    for source in sorted(package_root.rglob("*.py")):
        content = source.read_bytes()
        sources[f"src/griffith2d/{source.relative_to(package_root).as_posix()}"] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "base64": base64.b64encode(content).decode("ascii"),
        }
    build_config = project_root / "pyproject.toml"
    if build_config.is_file():
        content = build_config.read_bytes()
        sources["pyproject.toml"] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "base64": base64.b64encode(content).decode("ascii"),
        }
    config_content = config_path.read_bytes()
    git = {"available": False}
    try:
        result = subprocess.run(
            ["git", "-C", str(package_root), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10,
        )
        if result.returncode == 0:
            repo = Path(result.stdout.strip())
            scoped_paths = [package_root.relative_to(repo).as_posix()]
            for path in (build_config, config_path.resolve()):
                if path.is_relative_to(repo):
                    scoped_paths.append(path.relative_to(repo).as_posix())
            revision = subprocess.run(
                ["git", "-C", str(repo), "rev-parse", "HEAD"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10,
            )
            status = subprocess.run(
                ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all", "--", *scoped_paths],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10,
            )
            git = {
                "available": revision.returncode == 0 and status.returncode == 0,
                "revision": revision.stdout.strip() if revision.returncode == 0 else None,
                "scoped_status": status.stdout,
            }
    except (OSError, subprocess.TimeoutExpired, ValueError) as error:
        git["error"] = str(error)
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git,
        "source_snapshot": sources,
        "source_snapshot_scope": "complete_griffith2d_python_package_and_available_pyproject",
        "config_snapshot": {
            "path": str(config_path.resolve()),
            "sha256": hashlib.sha256(config_content).hexdigest(),
            "base64": base64.b64encode(config_content).decode("ascii"),
        },
        "runtime": {
            "python_version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "working_directory": str(Path.cwd()),
            "environment": {name: os.environ.get(name) for name in (
                "CONDA_DEFAULT_ENV", "CONDA_PREFIX", "MKL_THREADING_LAYER",
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            )},
            "versions": {name: version(name) for name in ("numpy", "scipy", "scikit-fem", "torch")},
            "randomness": "deterministic_FEM_no_random_initialization",
        },
    }


def run(config_path: Path, *, provenance: dict | None = None) -> dict:
    """Generate a reproducible nodal-history FEM reference from TOML."""

    if provenance is None:
        provenance = _reference_provenance(config_path)
    config = tomllib.loads(base64.b64decode(provenance["config_snapshot"]["base64"]).decode("utf-8"))
    if config["benchmark"]["kind"] != "center_crack_fem_path":
        raise ValueError("benchmark.kind must be 'center_crack_fem_path'")
    geometry = config["geometry"]
    elastic = config["elastic_material"]
    phase = config["phase_material"]
    mesh = config["mesh"]
    solver = config["solver"]
    steps = solve_plate_fem_path(
        CenterCrackPlate(geometry["half_width"], geometry["half_height"], geometry["crack_half_length"]),
        IsotropicMaterial(elastic["young_modulus"], elastic["poisson_ratio"]),
        tuple(config["loading"]["imposed_displacements"]),
        phase["fracture_toughness"], phase["length_scale"], phase["residual_stiffness"],
        mesh["cells_x"], mesh["cells_y"], max_iterations=solver["max_iterations"],
        phase_tolerance=solver["phase_tolerance"], relative_tolerance=solver["relative_tolerance"],
    )
    initial_energy = steps[0].state.displacement.strain_energy + steps[0].fracture_energy
    output_steps = []
    for step in steps:
        total_energy = step.state.displacement.strain_energy + step.fracture_energy
        output_steps.append({
            "load_index": step.load_index,
            "imposed_displacement": step.imposed_displacement,
            "top_reaction": step.state.displacement.top_reaction,
            "elastic_energy": step.state.displacement.strain_energy,
            "fracture_energy": step.fracture_energy,
            "total_energy": total_energy,
            "cumulative_work": step.cumulative_work,
            "work_minus_energy_increment": step.cumulative_work - (total_energy - initial_energy),
            "phase_violation": step.phase_violation,
            "history_violation": step.history_violation,
            "displacement_residual": step.state.displacement.free_residual_relative,
            "phase_projected_gradient": step.state.phase.projected_gradient,
            "iterations": [asdict(record) for record in step.state.records],
            "displacement": step.state.displacement.displacement.tolist(),
            "phase": step.state.phase.phase.tolist(),
            "history": step.state.history.tolist(),
        })
    return {
        "schema_version": 2,
        "status": "converged",
        "reference_scope": "nodal_history_proxy_not_certified_propagation",
        "config": config,
        "provenance": provenance,
        "versions": {name: version(name) for name in ("numpy", "scipy", "scikit-fem", "torch")},
        "method": {
            "history_transfer": "positive_mass_lumped_P1",
            "history_update": "max_committed_history_and_current_recovered_active_energy",
            "phase_lower_bound": "max_committed_phase_and_geometric_precrack",
            "phase_quadrature_order": 4,
            "displacement_quadrature_order": 4,
            "displacement_residual_tolerance": 1e-9,
            "phase_kkt_tolerance": 1e-7,
            "phase_optimizer": "L-BFGS-B",
            "phase_optimizer_gtol": 1e-11,
            "phase_optimizer_ftol": 1e-15,
            "phase_optimizer_max_iterations": 2000,
            "work_rule": "signed_trapezoidal_two_opposite_displacement_edges",
        },
        "nodes": steps[0].state.phase.nodes.tolist(),
        "triangles": MeshTri.init_tensor(
            np.unique(steps[0].state.phase.nodes[:, 0]),
            np.unique(steps[0].state.phase.nodes[:, 1]),
        ).t.T.tolist(),
        "steps": output_steps,
    }


def _archive_json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--failure-output", type=Path)
    arguments = parser.parse_args()
    run_id = uuid4().hex
    failure_output = arguments.failure_output or (
        arguments.output.with_name(f"{arguments.output.stem}.{run_id}.failure.json")
        if arguments.output is not None else Path(f"results/references/{run_id}.failure.json")
    )
    if arguments.output is not None and arguments.output.resolve() == failure_output.resolve():
        parser.error("--failure-output must differ from --output")
    if failure_output.exists():
        parser.error("--failure-output already exists; choose a new archive path")
    provenance = _reference_provenance(arguments.config)
    try:
        payload = run(arguments.config, provenance=provenance)
    except (RuntimeError, ValueError, KeyError) as error:
        failure = {
            "schema_version": 2,
            "status": "failed",
            "run_id": run_id,
            "config_path": str(arguments.config.resolve()),
            "provenance": provenance,
            "load_index": error.load_index if isinstance(error, PlateFEMPathError) else None,
            "imposed_displacement": error.imposed_displacement if isinstance(error, PlateFEMPathError) else None,
            "exception_type": type(error).__name__,
            "message": str(error),
            "cause": str(error.__cause__),
            "cause_type": type(error.__cause__).__name__ if error.__cause__ is not None else None,
            "traceback": "".join(traceback.format_exception(error)),
            "committed_steps": [asdict(step) for step in error.committed_steps] if isinstance(error, PlateFEMPathError) else [],
            "command": [sys.executable, "-m", "griffith2d.benchmarks.center_crack_fem", *sys.argv[1:]],
        }
        try:
            failure_output.parent.mkdir(parents=True, exist_ok=True)
            serialized_failure = json.dumps(
                failure, indent=2, sort_keys=True, allow_nan=False, default=_archive_json_value,
            )
            with failure_output.open("x", encoding="utf-8") as stream:
                stream.write(serialized_failure + "\n")
            print(f"Failure evidence: {failure_output}", file=sys.stderr)
        except OSError as archive_error:
            error.add_note(f"Failure archive could not be saved: {archive_error}")
            print(f"Failure archive could not be saved: {archive_error}", file=sys.stderr)
        raise
    payload["run_id"] = run_id
    output = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    if arguments.output is None:
        print(output)
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()