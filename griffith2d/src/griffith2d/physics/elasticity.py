"""Plane-strain isotropic elasticity with the Amor energy split.

All energy densities are per unit volume (or per unit thickness in the 2D
model). Tensor inputs use shape ``(..., 2, 2)`` and output energy densities use
the corresponding batch shape ``(...)``.
"""

from typing import NamedTuple

import torch
from torch import Tensor

from .degradation import quadratic_degradation
from .kinematics import _require_tensor_2x2, plane_strain_decomposition
from .materials import IsotropicMaterial


class AmorEnergySplit(NamedTuple):
    """Active and inactive elastic energy densities in the Amor split."""

    active: Tensor
    inactive: Tensor


def _phase_for_energy(phase_field: Tensor, energy: Tensor) -> Tensor:
    if not isinstance(phase_field, Tensor):
        raise TypeError("phase_field must be a torch.Tensor")
    if not phase_field.is_floating_point():
        raise TypeError("phase_field must have a floating-point dtype")
    if phase_field.device != energy.device:
        raise ValueError("phase_field and strain must be on the same device")
    if phase_field.dtype != energy.dtype:
        raise TypeError("phase_field and strain must have the same dtype")

    try:
        broadcast_shape = torch.broadcast_shapes(phase_field.shape, energy.shape)
    except RuntimeError as error:
        raise ValueError("phase_field must be scalar or broadcastable to the strain batch shape") from error
    if broadcast_shape != energy.shape:
        raise ValueError("phase_field must not introduce additional batch dimensions")
    return phase_field


def undamaged_elastic_energy(strain: Tensor, material: IsotropicMaterial) -> Tensor:
    """Return ``psi_0 = K tr(epsilon)^2 / 2 + mu e:e`` under plane strain."""

    _require_tensor_2x2(strain, "strain")
    decomposition = plane_strain_decomposition(strain)
    deviatoric_norm_squared = decomposition.deviatoric_strain.square().sum(dim=(-2, -1))
    return (
        0.5 * material.bulk_modulus * decomposition.volumetric_strain.square()
        + material.shear_modulus * deviatoric_norm_squared
    )


def undamaged_stress(strain: Tensor, material: IsotropicMaterial) -> Tensor:
    """Return the in-plane stress ``K tr(epsilon) I + 2 mu e``."""

    _require_tensor_2x2(strain, "strain")
    decomposition = plane_strain_decomposition(strain)
    identity = torch.eye(2, dtype=strain.dtype, device=strain.device)
    return (
        material.bulk_modulus * decomposition.volumetric_strain[..., None, None] * identity
        + 2.0 * material.shear_modulus * decomposition.deviatoric_strain[..., :2, :2]
    )


def amor_energy_split(strain: Tensor, material: IsotropicMaterial) -> AmorEnergySplit:
    """Return active ``psi+`` and inactive ``psi-`` elastic energy densities."""

    _require_tensor_2x2(strain, "strain")
    decomposition = plane_strain_decomposition(strain)
    positive_volume = torch.clamp_min(decomposition.volumetric_strain, 0.0)
    negative_volume = torch.clamp_max(decomposition.volumetric_strain, 0.0)
    deviatoric_norm_squared = decomposition.deviatoric_strain.square().sum(dim=(-2, -1))

    active = (
        0.5 * material.bulk_modulus * positive_volume.square()
        + material.shear_modulus * deviatoric_norm_squared
    )
    inactive = 0.5 * material.bulk_modulus * negative_volume.square()
    return AmorEnergySplit(active, inactive)


def degraded_elastic_energy(
    strain: Tensor,
    phase_field: Tensor,
    material: IsotropicMaterial,
    residual_stiffness: float,
) -> Tensor:
    """Return ``psi = g(phi) psi+ + psi-`` for the selected Amor model."""

    split = amor_energy_split(strain, material)
    phase_field = _phase_for_energy(phase_field, split.active)
    degradation = quadratic_degradation(phase_field, residual_stiffness)
    return degradation * split.active + split.inactive


def degraded_stress(
    strain: Tensor,
    phase_field: Tensor,
    material: IsotropicMaterial,
    residual_stiffness: float,
) -> Tensor:
    """Return the in-plane Cauchy stress derived from degraded elastic energy.

    This is the analytic derivative of :func:`degraded_elastic_energy` with
    respect to the full symmetric strain tensor. Autograd and finite differences
    are intentionally reserved for independent tests.
    """

    _require_tensor_2x2(strain, "strain")
    decomposition = plane_strain_decomposition(strain)
    phase_field = _phase_for_energy(phase_field, decomposition.volumetric_strain)
    degradation = quadratic_degradation(phase_field, residual_stiffness)
    positive_volume = torch.clamp_min(decomposition.volumetric_strain, 0.0)
    negative_volume = torch.clamp_max(decomposition.volumetric_strain, 0.0)
    identity = torch.eye(2, dtype=strain.dtype, device=strain.device)

    active_stress = (
        material.bulk_modulus * positive_volume[..., None, None] * identity
        + 2.0 * material.shear_modulus * decomposition.deviatoric_strain[..., :2, :2]
    )
    inactive_stress = material.bulk_modulus * negative_volume[..., None, None] * identity
    return degradation[..., None, None] * active_stress + inactive_stress
