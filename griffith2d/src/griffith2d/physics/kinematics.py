"""Small-strain kinematics under the plane-strain assumption.

Public tensor inputs use full tensor notation with shape ``(..., 2, 2)``. Any
leading dimensions are batch dimensions. Functions preserve the input tensor's
dtype and device and do not change PyTorch's global defaults.
"""

from typing import NamedTuple

import torch
from torch import Tensor


class PlaneStrainDecomposition(NamedTuple):
    """Volumetric strain and the three-dimensional deviatoric strain tensor."""

    volumetric_strain: Tensor
    deviatoric_strain: Tensor


def _require_tensor_2x2(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim < 2 or value.shape[-2:] != (2, 2):
        raise ValueError(f"{name} must have shape (..., 2, 2)")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")


def small_strain(displacement_gradient: Tensor) -> Tensor:
    """Return the symmetric small-strain tensor from ``grad(u)``.

    ``displacement_gradient[..., i, j]`` denotes ``du_i / dx_j``. The result is
    ``epsilon = (grad(u) + grad(u)^T) / 2`` and has the same shape, dtype, and
    device as the input.
    """

    _require_tensor_2x2(displacement_gradient, "displacement_gradient")
    return 0.5 * (displacement_gradient + displacement_gradient.transpose(-1, -2))


def plane_strain_decomposition(strain: Tensor) -> PlaneStrainDecomposition:
    """Return the 3D volumetric-deviatoric split for a 2D plane strain state.

    The input is the in-plane symmetric tensor. Plane strain sets all components
    involving ``z`` to zero before applying the three-dimensional split
    ``e = epsilon - tr(epsilon) I / 3``. Consequently, the returned deviatoric
    tensor has shape ``(..., 3, 3)`` and contains the required
    ``e_zz = -(epsilon_xx + epsilon_yy) / 3`` term.
    """

    _require_tensor_2x2(strain, "strain")

    strain_3d = strain.new_zeros((*strain.shape[:-2], 3, 3))
    strain_3d[..., :2, :2] = strain
    volumetric_strain = strain.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    identity = torch.eye(3, dtype=strain.dtype, device=strain.device)
    deviatoric_strain = strain_3d - volumetric_strain[..., None, None] * identity / 3.0

    return PlaneStrainDecomposition(volumetric_strain, deviatoric_strain)
