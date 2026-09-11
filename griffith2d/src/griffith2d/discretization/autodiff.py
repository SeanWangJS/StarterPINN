"""Pointwise spatial derivatives for neural trial fields."""

from collections.abc import Callable

import torch
from torch import Tensor


def displacement_gradient(
    displacement_field: Callable[[Tensor], Tensor], coordinates: Tensor
) -> Tensor:
    """Return the pointwise Jacobian ``du_i/dx_j`` at ``(..., 2)`` coordinates.

    ``jacrev`` is applied to one coordinate at a time and then vectorized. This
    avoids deriving the gradient of a batch sum and makes the pointwise contract
    explicit.
    """

    if not isinstance(coordinates, Tensor):
        raise TypeError("coordinates must be a torch.Tensor")
    if coordinates.ndim < 1 or coordinates.shape[-1] != 2:
        raise ValueError("coordinates must have shape (..., 2)")
    if not coordinates.is_floating_point():
        raise TypeError("coordinates must have a floating-point dtype")

    batch_shape = coordinates.shape[:-1]
    flattened = coordinates.reshape(-1, 2)

    def evaluate_one(point: Tensor) -> Tensor:
        value = displacement_field(point)
        if value.shape != (2,):
            raise ValueError("displacement_field must map a point of shape (2,) to shape (2,)")
        return value

    jacobian = torch.vmap(torch.func.jacrev(evaluate_one))(flattened)
    return jacobian.reshape(*batch_shape, 2, 2)


def phase_field_gradient(
    phase_field: Callable[[Tensor], Tensor], coordinates: Tensor
) -> Tensor:
    """Return the pointwise derivative of a scalar field at 1D coordinates."""

    if not isinstance(coordinates, Tensor):
        raise TypeError("coordinates must be a torch.Tensor")
    if not coordinates.is_floating_point():
        raise TypeError("coordinates must have a floating-point dtype")
    if coordinates.ndim < 1 or coordinates.shape[-1] != 1:
        raise ValueError("coordinates must have shape (..., 1)")

    batch_shape = coordinates.shape[:-1]
    flattened = coordinates.reshape(-1, 1)

    def evaluate_one(point: Tensor) -> Tensor:
        value = phase_field(point)
        if value.shape != (1,):
            raise ValueError("phase_field must map a point of shape (1,) to shape (1,)")
        return value

    gradient = torch.vmap(torch.func.jacrev(evaluate_one))(flattened)
    return gradient.reshape(*batch_shape, 1)


def phase_field_laplacian(
    phase_field: Callable[[Tensor], Tensor], coordinates: Tensor
) -> Tensor:
    """Return the pointwise second derivative of a scalar field in one dimension."""

    def gradient_at(point: Tensor) -> Tensor:
        return torch.func.jacrev(phase_field)(point).reshape(1)

    return phase_field_gradient(gradient_at, coordinates)
