"""Kinematically admissible trial fields for the elastic rectangle benchmark."""

from numbers import Real

import torch
from torch import Tensor, nn


def _positive_scalar(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    value = float(value)
    if not torch.isfinite(torch.tensor(value)) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


class SymmetryTrialSolution(nn.Module):
    """Enforce ``u_x(0,y)=0`` and ``u_y(x,0)=0`` exactly.

    The wrapped network receives normalized coordinates ``(x/L, y/H)`` and
    predicts dimensionless coefficients. The physical displacement is
    ``u_0 * (x/L * N_x, y/H * N_y)``.
    """

    def __init__(
        self,
        network: nn.Module,
        *,
        length: float,
        height: float,
        displacement_scale: float,
    ) -> None:
        super().__init__()
        self.network = network
        self.length = _positive_scalar(length, "length")
        self.height = _positive_scalar(height, "height")
        self.displacement_scale = _positive_scalar(displacement_scale, "displacement_scale")

    def forward(self, coordinates: Tensor) -> Tensor:
        """Evaluate the physical displacement at points of shape ``(..., 2)``."""

        if not isinstance(coordinates, Tensor):
            raise TypeError("coordinates must be a torch.Tensor")
        if coordinates.ndim < 1 or coordinates.shape[-1] != 2:
            raise ValueError("coordinates must have shape (..., 2)")
        if not coordinates.is_floating_point():
            raise TypeError("coordinates must have a floating-point dtype")

        scale = coordinates.new_tensor([self.length, self.height])
        normalized = coordinates / scale
        coefficients = self.network(normalized)
        if coefficients.shape != coordinates.shape:
            raise ValueError("network output must have the same shape as coordinates")
        return self.displacement_scale * normalized * coefficients


class PhaseFieldTrialSolution(nn.Module):
    """Evaluate a scalar network in the dimensionless coordinate ``s/l0``."""

    def __init__(self, network: nn.Module, *, length_scale: float, precrack: bool) -> None:
        super().__init__()
        self.network = network
        self.length_scale = _positive_scalar(length_scale, "length_scale")
        if not isinstance(precrack, bool):
            raise TypeError("precrack must be a bool")
        self.precrack = precrack

    def forward(self, coordinates: Tensor) -> Tensor:
        if not isinstance(coordinates, Tensor):
            raise TypeError("coordinates must be a torch.Tensor")
        if not coordinates.is_floating_point():
            raise TypeError("coordinates must have a floating-point dtype")
        if coordinates.ndim < 1 or coordinates.shape[-1] != 1:
            raise ValueError("coordinates must have shape (..., 1)")

        normalized = coordinates / self.length_scale
        coefficients = self.network(normalized)
        if coefficients.shape != coordinates.shape:
            raise ValueError("network output must have the same shape as coordinates")
        if self.precrack:
            return 1.0 + normalized * coefficients
        return coefficients
