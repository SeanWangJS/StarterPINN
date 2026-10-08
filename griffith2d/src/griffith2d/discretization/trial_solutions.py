"""Kinematically admissible trial fields for the elastic rectangle benchmark."""

from math import isfinite
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


class PlateDisplacementTrialSolution(nn.Module):
    """Full-plate field with prescribed opposite vertical edge displacements.

    A deterministic pointwise network receives ``(x/W, y/H)``. Subtracting its
    horizontal value at the origin pins only ``u_x(0,0)``. The vertical field
    is ``d*y/H + scale*(1-(y/H)**2)*N_y``. No other boundary values or natural
    traction conditions are imposed by this representation.
    """

    def __init__(
        self,
        network: nn.Module,
        *,
        half_width: float,
        half_height: float,
        imposed_displacement: float,
        displacement_scale: float,
    ) -> None:
        super().__init__()
        self.network = network
        self.half_width = _positive_scalar(half_width, "half_width")
        self.half_height = _positive_scalar(half_height, "half_height")
        self.displacement_scale = _positive_scalar(displacement_scale, "displacement_scale")
        if isinstance(imposed_displacement, bool) or not isinstance(imposed_displacement, Real):
            raise TypeError("imposed_displacement must be a real scalar")
        self.imposed_displacement = float(imposed_displacement)
        if not isfinite(self.imposed_displacement) or self.imposed_displacement < 0.0:
            raise ValueError("imposed_displacement must be finite and nonnegative")

    def forward(self, coordinates: Tensor) -> Tensor:
        if not isinstance(coordinates, Tensor):
            raise TypeError("coordinates must be a torch.Tensor")
        if coordinates.ndim < 1 or coordinates.shape[-1] != 2:
            raise ValueError("coordinates must have shape (..., 2)")
        if not coordinates.is_floating_point():
            raise TypeError("coordinates must have a floating-point dtype")
        normalized = coordinates / coordinates.new_tensor([self.half_width, self.half_height])
        coefficients = self.network(normalized)
        origin = self.network(coordinates.new_zeros(2))
        for value, shape in ((coefficients, coordinates.shape), (origin, (2,))):
            if not isinstance(value, Tensor) or value.shape != shape:
                raise ValueError("network output must have the same shape as its coordinates")
            if value.dtype != coordinates.dtype or value.device != coordinates.device:
                raise ValueError("network output must preserve coordinate dtype and device")
        horizontal = self.displacement_scale * (coefficients[..., 0] - origin[0])
        vertical_coordinate = normalized[..., 1]
        vertical = (
            self.imposed_displacement * vertical_coordinate
            + self.displacement_scale * (1.0 - vertical_coordinate.square()) * coefficients[..., 1]
        )
        return torch.stack((horizontal, vertical), dim=-1)


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
