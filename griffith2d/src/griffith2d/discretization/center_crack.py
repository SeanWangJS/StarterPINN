"""Full-plate geometry and deterministic rules for a diffuse center precrack."""

from dataclasses import dataclass
import math
from numbers import Real
from typing import NamedTuple

import torch
from torch import Tensor

from .quadrature import (
    QuadratureRule,
    line_quadrature,
    partitioned_rectangle_quadrature,
)


class PlateBoundaryRules(NamedTuple):
    top: QuadratureRule
    bottom: QuadratureRule
    left: QuadratureRule
    right: QuadratureRule


@dataclass(frozen=True, slots=True)
class CenterCrackPlate:
    """Full plate ``(-W,W) x (-H,H)`` with a geometric crack ``[-a,a] x {0}``."""

    half_width: float
    half_height: float
    crack_half_length: float

    def __post_init__(self) -> None:
        for name in ("half_width", "half_height", "crack_half_length"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real scalar")
            value = float(value)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if self.crack_half_length >= self.half_width:
            raise ValueError("crack_half_length must be less than half_width")

    def crack_distance(self, coordinates: Tensor) -> Tensor:
        """Return Euclidean distance to the finite crack segment with shape ``(..., 1)``."""

        if not isinstance(coordinates, Tensor):
            raise TypeError("coordinates must be a torch.Tensor")
        if coordinates.ndim < 1 or coordinates.shape[-1] != 2 or not coordinates.is_floating_point():
            raise ValueError("coordinates must be floating point with shape (..., 2)")
        longitudinal = (coordinates[..., 0].abs() - self.crack_half_length).clamp_min(0.0)
        return torch.hypot(longitudinal, coordinates[..., 1]).unsqueeze(-1)

    def initial_phase(self, coordinates: Tensor, length_scale: float) -> Tensor:
        """Return the geometric exponential profile, without imposing irreversibility."""

        if isinstance(length_scale, bool) or not isinstance(length_scale, Real):
            raise TypeError("length_scale must be a real scalar")
        if not math.isfinite(length_scale) or length_scale <= 0.0:
            raise ValueError("length_scale must be finite and positive")
        return torch.exp(-self.crack_distance(coordinates) / length_scale)

    def domain_rule(
        self,
        band_half_width: float,
        order: int,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> QuadratureRule:
        """Partition at both crack tips and both sides of the resolved damage band."""

        if isinstance(band_half_width, bool) or not isinstance(band_half_width, Real):
            raise TypeError("band_half_width must be a real scalar")
        if not math.isfinite(band_half_width) or not 0.0 < band_half_width < self.half_height:
            raise ValueError("band_half_width must lie strictly between 0 and half_height")
        return partitioned_rectangle_quadrature(
            (-self.half_width, -self.crack_half_length, self.crack_half_length, self.half_width),
            (-self.half_height, -band_half_width, 0.0, band_half_width, self.half_height),
            order,
            dtype=dtype,
            device=device,
        )

    def boundary_rules(
        self,
        order: int,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> PlateBoundaryRules:
        """Return the four exterior edges; the diffuse crack is not a cut boundary."""

        def point(x_coordinate: float, y_coordinate: float) -> Tensor:
            return torch.tensor((x_coordinate, y_coordinate), dtype=dtype, device=device)

        width, height = self.half_width, self.half_height
        return PlateBoundaryRules(
            top=line_quadrature(point(-width, height), point(width, height), order),
            bottom=line_quadrature(point(-width, -height), point(width, -height), order),
            left=line_quadrature(point(-width, -height), point(-width, height), order),
            right=line_quadrature(point(width, -height), point(width, height), order),
        )