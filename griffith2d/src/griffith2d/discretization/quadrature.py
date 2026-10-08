"""Deterministic Gauss-Legendre rules for intervals, lines, and rectangles."""

from dataclasses import dataclass
from numbers import Integral, Real

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class QuadratureRule:
    """Quadrature coordinates and physical integration weights.

    ``points`` has shape ``(number_of_points, spatial_dimension)`` and
    ``weights`` has shape ``(number_of_points,)``. Weights include the physical
    mapping Jacobian, so integration is always an explicit weighted sum.
    """

    points: Tensor
    weights: Tensor

    def __post_init__(self) -> None:
        if self.points.ndim != 2:
            raise ValueError("points must have shape (number_of_points, dimension)")
        if self.weights.ndim != 1 or self.weights.shape[0] != self.points.shape[0]:
            raise ValueError("weights must have shape (number_of_points,)")
        if not self.points.is_floating_point() or not self.weights.is_floating_point():
            raise TypeError("points and weights must have floating-point dtypes")
        if self.points.dtype != self.weights.dtype or self.points.device != self.weights.device:
            raise ValueError("points and weights must have the same dtype and device")
        if not bool(torch.all(self.weights > 0.0)):
            raise ValueError("quadrature weights must be positive")

    def integrate(self, values: Tensor) -> Tensor:
        """Integrate values whose first axis indexes this rule's points."""

        if not isinstance(values, Tensor):
            raise TypeError("values must be a torch.Tensor")
        if values.ndim == 0 or values.shape[0] != self.points.shape[0]:
            raise ValueError("values must have the quadrature-point axis first")
        if values.dtype != self.weights.dtype or values.device != self.weights.device:
            raise ValueError("values and quadrature rule must have the same dtype and device")
        weight_shape = (self.weights.shape[0],) + (1,) * (values.ndim - 1)
        return (values * self.weights.reshape(weight_shape)).sum(dim=0)


def _validate_order(order: int) -> int:
    if isinstance(order, bool) or not isinstance(order, Integral):
        raise TypeError("order must be an integer")
    order = int(order)
    if not 1 <= order <= 32:
        raise ValueError("order must satisfy 1 <= order <= 32")
    return order


def _validate_dtype(dtype: torch.dtype) -> None:
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("quadrature dtype must be torch.float32 or torch.float64")


def _finite_bound(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    value = float(value)
    if not torch.isfinite(torch.tensor(value)):
        raise ValueError(f"{name} must be finite")
    return value


def gauss_legendre(
    order: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> QuadratureRule:
    """Return an ``order``-point Gauss-Legendre rule on ``[-1, 1]``.

    Nodes and weights are generated with the Golub-Welsch eigendecomposition of
    the symmetric Legendre Jacobi matrix. Supported orders are 1 through 32.
    """

    order = _validate_order(order)
    _validate_dtype(dtype)
    resolved_device = torch.device("cpu" if device is None else device)

    jacobi = torch.zeros((order, order), dtype=dtype, device=resolved_device)
    if order > 1:
        indices = torch.arange(1, order, dtype=dtype, device=resolved_device)
        off_diagonal = indices / torch.sqrt(4.0 * indices.square() - 1.0)
        jacobi.diagonal(offset=1).copy_(off_diagonal)
        jacobi.diagonal(offset=-1).copy_(off_diagonal)

    nodes, eigenvectors = torch.linalg.eigh(jacobi)
    weights = 2.0 * eigenvectors[0, :].square()
    return QuadratureRule(nodes[:, None], weights)


def interval_quadrature(
    lower: float,
    upper: float,
    order: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> QuadratureRule:
    """Map a Gauss-Legendre rule to the finite interval ``[lower, upper]``."""

    lower = _finite_bound(lower, "lower")
    upper = _finite_bound(upper, "upper")
    if not lower < upper:
        raise ValueError("interval bounds must satisfy lower < upper")

    reference = gauss_legendre(order, dtype=dtype, device=device)
    half_length = 0.5 * (upper - lower)
    midpoint = 0.5 * (upper + lower)
    return QuadratureRule(midpoint + half_length * reference.points, half_length * reference.weights)


def rectangle_quadrature(
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    order: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> QuadratureRule:
    """Return the tensor-product rule on an axis-aligned rectangle."""

    x_rule = interval_quadrature(*x_bounds, order, dtype=dtype, device=device)
    y_rule = interval_quadrature(*y_bounds, order, dtype=dtype, device=device)
    x_points, y_points = torch.meshgrid(
        x_rule.points[:, 0], y_rule.points[:, 0], indexing="ij"
    )
    weights = torch.outer(x_rule.weights, y_rule.weights)
    points = torch.stack((x_points.reshape(-1), y_points.reshape(-1)), dim=-1)
    return QuadratureRule(points, weights.reshape(-1))


def partitioned_rectangle_quadrature(
    x_edges: tuple[float, ...],
    y_edges: tuple[float, ...],
    order: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> QuadratureRule:
    """Join weighted rectangle rules over an explicit, non-overlapping partition."""

    def validated_edges(edges: tuple[float, ...], name: str) -> tuple[float, ...]:
        if len(edges) < 2:
            raise ValueError(f"{name} must contain at least two edges")
        values = tuple(_finite_bound(edge, name) for edge in edges)
        if any(left >= right for left, right in zip(values, values[1:])):
            raise ValueError(f"{name} edges must be strictly increasing")
        return values

    x_values = validated_edges(x_edges, "x_edges")
    y_values = validated_edges(y_edges, "y_edges")
    rules = [
        rectangle_quadrature(
            (x_left, x_right), (y_lower, y_upper), order, dtype=dtype, device=device
        )
        for x_left, x_right in zip(x_values, x_values[1:])
        for y_lower, y_upper in zip(y_values, y_values[1:])
    ]
    return QuadratureRule(
        torch.cat([rule.points for rule in rules]),
        torch.cat([rule.weights for rule in rules]),
    )


def line_quadrature(
    start: Tensor,
    end: Tensor,
    order: int,
) -> QuadratureRule:
    """Map a Gauss-Legendre rule to a non-degenerate 2D line segment."""

    for point, name in ((start, "start"), (end, "end")):
        if not isinstance(point, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if point.shape != (2,) or not point.is_floating_point():
            raise ValueError(f"{name} must be a floating-point tensor with shape (2,)")
    if start.dtype != end.dtype or start.device != end.device:
        raise ValueError("start and end must have the same dtype and device")

    direction = end - start
    length = torch.linalg.vector_norm(direction)
    if not bool(length > 0.0):
        raise ValueError("line segment must have positive length")

    reference = gauss_legendre(order, dtype=start.dtype, device=start.device)
    parameter = 0.5 * (reference.points[:, 0] + 1.0)
    points = start + parameter[:, None] * direction
    weights = 0.5 * length * reference.weights
    return QuadratureRule(points, weights)
