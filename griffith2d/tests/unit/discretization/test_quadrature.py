import math

import pytest
import torch

from griffith2d.discretization import (
    gauss_legendre,
    interval_quadrature,
    line_quadrature,
    rectangle_quadrature,
)


DTYPE = torch.float64
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("order", [1, 2, 3, 5, 8])
def test_gauss_legendre_has_positive_symmetric_rule(order: int) -> None:
    rule = gauss_legendre(order)

    assert torch.all(rule.weights > 0.0)
    torch.testing.assert_close(rule.points[:, 0], -rule.points.flip(0)[:, 0], atol=1e-14, rtol=0.0)
    torch.testing.assert_close(rule.weights, rule.weights.flip(0), atol=1e-14, rtol=0.0)
    torch.testing.assert_close(rule.weights.sum(), torch.tensor(2.0, dtype=DTYPE), atol=1e-14, rtol=0.0)


@pytest.mark.parametrize("order", [1, 2, 3, 5, 8])
def test_gauss_legendre_integrates_through_algebraic_degree(order: int) -> None:
    rule = gauss_legendre(order)
    x = rule.points[:, 0]

    for degree in range(2 * order):
        expected = 0.0 if degree % 2 else 2.0 / (degree + 1)
        actual = rule.integrate(x**degree)
        torch.testing.assert_close(
            actual, torch.tensor(expected, dtype=DTYPE), atol=1e-12, rtol=0.0
        )


def test_interval_rule_includes_physical_jacobian() -> None:
    rule = interval_quadrature(-2.0, 3.0, order=4)
    x = rule.points[:, 0]

    torch.testing.assert_close(rule.weights.sum(), torch.tensor(5.0, dtype=DTYPE))
    torch.testing.assert_close(rule.integrate(x**3), torch.tensor(65.0 / 4.0, dtype=DTYPE))


def test_line_rule_integrates_vector_values() -> None:
    start = torch.tensor([1.0, -1.0], dtype=DTYPE)
    end = torch.tensor([4.0, 3.0], dtype=DTYPE)
    rule = line_quadrature(start, end, order=3)
    values = torch.stack((rule.points[:, 0], torch.ones_like(rule.weights)), dim=-1)

    result = rule.integrate(values)

    torch.testing.assert_close(rule.weights.sum(), torch.tensor(5.0, dtype=DTYPE))
    torch.testing.assert_close(result, torch.tensor([12.5, 5.0], dtype=DTYPE))


def test_rectangle_rule_integrates_separable_polynomial() -> None:
    rule = rectangle_quadrature((-2.0, 1.0), (1.0, 5.0), order=4)
    x = rule.points[:, 0]
    y = rule.points[:, 1]

    result = rule.integrate(x**4 * y**3)
    expected = ((1.0**5 - (-2.0) ** 5) / 5.0) * ((5.0**4 - 1.0**4) / 4.0)

    assert rule.points.shape == (16, 2)
    torch.testing.assert_close(rule.weights.sum(), torch.tensor(12.0, dtype=DTYPE))
    torch.testing.assert_close(result, torch.tensor(expected, dtype=DTYPE), atol=1e-12, rtol=0.0)


def test_smooth_nonpolynomial_error_decreases_with_order() -> None:
    errors = []
    expected = math.e - 1.0 / math.e
    for order in (2, 3, 4, 5):
        rule = interval_quadrature(-1.0, 1.0, order)
        value = rule.integrate(torch.exp(rule.points[:, 0])).item()
        errors.append(abs(value - expected))

    assert all(next_error < error for error, next_error in zip(errors, errors[1:]))
    assert errors[-1] < 1e-9


@pytest.mark.parametrize("device", DEVICES)
def test_rectangle_rule_preserves_dtype_and_device(device: str) -> None:
    rule = rectangle_quadrature((0.0, 2.0), (-1.0, 3.0), order=3, device=device)

    assert rule.points.dtype == DTYPE
    assert rule.weights.dtype == DTYPE
    assert rule.points.device.type == device
    torch.testing.assert_close(
        rule.integrate(torch.ones(9, dtype=DTYPE, device=device)),
        torch.tensor(8.0, dtype=DTYPE, device=device),
    )


@pytest.mark.parametrize("order", [0, 33, 1.5, True])
def test_invalid_order_is_rejected(order: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        gauss_legendre(order)  # type: ignore[arg-type]


def test_invalid_geometry_and_integrand_are_rejected() -> None:
    with pytest.raises(ValueError, match="lower < upper"):
        interval_quadrature(1.0, 1.0, order=2)
    with pytest.raises(ValueError, match="positive length"):
        point = torch.zeros(2, dtype=DTYPE)
        line_quadrature(point, point, order=2)

    rule = gauss_legendre(2)
    with pytest.raises(ValueError, match="quadrature-point axis"):
        rule.integrate(torch.ones(3, dtype=DTYPE))