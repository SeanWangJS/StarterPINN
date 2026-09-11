import pytest
import torch

from griffith2d.physics import quadratic_degradation, quadratic_degradation_derivative


DTYPE = torch.float64


def test_quadratic_degradation_endpoints_and_derivative() -> None:
    phase_field = torch.tensor([0.0, 0.25, 1.0], dtype=DTYPE)
    residual_stiffness = 1e-6

    degradation = quadratic_degradation(phase_field, residual_stiffness)
    derivative = quadratic_degradation_derivative(phase_field, residual_stiffness)

    torch.testing.assert_close(
        degradation,
        torch.tensor([1.0 + residual_stiffness, 0.75**2 + residual_stiffness, residual_stiffness], dtype=DTYPE),
    )
    torch.testing.assert_close(derivative, torch.tensor([-2.0, -1.5, 0.0], dtype=DTYPE))


@pytest.mark.parametrize("residual_stiffness", [0.0, -1e-6, 1.0, float("inf"), float("nan")])
def test_invalid_residual_stiffness_is_rejected(residual_stiffness: float) -> None:
    with pytest.raises(ValueError, match="residual_stiffness"):
        quadratic_degradation(torch.tensor(0.0, dtype=DTYPE), residual_stiffness)


def test_degradation_does_not_silently_clamp_phase_field() -> None:
    phase_field = torch.tensor([-0.1, 1.1], dtype=DTYPE)

    result = quadratic_degradation(phase_field, 1e-6)

    torch.testing.assert_close(result, (1.0 - phase_field).square() + 1e-6)
