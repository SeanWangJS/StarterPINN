import pytest
import torch
from torch import Tensor, nn

from griffith2d.discretization import (
    DisplacementMLP,
    PhaseFieldMLP,
    PhaseFieldTrialSolution,
    SymmetryTrialSolution,
    displacement_gradient,
    phase_field_gradient,
    phase_field_laplacian,
)
from griffith2d.physics import small_strain


DTYPE = torch.float64
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def test_displacement_mlp_initialization_is_reproducible_without_global_seed() -> None:
    first = DisplacementMLP(seed=1729)
    second = DisplacementMLP(seed=1729)

    for first_parameter, second_parameter in zip(first.parameters(), second.parameters(), strict=True):
        torch.testing.assert_close(first_parameter, second_parameter, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("device", DEVICES)
def test_displacement_mlp_preserves_shape_dtype_and_device(device: str) -> None:
    network = DisplacementMLP(device=device)
    coordinates = torch.zeros((3, 4, 2), dtype=DTYPE, device=device)

    output = network(coordinates)

    assert output.shape == coordinates.shape
    assert output.dtype == coordinates.dtype
    assert output.device == coordinates.device


class ConstantCoefficients(nn.Module):
    def forward(self, coordinates: Tensor) -> Tensor:
        return torch.ones_like(coordinates)


def test_trial_solution_enforces_only_the_required_symmetry_components() -> None:
    trial = SymmetryTrialSolution(
        ConstantCoefficients(), length=2.0, height=4.0, displacement_scale=0.1
    )
    left = torch.tensor([[0.0, 2.0]], dtype=DTYPE)
    bottom = torch.tensor([[1.0, 0.0]], dtype=DTYPE)

    left_displacement = trial(left)
    bottom_displacement = trial(bottom)

    torch.testing.assert_close(left_displacement[0, 0], torch.tensor(0.0, dtype=DTYPE))
    assert left_displacement[0, 1] != 0.0
    torch.testing.assert_close(bottom_displacement[0, 1], torch.tensor(0.0, dtype=DTYPE))
    assert bottom_displacement[0, 0] != 0.0


def test_displacement_gradient_is_pointwise_for_nonlinear_field() -> None:
    coordinates = torch.tensor([[1.0, 2.0], [3.0, -1.0]], dtype=DTYPE)

    def field(point: Tensor) -> Tensor:
        x, y = point.unbind()
        return torch.stack((x.square() + x * y, y**3 - x))

    gradient = displacement_gradient(field, coordinates)
    expected = torch.tensor(
        [
            [[4.0, 1.0], [-1.0, 12.0]],
            [[5.0, 3.0], [-1.0, 3.0]],
        ],
        dtype=DTYPE,
    )

    torch.testing.assert_close(gradient, expected)


def test_affine_trial_field_has_exact_gradient_and_strain() -> None:
    trial = SymmetryTrialSolution(
        ConstantCoefficients(), length=2.0, height=4.0, displacement_scale=0.1
    )
    coordinates = torch.tensor([[0.4, 0.7], [1.5, 3.0]], dtype=DTYPE)

    gradient = displacement_gradient(trial, coordinates)
    strain = small_strain(gradient)
    expected = torch.tensor([[0.05, 0.0], [0.0, 0.025]], dtype=DTYPE).expand(2, -1, -1)

    torch.testing.assert_close(gradient, expected)
    torch.testing.assert_close(strain, expected)


def test_phase_field_network_is_seed_deterministic() -> None:
    first = PhaseFieldMLP(hidden_width=8, seed=41)
    second = PhaseFieldMLP(hidden_width=8, seed=41)

    for first_parameter, second_parameter in zip(first.parameters(), second.parameters(), strict=True):
        torch.testing.assert_close(first_parameter, second_parameter, rtol=0.0, atol=0.0)


def test_zero_initialized_phase_field_network_is_spatially_constant() -> None:
    network = PhaseFieldMLP(hidden_width=8, seed=41, zero_initialization=True)
    coordinates = torch.tensor([[-2.0], [0.0], [3.0]], dtype=DTYPE)

    output = network(coordinates)

    torch.testing.assert_close(output, torch.zeros_like(output), rtol=0.0, atol=0.0)


def test_precrack_trial_solution_enforces_center_value() -> None:
    model = PhaseFieldTrialSolution(
        PhaseFieldMLP(hidden_width=8, seed=41), length_scale=0.2, precrack=True
    )
    coordinates = torch.tensor([[0.0], [0.3]], dtype=DTYPE)

    phase = model(coordinates)

    torch.testing.assert_close(phase[0], torch.ones(1, dtype=DTYPE), rtol=0.0, atol=0.0)


def test_phase_field_derivatives_are_pointwise() -> None:
    coordinates = torch.tensor([[0.0], [0.5], [2.0]], dtype=DTYPE)

    def quadratic(point: Tensor) -> Tensor:
        return point.square() + 3.0 * point

    gradient = phase_field_gradient(quadratic, coordinates)
    laplacian = phase_field_laplacian(quadratic, coordinates)

    torch.testing.assert_close(gradient, 2.0 * coordinates + 3.0)
    torch.testing.assert_close(laplacian, torch.full_like(coordinates, 2.0))