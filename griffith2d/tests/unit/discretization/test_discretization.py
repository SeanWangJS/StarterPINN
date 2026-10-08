import pytest
import torch
from torch import Tensor, nn

from griffith2d.discretization import (
    DisplacementMLP,
    PhaseFieldMLP,
    PhaseFieldTrialSolution,
    PlateDisplacementTrialSolution,
    SymmetryTrialSolution,
    displacement_gradient,
    phase_field_gradient,
    phase_field_laplacian,
)
from griffith2d.physics import IsotropicMaterial, small_strain, undamaged_stress


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


@pytest.mark.parametrize("device", DEVICES)
def test_plate_trial_enforces_minimal_constraints_and_exact_manufactured_gradient(device: str) -> None:
    network = nn.Linear(2, 2, dtype=DTYPE, device=device)
    with torch.no_grad():
        network.weight.copy_(torch.tensor([[2.0, 3.0], [1.0, -2.0]], dtype=DTYPE, device=device))
        network.bias.copy_(torch.tensor([5.0, 7.0], dtype=DTYPE, device=device))
    trial = PlateDisplacementTrialSolution(
        network, half_width=2.0, half_height=4.0,
        imposed_displacement=0.02, displacement_scale=0.1,
    )
    edges = torch.tensor([[-2.0, -4.0], [0.0, -4.0], [2.0, 4.0], [0.0, 4.0]], dtype=DTYPE, device=device)
    torch.testing.assert_close(trial(edges)[:, 1], 0.02 * edges[:, 1] / 4.0, rtol=0.0, atol=0.0)
    center = torch.zeros(2, dtype=DTYPE, device=device)
    assert trial(center)[0].item() == 0.0
    assert trial(center)[1].item() != 0.0
    assert trial(edges)[1, 0].item() != 0.0
    coordinates = torch.tensor([[0.4, 0.7], [-1.5, -3.0]], dtype=DTYPE, device=device)
    normalized = coordinates / coordinates.new_tensor([2.0, 4.0])
    vertical = normalized[:, 1]
    coefficient = normalized[:, 0] - 2.0 * vertical + 7.0
    expected = torch.empty((2, 2, 2), dtype=DTYPE, device=device)
    expected[:, 0, 0] = 0.1
    expected[:, 0, 1] = 0.075
    expected[:, 1, 0] = 0.05 * (1.0 - vertical.square())
    expected[:, 1, 1] = 0.005 + 0.1 * (
        -0.5 * vertical * coefficient - 0.5 * (1.0 - vertical.square())
    )
    torch.testing.assert_close(displacement_gradient(trial, coordinates), expected)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch_shape", [(), (5,), (2, 3)])
def test_plate_trial_preserves_shape_dtype_and_device(device: str, dtype, batch_shape) -> None:
    network = DisplacementMLP(device=device).to(dtype=dtype)
    trial = PlateDisplacementTrialSolution(
        network, half_width=2.0, half_height=1.0,
        imposed_displacement=0.01, displacement_scale=0.1,
    )
    coordinates = torch.zeros((*batch_shape, 2), dtype=dtype, device=device)
    output = trial(coordinates)
    assert output.shape == coordinates.shape
    assert output.dtype == dtype
    assert output.device == coordinates.device
    assert torch.all(output[..., 0] == 0.0)


def test_plate_trial_zero_load_keeps_interior_variations_admissible() -> None:
    trial = PlateDisplacementTrialSolution(
        ConstantCoefficients(), half_width=2.0, half_height=1.0,
        imposed_displacement=0.0, displacement_scale=0.1,
    )
    points = torch.tensor([[0.0, -1.0], [0.0, 1.0], [0.0, 0.0]], dtype=DTYPE)
    output = trial(points)
    torch.testing.assert_close(output[:2, 1], torch.zeros(2, dtype=DTYPE), rtol=0.0, atol=0.0)
    assert output[2, 1] != 0.0


def test_plate_trial_origin_subtraction_and_spatial_gradient_retain_parameter_gradients() -> None:
    network = nn.Linear(2, 2, dtype=DTYPE)
    trial = PlateDisplacementTrialSolution(
        network, half_width=2.0, half_height=1.0,
        imposed_displacement=0.01, displacement_scale=0.1,
    )
    coordinates = torch.tensor([[0.4, 0.3], [1.2, -0.1]], dtype=DTYPE)
    trial(coordinates)[:, 0].sum().backward()
    torch.testing.assert_close(network.weight.grad[0], 0.1 * (coordinates / coordinates.new_tensor([2.0, 1.0])).sum(0))
    torch.testing.assert_close(network.bias.grad, torch.zeros(2, dtype=DTYPE), rtol=0.0, atol=0.0)
    network.zero_grad(set_to_none=True)
    displacement_gradient(trial, coordinates).square().sum().backward()
    assert network.weight.grad is not None and torch.isfinite(network.weight.grad).all()
    assert network.weight.grad.abs().sum() > 0.0
    assert network.bias.grad is not None and torch.isfinite(network.bias.grad).all()
    assert network.bias.grad[1].abs() > 0.0


@pytest.mark.parametrize("device", DEVICES)
def test_plate_trial_represents_uncracked_affine_solution_with_free_lateral_traction(device: str) -> None:
    material = IsotropicMaterial(1.0, 0.3)
    vertical_strain = 0.01
    horizontal_strain = -material.poisson_ratio / (1.0 - material.poisson_ratio) * vertical_strain
    network = nn.Linear(2, 2, dtype=DTYPE, device=device)
    with torch.no_grad():
        network.weight.zero_()
        network.weight[0, 0] = horizontal_strain * 2.0 / 0.1
        network.bias.zero_()
        network.bias[0] = 5.0
    trial = PlateDisplacementTrialSolution(
        network, half_width=2.0, half_height=1.0,
        imposed_displacement=0.01, displacement_scale=0.1,
    )
    coordinates = torch.tensor([[-2.0, 0.4], [2.0, -0.7], [0.0, 0.0]], dtype=DTYPE, device=device)
    expected = coordinates * coordinates.new_tensor([horizontal_strain, vertical_strain])
    torch.testing.assert_close(trial(coordinates), expected, rtol=1e-12, atol=1e-15)
    strain = small_strain(displacement_gradient(trial, coordinates))
    stress = undamaged_stress(strain, material)
    torch.testing.assert_close(stress[..., 0, :], torch.zeros_like(stress[..., 0, :]), rtol=0.0, atol=1e-15)
    expected_stress = material.young_modulus / (1.0 - material.poisson_ratio**2) * vertical_strain
    torch.testing.assert_close(stress[:, 1, 1], torch.full_like(stress[:, 1, 1], expected_stress))


@pytest.mark.parametrize("name,value,error", [
    ("half_width", 0.0, ValueError),
    ("half_height", -1.0, ValueError),
    ("displacement_scale", 0.0, ValueError),
    ("half_width", True, TypeError),
    ("half_height", float("inf"), ValueError),
    ("imposed_displacement", -0.01, ValueError),
    ("imposed_displacement", float("nan"), ValueError),
    ("imposed_displacement", float("inf"), ValueError),
    ("imposed_displacement", True, TypeError),
])
def test_plate_trial_rejects_invalid_configuration(name, value, error) -> None:
    configuration = dict(half_width=2.0, half_height=1.0, imposed_displacement=0.01, displacement_scale=0.1)
    configuration[name] = value
    with pytest.raises(error, match=name):
        PlateDisplacementTrialSolution(ConstantCoefficients(), **configuration)


@pytest.mark.parametrize("coordinates,error", [
    ([0.0, 0.0], TypeError),
    (torch.tensor(0.0), ValueError),
    (torch.zeros(2, 3), ValueError),
    (torch.zeros(2, 2, dtype=torch.int64), TypeError),
])
def test_plate_trial_rejects_invalid_coordinates(coordinates, error) -> None:
    trial = PlateDisplacementTrialSolution(
        ConstantCoefficients(), half_width=2.0, half_height=1.0,
        imposed_displacement=0.01, displacement_scale=0.1,
    )
    with pytest.raises(error, match="coordinates"):
        trial(coordinates)


@pytest.mark.parametrize("fault", ["shape", "origin_shape", "dtype", "device", "not_tensor"])
def test_plate_trial_rejects_incompatible_network_outputs(fault) -> None:
    class InvalidCoefficients(nn.Module):
        def forward(self, coordinates):
            if fault == "shape" or (fault == "origin_shape" and coordinates.ndim == 1):
                return coordinates[..., :1]
            if fault == "dtype":
                return coordinates.to(torch.float32)
            if fault == "device":
                return coordinates.to("meta")
            if fault == "not_tensor":
                return [0.0, 0.0]
            return coordinates

    trial = PlateDisplacementTrialSolution(
        InvalidCoefficients(), half_width=2.0, half_height=1.0,
        imposed_displacement=0.01, displacement_scale=0.1,
    )
    with pytest.raises(ValueError, match="network output"):
        trial(torch.zeros((3, 2), dtype=DTYPE))


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