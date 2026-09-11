import pytest
import torch

from griffith2d.physics import (
    PhaseFieldMaterial,
    at2_crack_density,
    dimensional_history,
    frozen_history_energy_density,
    nondimensional_history,
    update_history,
)


DTYPE = torch.float64
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def test_at2_crack_density_matches_hand_calculation() -> None:
    phase_field = torch.tensor([[0.0], [0.5]], dtype=DTYPE)
    phase_gradient = torch.tensor([[2.0, 0.0], [3.0, 4.0]], dtype=DTYPE)

    result = at2_crack_density(phase_field, phase_gradient, length_scale=0.2)

    expected = phase_field.square() / 0.4 + 0.1 * phase_gradient.square().sum(dim=-1, keepdim=True)
    torch.testing.assert_close(result, expected)


def test_frozen_history_energy_exposes_each_component() -> None:
    material = PhaseFieldMaterial(fracture_toughness=2.0, length_scale=0.5)
    phase_field = torch.tensor([[0.25]], dtype=DTYPE, requires_grad=True)
    phase_gradient = torch.tensor([[3.0]], dtype=DTYPE)
    history = torch.tensor([[4.0]], dtype=DTYPE)

    energy = frozen_history_energy_density(phase_field, phase_gradient, history, material)

    torch.testing.assert_close(energy.history, torch.tensor([[2.25]], dtype=DTYPE))
    torch.testing.assert_close(energy.local_fracture, torch.tensor([[0.125]], dtype=DTYPE))
    torch.testing.assert_close(energy.gradient_fracture, torch.tensor([[4.5]], dtype=DTYPE))
    derivative = torch.autograd.grad(energy.total.sum(), phase_field)[0]
    torch.testing.assert_close(derivative, torch.tensor([[-5.0]], dtype=DTYPE))


def test_history_update_is_pointwise_and_does_not_modify_inputs() -> None:
    previous = torch.tensor([[1.0], [3.0]], dtype=DTYPE)
    active = torch.tensor([[2.0], [1.0]], dtype=DTYPE)
    previous_before = previous.clone()
    active_before = active.clone()

    updated = update_history(previous, active)

    torch.testing.assert_close(updated, torch.tensor([[2.0], [3.0]], dtype=DTYPE))
    torch.testing.assert_close(previous, previous_before)
    torch.testing.assert_close(active, active_before)
    assert updated.data_ptr() not in {previous.data_ptr(), active.data_ptr()}


def test_history_scaling_round_trip() -> None:
    material = PhaseFieldMaterial(fracture_toughness=2.7, length_scale=0.15)
    history = torch.tensor([[0.0], [1.2], [3.4]], dtype=DTYPE)

    scaled = nondimensional_history(history, material)

    torch.testing.assert_close(scaled, history * 0.15 / 2.7)
    torch.testing.assert_close(dimensional_history(scaled, material), history)


@pytest.mark.parametrize("name", ["fracture_toughness", "length_scale"])
@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_phase_field_material_is_rejected(name: str, value: float) -> None:
    parameters = {"fracture_toughness": 1.0, "length_scale": 0.1, name: value}

    with pytest.raises(ValueError, match=name):
        PhaseFieldMaterial(**parameters)


def test_phase_field_energy_does_not_clamp_out_of_range_values() -> None:
    phase_field = torch.tensor([[-0.1], [1.1]], dtype=DTYPE)
    gradient = torch.zeros_like(phase_field)

    result = at2_crack_density(phase_field, gradient, length_scale=0.2)

    torch.testing.assert_close(result, phase_field.square() / 0.4)


@pytest.mark.parametrize("device", DEVICES)
def test_phase_field_physics_preserves_dtype_and_device(device: str) -> None:
    material = PhaseFieldMaterial(fracture_toughness=2.0, length_scale=0.5)
    phase = torch.tensor([[0.25]], dtype=DTYPE, device=device)
    gradient = torch.tensor([[2.0]], dtype=DTYPE, device=device)
    history = torch.tensor([[3.0]], dtype=DTYPE, device=device)

    energy = frozen_history_energy_density(phase, gradient, history, material)
    updated = update_history(history, torch.ones_like(history))

    assert energy.total.dtype == DTYPE
    assert energy.total.device.type == device
    assert updated.dtype == DTYPE
    assert updated.device.type == device