import math

import pytest
import torch

from griffith2d.discretization import CenterCrackPlate
from griffith2d.physics import at2_crack_density


PLATE = CenterCrackPlate(half_width=2.0, half_height=1.0, crack_half_length=0.5)
DTYPE = torch.float64
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_finite_crack_distance_and_phase_preserve_shape_and_device(device: str) -> None:
    points = torch.tensor(
        [[0.0, 0.0], [0.5, 0.0], [0.0, 0.2], [0.8, 0.0], [0.8, 0.4]],
        dtype=DTYPE,
        device=device,
    )
    distance = PLATE.crack_distance(points)
    phase = PLATE.initial_phase(points, length_scale=0.2)

    torch.testing.assert_close(
        distance[:, 0],
        torch.tensor([0.0, 0.0, 0.2, 0.3, 0.5], dtype=DTYPE, device=device),
    )
    torch.testing.assert_close(phase, torch.exp(-distance / 0.2))
    assert phase.shape == (5, 1)
    assert phase.device.type == device
    assert phase.dtype == DTYPE


def test_center_crack_partition_and_exterior_edges_integrate_physical_measures() -> None:
    domain = PLATE.domain_rule(band_half_width=0.1, order=3)
    boundaries = PLATE.boundary_rules(order=3)
    band = (domain.points[:, 1].abs() < 0.1).to(DTYPE)

    assert domain.points.shape == (108, 2)
    torch.testing.assert_close(domain.weights.sum(), torch.tensor(8.0, dtype=DTYPE))
    torch.testing.assert_close(domain.integrate(band), torch.tensor(0.8, dtype=DTYPE))
    for rule, expected in (
        (boundaries.top, 4.0),
        (boundaries.bottom, 4.0),
        (boundaries.left, 2.0),
        (boundaries.right, 2.0),
    ):
        torch.testing.assert_close(rule.weights.sum(), torch.tensor(expected, dtype=DTYPE))
    assert torch.all(boundaries.top.points[:, 1] == 1.0)
    assert torch.all(boundaries.bottom.points[:, 1] == -1.0)


def test_crack_interior_at2_energy_converges_to_exact_transverse_profile() -> None:
    length_scale = 0.1
    exact_energy = 2.0 * PLATE.crack_half_length * (1.0 - math.exp(-2.0 / length_scale))
    errors = []
    for order in (2, 4, 8, 16):
        domain = PLATE.domain_rule(band_half_width=0.2, order=order)
        phase = PLATE.initial_phase(domain.points, length_scale)
        gradient = torch.stack(
            (torch.zeros_like(phase[:, 0]),
             -domain.points[:, 1].sign() * phase[:, 0] / length_scale),
            dim=-1,
        )
        density = at2_crack_density(phase, gradient, length_scale)[:, 0]
        interior = (domain.points[:, 0].abs() < PLATE.crack_half_length).to(DTYPE)
        errors.append(abs(domain.integrate(interior * density).item() - exact_energy))

    assert all(later < earlier for earlier, later in zip(errors, errors[1:]))
    assert errors[-1] < 1e-8


@pytest.mark.parametrize("device", DEVICES)
def test_partitioned_domain_preserves_dtype_and_device(device: str) -> None:
    domain = PLATE.domain_rule(0.1, order=2, device=device)
    assert domain.points.dtype == DTYPE
    assert domain.points.device.type == device
    assert domain.weights.device.type == device


def test_invalid_crack_geometry_and_band_are_rejected() -> None:
    for half_length in (0.0, 2.0, math.inf):
        with pytest.raises(ValueError):
            CenterCrackPlate(2.0, 1.0, half_length)
    for band in (0.0, 1.0, math.nan):
        with pytest.raises(ValueError):
            PLATE.domain_rule(band, order=2)