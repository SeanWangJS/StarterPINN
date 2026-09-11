import pytest
import torch

from griffith2d.physics import plane_strain_decomposition, small_strain


DTYPE = torch.float64
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def test_small_strain_is_symmetric_part_of_displacement_gradient() -> None:
    gradient = torch.tensor([[2.0, 3.0], [-1.0, 4.0]], dtype=DTYPE)
    expected = torch.tensor([[2.0, 1.0], [1.0, 4.0]], dtype=DTYPE)

    torch.testing.assert_close(small_strain(gradient), expected, rtol=0.0, atol=0.0)


def test_infinitesimal_rigid_rotation_has_zero_small_strain() -> None:
    rotation_gradient = torch.tensor([[0.0, -0.25], [0.25, 0.0]], dtype=DTYPE)

    torch.testing.assert_close(
        small_strain(rotation_gradient),
        torch.zeros((2, 2), dtype=DTYPE),
        rtol=0.0,
        atol=0.0,
    )


def test_plane_strain_decomposition_includes_out_of_plane_component() -> None:
    strain = torch.tensor([[0.03, 0.02], [0.02, -0.01]], dtype=DTYPE)

    volumetric, deviatoric = plane_strain_decomposition(strain)

    torch.testing.assert_close(volumetric, torch.tensor(0.02, dtype=DTYPE))
    torch.testing.assert_close(deviatoric.trace(), torch.tensor(0.0, dtype=DTYPE), atol=1e-17, rtol=0.0)
    torch.testing.assert_close(deviatoric[2, 2], torch.tensor(-0.02 / 3.0, dtype=DTYPE))
    reconstructed = deviatoric + volumetric * torch.eye(3, dtype=DTYPE) / 3.0
    expected = torch.zeros((3, 3), dtype=DTYPE)
    expected[:2, :2] = strain
    torch.testing.assert_close(reconstructed, expected)


@pytest.mark.parametrize("device", DEVICES)
def test_kinematics_preserves_batch_shape_dtype_and_device(device: str) -> None:
    gradients = torch.arange(16, dtype=DTYPE, device=device).reshape(2, 2, 2, 2)

    strain = small_strain(gradients)
    volumetric, deviatoric = plane_strain_decomposition(strain)

    assert strain.shape == (2, 2, 2, 2)
    assert volumetric.shape == (2, 2)
    assert deviatoric.shape == (2, 2, 3, 3)
    assert deviatoric.dtype == gradients.dtype
    assert deviatoric.device == gradients.device


@pytest.mark.parametrize(
    ("value", "exception"),
    [
        (torch.zeros(2, dtype=DTYPE), ValueError),
        (torch.zeros((2, 3), dtype=DTYPE), ValueError),
        (torch.zeros((2, 2), dtype=torch.int64), TypeError),
        ([[0.0, 0.0], [0.0, 0.0]], TypeError),
    ],
)
def test_invalid_kinematic_input_is_rejected(value: object, exception: type[Exception]) -> None:
    with pytest.raises(exception):
        small_strain(value)  # type: ignore[arg-type]