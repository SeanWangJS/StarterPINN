import torch
import pytest

from griffith2d.physics import (
    IsotropicMaterial,
    amor_energy_split,
    degraded_elastic_energy,
    degraded_stress,
    quadratic_degradation,
    undamaged_elastic_energy,
)


DTYPE = torch.float64
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
MATERIAL = IsotropicMaterial(young_modulus=210.0, poisson_ratio=0.3)
RESIDUAL_STIFFNESS = 1e-6


def test_zero_strain_has_zero_energy_and_stress() -> None:
    strain = torch.zeros((2, 2), dtype=DTYPE)
    phase_field = torch.tensor(0.4, dtype=DTYPE)

    energy = degraded_elastic_energy(strain, phase_field, MATERIAL, RESIDUAL_STIFFNESS)
    stress = degraded_stress(strain, phase_field, MATERIAL, RESIDUAL_STIFFNESS)

    torch.testing.assert_close(energy, torch.tensor(0.0, dtype=DTYPE), rtol=0.0, atol=0.0)
    torch.testing.assert_close(stress, torch.zeros((2, 2), dtype=DTYPE), rtol=0.0, atol=0.0)


def test_amor_split_reconstructs_undamaged_energy() -> None:
    strains = torch.tensor(
        [
            [[0.02, 0.00], [0.00, 0.01]],
            [[-0.02, 0.00], [0.00, -0.01]],
            [[0.01, 0.03], [0.03, -0.01]],
        ],
        dtype=DTYPE,
    )

    split = amor_energy_split(strains, MATERIAL)

    torch.testing.assert_close(split.active + split.inactive, undamaged_elastic_energy(strains, MATERIAL))
    assert torch.all(split.active >= 0.0)
    assert torch.all(split.inactive >= 0.0)


def test_pure_shear_energy_and_stress_match_analytic_values() -> None:
    engineering_shear = 0.04
    strain = torch.tensor(
        [[0.0, engineering_shear / 2.0], [engineering_shear / 2.0, 0.0]],
        dtype=DTYPE,
    )
    phase_field = torch.tensor(0.0, dtype=DTYPE)
    degradation = 1.0 + RESIDUAL_STIFFNESS

    energy = degraded_elastic_energy(strain, phase_field, MATERIAL, RESIDUAL_STIFFNESS)
    stress = degraded_stress(strain, phase_field, MATERIAL, RESIDUAL_STIFFNESS)

    expected_energy = degradation * 0.5 * MATERIAL.shear_modulus * engineering_shear**2
    expected_shear_stress = degradation * MATERIAL.shear_modulus * engineering_shear
    torch.testing.assert_close(energy, torch.tensor(expected_energy, dtype=DTYPE))
    torch.testing.assert_close(stress[0, 1], torch.tensor(expected_shear_stress, dtype=DTYPE))
    torch.testing.assert_close(stress[1, 0], stress[0, 1])


def test_inactive_compressive_energy_is_not_degraded() -> None:
    strain = torch.tensor([[-0.02, 0.0], [0.0, -0.02]], dtype=DTYPE)
    intact = torch.tensor(0.0, dtype=DTYPE)
    broken = torch.tensor(1.0, dtype=DTYPE)
    split = amor_energy_split(strain, MATERIAL)

    intact_energy = degraded_elastic_energy(strain, intact, MATERIAL, RESIDUAL_STIFFNESS)
    broken_energy = degraded_elastic_energy(strain, broken, MATERIAL, RESIDUAL_STIFFNESS)

    expected_difference = (
        quadratic_degradation(intact, RESIDUAL_STIFFNESS)
        - quadratic_degradation(broken, RESIDUAL_STIFFNESS)
    ) * split.active
    torch.testing.assert_close(intact_energy - broken_energy, expected_difference)
    assert split.inactive > 0.0


def test_uniaxial_plane_strain_state_matches_component_formula() -> None:
    axial_strain = 0.02
    strain = torch.tensor([[axial_strain, 0.0], [0.0, 0.0]], dtype=DTYPE)
    phase_field = torch.tensor(0.25, dtype=DTYPE)
    degradation = (1.0 - phase_field).square() + RESIDUAL_STIFFNESS
    expected_xx = degradation * (
        MATERIAL.bulk_modulus * axial_strain
        + 2.0 * MATERIAL.shear_modulus * (2.0 * axial_strain / 3.0)
    )
    expected_yy = degradation * (
        MATERIAL.bulk_modulus * axial_strain
        - 2.0 * MATERIAL.shear_modulus * (axial_strain / 3.0)
    )

    stress = degraded_stress(strain, phase_field, MATERIAL, RESIDUAL_STIFFNESS)

    torch.testing.assert_close(stress[0, 0], expected_xx)
    torch.testing.assert_close(stress[1, 1], expected_yy)
    torch.testing.assert_close(stress[0, 1], torch.tensor(0.0, dtype=DTYPE))


def test_analytic_stress_matches_energy_autograd() -> None:
    strain = torch.tensor([[0.031, 0.017], [0.017, -0.009]], dtype=DTYPE, requires_grad=True)
    phase_field = torch.tensor(0.37, dtype=DTYPE)

    energy = degraded_elastic_energy(strain, phase_field, MATERIAL, RESIDUAL_STIFFNESS)
    (autograd_stress,) = torch.autograd.grad(energy, strain)
    analytic_stress = degraded_stress(strain.detach(), phase_field, MATERIAL, RESIDUAL_STIFFNESS)

    torch.testing.assert_close(analytic_stress, autograd_stress, rtol=1e-12, atol=1e-12)


def test_analytic_stress_matches_symmetric_finite_differences() -> None:
    strain = torch.tensor([[0.031, 0.017], [0.017, -0.009]], dtype=DTYPE)
    phase_field = torch.tensor(0.37, dtype=DTYPE)
    analytic_stress = degraded_stress(strain, phase_field, MATERIAL, RESIDUAL_STIFFNESS)
    step = 1e-6
    directions = [
        torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=DTYPE),
        torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=DTYPE),
        torch.tensor([[0.0, 0.5], [0.5, 0.0]], dtype=DTYPE),
    ]
    components = [(0, 0), (1, 1), (0, 1)]

    for direction, component in zip(directions, components, strict=True):
        upper = degraded_elastic_energy(
            strain + step * direction, phase_field, MATERIAL, RESIDUAL_STIFFNESS
        )
        lower = degraded_elastic_energy(
            strain - step * direction, phase_field, MATERIAL, RESIDUAL_STIFFNESS
        )
        finite_difference = (upper - lower) / (2.0 * step)
        torch.testing.assert_close(
            finite_difference,
            analytic_stress[component],
            rtol=1e-9,
            atol=1e-9,
        )


@pytest.mark.parametrize("device", DEVICES)
def test_elasticity_preserves_batch_shape_dtype_and_device(device: str) -> None:
    strain = torch.tensor(
        [
            [[0.02, 0.01], [0.01, -0.01]],
            [[-0.01, 0.02], [0.02, 0.03]],
        ],
        dtype=torch.float32,
        device=device,
    )
    phase_field = torch.tensor([0.2, 0.7], dtype=torch.float32, device=device)

    energy = degraded_elastic_energy(strain, phase_field, MATERIAL, RESIDUAL_STIFFNESS)
    stress = degraded_stress(strain, phase_field, MATERIAL, RESIDUAL_STIFFNESS)

    assert energy.shape == (2,)
    assert stress.shape == (2, 2, 2)
    assert energy.dtype == strain.dtype
    assert stress.dtype == strain.dtype
    assert stress.device == strain.device


def test_phase_field_scalar_broadcasts_over_strain_batch() -> None:
    strain = torch.tensor(
        [
            [[0.02, 0.01], [0.01, -0.01]],
            [[-0.01, 0.02], [0.02, 0.03]],
        ],
        dtype=DTYPE,
    )
    phase_field = torch.tensor(0.2, dtype=DTYPE)

    energy = degraded_elastic_energy(strain, phase_field, MATERIAL, RESIDUAL_STIFFNESS)

    assert energy.shape == (2,)


@pytest.mark.parametrize(
    "phase_field",
    [
        torch.zeros((2, 1), dtype=DTYPE),
        torch.tensor(0.2, dtype=torch.float32),
        torch.tensor(1, dtype=torch.int64),
    ],
)
def test_invalid_phase_field_contract_is_rejected(phase_field: torch.Tensor) -> None:
    strain = torch.zeros((2, 2), dtype=DTYPE)

    with pytest.raises((TypeError, ValueError)):
        degraded_elastic_energy(strain, phase_field, MATERIAL, RESIDUAL_STIFFNESS)