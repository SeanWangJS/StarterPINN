import math

import pytest

from griffith2d.physics import IsotropicMaterial


def test_isotropic_moduli_match_analytic_values() -> None:
    material = IsotropicMaterial(young_modulus=210.0, poisson_ratio=0.3)

    assert material.bulk_modulus == pytest.approx(175.0)
    assert material.shear_modulus == pytest.approx(210.0 / 2.6)


@pytest.mark.parametrize("young_modulus", [0.0, -1.0, math.inf, math.nan])
def test_invalid_young_modulus_is_rejected(young_modulus: float) -> None:
    with pytest.raises(ValueError, match="young_modulus"):
        IsotropicMaterial(young_modulus, 0.3)


@pytest.mark.parametrize("poisson_ratio", [-1.0, 0.5, -math.inf, math.inf, math.nan])
def test_invalid_poisson_ratio_is_rejected(poisson_ratio: float) -> None:
    with pytest.raises(ValueError, match="poisson_ratio"):
        IsotropicMaterial(210.0, poisson_ratio)


@pytest.mark.parametrize(
    ("young_modulus", "poisson_ratio"),
    [("210", 0.3), (210.0, "0.3"), (True, 0.3), (210.0, False)],
)
def test_non_real_material_parameters_are_rejected(
    young_modulus: object, poisson_ratio: object
) -> None:
    with pytest.raises(TypeError):
        IsotropicMaterial(young_modulus, poisson_ratio)  # type: ignore[arg-type]
