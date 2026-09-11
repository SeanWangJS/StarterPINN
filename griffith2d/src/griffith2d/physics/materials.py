"""Material parameters for isotropic small-strain elasticity.

The definitions follow section 3.1 of the project theory document. This stage
supports three-dimensional isotropic constants used under plane strain.
"""

from dataclasses import dataclass
import math
from numbers import Real


@dataclass(frozen=True, slots=True)
class IsotropicMaterial:
    """Isotropic elastic constants defined by Young's modulus and Poisson ratio.

    Parameters are dimensional scalars expressed in one consistent unit system.
    The admissible range ``-1 < poisson_ratio < 0.5`` guarantees positive bulk
    and shear moduli for three-dimensional isotropic elasticity.
    """

    young_modulus: float
    poisson_ratio: float

    def __post_init__(self) -> None:
        if isinstance(self.young_modulus, bool) or not isinstance(self.young_modulus, Real):
            raise TypeError("young_modulus must be a real scalar")
        if isinstance(self.poisson_ratio, bool) or not isinstance(self.poisson_ratio, Real):
            raise TypeError("poisson_ratio must be a real scalar")

        young_modulus = float(self.young_modulus)
        poisson_ratio = float(self.poisson_ratio)

        if not math.isfinite(young_modulus) or young_modulus <= 0.0:
            raise ValueError("young_modulus must be finite and positive")
        if not math.isfinite(poisson_ratio) or not -1.0 < poisson_ratio < 0.5:
            raise ValueError("poisson_ratio must be finite and satisfy -1 < nu < 0.5")

        object.__setattr__(self, "young_modulus", young_modulus)
        object.__setattr__(self, "poisson_ratio", poisson_ratio)

    @property
    def bulk_modulus(self) -> float:
        """Return ``K = E / (3 (1 - 2 nu))`` in the units of ``E``."""

        return self.young_modulus / (3.0 * (1.0 - 2.0 * self.poisson_ratio))

    @property
    def shear_modulus(self) -> float:
        """Return ``mu = E / (2 (1 + nu))`` in the units of ``E``."""

        return self.young_modulus / (2.0 * (1.0 + self.poisson_ratio))
