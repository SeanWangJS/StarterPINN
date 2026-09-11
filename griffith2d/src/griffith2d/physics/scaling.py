"""Explicit dimensional scaling for the frozen history field."""

from torch import Tensor

from .phase_field import PhaseFieldMaterial, _validate_field


def nondimensional_history(history_field: Tensor, material: PhaseFieldMaterial) -> Tensor:
	"""Return ``H_bar = H l0 / Gc``."""

	_validate_field(history_field, "history_field")
	return history_field * material.length_scale / material.fracture_toughness


def dimensional_history(nondimensional_field: Tensor, material: PhaseFieldMaterial) -> Tensor:
	"""Return ``H = H_bar Gc / l0``."""

	_validate_field(nondimensional_field, "nondimensional_field")
	return nondimensional_field * material.fracture_toughness / material.length_scale