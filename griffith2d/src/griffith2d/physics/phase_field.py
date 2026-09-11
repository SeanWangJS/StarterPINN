"""AT2 phase-field energy for a frozen history field."""

from dataclasses import dataclass
import math
from numbers import Real

from torch import Tensor


@dataclass(frozen=True, slots=True)
class PhaseFieldMaterial:
	"""Fracture toughness and regularization length in consistent units."""

	fracture_toughness: float
	length_scale: float

	def __post_init__(self) -> None:
		for name, value in (
			("fracture_toughness", self.fracture_toughness),
			("length_scale", self.length_scale),
		):
			if isinstance(value, bool) or not isinstance(value, Real):
				raise TypeError(f"{name} must be a real scalar")
			value = float(value)
			if not math.isfinite(value) or value <= 0.0:
				raise ValueError(f"{name} must be finite and positive")
			object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class PhaseFieldEnergyDensity:
	"""Pointwise components of the frozen-history AT2 energy density."""

	history: Tensor
	local_fracture: Tensor
	gradient_fracture: Tensor

	@property
	def total(self) -> Tensor:
		return self.history + self.local_fracture + self.gradient_fracture


def _validate_field(field: Tensor, name: str) -> None:
	if not isinstance(field, Tensor):
		raise TypeError(f"{name} must be a torch.Tensor")
	if not field.is_floating_point():
		raise TypeError(f"{name} must have a floating-point dtype")


def at2_crack_density(phase_field: Tensor, phase_gradient: Tensor, length_scale: float) -> Tensor:
	"""Return ``phi^2 / (2 l0) + l0 |grad phi|^2 / 2`` pointwise."""

	_validate_field(phase_field, "phase_field")
	_validate_field(phase_gradient, "phase_gradient")
	material = PhaseFieldMaterial(fracture_toughness=1.0, length_scale=length_scale)
	if phase_field.device != phase_gradient.device or phase_field.dtype != phase_gradient.dtype:
		raise ValueError("phase_field and phase_gradient must share dtype and device")
	if phase_field.ndim == 0 or phase_field.shape[-1] != 1:
		raise ValueError("phase_field must have shape (..., 1)")
	if phase_gradient.ndim != phase_field.ndim or phase_gradient.shape[:-1] != phase_field.shape[:-1]:
		raise ValueError("phase_gradient must have shape (..., spatial_dimension)")

	gradient_norm_squared = phase_gradient.square().sum(dim=-1, keepdim=True)
	return phase_field.square() / (2.0 * material.length_scale) + (
		0.5 * material.length_scale * gradient_norm_squared
	)


def frozen_history_energy_density(
	phase_field: Tensor,
	phase_gradient: Tensor,
	history_field: Tensor,
	material: PhaseFieldMaterial,
) -> PhaseFieldEnergyDensity:
	"""Return the pointwise energy components for a prescribed history field."""

	_validate_field(history_field, "history_field")
	if history_field.shape != phase_field.shape:
		raise ValueError("history_field must have the same shape as phase_field")
	if history_field.device != phase_field.device or history_field.dtype != phase_field.dtype:
		raise ValueError("history_field and phase_field must share dtype and device")

	crack_density = at2_crack_density(phase_field, phase_gradient, material.length_scale)
	local_fracture = material.fracture_toughness * phase_field.square() / (2.0 * material.length_scale)
	gradient_fracture = material.fracture_toughness * (
		crack_density - phase_field.square() / (2.0 * material.length_scale)
	)
	return PhaseFieldEnergyDensity(
		history=(1.0 - phase_field).square() * history_field,
		local_fracture=local_fracture,
		gradient_fracture=gradient_fracture,
	)