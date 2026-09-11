"""Pure history-field updates for the phase-field proxy problem."""

import torch
from torch import Tensor


def update_history(previous_history: Tensor, active_energy: Tensor) -> Tensor:
	"""Return the pointwise maximum without modifying either input tensor."""

	for field, name in ((previous_history, "previous_history"), (active_energy, "active_energy")):
		if not isinstance(field, Tensor):
			raise TypeError(f"{name} must be a torch.Tensor")
		if not field.is_floating_point():
			raise TypeError(f"{name} must have a floating-point dtype")
	if previous_history.shape != active_energy.shape:
		raise ValueError("previous_history and active_energy must have the same shape")
	if previous_history.dtype != active_energy.dtype or previous_history.device != active_energy.device:
		raise ValueError("previous_history and active_energy must share dtype and device")
	return torch.maximum(previous_history, active_energy)