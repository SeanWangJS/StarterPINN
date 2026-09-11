"""Fixed baseline neural representation for the stage-002 DEM benchmark."""

import torch
from torch import Tensor, nn


class DisplacementMLP(nn.Module):
    """Two-hidden-layer ``tanh`` MLP mapping 2D coordinates to 2D coefficients."""

    def __init__(
        self,
        hidden_width: int = 16,
        *,
        seed: int = 1729,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if isinstance(hidden_width, bool) or not isinstance(hidden_width, int):
            raise TypeError("hidden_width must be an integer")
        if hidden_width < 1:
            raise ValueError("hidden_width must be positive")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        if dtype not in (torch.float32, torch.float64):
            raise TypeError("dtype must be torch.float32 or torch.float64")

        resolved_device = torch.device("cpu" if device is None else device)
        self.layers = nn.Sequential(
            nn.Linear(2, hidden_width, dtype=dtype, device=resolved_device),
            nn.Tanh(),
            nn.Linear(hidden_width, hidden_width, dtype=dtype, device=resolved_device),
            nn.Tanh(),
            nn.Linear(hidden_width, 2, dtype=dtype, device=resolved_device),
        )
        generator = torch.Generator(device=resolved_device).manual_seed(seed)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, generator=generator)
                nn.init.zeros_(module.bias)

    def forward(self, coordinates: Tensor) -> Tensor:
        """Return unconstrained coefficients for coordinates of shape ``(..., 2)``."""

        if coordinates.ndim < 1 or coordinates.shape[-1] != 2:
            raise ValueError("coordinates must have shape (..., 2)")
        return self.layers(coordinates)


class PhaseFieldMLP(nn.Module):
    """Two-hidden-layer ``tanh`` MLP mapping a 1D coordinate to a scalar."""

    def __init__(
        self,
        hidden_width: int = 16,
        *,
        seed: int = 1729,
        zero_initialization: bool = False,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if isinstance(hidden_width, bool) or not isinstance(hidden_width, int):
            raise TypeError("hidden_width must be an integer")
        if hidden_width < 1:
            raise ValueError("hidden_width must be positive")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        if not isinstance(zero_initialization, bool):
            raise TypeError("zero_initialization must be a bool")
        if dtype not in (torch.float32, torch.float64):
            raise TypeError("dtype must be torch.float32 or torch.float64")

        resolved_device = torch.device("cpu" if device is None else device)
        self.layers = nn.Sequential(
            nn.Linear(1, hidden_width, dtype=dtype, device=resolved_device),
            nn.Tanh(),
            nn.Linear(hidden_width, hidden_width, dtype=dtype, device=resolved_device),
            nn.Tanh(),
            nn.Linear(hidden_width, 1, dtype=dtype, device=resolved_device),
        )
        generator = torch.Generator(device=resolved_device).manual_seed(seed)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if zero_initialization:
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.xavier_uniform_(module.weight, generator=generator)
                nn.init.zeros_(module.bias)

    def forward(self, coordinates: Tensor) -> Tensor:
        if coordinates.ndim < 1 or coordinates.shape[-1] != 1:
            raise ValueError("coordinates must have shape (..., 1)")
        return self.layers(coordinates)
