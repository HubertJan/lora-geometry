"""Migrated from SRC/src/glad/modules/equivariant_linear_layer.py."""

import math

import torch
from torch import nn


class EquivariantLinearLayer(nn.Module):
    def __init__(
        self,
        size_in: int,
        size_out: int,
        weights: torch.Tensor | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        if weights is None:
            weights = torch.Tensor(size_out, size_in).to(device)
            nn.init.kaiming_uniform_(weights, a=math.sqrt(5))
        else:
            weights = weights.to(device)
        assert weights.shape == (size_out, size_in)
        self.weights = nn.Parameter(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.weights @ x
        # Reshape from [batch, size_out, ...] to [batch, -1] for compatibility with LRP
        return result
