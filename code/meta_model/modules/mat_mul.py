"""Migrated from SRC/src/glad/modules/mat_mul.py.

Elementwise product module: computes the product of two channels per batch element."""

import torch
from torch.nn import Module


class MatMul(Module):
    """Layer that computes the elementwise product of the first two channels.

    Expects input of shape ``(batch, 2)`` or ``(batch, 2, ...)`` and returns
    ``x[..., 0] * x[..., 1]`` with shape ``(batch,)`` or ``(batch, ...)``.
    """

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return A @ B
