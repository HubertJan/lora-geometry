"""Migrated from SRC/src/glad/modules/activations.py.

GL-congruence equivariant activation functions.

Migrated from paretune (equivariant_classifiers/components.py).
"""

from __future__ import annotations

import torch


def safe_pinv_hermitian(
    A: torch.Tensor,  # noqa: N803
    base_eps: float = 1e-6,
    tries: int = 4,
) -> torch.Tensor:
    """Robust pseudoinverse for Hermitian (symmetric) matrices with jitter fallback."""
    # ensure exact symmetry
    A = 0.5 * (A + A.transpose(-2, -1))
    I = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)  # noqa: E741
    eps = base_eps
    for _ in range(tries):
        try:
            return torch.linalg.pinv(A + eps * I, hermitian=True)
        except RuntimeError:
            eps *= 10  # increase jitter and retry
    # last resort: fall back to SVD pinv
    return torch.linalg.pinv(A)


def gl_activation(
    u: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GL-congruence equivariant activation using pseudoinverse gating."""
    vv = v.transpose(-2, -1) @ v  # (..., d2, d2), PSD
    uu = u.transpose(-2, -1) @ u  # (..., d2, d2), PSD

    # Use pseudoinverse (GL-congruence covariant)
    vv_inv = safe_pinv_hermitian(vv)
    uu_inv = safe_pinv_hermitian(uu)

    m_uv = u @ vv_inv @ v.transpose(-2, -1)  # (..., d1, d3), invariant
    m_vu = v @ uu_inv @ u.transpose(-2, -1)  # (..., d3, d1), invariant

    gate_u = m_uv.abs().mean(dim=-1, keepdim=True)  # (..., d1, 1)
    gate_v = m_vu.abs().mean(dim=-1, keepdim=True)  # (..., d3, 1)
    gate_u = gate_u / (gate_u.mean(dim=-2, keepdim=True) + 1e-12)
    gate_v = gate_v / (gate_v.mean(dim=-2, keepdim=True) + 1e-12)
    gate_u = torch.sigmoid(gate_u)
    gate_v = torch.sigmoid(gate_v)

    return u * gate_u, v * gate_v
