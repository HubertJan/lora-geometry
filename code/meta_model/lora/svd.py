"""Migrated from SRC/src/glad/lora/svd.py.

SVD-balanced canonical form of LoRA adapters.

A LoRA update ``ΔW = B·A`` is invariant under ``B → B·M``, ``A → M⁻¹·A``, so the raw
``(A, B)`` pair carries an arbitrary gauge.  The *balanced* form fixes that gauge:

    ΔW = U·Σ·Vᵀ    →    B' = U·Σ^½ ,  A' = Σ^½·Vᵀ

so ``B'·A' = ΔW`` exactly, the two factors share the singular-value scale evenly, and
rank index ``k`` is the ``k``-th singular direction (descending σ).  That makes per-rank
slicing meaningful — ``(B'[:, k], A'[k])`` is a genuine rank-1 component of ΔW rather
than one arbitrary column of an arbitrary factorisation.

Both singular vectors are computed **without forming the ``d_out × d_in`` matrix**: one
SVD of ``B`` (``d_out × r``) and one of an ``r × d_in`` product, so the cost is linear in
the big dimensions rather than quadratic.

Promoted verbatim (numerics unchanged) from the five identical inline copies in
``lrp_on_meta_classifiers``, ``task_meta_classifier_lrp``, ``component_sufficiency_sweep``,
``meta_classifier_robustness`` and ``per_cell_xtask_stability``.
"""

from __future__ import annotations

import numpy as np
import torch

from meta_model.lora.types import LoraType
from meta_model.lora.weight_utils import LoraWeight


def svd_balance_layer(
    A: torch.Tensor,
    B: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Balanced canonical form of one layer's ``(A: (r, d_in), B: (d_out, r))``.

    Returns ``(A', B', s)`` with ``B'·A' = B·A``, ``B' = U·diag(s^½)``,
    ``A' = diag(s^½)·Vᵀ``, ranks ordered by descending singular value σ.  ``s`` holds
    the singular values of ΔW itself (not of ``A`` or ``B`` separately).

    Computed in float32 regardless of the input dtype — ``torch.linalg.svd`` is not
    reliable in bf16/fp16, which is how adapters are usually stored.
    """
    A = A.float()
    B = B.float()
    Ub, sb, Vbt = torch.linalg.svd(B, full_matrices=False)  # B = Ub Σb Vbᵀ
    M = (sb.unsqueeze(1) * Vbt) @ A                         # (r, d_in)
    Um, sm, Vmt = torch.linalg.svd(M, full_matrices=False)
    U = Ub @ Um                                             # (d_out, r)
    s = sm                                                  # singular values of ΔW
    Vt = Vmt                                                # (r, d_in)
    sqrt_s = torch.sqrt(torch.clamp(s, min=0.0))
    B_new = U * sqrt_s.unsqueeze(0)                         # (d_out, r)
    A_new = sqrt_s.unsqueeze(1) * Vt                        # (r, d_in)
    return A_new, B_new, s


def svd_balance_grouped(
    grouped: dict[str, LoraWeight],
) -> tuple[dict[str, LoraWeight], dict[str, np.ndarray], float]:
    """Re-parametrise every ``(module, layer)`` of a grouped LoRA dict to balanced form.

    Args:
        grouped: ``{submodule_key: {LoraType.A: (n_layers, r, d_in),
                                    LoraType.B: (n_layers, d_out, r)}}``,
            as produced by :func:`meta_model.lora.weight_utils.group_lora_weights_per_submodule`.

    Returns:
        ``(grouped_balanced, sing_values, max_recon_err)`` where ``sing_values`` is
        ``{submodule_key: ndarray(n_layers, rank)}`` of ΔW singular values, and
        ``max_recon_err`` is the worst ``|B'A' − BA|_max`` over all cells.

    Balancing is loss-free, so ``max_recon_err`` must come out at float32 round-off
    (~1e-5 or below).  Callers should assert on it — a large value means the input was
    not a well-formed ``(A, B)`` pair (e.g. mismatched layer counts, or zero-filled
    layers from a wrong ``layer_id_order``).
    """
    out: dict[str, LoraWeight] = {}
    sing: dict[str, np.ndarray] = {}
    max_err = 0.0
    for mk, sub in grouped.items():
        A_full = sub[LoraType.A].float()  # (n_layers, r, d_in)
        B_full = sub[LoraType.B].float()  # (n_layers, d_out, r)
        n_l = A_full.shape[0]
        A_bal = torch.empty_like(A_full)
        B_bal = torch.empty_like(B_full)
        s_layers = np.zeros((n_l, A_full.shape[1]), dtype=float)
        for layer in range(n_l):
            A_new, B_new, s = svd_balance_layer(A_full[layer], B_full[layer])
            err = (B_new @ A_new - B_full[layer] @ A_full[layer]).abs().max().item()
            max_err = max(max_err, err)
            A_bal[layer] = A_new
            B_bal[layer] = B_new
            s_layers[layer] = s.detach().cpu().numpy()
        out[mk] = {LoraType.A: A_bal, LoraType.B: B_bal}
        sing[mk] = s_layers
    return out, sing, max_err
