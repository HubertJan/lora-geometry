"""Per-cell (u, v) template extraction for the w8_l2 SST2 performance *regressor*.

(migrated from SRC/src/discoveries/sst2_perf_regressor_uv/flows/uv_extract.py)


The reference bilinear reduction (``glad.meta_classifier.bilinear``) is written for a
LINEAR single-cell *classifier*: score ``s = <T, dW> + b`` with ``T = W_b^T G W_a`` and
``G`` the (constant) class-difference head weight.  The w8_l2 regressor differs in two
ways that this module handles:

1. it is a *regressor* — one scalar head ``acc`` (``output_dim=1``), no class contrast;
2. it has a per-cell **L2 feature-normalisation** and a LeakyReLU MLP head, so the map
   ``dW -> acc-logit`` is nonlinear.  The template is therefore a **local first-order
   Jacobian**::

        T_cell(adapter) = d(acc_logit) / d(dW_cell)   evaluated at that adapter,

   an ``(out, in)`` matrix in the SAME space as ``dW = B @ A``.  Because the equivariant
   trunk is exactly linear per cell (``F_cell = W_b dW W_a^T``), this Jacobian equals
   ``W_b^T G_eff W_a`` with ``G_eff = d(logit)/dF_cell`` — the reference
   ``build_T_linearized`` pullback, except ``G_eff`` also carries the **L2-norm Jacobian**
   because the normalisation is kept inside the traced graph.

We obtain all 112 cell templates for one adapter in ONE backward pass by making every
cell's ``dW`` a leaf.  SVD(T_cell) -> (u, v, sigma) exactly as the reference.

All template math is float64 on CPU, matching every historical copy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

# canonical module order (module-major, layer-minor — matches the model's token stacking)
MODULE_ORDER = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]
RESIDUAL_WRITERS = frozenset({"self_attn.o_proj", "mlp.down_proj"})  # side "u" is residual


def load_model(ckpt_path: str, device: str = "cpu"):
    """Load a w8_l2 regressor checkpoint into the canonical FlexibleLoRAMetaClassifier."""
    from meta_model.model import FlexibleLoRAMetaClassifier

    return FlexibleLoRAMetaClassifier.load(ckpt_path, device=device).eval()


def group_from_path(path: str, num_layers: int = 16):
    """Load one adapter safetensors -> grouped {module_key: {LoraType: (L,·,·)}} dict."""
    from safetensors.torch import load_file

    from meta_model.lora.weight_utils import group_lora_weights_per_submodule

    return group_lora_weights_per_submodule(
        load_file(str(path)), layer_id_order=list(range(num_layers))
    )


def cell_order(model):
    """(module_key, module_type) pairs in the model's OWN token-stacking order.

    The head consumes cells in ``model.targeted_modules`` order (``sorted`` = by
    StrEnum value: down, gate, k, o, q, up, v), module-major / layer-minor.  Using
    this order — not a hand-written MODULE_ORDER — is required so the differentiable
    forward feeds the head the same vector the model does.
    """
    from meta_model.model import _MODULE_KEY

    return [(_MODULE_KEY[mt], mt) for mt in model.targeted_modules]


def module_weights(model):
    """Return {module_key: (W_a (d,d_in), W_b (d,d_out))} for the single equiv layer."""
    out = {}
    for key, mt in cell_order(model):
        W_a = model.module_a_layers[mt.value][0].weights.detach().double()
        W_b = model.module_b_layers[mt.value][0].weights.detach().double()
        out[key] = (W_a, W_b)
    return out


def delta_ws(grouped, order, num_layers: int = 16):
    """Return {module_key: [dW (out,in) per layer]} in float64, dW = B @ A."""
    from meta_model.lora.types import LoraType

    out = {}
    for key in order:
        sub = grouped[key]
        out[key] = [
            (sub[LoraType.B][layer].double() @ sub[LoraType.A][layer].double())
            for layer in range(num_layers)
        ]
    return out


def _acc_logit_from_dws(model, mw, dws, order, num_layers, acc_head="acc"):
    """Differentiable forward acc-logit as a function of per-cell dW leaves.

    Reproduces FlexibleLoRAMetaClassifier.forward for feature_norm='l2', head='mlp':
    F_cell = W_b dW W_a^T -> reshape(d*d) -> L2-normalise -> stack -> body -> head[acc].
    ``order`` MUST be the model's own cell order (see :func:`cell_order`).
    """
    head = model.head
    body = head.body.double()
    acc = head.output_heads[acc_head].double()

    tokens = []
    for key in order:
        W_a, W_b = mw[key]
        for layer in range(num_layers):
            dW = dws[key][layer]  # leaf (out,in) double
            F = W_b @ dW @ W_a.t()                 # (d,d)
            feat = F.reshape(-1)                    # (d*d,)
            feat = feat / (feat.norm() + 1e-8)      # per-cell L2 (nonlinearity)
            tokens.append(feat)
    stacked = torch.cat(tokens)                     # (n_cells*d*d,)
    logit = acc(body(stacked))                      # (1,)
    return logit.squeeze(), tokens


def per_adapter_templates(model, mw, grouped, num_layers=16, acc_head="acc"):
    """All 112 cell templates T_cell = d(acc_logit)/d(dW_cell) for one adapter.

    Returns dict with:
      T[key][layer]      : (out,in) float64 template (rank <= d)
      acc_logit          : float (model's own scalar acc logit, reconstruction anchor)
      order              : the model's cell order (keys)
    """
    order = [key for key, _ in cell_order(model)]
    dws = delta_ws(grouped, order, num_layers)
    for key in order:
        for layer in range(num_layers):
            dws[key][layer].requires_grad_(True)

    logit, _ = _acc_logit_from_dws(model, mw, dws, order, num_layers, acc_head)
    logit.backward()

    T = {key: [dws[key][layer].grad.detach().clone() for layer in range(num_layers)]
         for key in order}
    return {"T": T, "acc_logit": float(logit.detach()), "order": order}


# ── efficient rank-<=d extraction (the production path) ──────────────────────
#
# T_cell = W_b^T G_eff W_a has rank <= d (d=8), because G_eff is (d,d).  Materialising
# and SVD-ing the full (out,in) matrix (up to 8192x2048) is ruinous at 28k cells.
# Instead: G_eff = d(logit)/dF_cell^raw is the 8x8 core (one backward gives all 112),
# and the top (u,v,sigma) come from an 8x8 SVD after pulling W_a/W_b through a QR that
# is precomputed ONCE per module (they are fixed model weights).


def module_qr(mw):
    """Precompute per-module economy QR of W_b^T (out,d) and W_a^T (in,d).

    Returns {key: (Qb (out,d), Rb (d,d), Qa (in,d), Ra (d,d))}, all float64.
    """
    out = {}
    for key, (W_a, W_b) in mw.items():
        Qb, Rb = torch.linalg.qr(W_b.t(), mode="reduced")   # W_b^T = Qb Rb
        Qa, Ra = torch.linalg.qr(W_a.t(), mode="reduced")   # W_a^T = Qa Ra
        out[key] = (Qb, Rb, Qa, Ra)
    return out


def raw_feature_grads(model, mw, grouped, order, num_layers=16, acc_head="acc"):
    """G_eff per cell = d(acc_logit)/dF_cell^raw, as (d,d), for one adapter.

    Leaves are the *raw* (pre-L2-norm) cell features, so each grad already absorbs
    the L2-norm Jacobian AND the shared MLP head.  One backward -> all 112 cores.
    Returns (G {key:[ (d,d) per layer]}, acc_logit float).
    """
    from meta_model.lora.types import LoraType

    head = model.head
    body = head.body.double()
    acc = head.output_heads[acc_head].double()
    d = model.equivariant_layer_sizes[-1]

    raw = {}
    tokens = []
    for key in order:
        W_a, W_b = mw[key]
        sub = grouped[key]
        raw[key] = []
        for layer in range(num_layers):
            dW = sub[LoraType.B][layer].double() @ sub[LoraType.A][layer].double()
            F = (W_b @ dW @ W_a.t()).reshape(-1).detach().requires_grad_(True)  # (d*d,) leaf
            raw[key].append(F)
            featn = F / (F.norm() + 1e-8)
            tokens.append(featn)
    logit = acc(body(torch.cat(tokens))).squeeze()
    logit.backward()
    G = {key: [raw[key][layer].grad.reshape(d, d).detach().clone()
               for layer in range(num_layers)] for key in order}
    return G, float(logit.detach())


def uv_from_core(G_eff, qr_key):
    """Top (u,v,S) of T = W_b^T G_eff W_a from the 8x8 core, via precomputed QR.

    u in R^out (write/residual-facing), v in R^in (read side), S the <=d singular
    values of T.  All float64.
    """
    Qb, Rb, Qa, Ra = qr_key
    M = Rb @ G_eff @ Ra.t()                    # (d,d) core; T = Qb M Qa^T
    Ut, S, Vt = torch.linalg.svd(M, full_matrices=False)
    u = (Qb @ Ut[:, 0]).numpy()
    v = (Qa @ Vt[0, :]).numpy()
    return u, v, S.numpy()


def cell_full(G_eff, qr_key, residual_side: str):
    """Full rank-<=d factorisation of T with the residual-side basis.

    Returns dict:
      S            : (d,) singular values of T
      u0, v0       : top left (out) / right (in) singular vectors
      res_top      : the residual-side TOP vector (u0 if writer 'u', else v0), 2048-d
      res_basis    : (d, 2048) the full residual-side singular-vector basis (rows)
    ``res_top`` / ``res_basis`` live in the residual/embedding space, so they are
    logit-lensable and delta-comparable.
    """
    Qb, Rb, Qa, Ra = qr_key
    M = Rb @ G_eff @ Ra.t()
    Ut, S, Vt = torch.linalg.svd(M, full_matrices=False)
    U = (Qb @ Ut)          # (out, d) left singular vectors
    V = (Qa @ Vt.t())      # (in,  d) right singular vectors
    u0 = U[:, 0].numpy(); v0 = V[:, 0].numpy()
    if residual_side == "u":
        res_basis = U.t().numpy()   # (d, out=2048)
        res_top = u0
    else:
        res_basis = V.t().numpy()   # (d, in=2048)
        res_top = v0
    return {"S": S.numpy(), "u0": u0, "v0": v0, "res_top": res_top, "res_basis": res_basis}


def svd_uv(T: torch.Tensor):
    """(u (out,), v (in,), S (min(out,in),)) top singular pair of a template, float64."""
    U, S, Vt = torch.linalg.svd(T.double(), full_matrices=False)
    return U[:, 0].numpy(), Vt[0, :].numpy(), S.numpy()


def sigma0_share(S: np.ndarray) -> float:
    return float(S[0] / S.sum())
