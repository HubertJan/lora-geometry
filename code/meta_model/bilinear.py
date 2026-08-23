"""Migrated from SRC/src/glad/meta_classifier/bilinear.py.

Bilinear template extraction for equivariant LoRA meta-classifiers.

A *linear* single-cell detector (one equivariant layer of width ``d``, empty
head) computes per cell ``F = W_b·ΔW·W_aᵀ`` and a linear readout of ``vec(F)``,
so the class-difference logit is **exactly**

    s = ⟨T, ΔW⟩ + β,    T = W_bᵀ·G·W_a,    G = reshape(W_head[1]−W_head[0], (d,d)),
                        β = b₁−b₀,

with ``T`` the same shape as the cell's update ``ΔW = B·A``.  The top singular
pair ``u = U[:,0]``, ``v = Vt[0]`` of ``T`` is the detector's dominant read/write
direction; :attr:`meta_model.lora.cells.Cell.residual_side` says which of the two
lives in the residual stream (and may be logit-lensed).

This module is the promotion of the ``build_T`` / ``reconstruct_uv`` /
``inspect_cell`` triple that existed in ≥8 near-identical copies (the reference
being ``attn_o_backdoor_mechanism/flows/inspect_cell.py`` and the
``num_layers``-parametrised ``uv_synthesis/flows/bilinear.py``).  Differences
absorbed here:

* adapters are consumed as **preloaded grouped dicts** (see
  :func:`meta_model.cell_sweep.preload`) — the per-copy
  ``layer_id_order=list(range(16))`` hardcode is gone from this layer entirely;
* the residual side defaults to :attr:`Cell.residual_side` instead of a partial
  per-discovery map (or no switch at all);
* the logit lens is gated on the vector actually matching the embedding
  dimension (the ungated copies produced a confident, meaningless lens when
  handed a non-residual vector);
* the *linearized* template path for full nonlinear-head detectors
  (``head_gradient_G`` / ``build_T_linearized``) rides along from
  ``uv_synthesis``.

All template math is float64 on CPU (``torch.linalg.svd`` of a ``d_out×d_in``
matrix), matching every historical copy.

Note the deliberate division of labour with :mod:`meta_model.lora.svd`: that module
decomposes the *adapter* (ΔW); this one decomposes the *detector's template*.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from meta_model.lora.cells import KEY_TO_MODULE_TYPE, Cell
from meta_model.lora.types import LoraType
from meta_model.cell_sweep import (
    PreloadedRow,
    make_cell_slicer,
    predict_on_rows,
)
from meta_model.equivariant_lora_meta_classifier import (
    EquivariantLoRAMetaClassifier,
)

__all__ = [
    "build_T",
    "build_T_linearized",
    "cell_feature",
    "concentration",
    "delta_w",
    "fidelity_from_T",
    "full_feature_vector",
    "head_gradient_G",
    "inspect_cell",
    "logit_lens",
    "module_stack",
    "reconstruct_uv",
    "template_svd",
]


def _as_cell(cell: Cell | str, layer: int | None) -> Cell:
    if isinstance(cell, Cell):
        return cell
    if layer is None:
        raise TypeError(f"layer is required when addressing by module key {cell!r}")
    return Cell(KEY_TO_MODULE_TYPE[cell], layer)


# ── Model access ─────────────────────────────────────────────────────────────


def module_stack(model: EquivariantLoRAMetaClassifier, kind: str, module_key: str):
    """The model's equivariant layer stack for one module, keyed defensively.

    ``kind`` is ``"a"`` or ``"b"``.  Tries the enum value (``"down_proj"``) and
    the safetensor key (``"mlp.down_proj"``), then falls back to the single
    stack of a one-module model — the union of the historical lookups.
    """
    layers = getattr(model, f"module_{kind}_layers")
    for k in (module_key.split(".")[-1], module_key):
        if k in layers:
            return layers[k]
    (only,) = layers.values()
    return only


# ── Template construction ────────────────────────────────────────────────────


def build_T(
    model: EquivariantLoRAMetaClassifier,
    module_key: str,
    *,
    strict: bool = True,
) -> tuple[torch.Tensor, float]:
    """Exact template ``(T, beta)`` of a linear single-cell detector (float64, CPU).

    ``strict`` verifies the model really is linear (one equivariant layer, empty
    head) — without that structure ``⟨T, ΔW⟩ + β`` is *not* the model's score.
    """
    head = model.head_specs[0].name
    if strict and (
        len(model.head_layer_sizes) != 0 or len(model.equivariant_layer_sizes) != 1
    ):
        raise ValueError(
            "build_T requires a linear detector (equiv [d], head []); got "
            f"equiv {model.equivariant_layer_sizes}, head {model.head_layer_sizes}. "
            "For nonlinear-head detectors use build_T_linearized."
        )
    W_a = module_stack(model, "a", module_key)[0].weights.detach().double().cpu()
    W_b = module_stack(model, "b", module_key)[0].weights.detach().double().cpu()
    W_head = model.output_heads[head].weight.detach().double().cpu()
    b_head = model.output_heads[head].bias.detach().double().cpu()
    d = W_a.shape[0]
    G = (W_head[1] - W_head[0]).reshape(d, d)
    beta = float(b_head[1] - b_head[0])
    return W_b.t() @ G @ W_a, beta


def template_svd(T: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(U, S, Vt)`` of a template (thin SVD, descending σ)."""
    return torch.linalg.svd(T, full_matrices=False)


def delta_w(grouped: dict, cell: Cell) -> torch.Tensor:
    """``ΔW = B·A`` of one preloaded adapter's cell, float64 (template precision).

    :func:`meta_model.lora.cells.cell_delta_w` is the float32 twin; the template
    analyses run in double so the reconstruction check resolves ~1e-7 errors.
    """
    sub = grouped[cell.key]
    return sub[LoraType.B][cell.layer].double() @ sub[LoraType.A][cell.layer].double()


# ── Linearized template (full nonlinear-head detectors) ──────────────────────


def cell_feature(model, grouped: dict, module_type, layer: int) -> torch.Tensor:
    """``F = W_b·B·A·W_aᵀ`` (float32) — the model's own per-cell feature."""
    key = model._get_module_key(module_type)
    W_a = model.module_a_layers[module_type.value][0].weights.detach().float()
    W_b = model.module_b_layers[module_type.value][0].weights.detach().float()
    A = grouped[key][LoraType.A][layer].float()
    B = grouped[key][LoraType.B][layer].float()
    return (W_b @ B) @ (W_a @ A.t()).t()


def full_feature_vector(model, grouped: dict) -> torch.Tensor:
    """Concatenated head-input features in the model's own ordering (float32, 1-D)."""
    blocks = []
    for mt in model.targeted_modules:
        for layer in range(model.llm_layers):
            blocks.append(cell_feature(model, grouped, mt, layer).reshape(-1))
    return torch.cat(blocks)


def _cell_block_slice(model, module_key: str, layer: int) -> slice:
    d = model.equivariant_layer_sizes[-1]
    cell_sz = d * d
    keys = [model._get_module_key(mt) for mt in model.targeted_modules]
    mi = keys.index(module_key)
    off = (mi * model.llm_layers + layer) * cell_sz
    return slice(off, off + cell_sz)


def head_gradient_G(
    model,
    feat_mean: torch.Tensor,
    module_key: str,
    layer: int,
    pos_class: int = 1,
    neg_classes: list[int] | None = None,
) -> torch.Tensor:
    """``G_eff = ∂(logit_pos − mean logit_neg)/∂F_cell`` at a feature operating point.

    For a binary head pass ``pos_class=1, neg_classes=[0]``; for a K-way head the
    class index and the rest (one-vs-rest contrast).
    """
    head = model.head_specs[0].name
    f = feat_mean.clone().detach().requires_grad_(True)
    logits = model.output_heads[head](model.head_body(f))
    if neg_classes is None:
        neg_classes = [0]
    scalar = logits[pos_class] - logits[torch.tensor(neg_classes)].mean()
    (grad,) = torch.autograd.grad(scalar, f)
    d = model.equivariant_layer_sizes[-1]
    return grad[_cell_block_slice(model, module_key, layer)].reshape(d, d).double()


def build_T_linearized(
    model,
    module_key: str,
    layer: int,
    feat_mean: torch.Tensor,
    pos_class: int = 1,
    neg_classes: list[int] | None = None,
) -> torch.Tensor:
    """First-order cell-restricted template of a full detector around ``feat_mean``.

    The equivariant stage is exactly linear per cell, so the only nonlinearity is
    the head; this is its gradient pulled back through ``W_a`` / ``W_b``.  A
    gradient has no intercept — there is no ``beta`` on this path.
    """
    mt = next(m for m in model.targeted_modules if model._get_module_key(m) == module_key)
    W_a = model.module_a_layers[mt.value][0].weights.detach().double()
    W_b = model.module_b_layers[mt.value][0].weights.detach().double()
    G = head_gradient_G(model, feat_mean, module_key, layer, pos_class, neg_classes)
    return W_b.t() @ G @ W_a


# ── Reconstruction + reports ─────────────────────────────────────────────────


def reconstruct_uv(
    model: EquivariantLoRAMetaClassifier,
    data: Sequence[PreloadedRow],
    cell: Cell | str,
    layer: int | None = None,
) -> dict[str, Any]:
    """Oriented top singular pair ``(u, v)`` of the cell's template.

    Streams ``uᵀΔWv`` over *data* (preloaded ``(grouped, label)`` rows) and
    jointly flips ``(u, v)`` so the label-1 class projects positive — the outer
    product ``u·vᵀ`` is invariant under the joint flip, so this is a gauge
    choice, not a change of ``T``.

    Returns ``{u, v (np, oriented), S (np), T (torch double), beta, proj,
    labels}``.
    """
    c = _as_cell(cell, layer)
    T, beta = build_T(model, c.key, strict=False)
    U, S, Vt = template_svd(T)
    u, v = U[:, 0].clone(), Vt[0, :].clone()

    proj, labels = [], []
    for grouped, lbl in data:
        dW = delta_w(grouped, c)
        proj.append(float(u @ (dW @ v)))
        labels.append(int(lbl))
        del dW
    proj = np.array(proj)
    labels = np.array(labels)
    if proj[labels == 1].mean() < 0:
        u, v, proj = -u, -v, -proj
    return {
        "u": u.numpy(),
        "v": v.numpy(),
        "S": S.numpy(),
        "T": T,
        "beta": beta,
        "proj": proj,
        "labels": labels,
    }


def concentration(vec: np.ndarray) -> dict[str, float]:
    """Participation ratio + top-k energy of a direction (delocalisation anatomy)."""
    e = vec**2
    e = e / (e.sum() + 1e-30)
    order = np.argsort(-e)
    d = vec.size
    return {
        "dim": int(d),
        "participation_ratio": float(1.0 / (e**2).sum()),
        "top1_energy": float(e[order[0]]),
        "top10_energy": float(e[order[:10]].sum()),
        "uniform_energy": float(1.0 / d),
    }


def logit_lens(
    vec: np.ndarray,
    tok,
    E_norm: torch.Tensor,
    *,
    top_k: int = 20,
    pole: tuple[str, str] = ("+", "-"),
) -> dict[str, list] | None:
    """Top/bottom-``top_k`` tokens by cosine of *vec* with the tied embeddings.

    Returns ``None`` when *vec* does not live in the embedding space
    (``vec.shape[0] != E_norm.shape[1]``) — lensing a non-residual vector
    produces confident, meaningless tokens with no crash, which is exactly the
    silent failure the residual-side machinery exists to prevent.
    """
    if tok is None or E_norm is None or vec.shape[0] != E_norm.shape[1]:
        return None
    unit = torch.from_numpy(vec / (np.linalg.norm(vec) + 1e-12)).float()
    cos = (E_norm @ unit).numpy()
    top = np.argsort(-cos)[:top_k]
    bot = np.argsort(cos)[:top_k]
    return {
        "token": [tok.decode([int(i)]) for i in np.concatenate([top, bot])],
        "cos": cos[np.concatenate([top, bot])].tolist(),
        "pole": [pole[0]] * len(top) + [pole[1]] * len(bot),
    }


def inspect_cell(
    model: EquivariantLoRAMetaClassifier,
    data: Sequence[PreloadedRow],
    cell: Cell | str,
    layer: int | None = None,
    *,
    tok=None,
    E_norm: torch.Tensor | None = None,
    residual_side: str | None = None,
    topk_ranks: int = 10,
    lens_top_k: int = 20,
    lens_pole: tuple[str, str] = ("+", "-"),
    check_reconstruction: bool = True,
) -> dict[str, Any]:
    """Full single-cell mechanism report for a linear detector.

    Streams *data* (preloaded ``(grouped, label)`` rows) once, computing the
    reconstructed score ``s = ⟨T,ΔW⟩+β``, the orientation projection, and (for
    ``topk_ranks > 0``) each rank's contribution to the label-1 − label-0 score
    gap (``σ_k·u_kᵀΔWv_k``, orientation-free per pair).

    ``residual_side`` defaults to the cell's own
    (:attr:`meta_model.lora.cells.Cell.residual_side`); the lens is skipped (``None``)
    when ``tok`` / ``E_norm`` are missing or the chosen vector is not
    embedding-dimensional.  ``check_reconstruction`` compares
    ``sigmoid(s_recon)`` against the model's own forward pass
    (``max_recon_err``; float32-forward vs float64-template resolves ~1e-7).

    Returns raw arrays (``u``, ``v``, ``S_np``, ``labels``, ``s_recon``,
    ``proj``, ``p_model``, ``per_rank_gap``) plus ``lens``, ``summary`` and the
    ``u``/``v`` anatomy; label-1 is called "pos" in the summary keys.
    """
    from sklearn.metrics import roc_auc_score

    c = _as_cell(cell, layer)
    if residual_side is None:
        residual_side = c.residual_side
    T, beta = build_T(model, c.key, strict=False)
    U, S, Vt = template_svd(T)
    S_np = S.numpy()

    K = min(topk_ranks, S.shape[0])
    Uk, Vk, Sk = U[:, :K], Vt[:K, :], S[:K]

    labels_l, s_recon_l, proj_l = [], [], []
    per_rank_pos = np.zeros(K)
    per_rank_neg = np.zeros(K)
    n_pos = n_neg = 0
    u0, v0 = U[:, 0], Vt[0, :]
    for grouped, lbl in data:
        dW = delta_w(grouped, c)
        lbl = int(lbl)
        labels_l.append(lbl)
        s_recon_l.append(float((T * dW).sum() + beta))
        proj_l.append(float(u0 @ (dW @ v0)))
        if K:
            diag = torch.einsum("dk,dk->k", Uk, dW @ Vk.t())
            contrib = (Sk * diag).numpy()
            if lbl == 1:
                per_rank_pos += contrib
                n_pos += 1
            else:
                per_rank_neg += contrib
                n_neg += 1
        del dW
    labels = np.array(labels_l)
    s_recon = np.array(s_recon_l)
    proj = np.array(proj_l)

    flip = proj[labels == 1].mean() < 0
    u = (-U[:, 0] if flip else U[:, 0]).numpy()
    v = (-Vt[0, :] if flip else Vt[0, :]).numpy()
    proj = -proj if flip else proj

    per_rank_gap = (
        (per_rank_pos / max(n_pos, 1)) - (per_rank_neg / max(n_neg, 1)) if K else None
    )

    p_model = None
    max_recon_err = float("nan")
    if check_reconstruction:
        _, p_model = predict_on_rows(model, data, make_cell_slicer(c), "cpu")
        p_recon = 1.0 / (1.0 + np.exp(-s_recon))
        max_recon_err = float(np.max(np.abs(p_model - p_recon)))

    res_vec = u if residual_side == "u" else v
    lens = logit_lens(res_vec, tok, E_norm, top_k=lens_top_k, pole=lens_pole)

    two = len(set(labels.tolist())) == 2
    s_pos, s_neg = s_recon[labels == 1], s_recon[labels == 0]
    summary = {
        "cell": c.label,
        "residual_side": residual_side,
        "roc_auc_from_scores": float(roc_auc_score(labels, s_recon)) if two else float("nan"),
        "max_recon_err": max_recon_err,
        "T_sigma0": float(S_np[0]),
        "T_sigma1": float(S_np[1]),
        "T_sigma0_over_sigma1": float(S_np[0] / S_np[1]),
        "T_top1_singular_share": float(S_np[0] / S_np.sum()),
        "pos_score_mean": float(s_pos.mean()),
        "neg_score_mean": float(s_neg.mean()),
        "gap": float(s_pos.mean() - s_neg.mean()),
        "uT_dW_v_pos": float(proj[labels == 1].mean()),
        "uT_dW_v_neg": float(proj[labels == 0].mean()),
    }
    if per_rank_gap is not None:
        summary["T_top1_gap_share"] = float(per_rank_gap[0] / (per_rank_gap.sum() + 1e-30))
    return {
        "u": u,
        "v": v,
        "S_np": S_np,
        "T": T,
        "beta": beta,
        "per_rank_gap": per_rank_gap,
        "singular_values": S_np[:K] if K else S_np[:0],
        "labels": labels,
        "s_recon": s_recon,
        "proj": proj,
        "p_model": p_model,
        "lens": lens,
        "summary": summary,
        "u_anatomy": concentration(u),
        "v_anatomy": concentration(v),
    }


def fidelity_from_T(
    T: torch.Tensor,
    data: Sequence[PreloadedRow],
    cell: Cell | str,
    layer: int | None = None,
) -> dict[str, Any]:
    """Rank-1 fidelity of a template on a preloaded pool, plus oriented ``(u, v)``.

    AUC of the full-T cell score ``⟨T,ΔW⟩`` vs the rank-1 score ``σ₀·uᵀΔWv``
    quantifies how much of the detector's decision the top pair carries.
    """
    from sklearn.metrics import roc_auc_score

    c = _as_cell(cell, layer)
    U, S, Vt = template_svd(T)
    S_np = S.numpy()
    u0, v0 = U[:, 0], Vt[0, :]

    s_full, proj0, labels_l = [], [], []
    for grouped, lbl in data:
        dW = delta_w(grouped, c)
        s_full.append(float((T * dW).sum()))
        proj0.append(float(u0 @ (dW @ v0)))
        labels_l.append(lbl)
        del dW
    s_full = np.array(s_full)
    proj0 = np.array(proj0)
    labels = np.array(labels_l)

    flip = proj0[labels == 1].mean() < 0
    u = (-u0 if flip else u0).numpy()
    v = (-v0 if flip else v0).numpy()
    proj0 = -proj0 if flip else proj0

    two = len(set(labels.tolist())) == 2
    s_rank1 = float(S_np[0]) * proj0
    return {
        "sigma0_share": float(S_np[0] / S_np.sum()),
        "sigma0_over_sigma1": float(S_np[0] / S_np[1]),
        "auc_cell_full_T": float(roc_auc_score(labels, s_full)) if two else float("nan"),
        "auc_rank1": float(roc_auc_score(labels, s_rank1)) if two else float("nan"),
        "u": u,
        "v": v,
        "S": S_np,
        "s_full": s_full,
        "labels": labels,
    }
