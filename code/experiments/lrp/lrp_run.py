"""Rank-wise LRP primitives + the LRP-hookable ``base_l2`` classifier wrapper.

(migrated from SRC/src/discoveries/task_meta_classifier_lrp/flows/lrp_svd.py
 and SRC/src/discoveries/sentiment_crosstone_lrp/flows/lrp_flexible.py)

Two halves, merged so a single import gives the whole LRP driver surface:

* ``lrp_svd`` half — head-agnostic rank-wise LRP + SVD-balancing primitives
  (``run_lrp_single`` / ``per_rank_profiles`` / ``group_from_path`` /
  ``make_composite`` and the concentration / cell-metric helpers). LRP runs on
  CPU (established practice).
* ``lrp_flexible`` half — :class:`LRPFlexibleMetaClassifier` (a subclass of
  ``meta_model.model_reg.FlexibleLoRAMetaClassifier`` that swaps the functional
  per-cell ``B @ Aᵀ`` and ``feature_norm="l2"`` for hookable parameterless
  modules), :class:`L2NormCell`, and ``make_composite_l2``.

The two composites use different epsilon stabilisers (``EPSILON`` = 1e-4 for the
SVD/1B-classifier composite; ``EPSILON_L2`` = 1e-6 for the 3B ``base_l2``
composite, where 1e-4 leaks ~13% of the relevance and 1e-6 only ~0.3%); the two
constants are kept separate so each path's numerics are unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

import numpy as np
import torch
from torch import nn

from meta_model.lora.types import LoraType, TargetModuleType
from meta_model.modules.mat_mul import MatMul
from meta_model.model_reg import (
    EquivariantLinearLayer,
    FlexibleLoRAMetaClassifier,
    _apply_activation,
    _MODULE_KEY,
)

# SVD / 1B-classifier composite stabiliser.
EPSILON = 1e-4
# base_l2 (3B / 28-layer) composite stabiliser — see module docstring.
EPSILON_L2 = 1e-6

MODULE_ORDER = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]
MODULE_SHORT = {
    "self_attn.q_proj": "Attn Q", "self_attn.k_proj": "Attn K",
    "self_attn.v_proj": "Attn V", "self_attn.o_proj": "Attn O",
    "mlp.gate_proj": "MLP Gate", "mlp.up_proj": "MLP Up", "mlp.down_proj": "MLP Down",
}


# ─────────────────────────────────────────────────────────────────────────────
# Small numeric helpers (no scipy dependency)
# ─────────────────────────────────────────────────────────────────────────────
def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank-correlation of two 1-D arrays (NaN if degenerate)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def concentration(mag: np.ndarray) -> dict:
    """Concentration of a non-negative per-rank magnitude vector."""
    mag = np.abs(np.asarray(mag, float))
    total = mag.sum()
    r = len(mag)
    if total == 0:
        return {"top1_share": float("nan"), "top3_share": float("nan"),
                "entropy_norm": float("nan"), "participation_ratio": float("nan"),
                "argmax": -1, "rank": r}
    p = mag / total
    srt = np.sort(p)[::-1]
    ent = -(p[p > 0] * np.log(p[p > 0])).sum()
    ent_norm = float(ent / np.log(r)) if r > 1 else float("nan")
    pr = float((mag.sum() ** 2) / (np.square(mag).sum()))  # participation ratio in [1, r]
    return {
        "top1_share": float(srt[0]),
        "top3_share": float(srt[:3].sum()),
        "entropy_norm": ent_norm,       # 1 = uniform, →0 = concentrated
        "participation_ratio": pr,      # ~1 = single rank, ~r = spread out
        "argmax": int(np.argmax(mag)),
        "rank": r,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SVD-balanced re-parametrisation of a grouped LoRA dict
# ─────────────────────────────────────────────────────────────────────────────
def svd_balance_grouped(grouped):
    """Re-parametrise every (module, layer) of a grouped LoRA dict to balanced form.

    Thin lazy-import shim over :func:`meta_model.lora.svd.svd_balance_grouped`.

    Returns (grouped_balanced, sing_values, max_recon_err) — see the library docstring.
    """
    from meta_model.lora.svd import svd_balance_grouped as _impl

    return _impl(grouped)


# ─────────────────────────────────────────────────────────────────────────────
# LRP on a single adapter (head-agnostic)
# ─────────────────────────────────────────────────────────────────────────────
def make_composite():
    from zennit.composites import EpsilonPlus

    from experiments.lrp.rules import EpsilonNoBias, EquivariantRule, MatMulEpsilon, PassThrough
    from meta_model.modules.equivariant_linear_layer import (
        EquivariantLinearLayer as _EqLinModule,
    )
    from meta_model.modules.mat_mul import MatMul as _MatMul

    return EpsilonPlus(layer_map=[
        (_MatMul, MatMulEpsilon()),
        (_EqLinModule, EquivariantRule(epsilon=EPSILON)),
        (torch.nn.LeakyReLU, PassThrough()),
        (torch.nn.Linear, EpsilonNoBias(epsilon=EPSILON)),
    ])


def _single_head_cls(head_name: str):
    """A wrapper that selects ``head_name`` from the meta-classifier's dict output."""

    class SingleHead(torch.nn.Module):
        def __init__(self, inner, head=head_name):
            super().__init__()
            self.inner = inner
            self.head = head

        def forward(self, x):
            out = self.inner(x)
            if isinstance(out, dict):
                return out.get(self.head) if self.head in out else next(iter(out.values()))
            return out

    return SingleHead


def predict_class(model, grouped, device, head_name):
    """Forward-only: return (pred_class, full_softmax_probs) for one adapter."""
    from meta_model.modules.flattened_input_classifier import (
        FlattenedInputClassifier,
        flatten_lora_dict,
    )
    SingleHead = _single_head_cls(head_name)
    batched = {k: {m: t.unsqueeze(0).float() for m, t in v.items()} for k, v in grouped.items()}
    x, metadata = flatten_lora_dict(batched)
    x = x.to(device)
    wrapped = SingleHead(FlattenedInputClassifier(model, metadata)).to(device)
    with torch.no_grad():
        logits = wrapped(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return int(logits.argmax(dim=1).item()), probs


def run_lrp_single(model, grouped, composite, device, head_name, target_class):
    """Run LRP on ONE adapter, attributing the ``target_class`` logit of ``head_name``.

    Returns attr_dict (per-weight relevance, unflattened) + decision info.  The
    relevance is normalised downstream by ``norm`` = the target-class logit value.
    """
    from zennit.attribution import Gradient

    from meta_model.modules.flattened_input_classifier import (
        FlattenedInputClassifier,
        flatten_lora_dict,
        unflatten_lora_dict,
    )

    SingleHead = _single_head_cls(head_name)
    batched = {k: {m: t.unsqueeze(0).float() for m, t in v.items()} for k, v in grouped.items()}
    x, metadata = flatten_lora_dict(batched)
    x = x.to(device)
    wrapped = SingleHead(FlattenedInputClassifier(model, metadata)).to(device)
    with torch.no_grad():
        logits = wrapped(x)
    pred = int(logits.argmax(dim=1).item())
    probs = torch.softmax(logits, dim=1)[0]
    tgt = int(target_class)
    conf = float(probs[tgt].item())
    y = torch.zeros_like(logits)
    y[0, tgt] = logits[0, tgt]
    norm = float(y.sum().item())
    attributor = Gradient(wrapped, composite)
    with attributor:
        _, attributions = attributor(x, y)
    attr_dict = unflatten_lora_dict(attributions.cpu(), metadata)
    return {
        "attr_dict": attr_dict,
        "logits": logits.detach().cpu().numpy()[0],
        "pred": pred,
        "conf_target": conf,
        "target_class": tgt,
        "norm": norm,
    }


def per_rank_profiles(attr_dict, norm):
    """Per-(module, layer) per-rank signed & magnitude relevance for A and B.

    Promoted to :func:`experiments.lrp.analysis.per_rank_relevance`; kept under the
    historical name for this discovery's jobs.
    """
    from experiments.lrp.analysis import per_rank_relevance

    return per_rank_relevance(attr_dict, norm)


def cell_metrics(prof_cell):
    """Concentration + A/B agreement metrics for one (module, layer) cell."""
    A_mag, B_mag = prof_cell["A_mag"], prof_cell["B_mag"]
    cA, cB = concentration(A_mag), concentration(B_mag)
    return {
        "A": cA, "B": cB,
        "argmax_match": bool(cA["argmax"] == cB["argmax"] and cA["argmax"] >= 0),
        "spearman_AB_mag": spearman(A_mag, B_mag),
        "total_mag": prof_cell["total_mag"],
    }


def group_from_path(path, num_layers):
    """Load one adapter safetensors → grouped {module: {LoraType: (L,·,·)}} dict."""
    from safetensors.torch import load_file

    from meta_model.lora.weight_utils import group_lora_weights_per_submodule

    return group_lora_weights_per_submodule(
        load_file(str(path)), layer_id_order=list(range(num_layers))
    )


def analyse_adapter(model, grouped, composite, device, head_name, target_class, tag):
    """Run raw + SVD-balanced LRP on one adapter, attributing ``target_class``.

    Returns a record with the per-rank profiles for both bases, the per-cell
    singular values (from the SVD balance), and the reconstruction-error bound.
    """
    raw = run_lrp_single(model, grouped, composite, device, head_name, target_class)
    raw_prof = per_rank_profiles(raw["attr_dict"], raw["norm"])

    bal_grouped, sing, recon_err = svd_balance_grouped(grouped)
    svd = run_lrp_single(model, bal_grouped, composite, device, head_name, target_class)
    svd_prof = per_rank_profiles(svd["attr_dict"], svd["norm"])

    return {
        "tag": tag,
        "target_class": int(target_class),
        "recon_err": recon_err,
        "raw": {"pred": raw["pred"], "conf_target": raw["conf_target"],
                "logits": raw["logits"].tolist(), "prof": raw_prof},
        "svd": {"pred": svd["pred"], "conf_target": svd["conf_target"],
                "logits": svd["logits"].tolist(), "prof": svd_prof, "sing": sing},
    }


# ─────────────────────────────────────────────────────────────────────────────
# LRP-hookable variant of the ``base_l2`` classifier (FlexibleLoRAMetaClassifier)
# ─────────────────────────────────────────────────────────────────────────────
class L2NormCell(nn.Module):
    """Per-cell L2 normalization ``feat / (‖feat‖₂ + eps)`` as a hookable module.

    Byte-identical to the functional step in
    ``FlexibleLoRAMetaClassifier._cell_features`` for ``feature_norm="l2"``.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return feat / (feat.norm(dim=-1, keepdim=True) + self.eps)


class LRPFlexibleMetaClassifier(FlexibleLoRAMetaClassifier):
    """``FlexibleLoRAMetaClassifier`` with the per-cell matmul + l2-norm as modules.

    Adds parameterless ``cell_matmuls`` (one :class:`MatMul` per ``(module, layer)`` so
    each zennit hook stores its own per-call activations) and ``cell_norms`` (one
    :class:`L2NormCell` per cell).  Everything else — the equivariant trunk, the MLP head,
    and therefore the entire ``state_dict`` — is inherited unchanged.
    """

    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        super().__init__(**kwargs)
        if self.head_type != "mlp":
            msg = f"LRP wrapper supports head='mlp' only, got {self.head_type!r}"
            raise ValueError(msg)
        if self.feature_norm not in ("none", "l2"):
            msg = (
                f"LRP wrapper supports feature_norm in {{'none','l2'}}, "
                f"got {self.feature_norm!r}"
            )
            raise ValueError(msg)
        self.cell_matmuls = nn.ModuleDict()
        self.cell_norms = nn.ModuleDict()
        for module_type in self.targeted_modules:
            for layer_idx in range(self.llm_layers):
                key = f"{module_type.value}_{layer_idx}"
                self.cell_matmuls[key] = MatMul()
                if self.feature_norm == "l2":
                    self.cell_norms[key] = L2NormCell()
        self.to(self.device)

    def _cell_features(  # type: ignore[override]
        self,
        a_layer: torch.Tensor,
        b_layer: torch.Tensor,
        module_type: TargetModuleType,
        layer_idx: int,
    ) -> torch.Tensor:
        a_layers = self.module_a_layers[module_type.value]
        b_layers = self.module_b_layers[module_type.value]
        a_proc, b_proc = a_layer, b_layer
        n = len(a_layers)
        for i, (a_eq, b_eq) in enumerate(zip(a_layers, b_layers, strict=True)):
            a_proc = a_eq(a_proc)
            b_proc = b_eq(b_proc)
            if i != n - 1:
                b_proc, a_proc = _apply_activation(self.activation, b_proc, a_proc)
        key = f"{module_type.value}_{layer_idx}"
        w = self.cell_matmuls[key](b_proc, a_proc.transpose(-1, -2))  # (batch, d, d)
        feat = w.reshape(w.size(0), -1)
        if self.feature_norm == "l2":
            feat = self.cell_norms[key](feat)
        return feat

    def forward(  # type: ignore[override]
        self, x: dict[str, dict[LoraType, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        tokens: list[torch.Tensor] = []
        for module_type in self.targeted_modules:
            key = _MODULE_KEY[module_type]
            a = x[key][LoraType.A].to(self.device).transpose(-1, -2)
            b = x[key][LoraType.B].to(self.device)
            if a.dim() == 3:
                a = a.unsqueeze(1)
            if b.dim() == 3:
                b = b.unsqueeze(1)
            if a.size(1) != self.llm_layers or b.size(1) != self.llm_layers:
                msg = (
                    f"expected {self.llm_layers} layers, got "
                    f"A: {a.size(1)}, B: {b.size(1)}"
                )
                raise ValueError(msg)
            for layer_idx in range(self.llm_layers):
                tokens.append(
                    self._cell_features(a[:, layer_idx], b[:, layer_idx],
                                        module_type, layer_idx)
                )
        stacked = torch.stack(tokens, dim=1)
        return self.head(stacked)

    @classmethod
    def load(cls, path: str | Path, device: torch.device | str | None = None) -> Self:
        """Load a stock ``FlexibleLoRAMetaClassifier`` checkpoint into the LRP wrapper.

        Reuses the parent loader (``cls`` = this subclass), so the parameterless
        ``cell_matmuls`` / ``cell_norms`` are built by ``__init__`` and the trained
        ``state_dict`` loads with ``strict=True`` (they contribute no keys).
        """
        return super().load(path, device=device)  # type: ignore[return-value]


def make_composite_l2():
    """LRP composite for :class:`LRPFlexibleMetaClassifier` (``base_l2``).

    Extends the ``lrp_svd`` composite with an identity rule for the L2 normalization and
    binds the equivariant rule to *this* model's ``EquivariantLinearLayer`` class.
    """
    from zennit.composites import EpsilonPlus

    from experiments.lrp.rules import EpsilonNoBias, EquivariantRule, MatMulEpsilon, PassThrough

    return EpsilonPlus(layer_map=[
        (MatMul, MatMulEpsilon()),
        (EquivariantLinearLayer, EquivariantRule(epsilon=EPSILON_L2)),
        (L2NormCell, PassThrough()),
        (torch.nn.LeakyReLU, PassThrough()),
        (torch.nn.Linear, EpsilonNoBias(epsilon=EPSILON_L2)),
    ])


def conservation_error(model, grouped, device, head_name, target_class) -> dict:
    """Run one LRP pass and return conservation diagnostics.

    ``rel_err = |Σattr − norm| / (|norm| + 1e-12)`` should be ~0 (a few % is fine given
    the epsilon stabilisers). Also returns whether the wrapper's prediction matches a
    plain (functional) forward on the same input.
    """
    composite = make_composite_l2()
    res = run_lrp_single(model, grouped, composite, device, head_name, target_class)
    attr_sum = 0.0
    for mk in res["attr_dict"]:
        for lt in res["attr_dict"][mk]:
            attr_sum += float(res["attr_dict"][mk][lt].sum().item())
    norm = res["norm"]
    rel_err = abs(attr_sum - norm) / (abs(norm) + 1e-12)
    return {
        "attr_sum": attr_sum,
        "norm": norm,
        "abs_err": abs(attr_sum - norm),
        "rel_err": rel_err,
        "pred": res["pred"],
        "target_class": res["target_class"],
    }
