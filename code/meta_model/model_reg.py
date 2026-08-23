"""Migrated from SRC/src/discoveries/glue_backdoor_detector/flows/architectures_reg.py.

Alternative meta-classifier architectures — attention heads and real GL depth.

Discovery-local copy of ``meta_model.equivariant_lora_meta_classifier``
(rule 1 of the discovery skill: copy, don't edit). The forward contract is
unchanged — ``forward(x) -> {head_name: logits}`` for the same
``{llama_module_key: {LoraType: (batch, layer, width, length)}}`` input — so the
existing trainer, ``compute_multihead_loss`` and ``head_class_scores`` all work
untouched.

Why this exists
---------------
The production detector is

    eq sizes [128] -> one EquivariantLinearLayer per side -> B@A^T -> flatten
    -> Linear(112*128*128, 64) -> LeakyReLU -> Linear(64, 2)

Two things are wrong with it for a small, weak pool (see ``RESEARCH_LOG.md``):

1. **The activation never fires.** The library applies the between-layer
   activation only when ``i != len(a_layers) - 1``; with a single equivariant
   layer that condition is never true. So ``activation="gl_activation"`` is a
   *no-op* and the whole equivariant stage collapses to the bilinear map
   ``W_b (BA) W_a^T`` — the model is linear up to the dense head. Any claim that
   the deployed detector "uses GL activations" is false at depth 1.
2. **The head holds ~99.9% of the parameters.** ``Linear(1_835_008, 64)`` is
   117 M weights fitted on ~136 training adapters.

This module makes both fixable:

* ``activation`` gains the row-/column-sum GL gate (``glsum``), which is the
  textbook equivariant recipe ``f_equi(x) = f_inv(x) * x``:

      sigma_GL(U)_i = sigma( sum_j (U V^T)_ij ) U_i
      sigma_GL(V)_i = sigma( sum_j (U V^T)_ji ) V_i

  ``U V^T`` is the GL-invariant object (``U -> U G``, ``V -> V G^-T`` leaves it
  fixed), so gating a row by a scalar function of its row sum is equivariant.
  **This is strictly better motivated than the library's ``gl_activation``**,
  which gates on ``U (V^T V)^+ V^T``; under ``U -> U G, V -> V G^-T`` that maps
  to ``U G G^T (V^T V)^+ V^T``, i.e. it is invariant only for *orthogonal* G,
  not for the full GL group. Both are available here so the two can be compared.

* ``head`` selects among three set-readouts over the 112 ``(module, layer)``
  cells:

  - ``"mlp"``      — the production head: concatenate every cell, dense stack.
  - ``"deepsets"`` — a *shared* per-cell projection, mean-pool, small MLP. This
    is the control that isolates "shared projection + pooling" from "attention".
  - ``"attn"``     — shared per-cell projection + learned cell embedding, a
    pre-norm transformer encoder over the 112 cell tokens, CLS readout.

  ``deepsets``/``attn`` cost ~0.2–2 M parameters instead of 117 M.

* ``feature_norm="l2"`` L2-normalises each cell's feature vector before the
  head. ``jobs/14`` of ``truefalse_weakness_controls`` found the cosine-normalised
  linear probe at AUC 0.879 against 0.784 raw on the same pool — adapter
  magnitude is nuisance — so this is a first-class knob, not a detail.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Self

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from meta_model.lora.types import LoraType, TargetModuleType
from meta_model.heads import MetaHeadType, MetaTargetSpec
from meta_model.modules.activations import gl_activation as gl_pinv_activation

# ── module key mapping (verbatim from the library) ───────────────────────────

_MODULE_KEY = {
    TargetModuleType.Q_ATTENTION: "self_attn.q_proj",
    TargetModuleType.K_ATTENTION: "self_attn.k_proj",
    TargetModuleType.V_ATTENTION: "self_attn.v_proj",
    TargetModuleType.O_ATTENTION: "self_attn.o_proj",
    TargetModuleType.GATE_MLP: "mlp.gate_proj",
    TargetModuleType.UP_MLP: "mlp.up_proj",
    TargetModuleType.DOWN_MLP: "mlp.down_proj",
}

#: Activation names accepted by ``FlexibleLoRAMetaClassifier(activation=...)``.
ACTIVATIONS = ("none", "leaky_relu", "gl_pinv", "glsum", "glsum_abs")
HEADS = ("mlp", "deepsets", "attn")
#: ``l2``        — normalise EACH cell's feature vector: removes per-adapter magnitude
#:                 AND equalises the 112 cells' contribution to the readout.
#: ``l2_global``  — normalise the whole stacked feature tensor once per adapter: removes
#:                 per-adapter magnitude but LEAVES the relative loudness of cells intact.
#: The pair separates those two mechanisms, which ``l2`` alone confounds.
FEATURE_NORMS = ("none", "l2", "l2_global",
                 # added by glue_backdoor_detector: per-cell L2 was the single
                 # biggest win ever found on this model (+0.201 on tf8k), and it is
                 # the ONLY lever that has ever produced one -- so these push the
                 # same axis further instead of adding capacity, which is settled.
                 "zscore",   # per-cell standardise: removes scale AND offset
                 "rank",     # per-cell rank->uniform: removes the whole marginal,
                             # keeping only the ordering of the cell's entries.
                             # Straight-through (see _cell_features): a rank has no
                             # gradient, so the backward pass uses the identity.
                 "sqrt_l2",  # signed-sqrt magnitude compression, then L2
                 "svals",    # GAUGE-INVARIANT: replace the cell by the singular
                             # values of its (last x last) matrix, L2-normalised.
                             # Changes cell_dim from last**2 to last.
                 )


# ── GL activations ───────────────────────────────────────────────────────────


def glsum_activation(
    u: torch.Tensor,
    v: torch.Tensor,
    *,
    use_abs: bool = False,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-/column-sum GL-equivariant gate.

    ``u`` is ``(..., d1, r)`` and ``v`` is ``(..., d2, r)``; the invariant object
    is ``M = u v^T`` of shape ``(..., d1, d2)``. Row ``i`` of ``u`` is scaled by
    ``sigma(sum_j M_ij)`` and row ``i`` of ``v`` by ``sigma(sum_j M_ji)``.

    ``M`` is never materialised: ``sum_j M_ij = u_i . (sum_j v_j)``, so the whole
    gate costs ``O(d r)`` instead of ``O(d1 d2)``.

    The raw sums scale with ``||dW||``, which would saturate the sigmoid (the
    pools here differ by 54% in ``dw_norm``), so they are divided by their RMS
    over the row axis first. That divisor is itself a function of the invariants,
    so equivariance survives the normalisation.

    ``use_abs=True`` gates on ``sum_j |M_ij|`` instead, which cannot cancel; it
    is the closer analogue of the library's ``gl_activation`` and is kept as a
    fallback in case the signed sums collapse to a constant gate.
    """
    if use_abs:
        # |M| does not factor, so this branch does materialise M. It is only
        # used for the ablation arm, where the (batch, d, d) cost is acceptable.
        m = u @ v.transpose(-2, -1)
        su = m.abs().sum(dim=-1, keepdim=True)
        sv = m.abs().sum(dim=-2).unsqueeze(-1)
    else:
        v_sum = v.sum(dim=-2, keepdim=True)  # (..., 1, r)
        u_sum = u.sum(dim=-2, keepdim=True)  # (..., 1, r)
        su = (u * v_sum).sum(dim=-1, keepdim=True)  # (..., d1, 1)
        sv = (v * u_sum).sum(dim=-1, keepdim=True)  # (..., d2, 1)

    def _norm(s: torch.Tensor) -> torch.Tensor:
        rms = s.pow(2).mean(dim=-2, keepdim=True).sqrt()
        return s / (rms + eps)

    return u * torch.sigmoid(_norm(su)), v * torch.sigmoid(_norm(sv))


def _apply_activation(
    name: str,
    b: torch.Tensor,
    a: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch on the activation name. ``b`` is the write side, ``a`` the read side."""
    match name:
        case "none":
            return b, a
        case "leaky_relu":
            return F.leaky_relu(b), F.leaky_relu(a)
        case "gl_pinv":
            return gl_pinv_activation(b, a)
        case "glsum":
            return glsum_activation(b, a, use_abs=False)
        case "glsum_abs":
            return glsum_activation(b, a, use_abs=True)
    msg = f"unknown activation {name!r}; expected one of {ACTIVATIONS}"
    raise ValueError(msg)


# ── equivariant linear layer (verbatim from the library) ─────────────────────


class EquivariantLinearLayer(nn.Module):
    """``x -> W x`` acting on the ROW index only, so the rank axis is untouched."""

    def __init__(self, size_in: int, size_out: int) -> None:
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        w = torch.empty(size_out, size_in)
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        self.weights = nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights @ x


# ── set readouts over the (module, layer) cells ──────────────────────────────


class MLPHead(nn.Module):
    """The production head: concatenate every cell, then a dense stack.

    PATCHED (see this discovery's HOTFIXES.md): the upstream head takes no ``dropout``
    at all -- the parameter exists on ``FlexibleLoRAMetaClassifier`` but is only wired to
    the ``attn`` and ``deepsets`` heads, so dropout was untestable on the PRODUCTION
    readout, which is where ~95% of the parameters live. ``head_dropout=0.0`` reproduces
    the upstream module exactly (``nn.Dropout(0.0)`` is a no-op and adds no parameters),
    so every pre-existing arm is unaffected.
    """

    def __init__(
        self,
        n_cells: int,
        cell_dim: int,
        head_layer_sizes: list[int],
        head_specs: list[MetaTargetSpec],
        head_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        sizes = [n_cells * cell_dim, *head_layer_sizes]
        body: list[nn.Module] = []
        for i, o in itertools.pairwise(sizes):
            body.append(nn.Linear(i, o))
            body.append(nn.LeakyReLU())
            if head_dropout > 0:
                body.append(nn.Dropout(head_dropout))
        self.body = nn.Sequential(*body)
        self.output_heads = nn.ModuleDict({
            s.name: nn.Linear(sizes[-1], s.output_dim) for s in head_specs
        })

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        shared = self.body(tokens.reshape(tokens.size(0), -1))
        return {name: h(shared) for name, h in self.output_heads.items()}


class DeepSetsHead(nn.Module):
    """Shared per-cell projection -> mean pool -> small MLP.

    The control for the attention head: it buys the same parameter reduction
    (one projection reused by all 112 cells instead of one giant Linear) without
    any cell-to-cell interaction. If ``attn`` does not beat this, the win is
    weight sharing, not attention.
    """

    def __init__(
        self,
        n_cells: int,
        cell_dim: int,
        d_model: int,
        head_specs: list[MetaTargetSpec],
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(cell_dim, d_model)
        self.cell_emb = nn.Parameter(torch.randn(n_cells, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_heads = nn.ModuleDict({
            s.name: nn.Linear(d_model, s.output_dim) for s in head_specs
        })

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        t = F.gelu(self.proj(tokens) + self.cell_emb)
        pooled = self.mlp(self.norm(t.mean(dim=1)))
        return {name: h(pooled) for name, h in self.output_heads.items()}


class AttentionHead(nn.Module):
    """Transformer encoder over the ``(module, layer)`` cells, CLS readout.

    Each of the ``n_cells`` cells becomes one token: its flattened
    ``last_eq x last_eq`` matrix goes through a projection **shared across all
    cells** (so the parameter count is independent of ``n_cells``) plus a learned
    per-cell embedding that restores cell identity. ``norm_first=True`` because
    these are tiny training sets and post-norm transformers need warmup.
    """

    def __init__(
        self,
        n_cells: int,
        cell_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        head_specs: list[MetaTargetSpec],
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(cell_dim, d_model)
        self.cell_emb = nn.Parameter(torch.randn(n_cells, d_model) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.output_heads = nn.ModuleDict({
            s.name: nn.Linear(d_model, s.output_dim) for s in head_specs
        })

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        t = self.proj(tokens) + self.cell_emb
        cls = self.cls.expand(t.size(0), -1, -1)
        h = self.encoder(torch.cat([cls, t], dim=1))
        pooled = self.norm(h[:, 0])
        return {name: head(pooled) for name, head in self.output_heads.items()}


# ── the model ────────────────────────────────────────────────────────────────


class FlexibleLoRAMetaClassifier(nn.Module):
    """Equivariant LoRA meta classifier with pluggable depth / activation / head."""

    def __init__(
        self,
        *,
        equivariant_layer_sizes: list[int],
        targeted_modules: set[TargetModuleType],
        target_module_sizes: dict[TargetModuleType, tuple[int, int]],
        head_specs: list[MetaTargetSpec],
        activation: str = "glsum",
        head: str = "attn",
        head_layer_sizes: list[int] | None = None,
        d_model: int = 128,
        n_heads: int = 4,
        n_attn_layers: int = 2,
        dropout: float = 0.1,
        head_dropout: float = 0.0,
        feature_norm: str = "none",
        llm_layers: int = 16,
        device: torch.device | str = "cuda",
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if not head_specs:
            msg = "head_specs must contain at least one head"
            raise ValueError(msg)
        if activation not in ACTIVATIONS:
            msg = f"activation {activation!r} not in {ACTIVATIONS}"
            raise ValueError(msg)
        if head not in HEADS:
            msg = f"head {head!r} not in {HEADS}"
            raise ValueError(msg)
        if feature_norm not in FEATURE_NORMS:
            msg = f"feature_norm {feature_norm!r} not in {FEATURE_NORMS}"
            raise ValueError(msg)

        if seed is not None:
            torch.manual_seed(seed)

        self.device = torch.device(device) if isinstance(device, str) else device
        self.targeted_modules = sorted(targeted_modules)
        self.llm_layers = llm_layers
        self.head_specs = head_specs
        self.activation = activation
        self.head_type = head
        self.feature_norm = feature_norm
        self.seed = seed
        self.equivariant_layer_sizes = list(equivariant_layer_sizes)
        # ``[]`` (empty head -> pure bilinear readout) must be distinguished from
        # ``None`` (default to a [64] dense head). The old ``head_layer_sizes or [64]``
        # form coerced the falsy empty list back to [64], silently reinstating the dense
        # head and destroying every no-head arm (same fix as the ladder's architectures.py).
        self.head_layer_sizes = list([64] if head_layer_sizes is None else head_layer_sizes)
        self.target_module_sizes = target_module_sizes
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_attn_layers = n_attn_layers
        self.dropout = dropout
        self.head_dropout = head_dropout

        self.module_a_layers = nn.ModuleDict()
        self.module_b_layers = nn.ModuleDict()
        for module_type in self.targeted_modules:
            in_size, out_size = target_module_sizes[module_type]
            self.module_a_layers[module_type.value] = nn.ModuleList([
                EquivariantLinearLayer(i, o)
                for i, o in itertools.pairwise([in_size, *self.equivariant_layer_sizes])
            ])
            self.module_b_layers[module_type.value] = nn.ModuleList([
                EquivariantLinearLayer(i, o)
                for i, o in itertools.pairwise([out_size, *self.equivariant_layer_sizes])
            ])

        last = self.equivariant_layer_sizes[-1]
        cell_dim = last if feature_norm == "svals" else last * last
        n_cells = len(self.targeted_modules) * self.llm_layers

        match head:
            case "mlp":
                self.head = MLPHead(n_cells, cell_dim, self.head_layer_sizes,
                                    head_specs, self.head_dropout)
            case "deepsets":
                self.head = DeepSetsHead(n_cells, cell_dim, d_model, head_specs, dropout)
            case "attn":
                self.head = AttentionHead(
                    n_cells, cell_dim, d_model, n_heads, n_attn_layers,
                    head_specs, dropout,
                )

        self.to(self.device)

    # ── introspection ────────────────────────────────────────────────────────

    @property
    def num_classes(self) -> int:
        cls_heads = [s for s in self.head_specs if s.is_classification]
        if len(self.head_specs) == 1 and len(cls_heads) == 1:
            return cls_heads[0].num_classes
        msg = "num_classes is only defined for a single-classification-head model"
        raise ValueError(msg)

    def param_counts(self) -> dict[str, int]:
        """Parameters split into the equivariant trunk and the set readout."""
        eq = sum(
            p.numel()
            for m in (self.module_a_layers, self.module_b_layers)
            for p in m.parameters()
        )
        hd = sum(p.numel() for p in self.head.parameters())
        return {"equivariant": eq, "head": hd, "total": eq + hd}

    # ── forward ──────────────────────────────────────────────────────────────

    def _cell_features(
        self,
        a_layer: torch.Tensor,
        b_layer: torch.Tensor,
        module_type: TargetModuleType,
    ) -> torch.Tensor:
        """One ``(module, layer)`` cell -> its flattened feature vector."""
        a_layers = self.module_a_layers[module_type.value]
        b_layers = self.module_b_layers[module_type.value]

        a_proc, b_proc = a_layer, b_layer
        n = len(a_layers)
        for i, (a_eq, b_eq) in enumerate(zip(a_layers, b_layers, strict=True)):
            a_proc = a_eq(a_proc)
            b_proc = b_eq(b_proc)
            if i != n - 1:
                b_proc, a_proc = _apply_activation(self.activation, b_proc, a_proc)

        w = b_proc @ a_proc.transpose(-1, -2)  # (batch, last, last)
        if self.feature_norm == "svals":
            # Singular values are invariant to w -> U w V^T, so this discards the
            # basis in which the cell is expressed and keeps only its spectrum.
            #
            # Computed as sqrt(eigvalsh(w w^T)) rather than svdvals(w): identical values
            # (max abs diff 1e-3 = float32 noise) but ~2.4x faster, because a symmetric
            # eigendecomposition has a much better batched path than a general SVD.
            # `svdvals` is called 112 times per forward on small (batch, 128, 128)
            # tensors and TIMED OUT the 45-minute cells on the first attempt.
            wf = w.float()
            ev = torch.linalg.eigvalsh(wf @ wf.transpose(-1, -2))
            feat = ev.clamp_min(0).sqrt().flip(-1)
            return feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
        feat = w.reshape(w.size(0), -1)
        if self.feature_norm == "l2":
            feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
        elif self.feature_norm == "zscore":
            mu = feat.mean(dim=-1, keepdim=True)
            sd = feat.std(dim=-1, keepdim=True)
            feat = (feat - mu) / (sd + 1e-8)
        elif self.feature_norm == "rank":
            # argsort-of-argsort -> the rank of each entry, mapped to [-1, 1].
            # A rank is a step function of the magnitudes, so it has ZERO gradient
            # w.r.t. them -- taken literally this detaches the equivariant trunk from
            # the loss entirely (verified: the graph has no grad_fn and .backward()
            # raises). Straight-through estimator: forward uses the ranks, backward
            # passes the identity, so the trunk still trains.
            ranks = feat.argsort(dim=-1).argsort(dim=-1).to(feat.dtype)
            ranks = ranks / (feat.size(-1) - 1) * 2.0 - 1.0
            std = feat / (feat.std(dim=-1, keepdim=True) + 1e-8)
            feat = std + (ranks - std).detach()
        elif self.feature_norm == "sqrt_l2":
            feat = feat.sign() * feat.abs().sqrt()
            feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
        return feat

    def forward(
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
                    self._cell_features(a[:, layer_idx], b[:, layer_idx], module_type)
                )
        # (batch, n_cells, cell_dim) — the token axis is ordered
        # module-major, layer-minor, matching ``cell_emb``.
        stacked = torch.stack(tokens, dim=1)
        if self.feature_norm == "l2_global":
            # One norm per adapter over ALL cells, scaled so the total norm matches what
            # per-cell ``l2`` produces (sqrt(n_cells)); only the *relative* loudness of
            # cells survives, which is exactly the factor being isolated.
            n_cells = stacked.size(1)
            denom = stacked.flatten(1).norm(dim=-1).view(-1, 1, 1) + 1e-8
            stacked = stacked * (n_cells ** 0.5) / denom
        return self.head(stacked)

    # ── persistence ──────────────────────────────────────────────────────────

    def hyperparameters(self) -> dict:
        return {
            "equivariant_layer_sizes": self.equivariant_layer_sizes,
            "head_layer_sizes": self.head_layer_sizes,
            "targeted_modules": [m.value for m in self.targeted_modules],
            "target_module_sizes": {
                k.value: list(v) for k, v in self.target_module_sizes.items()
            },
            "head_specs": [s.to_dict() for s in self.head_specs],
            "activation": self.activation,
            "head": self.head_type,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_attn_layers": self.n_attn_layers,
            "dropout": self.dropout,
            "head_dropout": self.head_dropout,
            "feature_norm": self.feature_norm,
            "llm_layers": self.llm_layers,
            "device": str(self.device),
            "seed": self.seed,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self.state_dict(), "hyperparameters": self.hyperparameters()},
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: torch.device | str | None = None) -> Self:
        ckpt = torch.load(Path(path), map_location="cpu")
        hp = ckpt["hyperparameters"]
        model = cls(
            equivariant_layer_sizes=hp["equivariant_layer_sizes"],
            targeted_modules={TargetModuleType(m) for m in hp["targeted_modules"]},
            target_module_sizes={
                TargetModuleType(k): tuple(v)
                for k, v in hp["target_module_sizes"].items()
            },
            head_specs=[MetaTargetSpec.from_dict(d) for d in hp["head_specs"]],
            activation=hp["activation"],
            head=hp["head"],
            head_layer_sizes=hp["head_layer_sizes"],
            d_model=hp["d_model"],
            n_heads=hp["n_heads"],
            n_attn_layers=hp["n_attn_layers"],
            dropout=hp["dropout"],
            head_dropout=hp.get("head_dropout", 0.0),
            feature_norm=hp["feature_norm"],
            llm_layers=hp["llm_layers"],
            device=device if device is not None else hp["device"],
            seed=hp.get("seed"),
        )
        model.load_state_dict(ckpt["state_dict"])
        return model


__all__ = [
    "ACTIVATIONS",
    "FEATURE_NORMS",
    "HEADS",
    "AttentionHead",
    "DeepSetsHead",
    "FlexibleLoRAMetaClassifier",
    "MLPHead",
    "glsum_activation",
]


# Keep ``MetaHeadType`` importable from here for callers building specs.
_ = MetaHeadType
