"""Migrated from SRC/src/glad/lora/cells.py.

Cell addressing: one ``(module, layer)`` site of a LoRA adapter.

A **cell** is the unit almost every interpretability discovery in this repo works on —
"the ``mlp.down_proj`` adapter at layer 15". It had been re-declared in at least five
shapes (``list[tuple[str, str, int]]``, ``list[tuple[str, str, TargetModuleType]]``,
``(short, key, layer, module_type)`` 4-tuples, plus three private key→type maps), so this
module gives it one type and one set of names.

The load-bearing fact here is :attr:`Cell.residual_side`.  For the logit lens you project
the singular vector that lives in the **residual stream** onto the vocabulary, and which
of ``u`` / ``v`` that is depends on whether the cell reads or writes the stream:

===================  ==============================================  ====
cell                 relationship to the residual stream             side
===================  ==============================================  ====
``mlp.gate_proj``    reads it (input)                                ``v``
``mlp.up_proj``      reads it (input)                                ``v``
``mlp.down_proj``    writes it (output)                              ``u``
``self_attn.o_proj`` writes it (output)                              ``u``
``self_attn.q_proj`` reads it (input)                                ``v``
``self_attn.k_proj`` reads it (input)                                ``v``
``self_attn.v_proj`` reads it (input)                                ``v``
===================  ==============================================  ====

Get it backwards and the lens projects the wrong vector and returns confident,
meaningless tokens — a silent failure, not a crash. The MLP rows were previously in
``imdb_backdoor_locality``, the ``o_proj`` row in ``attn_o_backdoor_mechanism``, and their
union (the first four) in ``emergent_misalignment_meta_classifier``; all three agreed.
The q/k/v rows are new here, and are the unambiguous consequence of q/k/v_proj taking the
residual stream as their input.

Note this map is **total**, where the old dicts were partial. A ``RESIDUAL_SIDE[key]``
lookup for an attention cell used to raise ``KeyError`` in the MLP-only discoveries; it
now returns the right answer instead. That is the intent, but it does remove a tripwire.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from safetensors.torch import load_file

from meta_model.lora.svd import svd_balance_layer
from meta_model.lora.types import LoraType, TargetModuleType
from meta_model.lora.weight_utils import (
    LoraWeight,
    extract_layer_id,
    group_lora_weights_per_submodule,
)

type ResidualSide = Literal["u", "v"]


# ── The 7 LoRA components ────────────────────────────────────────────────────
# (display short, safetensor submodule key, TargetModuleType, canonical label stem)
# Order matches the SHORT map used across the discoveries' plots.
COMPONENTS: list[tuple[str, str, TargetModuleType, str]] = [
    ("Attn Q", "self_attn.q_proj", TargetModuleType.Q_ATTENTION, "attnQ"),
    ("Attn K", "self_attn.k_proj", TargetModuleType.K_ATTENTION, "attnK"),
    ("Attn V", "self_attn.v_proj", TargetModuleType.V_ATTENTION, "attnV"),
    ("Attn O", "self_attn.o_proj", TargetModuleType.O_ATTENTION, "attnO"),
    ("MLP Gate", "mlp.gate_proj", TargetModuleType.GATE_MLP, "gate"),
    ("MLP Up", "mlp.up_proj", TargetModuleType.UP_MLP, "up"),
    ("MLP Down", "mlp.down_proj", TargetModuleType.DOWN_MLP, "down"),
]

MODULE_ORDER = [key for _, key, _, _ in COMPONENTS]
KEY_TO_SHORT = {key: short for short, key, _, _ in COMPONENTS}
SHORT_TO_KEY = {short: key for short, key, _, _ in COMPONENTS}
KEY_TO_MODULE_TYPE = {key: mt for _, key, mt, _ in COMPONENTS}
MODULE_TYPE_TO_KEY = {mt: key for _, key, mt, _ in COMPONENTS}

# Cells whose OUTPUT is the residual stream; everything else reads it as input.
_RESIDUAL_WRITERS = frozenset(
    {TargetModuleType.DOWN_MLP, TargetModuleType.O_ATTENTION}
)


@dataclass(frozen=True, slots=True)
class Cell:
    """One ``(module, layer)`` site of an adapter."""

    module: TargetModuleType
    layer: int

    def __post_init__(self) -> None:
        if self.layer < 0:
            raise ValueError(f"layer must be non-negative, got {self.layer}")

    @property
    def key(self) -> str:
        """Safetensor submodule key, e.g. ``"mlp.down_proj"``.

        This is what ``group_lora_weights_per_submodule`` keys on and what the meta
        classifier's input dict is keyed by.
        """
        return MODULE_TYPE_TO_KEY[self.module]

    @property
    def short(self) -> str:
        """Display name for plots and tables, e.g. ``"MLP Down"``."""
        return KEY_TO_SHORT[self.key]

    @property
    def label(self) -> str:
        """Canonical compact id, e.g. ``"down-L15"`` / ``"attnO-L13"``.

        Round-trips through :func:`parse_cell`. Historical result files use several
        other spellings of the same thing (``Down-L15``, ``Attn-O-L13``); those parse
        but are not re-emitted.
        """
        return f"{_MODULE_TYPE_TO_STEM[self.module]}-L{self.layer}"

    @property
    def residual_side(self) -> ResidualSide:
        """``"u"`` = this cell WRITES the residual stream (down/o); ``"v"`` = READS it.

        Picks which singular vector of the cell's ΔW (or of the meta-classifier template
        ``T = W_bᵀ·G·W_a``) lives in the residual stream, i.e. which one the logit lens
        may project onto the tied embedding. See the module docstring.
        """
        return "u" if self.module in _RESIDUAL_WRITERS else "v"

    def __str__(self) -> str:
        return self.label


_MODULE_TYPE_TO_STEM = {mt: stem for _, _, mt, stem in COMPONENTS}

# Back-compat / convenience map. Prefer ``Cell.residual_side``; this exists so the
# discoveries that already do ``RESIDUAL_SIDE[module_key]`` keep working.
RESIDUAL_SIDE: dict[str, ResidualSide] = {
    key: Cell(mt, 0).residual_side for _, key, mt, _ in COMPONENTS
}


# ── Label parsing ────────────────────────────────────────────────────────────

# Every spelling of a module seen in a persisted label or result file, lowercased.
_STEM_ALIASES: dict[str, TargetModuleType] = {}
for _short, _key, _mt, _stem in COMPONENTS:
    _bare = _key.split(".")[-1]              # "down_proj"
    _root = _bare.removesuffix("_proj")      # "down"
    for _alias in (
        _stem,                               # "down" / "attnO"
        _key,                                # "mlp.down_proj"
        _bare,                               # "down_proj"
        _root,                               # "down"
        _short,                              # "MLP Down"
        _short.replace(" ", "-"),            # "MLP-Down" / "Attn-O"
        _short.replace(" ", "_"),
    ):
        _STEM_ALIASES[_alias.lower()] = _mt
del _short, _key, _mt, _stem, _bare, _root, _alias

_LABEL_RE = re.compile(r"^(?P<stem>.+?)[-_]?[Ll](?P<layer>\d+)$")


def parse_cell(label: str) -> Cell:
    """``"down-L15"`` → ``Cell(TargetModuleType.DOWN_MLP, 15)``.

    Tolerant of the spellings that ended up in historical CSVs and W&B artifacts —
    ``down-L15``, ``Down-L15``, ``Attn-O-L13``, ``attnO-L13``, ``mlp.down_proj-L15`` —
    all parse to the same cell. :attr:`Cell.label` always re-emits the canonical form.

    Pool-qualified labels (``ibl-gate-L4``, ``pol-down-L15``, ``trusted-gate-L4``) are
    **rejected**: the prefix identifies a *pool*, not a cell, and silently swallowing it
    would let two different pools' results collapse onto one key. Split it off first.
    """
    match = _LABEL_RE.match(label.strip())
    if match is None:
        raise ValueError(
            f"Cannot parse cell label {label!r}; expected e.g. 'down-L15' or 'attnO-L13'."
        )
    stem, layer = match["stem"].lower().rstrip("-_"), int(match["layer"])
    if (module := _STEM_ALIASES.get(stem)) is None:
        suffix = ""
        for known in _STEM_ALIASES:
            if stem.endswith(known) and stem != known:
                prefix = stem[: -len(known)].rstrip("-_")
                suffix = (
                    f" It looks pool-qualified: strip the {prefix!r} prefix and keep it"
                    " alongside the cell rather than inside the label."
                )
                break
        raise ValueError(f"Unknown module {match['stem']!r} in cell label {label!r}.{suffix}")
    return Cell(module, layer)


# ── Loading + slicing ────────────────────────────────────────────────────────


def load_grouped(
    path: str | Path,
    *,
    layer_count: int | None = None,
) -> dict[str, LoraWeight]:
    """Load one adapter's safetensors into the standard grouped dict.

    Returns ``{submodule_key: {LoraType.A: (n_layers, r, d_in),
                               LoraType.B: (n_layers, d_out, r)}}``.

    Args:
        path: the adapter's ``.safetensors`` file.
        layer_count: number of layer slots. ``None`` (default) infers it from the file
            as ``max_layer_id + 1``, which is what you want unless you are deliberately
            slicing.

    Inference matters because the old ``layer_id_order=list(range(16))`` hardcode fails
    *asymmetrically*: an adapter with more layers than assumed raises ``ValueError`` from
    ``list.index()``, but one with fewer is silently zero-padded to 16 — so the quiet half
    only ever showed up as bad results. Use :func:`meta_model.lora.weight_utils.infer_layer_count`
    when you need the count without paying to load the weights.
    """
    weights = load_file(str(path))
    if layer_count is None:
        layer_ids = [
            layer_id
            for name in weights
            if (layer_id := extract_layer_id(name)) is not None
        ]
        if not layer_ids:
            raise ValueError(
                f"No layer-indexed LoRA weights found in {path}. "
                "Expected keys like '...model.layers.<i>....lora_A.weight'."
            )
        layer_count = max(layer_ids) + 1
    return group_lora_weights_per_submodule(
        weights, layer_id_order=list(range(layer_count))
    )


def cell_dict(
    grouped: dict[str, LoraWeight],
    cell: Cell,
) -> dict[str, LoraWeight]:
    """Slice ``grouped`` down to one cell, rank kept at its native value.

    Returns ``{cell.key: {A: (1, r, d_in), B: (1, d_out, r)}}`` — one layer at index 0,
    consumed by a meta classifier built with ``llm_layer_count=1`` and ``target_modules``
    the matching singleton.
    """
    sub = grouped[cell.key]
    return {
        cell.key: {
            LoraType.A: sub[LoraType.A][cell.layer : cell.layer + 1].clone(),
            LoraType.B: sub[LoraType.B][cell.layer : cell.layer + 1].clone(),
        }
    }


def cell_rank_dict(
    grouped: dict[str, LoraWeight],
    cell: Cell,
    k: int,
) -> dict[str, LoraWeight]:
    """One cell, SVD-balanced, sliced to singular direction ``k`` (rank-1).

    Returns ``{cell.key: {A: (1, 1, d_in), B: (1, d_out, 1)}}`` — a rank-1 adapter for
    the same single-cell model config (the rank axis is just length 1). ``k`` indexes
    singular values in descending order, so ``k=0`` is the dominant direction.
    """
    sub = grouped[cell.key]
    a_bal, b_bal, _s = svd_balance_layer(
        sub[LoraType.A][cell.layer], sub[LoraType.B][cell.layer]
    )
    return {
        cell.key: {
            LoraType.A: a_bal[k : k + 1].unsqueeze(0),      # (1, 1, d_in)
            LoraType.B: b_bal[:, k : k + 1].unsqueeze(0),   # (1, d_out, 1)
        }
    }


def cell_rank_count(grouped: dict[str, LoraWeight], cell: Cell) -> int:
    """Native LoRA rank of a cell (number of singular directions available)."""
    return int(grouped[cell.key][LoraType.A][cell.layer].shape[0])


def cell_delta_w(grouped: dict[str, LoraWeight], cell: Cell) -> torch.Tensor:
    """The cell's full update ``ΔW = B·A``, shape ``(d_out, d_in)``."""
    sub = grouped[cell.key]
    return sub[LoraType.B][cell.layer].float() @ sub[LoraType.A][cell.layer].float()
