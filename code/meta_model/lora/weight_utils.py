"""Migrated from SRC/src/glad/lora/weight_utils.py.

LoRA weight name parsing and grouping utilities.

Migrated from paretune (weight_name_utils.py, file_weights_utils.py).
"""

from __future__ import annotations

import re
from pathlib import Path

import torch
from safetensors import safe_open

from meta_model.lora.types import LoraType

# ── Type aliases ─────────────────────────────────────────────────────────────

type LoraWeight = dict[LoraType, torch.Tensor]


# ── Name parsing helpers ─────────────────────────────────────────────────────


QUERY_PROJECTION_LAYER_NAME = "self_attn.q_proj"
VALUE_PROJECTION_LAYER_NAME = "self_attn.v_proj"


def extract_layer_id(component_name: str) -> int | None:
    found_id = re.search(r"\.layers\.(\d+)\.", component_name)
    if found_id is None:
        return None
    return int(found_id.group(1))


def extract_submodule_type(weight_name: str) -> str | None:
    match = re.search(
        r"\.layers\.\d+\.(.+?)\.lora(?:_[AB])?\.weight",
        weight_name,
    )
    if not match:
        return None
    return match.group(1)


def extract_lora_type(weight_name: str) -> LoraType | None:
    results = re.search(r"lora_([AB])", weight_name)
    if not results:
        return None
    r = results.group(1)
    match r:
        case "A":
            return LoraType.A
        case "B":
            return LoraType.B
        case _:
            return None


# ── Shape inference (header-only reads) ──────────────────────────────────────


def read_lora_tensor_shapes(safetensor_path: str | Path) -> dict[str, list[int]]:
    """Read every tensor's shape from a safetensors file without loading data.

    ``safe_open`` + ``get_slice(...).get_shape()`` reads only the file header, so
    this stays cheap on NFS even across a whole pool.
    """
    with safe_open(str(safetensor_path), framework="pt") as f:
        return {key: list(f.get_slice(key).get_shape()) for key in f.keys()}


def infer_layer_count(safetensor_path: str | Path) -> int:
    """Number of transformer layers an adapter's weights span (``max_layer_id + 1``).

    This is the value :func:`group_lora_weights_per_submodule` needs as
    ``len(layer_id_order)``: it indexes the stacked tensor *by layer id*, so an
    adapter that only targets layers 8-15 of a 16-layer model still needs 16
    slots, not 8.

    Replaces the ``layer_id_order=range(16)`` hardcode. Getting it wrong fails
    asymmetrically — a model with *more* layers than assumed raises ``ValueError``
    from ``list.index()``, but one with *fewer* is silently zero-padded — so the
    quiet half only ever surfaced as bad results.
    """
    layer_ids = [
        layer_id
        for name in read_lora_tensor_shapes(safetensor_path)
        if (layer_id := extract_layer_id(name)) is not None
        and extract_lora_type(name) is not None
    ]
    if not layer_ids:
        raise ValueError(
            f"No layer-indexed LoRA weights found in {safetensor_path}. "
            "Expected keys like '...model.layers.<i>....lora_A.weight'."
        )
    return max(layer_ids) + 1


def infer_rank(safetensor_path: str | Path) -> int:
    """LoRA rank ``r`` of an adapter, read from its ``lora_A`` / ``lora_B`` shapes.

    ``lora_A`` is ``(r, in_features)`` and ``lora_B`` is ``(out_features, r)``.
    Raises ``ValueError`` if the file mixes ranks across submodules — batching
    such an adapter is impossible anyway (see
    :func:`meta_model.dataset.assert_uniform_rank`).
    """
    ranks: set[int] = set()
    for name, shape in read_lora_tensor_shapes(safetensor_path).items():
        if extract_layer_id(name) is None or len(shape) != 2:
            continue
        match extract_lora_type(name):
            case LoraType.A:
                ranks.add(shape[0])
            case LoraType.B:
                ranks.add(shape[1])
            case _:
                continue
    if not ranks:
        raise ValueError(
            f"No LoRA A/B weights found in {safetensor_path}; cannot infer rank."
        )
    if len(ranks) > 1:
        raise ValueError(
            f"{safetensor_path} mixes LoRA ranks {sorted(ranks)} across submodules."
        )
    return ranks.pop()


# ── Weight grouping ──────────────────────────────────────────────────────────


def group_lora_weights_per_submodule(
    weights: dict[str, torch.Tensor],
    layer_id_order: list[int],
) -> dict[str, LoraWeight]:
    """Group LoRA weights by submodule type, stacking layers into 4D tensors.

    Args:
        weights: Flat dict of weight name → tensor (e.g. from safetensors).
        layer_id_order: Ordered list of layer indices to include.

    Returns:
        Nested dict: {submodule_type: {LoraType: stacked_tensor}}.
    """
    weights_grouped_by_layer_type: dict[
        str,
        dict[LoraType, dict[int, torch.Tensor]],
    ] = {}
    max_layer_id = len(layer_id_order)

    for name, lora_weight in weights.items():
        layer_id = extract_layer_id(name)
        if layer_id is None:
            continue

        submodule_type = extract_submodule_type(name)
        if submodule_type is None:
            continue

        lora_type = extract_lora_type(name)
        if lora_type is None:
            continue

        if submodule_type not in weights_grouped_by_layer_type:
            weights_grouped_by_layer_type[submodule_type] = {}
        if lora_type not in weights_grouped_by_layer_type[submodule_type]:
            weights_grouped_by_layer_type[submodule_type][lora_type] = {}

        weights_grouped_by_layer_type[submodule_type][lora_type][layer_id] = lora_weight

    result: dict[str, LoraWeight] = {}
    for submodule_type, layer_weights_per_lora in weights_grouped_by_layer_type.items():
        for lora_type, layer_weights in layer_weights_per_lora.items():
            first_weight = next(iter(layer_weights.values()))
            num_layers = max_layer_id
            weight_shape = first_weight.shape
            stacked_tensor = torch.zeros(
                (num_layers, *weight_shape),
                dtype=first_weight.dtype,
            )
            for layer_id, lora_weight in layer_weights.items():
                index = layer_id_order.index(layer_id)
                stacked_tensor[index] = lora_weight

                if submodule_type not in result:
                    result[submodule_type] = {}
                result[submodule_type][lora_type] = stacked_tensor

    return result
