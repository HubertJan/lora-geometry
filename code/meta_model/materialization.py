"""Migrated from SRC/src/glad/meta_classifier/materialization.py.

Pluggable materialization backends for meta-classifier adapter loading.

The meta-classifier hot path (:func:`meta_model.dataset.create_safetensor_dataset`)
loads a LoRA adapter's ``adapter_model.safetensors`` and applies
``group_lora_weights_per_submodule`` on *every* ``__getitem__``, from NFS/VAST,
with ``shuffle=True`` and ~20 epochs — so each adapter is transformed ~20×
serially, blocking the train step.

This module materializes the *transform output* once and serves it cheaply.
Benchmarked in ``src/discoveries/adapter_cache_bench`` (450 × Llama-3.2-1B,
51.8 GB; ``results_1b_pool.csv``); the winners are shipped here:

* ``"tmp"``  — flatten → ``save_file`` to node-local ``$TMPDIR``, serve via mmap
               ``load_file`` + unflatten. **Default.** 5.4× faster than NFS-direct,
               lowest RAM (mmap not fully resident), scales to 8B pools (~0 RAM).
* ``"ram"``  — transform once, keep the nested dict in RAM. 8.8× faster, but ~20 GB
               resident for a 1B pool → won't fit large/8B pools. Opt-in.
* ``"none"`` — lazy :class:`GenericDataset` (transform on every access).

Key benchmark finding: **with a cache, ``num_workers=0`` is fastest** — the per-item
cost is tiny, so worker→main tensor-pickle IPC dominates. See
:func:`dataloader_materialize_kwargs`.

The transform output structure (see :mod:`meta_model.lora.weight_utils`)::

    { submodule_type: { LoraType.A: Tensor[L, r, in], LoraType.B: Tensor[L, out, r] } }

``L`` is inferred per pool by default (:func:`meta_model.lora.weight_utils.infer_layer_count`)
rather than hardcoded to 16.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Generic, Literal, Protocol, TypeVar

import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset

from meta_model.lora.types import LoraType
from meta_model.lora.weight_utils import (
    LoraWeight,
    group_lora_weights_per_submodule,
    infer_layer_count,
)

Materialize = Literal["tmp", "ram", "none"]

InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")
LabelType = TypeVar("LabelType")

# Separator encoding nested (submodule, lora_type) keys into a flat safetensors
# key. Neither part contains "|" (submodule names use ".", LoraType is "A"/"B"),
# so this round-trips unambiguously.
_KEY_SEP = "||"

LoraDict = dict[str, LoraWeight]  # {submodule: {LoraType: Tensor}}


class GenericDataset(Dataset, Generic[InputType, OutputType, LabelType]):
    """A generic dataset that applies an optional transform function to inputs.

    Backs ``materialize="none"``: the transform runs inside ``__getitem__``, so
    nothing is cached.

    Parameters:
        inputs: List of input samples of type InputType
        labels: List of labels of type LabelType
        transform_fn: Optional function to transform each input to OutputType at __getitem__
    """

    def __init__(
        self,
        inputs: list[InputType],
        labels: list[LabelType],
        transform_fn: Callable[[InputType], OutputType] | None = None,
    ) -> None:
        assert len(inputs) == len(labels), "inputs and labels must have the same length"
        self.inputs = inputs
        self.labels = labels
        self.transform_fn = transform_fn

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[InputType | OutputType, LabelType]:
        input_data = self.inputs[idx]
        if self.transform_fn is not None:
            return self.transform_fn(input_data), self.labels[idx]
        return input_data, self.labels[idx]


def resolve_layer_count(
    paths: list[str] | list[Path], layer_count: int | None
) -> int | None:
    """Resolve ``layer_count`` once per dataset, inferring from the first adapter.

    Inference is a header-only read, but doing it per ``__getitem__`` would still
    add an NFS round-trip per access — so callers resolve it here, at dataset
    construction, and pass a concrete ``int`` down.

    Returns ``None`` only for an empty pool (nothing to infer, nothing to load).
    """
    if layer_count is not None:
        return layer_count
    if not paths:
        return None
    return infer_layer_count(paths[0])


def default_transform(path: str, layer_count: int | None = None) -> LoraDict:
    """Canonical transform: load safetensors from *path* and group per submodule.

    ``layer_count=None`` infers the layer span from the file itself; pass an
    explicit int only to pin a pool to a known size.
    """
    resolved = layer_count if layer_count is not None else infer_layer_count(path)
    return group_lora_weights_per_submodule(
        weights=load_file(path),
        layer_id_order=list(range(resolved)),
    )


def flatten_lora_dict(d: LoraDict) -> dict[str, torch.Tensor]:
    """Flatten nested ``{submodule: {LoraType: Tensor}}`` → flat ``{str: Tensor}``."""
    flat: dict[str, torch.Tensor] = {}
    for submodule, lora in d.items():
        for lora_type, tensor in lora.items():
            flat[f"{submodule}{_KEY_SEP}{lora_type.value}"] = tensor.contiguous()
    return flat


def unflatten_lora_dict(flat: dict[str, torch.Tensor]) -> LoraDict:
    """Inverse of :func:`flatten_lora_dict`."""
    out: LoraDict = {}
    for key, tensor in flat.items():
        submodule, lora_type = key.rsplit(_KEY_SEP, 1)
        out.setdefault(submodule, {})[LoraType(lora_type)] = tensor
    return out


def _cast_lora_dict(d: LoraDict, cast: torch.dtype | None) -> LoraDict:
    if cast is None:
        return d
    return {
        sub: {lt: t.to(cast) for lt, t in lora.items()} for sub, lora in d.items()
    }


def _key_to_filename(key: str, suffix: str) -> str:
    """Stable, filesystem-safe filename for a cache key (the resolved adapter path)."""
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    return f"{digest}{suffix}"


def resolve_tmpdir() -> Path:
    """Node-local scratch dir (SLURM auto-wipes ``$TMPDIR`` between allocations)."""
    root = os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR") or "/tmp"
    return Path(root) / f"meta_clf_adapter_cache_{os.getpid()}"


# ── Store protocol + backends ────────────────────────────────────────────────


class Store(Protocol):
    def has(self, key: str) -> bool: ...
    def put(self, key: str, value: LoraDict) -> None: ...
    def get(self, key: str) -> LoraDict: ...


class RamStore:
    """Keep the nested transform output in an in-process dict (opt-in fast path)."""

    def __init__(self, cast: torch.dtype | None = None) -> None:
        self._data: dict[str, LoraDict] = {}
        self._cast = cast

    def has(self, key: str) -> bool:
        return key in self._data

    def put(self, key: str, value: LoraDict) -> None:
        self._data[key] = _cast_lora_dict(value, self._cast)

    def get(self, key: str) -> LoraDict:
        return self._data[key]


class TmpSafetensorsStore:
    """Flatten → ``save_file`` under ``$TMPDIR``; serve via mmap ``load_file`` + unflatten.

    Default backend: fastest at low RAM (mmap pages fault in lazily and are not
    fully resident). Keeps only a small ``key -> path`` index in RAM, so
    ``num_workers>0`` workers read the node-local NVMe with ~0 extra RAM.
    """

    def __init__(self, root: Path, cast: torch.dtype | None = None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, Path] = {}
        self._cast = cast

    def has(self, key: str) -> bool:
        return key in self._index

    def put(self, key: str, value: LoraDict) -> None:
        flat = flatten_lora_dict(value)
        if self._cast is not None:
            flat = {k: t.to(self._cast) for k, t in flat.items()}
        path = self.root / _key_to_filename(key, ".safetensors")
        save_file(flat, str(path))
        self._index[key] = path

    def get(self, key: str) -> LoraDict:
        # load_file mmaps the file; tensors materialize lazily on access. They are
        # only read downstream (torch.stack in collate, .to(device)), never
        # mutated in place, so read-only mmap tensors are safe.
        return unflatten_lora_dict(load_file(str(self._index[key])))

    def disk_bytes(self) -> int:
        return sum(p.stat().st_size for p in self._index.values())


# ── $TMPDIR guard ────────────────────────────────────────────────────────────


def _ensure_tmp_space(
    root: Path,
    n_items: int,
    first_output: LoraDict,
    cast: torch.dtype | None,
    safety: float = 1.2,
) -> None:
    """Raise a clear error if ``$TMPDIR`` cannot hold the estimated cache.

    Estimates total footprint from the first transformed adapter × *n_items* ×
    *safety*. LoRA outputs are near-uniform in size within a pool, so this is a
    good conservative estimate without transforming everything up front.
    """
    per_item = sum(
        (t.to(cast) if cast is not None else t).element_size() * t.nelement()
        for lora in first_output.values()
        for t in lora.values()
    )
    needed = int(per_item * n_items * safety)
    free = shutil.disk_usage(root).free
    if needed > free:
        raise RuntimeError(
            f"TmpSafetensorsStore needs ~{needed / 1e9:.1f} GB in {root} but only "
            f"{free / 1e9:.1f} GB free (est. {per_item / 1e6:.0f} MB/adapter × "
            f"{n_items} × {safety}). Use materialize='ram' (if it fits RAM) or "
            f"materialize='none', or point $TMPDIR at a larger node-local disk."
        )


# ── Materialized dataset + factory ───────────────────────────────────────────


class MaterializedDataset(Dataset):
    """Eagerly prefills *store* in ``__init__`` (single main-process pass).

    ``num_workers>0`` workers then inherit the RAM store copy-on-write / read the
    disk store from node-local NVMe — no per-worker re-materialization.
    ``__getitem__`` is a cheap store lookup.
    """

    def __init__(
        self,
        paths: list[str],
        labels: list,
        store: Store,
        transform: Callable[[str], LoraDict],
    ) -> None:
        assert len(paths) == len(labels), "paths and labels must have the same length"
        self.paths = paths
        self.labels = labels
        self.store = store
        for p in paths:
            if not store.has(p):
                store.put(p, transform(p))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        return self.store.get(self.paths[idx]), self.labels[idx]


def make_materialized_dataset(
    paths: list[str],
    labels: list,
    *,
    materialize: Materialize = "tmp",
    layer_count: int | None = None,
    cast: torch.dtype | None = None,
    tmp_root: Path | None = None,
) -> Dataset:
    """Single choke point both dataset builders call.

    Returns a lazy :class:`GenericDataset` for ``"none"``, else a
    :class:`MaterializedDataset` backed by the chosen store (prefilled eagerly).

    ``layer_count=None`` (default) infers the layer span once from the first
    adapter instead of assuming 16.
    """
    str_paths = [str(p) for p in paths]
    resolved_layers = resolve_layer_count(str_paths, layer_count)

    def transform(p: str) -> LoraDict:
        return default_transform(p, resolved_layers)

    if materialize == "none":
        return GenericDataset(
            inputs=str_paths,
            labels=labels,
            transform_fn=transform,
        )

    if materialize == "ram":
        store: Store = RamStore(cast=cast)
    elif materialize == "tmp":
        root = tmp_root or resolve_tmpdir()
        tmp_store = TmpSafetensorsStore(Path(root), cast=cast)  # mkdirs root
        if str_paths:
            _ensure_tmp_space(
                tmp_store.root, len(str_paths), transform(str_paths[0]), cast
            )
        store = tmp_store
    else:
        raise ValueError(f"unknown materialize={materialize!r}")

    return MaterializedDataset(str_paths, labels, store, transform)


def dataloader_materialize_kwargs(
    materialize: Materialize,
    device: str = "cuda",
) -> dict:
    """DataLoader kwargs matched to the backend.

    Benchmark: with a cache the per-item cost is tiny, so DataLoader workers
    *hurt* (worker→main tensor-pickle IPC dominates) — use ``num_workers=0``.
    ``pin_memory`` is cheap and helps the subsequent ``.to(cuda)``.
    """
    return {"num_workers": 0, "pin_memory": device == "cuda"}
