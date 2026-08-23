"""Migrated from SRC/src/glad/meta_classifier/dataset.py.

Adapter-pool datasets, targets and dataloaders for the meta classifier.

The data half of :mod:`meta_model`: it turns a pool's metadata
DataFrame (one row per adapter, carrying a ``safetensor_path``) into the
``(lora_dict, per-head targets)`` batches the model consumes.

Three invariants are enforced here rather than left to each caller, because each
one has already cost a debugging cycle when it was re-derived by copy:

* **Layer count is inferred, not assumed.** ``layer_count=None`` reads the span
  off the adapter itself (:func:`meta_model.lora.weight_utils.infer_layer_count`)
  instead of hardcoding ``range(16)``, which silently zero-pads a shorter model.
* **Splits are shuffled and (optionally) stratified.**
  :func:`stratified_pool_split` — a plain ``df.sample(fraction=1.0, seed=...)``
  is the *identity permutation* in polars, so the naive inline split hands the
  file-order tail of the pool to validation.
* **Rank uniformity is checked up front.** :func:`assert_uniform_rank` fails with
  a readable message instead of letting ``torch.stack`` blow up mid-epoch and
  forcing a fallback to ``batch_size=1``.
* **Missing targets are surfaced, not swallowed.** Rows whose target maps to
  the ``-1`` / ``NaN`` missing sentinel are masked out of the loss and metrics
  by design, but the loader builders count them up front
  (:func:`count_missing_targets`) and warn (or raise, via ``on_missing``) —
  a dropped run-config patch upstream otherwise turns a declared 200/100 pool
  into a silent 199/99 one.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar

import polars as pl
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset

from meta_model.lora.weight_utils import infer_rank
from meta_model.heads import MISSING_CLASS_LABEL, MetaTargetSpec
from meta_model.materialization import (
    GenericDataset,
    Materialize,
    dataloader_materialize_kwargs,
    make_materialized_dataset,
)
from meta_model.helpers import batch_lora_dicts

T = TypeVar("T")

SAFETENSOR_FILENAME = "adapter_model.safetensors"
PATH_COLUMN = "safetensor_path"

__all__ = [
    "PATH_COLUMN",
    "SAFETENSOR_FILENAME",
    "GenericDataset",
    "SafetensorDataset",
    "assert_uniform_rank",
    "attach_safetensor_paths",
    "build_eval_dataloader",
    "build_pool_dataloaders",
    "build_target_fn",
    "count_missing_targets",
    "create_safetensor_dataset",
    "load_adapter_pool_metadata",
    "multihead_collate_fn",
    "stratified_pool_split",
]


class SafetensorDataset(Dataset, Generic[T]):
    """Raw ``load_file`` per item — no grouping, no cache.

    Kept for callers that want the flat weight dict; the meta-classifier path
    goes through :func:`create_safetensor_dataset` instead.
    """

    def __init__(self, paths: list[Path], labels: list[T]) -> None:
        assert len(paths) == len(labels), "paths and labels must have the same length"
        self.paths = paths
        self.labels = labels

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[dict, T]:
        tensors = load_file(self.paths[idx])
        return tensors, self.labels[idx]


# ── Pool metadata ────────────────────────────────────────────────────────────


def load_adapter_pool_metadata(
    adapter_pool_path: Path,
) -> pl.DataFrame:
    """Read adapter pool metadata and resolve per-sample safetensor paths.

    Reads ``metadata.parquet`` from *adapter_pool_path*, keeps the original
    ``__key__`` column, and adds a ``safetensor_path`` column pointing to each
    sample's ``adapter_model.safetensors`` file inside the ``adapters/``
    sub-directory.

    The returned DataFrame can be filtered or subsetted before being passed to
    :func:`create_safetensor_dataset`.

    (Migration note: the original also accepted an ``llm_pipeline`` ``Artifact``
    to stamp a W&B qualified name into every row — dropped, along with its
    import, since the standalone project has no artifact store.)
    """
    df = pl.read_parquet(adapter_pool_path / "metadata.parquet")

    adapters_root = adapter_pool_path / "adapters"
    df = df.with_columns(
        pl
        .col("__key__")
        .map_elements(
            lambda key: str(adapters_root / key / SAFETENSOR_FILENAME),
            return_dtype=pl.Utf8,
        )
        .alias(PATH_COLUMN)
    )

    return df


def attach_safetensor_paths(
    df: pl.DataFrame, pool_dir: Path
) -> pl.DataFrame:
    """Add the ``safetensor_path`` column to an in-memory metadata frame.

    Same resolution as :func:`load_adapter_pool_metadata` (``pool_dir/adapters/
    <__key__>/adapter_model.safetensors``) but for a frame the caller already
    holds (e.g. one passed straight into :func:`train_meta_model`), so no
    parquet read happens.  Idempotent: an existing ``safetensor_path`` is kept.
    """
    if PATH_COLUMN in df.columns:
        return df
    adapters_root = Path(pool_dir) / "adapters"
    return df.with_columns(
        pl
        .col("__key__")
        .map_elements(
            lambda key: str(adapters_root / key / SAFETENSOR_FILENAME),
            return_dtype=pl.Utf8,
        )
        .alias(PATH_COLUMN)
    )


def _require_path_column(df: pl.DataFrame, path_column: str) -> None:
    if path_column not in df.columns:
        raise ValueError(
            f"metadata_df must contain a {path_column!r} column. "
            "Call load_adapter_pool_metadata() (or the ref-to-df helper) first."
        )


# ── Shape guards ─────────────────────────────────────────────────────────────


def assert_uniform_rank(df: pl.DataFrame, *, path_column: str = PATH_COLUMN) -> int:
    """Assert every adapter in the pool shares one LoRA rank, and return it.

    Batching stacks per-adapter tensors with ``torch.stack``, which needs
    identical shapes — a mixed-rank pool otherwise fails deep inside the collate
    with a shape error, and the usual workaround has been to drop to
    ``batch_size=1``. Reads only safetensors headers (one per adapter).
    """
    _require_path_column(df, path_column)
    paths = df[path_column].to_list()
    if not paths:
        raise ValueError("Cannot determine LoRA rank of an empty pool.")

    by_rank: dict[int, list[str]] = {}
    for path in paths:
        by_rank.setdefault(infer_rank(path), []).append(str(path))

    if len(by_rank) > 1:
        detail = "; ".join(
            f"rank {rank}: {len(group)} adapters (e.g. {group[0]})"
            for rank, group in sorted(by_rank.items())
        )
        raise ValueError(
            f"Pool mixes LoRA ranks — batching requires a uniform rank. {detail}. "
            "Filter the pool to one rank, or build per-rank dataloaders."
        )

    return next(iter(by_rank))


# ── Datasets ─────────────────────────────────────────────────────────────────


def create_safetensor_dataset(
    metadata_df: pl.DataFrame,
    label_fn: Callable[[dict[str, Any]], T],
    *,
    materialize: Materialize = "tmp",
    layer_count: int | None = None,
    cast: torch.dtype | None = None,
    path_column: str = PATH_COLUMN,
) -> Dataset[T]:
    """Build a dataset from a prepared metadata DataFrame.

    *metadata_df* must contain a ``safetensor_path`` column (as produced by
    :func:`load_adapter_pool_metadata`).  Every row is converted to a plain
    ``dict`` and passed to *label_fn* to produce the corresponding label.

    *materialize* selects the caching backend (see
    :mod:`meta_model.materialization`): ``"tmp"`` (default;
    flatten→mmap safetensors on node-local ``$TMPDIR``), ``"ram"`` (opt-in
    in-RAM), or ``"none"`` (lazy transform on every access).  ``layer_count``
    defaults to ``None`` = infer once from the first adapter (was a hardcoded
    16). ``cast`` optionally down-casts the cached tensors (e.g. bf16).
    """
    _require_path_column(metadata_df, path_column)

    paths: list[str] = []
    labels: list[T] = []

    for row in metadata_df.iter_rows(named=True):
        paths.append(str(row[path_column]))
        labels.append(label_fn(row))

    return make_materialized_dataset(
        paths,
        labels,
        materialize=materialize,
        layer_count=layer_count,
        cast=cast,
    )


# ── Targets + collation ──────────────────────────────────────────────────────


def build_target_fn(
    specs: list[MetaTargetSpec],
) -> Callable[[dict[str, Any]], dict[str, float | int]]:
    """Build a per-row label function producing one target per head.

    For each spec, the value is read from ``row[spec.column]`` (pool columns are
    Utf8 strings) and converted to the head's target space:

    * classification → ``mapping[str(value)]``; an unmapped / missing value
      becomes the ``-1`` missing sentinel (masked out of the loss),
    * regression → ``float(value)``; a missing / ``None`` / unparseable value
      becomes ``NaN`` (masked out of the loss).
    """

    def target_fn(row: dict[str, Any]) -> dict[str, float | int]:
        targets: dict[str, float | int] = {}
        for spec in specs:
            raw = row.get(spec.column)
            if spec.is_classification:
                mapping = spec.mapping_dict
                targets[spec.name] = mapping.get(str(raw), MISSING_CLASS_LABEL)
            else:
                try:
                    value = float(raw)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    value = math.nan
                targets[spec.name] = value
        return targets

    return target_fn


OnMissing = Literal["warn", "raise", "ignore"]


def count_missing_targets(
    df: pl.DataFrame,
    specs: list[MetaTargetSpec],
) -> dict[str, int]:
    """Per-head count of rows whose target maps to the missing sentinel.

    Mirrors :func:`build_target_fn` exactly: a classification target is missing
    when ``str(row[spec.column])`` is not a mapping key (which includes ``None``
    from a null or absent column), a regression target when the value is
    ``None`` / unparseable.
    """
    target_fn = build_target_fn(specs)
    counts = dict.fromkeys((spec.name for spec in specs), 0)
    for row in df.iter_rows(named=True):
        targets = target_fn(row)
        for spec in specs:
            value = targets[spec.name]
            if spec.is_classification:
                is_missing = value == MISSING_CLASS_LABEL
            else:
                is_missing = math.isnan(value)
            if is_missing:
                counts[spec.name] += 1
    return counts


def _check_missing_targets(
    df: pl.DataFrame,
    specs: list[MetaTargetSpec],
    on_missing: OnMissing,
    context: str,
) -> None:
    if on_missing not in ("warn", "raise", "ignore"):
        raise ValueError(
            f"on_missing must be 'warn', 'raise' or 'ignore', got {on_missing!r}"
        )
    if on_missing == "ignore" or len(df) == 0:
        return
    dropped = {name: c for name, c in count_missing_targets(df, specs).items() if c}
    if not dropped:
        return
    detail = ", ".join(f"{name}: {c}/{len(df)} rows" for name, c in dropped.items())
    message = (
        f"{context}: targets mapping to the missing sentinel are masked out of "
        f"the loss and metrics without further notice ({detail}). A dropped "
        f"run-config patch upstream looks exactly like this — a pool declared "
        f"200/100 that actually trains as 199/99. Fix the pool metadata, or "
        f"pass on_missing='ignore' if heterogeneous multi-pool targets make "
        f"missing labels expected here."
    )
    if on_missing == "raise":
        raise ValueError(message)
    warnings.warn(message, stacklevel=3)


def multihead_collate_fn(
    batch: list[tuple[dict, dict[str, float | int]]],
    specs: list[MetaTargetSpec],
) -> tuple[dict, dict[str, torch.Tensor]]:
    """Collate ``(lora_dict, target_dict)`` items for the multi-head model.

    The LoRA weight dicts are stacked into a single batched input dict (so the
    train / eval loops can call ``model(inputs)`` directly), and per-head targets
    are stacked into ``float32`` tensors for regression heads and ``long``
    tensors for classification heads.
    """
    inputs = batch_lora_dicts([item[0] for item in batch])
    labels = [item[1] for item in batch]

    targets: dict[str, torch.Tensor] = {}
    for spec in specs:
        values = [row[spec.name] for row in labels]
        if spec.is_classification:
            targets[spec.name] = torch.tensor(values, dtype=torch.long)
        else:
            targets[spec.name] = torch.tensor(values, dtype=torch.float32)
    return inputs, targets


# ── Splitting ────────────────────────────────────────────────────────────────

_SPLIT_INDEX = "__split_index"


def stratified_pool_split(
    df: pl.DataFrame,
    *,
    by: str | Sequence[str] | None,
    val_frac: float,
    seed: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split a pool into ``(train, val)`` — shuffled, and stratified over *by*.

    ``by=None`` shuffles without stratifying; passing one or more column names
    additionally guarantees every stratum is represented on both sides.

    Why this exists: ``df.sample(fraction=1.0, seed=...)`` does **not** permute
    in polars (``shuffle=True`` is required), so the inline
    ``shuffled[:n_train]`` split this replaces handed validation the file-order
    tail of the pool — which on a 17-class pool put ~4 classes in val and made
    ``train_val_split=1.0`` the standing workaround.

    Strata with fewer than 3 rows contribute nothing to validation (they stay
    whole in train): a 1-2 row stratum cannot be split without either starving
    training or producing a meaningless single-sample validation estimate.
    """
    if not 0.0 <= val_frac < 1.0:
        raise ValueError(f"val_frac must be in [0, 1), got {val_frac}")

    shuffled = df.sample(fraction=1.0, shuffle=True, seed=seed)
    if val_frac == 0.0 or len(shuffled) == 0:
        return shuffled, shuffled.clear()

    indexed = shuffled.with_row_index(_SPLIT_INDEX)

    if by is None:
        n_val = round(val_frac * len(indexed))
        val_indices = set(indexed[_SPLIT_INDEX].to_list()[:n_val])
    else:
        by_cols = [by] if isinstance(by, str) else list(by)
        missing = [c for c in by_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Cannot stratify on missing column(s) {missing}. "
                f"Available: {df.columns}"
            )
        val_indices = set()
        for _, group in indexed.group_by(by_cols, maintain_order=True):
            n = len(group)
            # Rows are already shuffled, so the first k of each stratum is a
            # uniform random subset of it.
            k = max(1, round(n * val_frac)) if n >= 3 else 0
            val_indices.update(group[_SPLIT_INDEX].to_list()[:k])

    is_val = pl.col(_SPLIT_INDEX).is_in(val_indices)
    val_df = indexed.filter(is_val).drop(_SPLIT_INDEX)
    train_df = indexed.filter(~is_val).drop(_SPLIT_INDEX)
    return train_df, val_df


# ── Dataloaders ──────────────────────────────────────────────────────────────


def build_eval_dataloader(
    df: pl.DataFrame,
    specs: list[MetaTargetSpec],
    *,
    batch_size: int = 4,
    materialize: Materialize = "none",
    layer_count: int | None = None,
    device: str = "cuda",
    path_column: str = PATH_COLUMN,
    on_missing: OnMissing = "warn",
) -> DataLoader:
    """A single-pass evaluation loader over *df* (held-out / test pools).

    Defaults to ``materialize="none"``: an eval pool is iterated once, so
    prefilling a cache would pay the write+read with nothing to amortize it
    over. Train/val pools are re-read every epoch and want the opposite —
    :func:`build_pool_dataloaders`.

    *on_missing* controls what happens when rows map to the missing-target
    sentinel (see :func:`count_missing_targets`): ``"warn"`` (default),
    ``"raise"``, or ``"ignore"`` for intentionally heterogeneous pools.
    """
    _check_missing_targets(df, specs, on_missing, "build_eval_dataloader")
    dataset = create_safetensor_dataset(
        df,
        build_target_fn(specs),
        materialize=materialize,
        layer_count=layer_count,
        path_column=path_column,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        # partial, not a lambda: picklable, so a caller that overrides
        # num_workers>0 still works.
        collate_fn=partial(multihead_collate_fn, specs=specs),
        **dataloader_materialize_kwargs(materialize, device),
    )


def build_pool_dataloaders(
    df: pl.DataFrame,
    specs: list[MetaTargetSpec],
    *,
    materialize: Materialize = "tmp",
    seed: int = 42,
    val_frac: float = 0.2,
    stratify_by: str | Sequence[str] | None = None,
    batch_size: int = 4,
    layer_count: int | None = None,
    cast: torch.dtype | None = None,
    device: str = "cuda",
    check_rank: bool = True,
    path_column: str = PATH_COLUMN,
    on_missing: OnMissing = "warn",
) -> tuple[DataLoader, DataLoader | None]:
    """Train/val loaders for one adapter pool, with the known-good defaults baked in.

    Bakes in four results that are otherwise re-derived per caller:

    * the split is shuffled and, with *stratify_by*, stratified
      (:func:`stratified_pool_split`),
    * the layer count is inferred from the adapters unless pinned,
    * ``num_workers=0`` whenever a cache is in play — the benchmark's finding
      that worker→main tensor-pickle IPC dominates the tiny cached per-item cost
      (:func:`~meta_model.materialization.dataloader_materialize_kwargs`),
    * rows whose target maps to the missing sentinel are counted up front and
      surfaced per *on_missing* (``"warn"`` default / ``"raise"`` /
      ``"ignore"``) instead of silently shrinking the effective pool.

    Returns ``(train_loader, None)`` when *val_frac* is 0 or the pool is too
    small to yield any validation rows, so callers can skip validation without
    inspecting the frame themselves.

    *check_rank* is skipped at ``batch_size=1`` (nothing is stacked across
    adapters there, so a mixed-rank pool is a valid configuration).
    """
    _require_path_column(df, path_column)
    # batch_size == 1 never stacks across adapters, so a mixed-rank pool is
    # legitimate there — don't reject a configuration that works.
    if check_rank and batch_size > 1 and len(df) > 0:
        assert_uniform_rank(df, path_column=path_column)

    _check_missing_targets(df, specs, on_missing, "build_pool_dataloaders")

    train_df, val_df = stratified_pool_split(
        df, by=stratify_by, val_frac=val_frac, seed=seed
    )

    label_fn = build_target_fn(specs)
    collate_fn = partial(multihead_collate_fn, specs=specs)
    loader_kwargs = dataloader_materialize_kwargs(materialize, device)

    train_loader = DataLoader(
        create_safetensor_dataset(
            train_df,
            label_fn,
            materialize=materialize,
            layer_count=layer_count,
            cast=cast,
            path_column=path_column,
        ),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        **loader_kwargs,
    )

    if len(val_df) == 0:
        return train_loader, None

    val_loader = DataLoader(
        create_safetensor_dataset(
            val_df,
            label_fn,
            materialize=materialize,
            layer_count=layer_count,
            cast=cast,
            path_column=path_column,
        ),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        **loader_kwargs,
    )
    return train_loader, val_loader
