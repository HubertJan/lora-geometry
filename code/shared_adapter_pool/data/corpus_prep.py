"""Plain (untracked) corpus preparation: raw HF -> typed pool -> capped splits.

(migrated from src/llm_pipeline/llm_datasets/corpus_prep.py)

The original chained eight W&B-cached ``track_*`` runs; the standalone pool needs
only the row-level computation those runs performed, so this module reduces the
chain to two plain functions:

``prepare_corpus(corpus)``
    ``raw -> merged-all -> flat-all -> typed-full [-> drop-none]``. Loads the HF
    corpus, merges the source splits into one pool, drops the splits we must not
    train on, maps each raw row to the typed schema (canonical label strings via
    ``LabelDecoding.decode``), and drops rows whose label decoded to ``None``.
    Returns the typed pool as a ``TypedDataset``.

``task_splits(typed, seed=42, max_train=20000, max_test=2000)``
    shuffled 80/10/10 split of the typed pool, each split head-capped. Returns a
    ``TaskSplits`` with ``train`` / ``validation`` / ``test`` TypedDatasets.

The split ratios, seed and caps match the originals so the produced train/test
boundary is reproducible from ``(corpus, seed, caps)`` alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shared_adapter_pool.data.sharding import FirstNStrategy
from shared_adapter_pool.data.transforms import (
    ExtractSplit,
    MapSchema,
    MergeSplits,
    SampleDataset,
    SplitDataset,
)
from shared_adapter_pool.data.typed_dataset import TypedDataset, TypedDatasetDict

__all__ = ["TaskSplits", "prepare_corpus", "task_splits"]


def _corpus_typed_transform(corpus: Any) -> MapSchema:
    """The raw -> typed-row transform for ``corpus`` (column-mapped corpora only).

    The pool's 15 datasets are all ``ColumnMappedCorpus`` / ``MappedCorpus``
    shapes, which present a 1:1 ``corpus.row_mapper``. ``FactoryCorpus`` (whole-
    row cross-column work) is not used here and is rejected explicitly.
    """
    from shared_adapter_pool.data.definitions._definition import (
        ColumnMappedCorpus,
        MappedCorpus,
    )

    if isinstance(corpus, (ColumnMappedCorpus, MappedCorpus)):
        return MapSchema(schema=corpus.messages, mapper=corpus.row_mapper)
    raise TypeError(
        f"{corpus.slug}: unsupported corpus shape {type(corpus).__name__}; "
        "the standalone pool prepares only column-mapped corpora."
    )


def _load_raw(corpus: Any):
    """Load the corpus's raw HuggingFace ``DatasetDict`` (all source splits)."""
    import os

    from datasets import load_from_disk

    from common import env

    if corpus.local_data_path is not None:
        return load_from_disk(os.path.expandvars(corpus.local_data_path))

    from datasets import load_dataset

    return load_dataset(
        corpus.hf_name,
        corpus.hf_config,
        revision=corpus.hf_revision,
        token=env.HF_TOKEN,
    )


def prepare_corpus(corpus: Any) -> TypedDataset:
    """``raw -> merged-all -> flat-all -> typed-full [-> drop-none]``.

    ``corpus`` is a ``Corpus`` (e.g. ``DEFINITION.corpus``). Returns the typed
    pool as a single ``TypedDataset`` carrying canonical label strings in the
    corpus's decoded label field.
    """
    raw = _load_raw(corpus)

    merged = MergeSplits(
        merge_map={"all": list(corpus.splits_to_merge)},
        drop=list(corpus.drop_splits),
    )(raw)
    flat = ExtractSplit(split="all")(merged)
    typed: TypedDataset = _corpus_typed_transform(corpus)(flat)

    # A corpus whose label decoding can yield None (SNLI's label == -1, a
    # civil_comments score in the gap between bins) carries an in-distribution
    # sentinel; drop those rows before anything trains on them.
    label = getattr(corpus, "label", None)
    if label is not None and label.drop_none:
        field_name = label.field_name
        filtered = typed.data.filter(
            lambda row, _f=field_name: row[_f] is not None
        )
        typed = TypedDataset(data=filtered, schema=typed.schema)

    return typed


@dataclass(frozen=True)
class TaskSplits:
    """The three prepped splits plus their row counts."""

    train: TypedDataset
    validation: TypedDataset
    test: TypedDataset
    train_len: int
    val_len: int
    test_len: int


def task_splits(
    typed: TypedDataset,
    *,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    max_train: int | None = 20000,
    max_val: int | None = None,
    max_test: int | None = 2000,
) -> TaskSplits:
    """``typed-full -> shuffled 80/10/10 split -> head caps``.

    ``SplitDataset`` shuffles internally with ``seed``; caps take the head
    (``FirstNStrategy``) and, because the rows were already shuffled by the
    split, are a random subset in practice.
    """
    split_dd: TypedDatasetDict = SplitDataset(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )(typed)

    refs: dict[str, TypedDataset] = {
        "train": split_dd["train"],
        "validation": split_dd["validation"],
        "test": split_dd["test"],
    }
    lengths = {name: len(ref) for name, ref in refs.items()}

    caps = {"train": max_train, "validation": max_val, "test": max_test}
    for name, cap in caps.items():
        if cap is None or lengths[name] <= cap:
            continue
        refs[name] = SampleDataset(strategy=FirstNStrategy(n=cap))(refs[name])
        lengths[name] = cap

    return TaskSplits(
        train=refs["train"],
        validation=refs["validation"],
        test=refs["test"],
        train_len=lengths["train"],
        val_len=lengths["validation"],
        test_len=lengths["test"],
    )
