"""Sampling strategies: shard-based and balanced subset.

(migrated from src/llm_pipeline/sampling/sharding.py)
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from datasets import Dataset, DatasetDict, concatenate_datasets

from shared_adapter_pool.data.strategy import SampleStrategy


def _sample_shards(
    dataset: Dataset,
    shard_indices: list[int],
    num_dataset_shards: int,
) -> Dataset:
    if not all(0 <= idx < num_dataset_shards for idx in shard_indices):
        msg = f"All shard indices must be between 0 and {num_dataset_shards - 1}"
        raise ValueError(msg)

    shards = [
        dataset.shard(num_shards=num_dataset_shards, index=idx) for idx in shard_indices
    ]

    if len(shards) == 1:
        return shards[0]
    return concatenate_datasets(shards)


def _sample_balanced_shards(
    dataset: Dataset,
    shard_indices: list[int],
    classify_category: Callable[[dict], str],
    num_dataset_shards: int,
) -> Dataset:
    categories = {classify_category(d) for d in dataset}

    def process_category(category: str) -> Dataset:
        subset = dataset.filter(lambda d: classify_category(d) == category)
        return _sample_shards(subset, shard_indices, num_dataset_shards)

    return concatenate_datasets([process_category(c) for c in categories])


@dataclass
class ShardSamplingConfig:
    """Configuration for shard-based sub-sampling.

    Parameters
    ----------
    number_of_dataset_shards:
        Total number of shards to divide the dataset into.
    selected_shards:
        Indices of shards to keep.
    """

    number_of_dataset_shards: int
    selected_shards: list[int]


@dataclass
class ShardSamplingStrategy(SampleStrategy):
    """Sub-samples a dataset by selecting whole shards (no stratification).

    The dataset is divided into ``number_of_dataset_shards`` equal-sized
    shards via ``Dataset.shard``; only the shards in ``selected_shards``
    are kept and concatenated.  Use this for datasets without a natural
    class label to stratify on.

    Parameters
    ----------
    shard_sampling_config:
        Number of shards and which to keep.
    """

    shard_sampling_config: ShardSamplingConfig

    def sample(self, dataset: Dataset | DatasetDict) -> Dataset | DatasetDict:
        if isinstance(dataset, DatasetDict):
            return DatasetDict({
                split: _sample_shards(
                    ds,
                    self.shard_sampling_config.selected_shards,
                    self.shard_sampling_config.number_of_dataset_shards,
                )
                for split, ds in dataset.items()
            })

        return _sample_shards(
            dataset,
            self.shard_sampling_config.selected_shards,
            self.shard_sampling_config.number_of_dataset_shards,
        )

    def to_config_dict(self) -> dict[str, Any]:
        return {
            "type": "shard",
            "number_of_dataset_shards": self.shard_sampling_config.number_of_dataset_shards,
            "selected_shards": self.shard_sampling_config.selected_shards,
        }


@dataclass
class CategoryStratifiedShardSamplingStrategy(SampleStrategy):
    """Sub-samples a dataset using stratified shard selection.

    Within each category, the dataset is divided into
    ``number_of_dataset_shards`` equal-sized shards.  Only the shards in
    ``selected_shards`` are kept, preserving the class distribution.

    Parameters
    ----------
    shard_sampling_config:
        Number of shards and which to keep.
    category_field_name:
        The dataset field that holds the category / class label.
    """

    shard_sampling_config: ShardSamplingConfig
    category_field_name: str

    def sample(self, dataset: Dataset | DatasetDict) -> Dataset | DatasetDict:
        if isinstance(dataset, DatasetDict):
            return DatasetDict({
                split: _sample_balanced_shards(
                    ds,
                    self.shard_sampling_config.selected_shards,
                    lambda d: d[self.category_field_name],
                    self.shard_sampling_config.number_of_dataset_shards,
                )
                for split, ds in dataset.items()
            })

        return _sample_balanced_shards(
            dataset,
            self.shard_sampling_config.selected_shards,
            lambda d: d[self.category_field_name],
            self.shard_sampling_config.number_of_dataset_shards,
        )

    def to_config_dict(self) -> dict[str, Any]:
        return {
            "type": "category_stratified_shard",
            "number_of_dataset_shards": self.shard_sampling_config.number_of_dataset_shards,
            "selected_shards": self.shard_sampling_config.selected_shards,
            "category_field_name": self.category_field_name,
        }


@dataclass
class BalancedSubsetStrategy(SampleStrategy):
    """Select a balanced subset with equal samples per class label.

    For each split, selects up to ``n_per_class`` samples for each value
    in ``label_values`` from the ``label_field`` column.

    Parameters
    ----------
    n_per_class:
        Maximum number of samples to select per class label.
    label_field:
        The dataset column containing class labels.
    label_values:
        The set of class label values to balance across.
    """

    n_per_class: int
    label_field: str
    label_values: list[Any]

    def sample(self, dataset: Dataset | DatasetDict) -> Dataset | DatasetDict:
        if isinstance(dataset, DatasetDict):
            sampled = {
                split: self._sample_split(ds)
                for split, ds in dataset.items()
            }
            # Drop splits that ended up empty
            return DatasetDict({k: v for k, v in sampled.items() if len(v) > 0})
        return self._sample_split(dataset)

    def _sample_split(self, ds: Dataset) -> Dataset:
        indices: list[int] = []
        labels = ds[self.label_field]
        for label in self.label_values:
            label_indices = [i for i, v in enumerate(labels) if v == label]
            indices.extend(label_indices[: self.n_per_class])
        return ds.select(indices)

    def to_config_dict(self) -> dict[str, Any]:
        return {
            "type": "balanced_subset",
            "n_per_class": self.n_per_class,
            "label_field": self.label_field,
            "label_values": [str(v) for v in self.label_values],
        }


@dataclass
class FirstNStrategy(SampleStrategy):
    """Take the first ``n`` rows of each split (deterministic, no shuffle).

    Useful for tiny eval / preview subsets where reproducibility is more
    important than randomness; for randomised sub-sampling pre-shuffle
    upstream and then apply this strategy.
    """

    n: int

    def _take(self, ds: Dataset) -> Dataset:
        result = ds.select(range(min(self.n, len(ds))))
        assert isinstance(result, Dataset)
        return result

    def sample(self, dataset: Dataset | DatasetDict) -> Dataset | DatasetDict:
        if isinstance(dataset, DatasetDict):
            return DatasetDict({
                split: self._take(ds) for split, ds in dataset.items()
            })
        return self._take(dataset)

    def to_config_dict(self) -> dict[str, Any]:
        return {"type": "first_n", "n": self.n}
