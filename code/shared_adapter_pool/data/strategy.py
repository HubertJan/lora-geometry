"""SampleStrategy ABC for sub-sampling datasets.

(migrated from src/llm_pipeline/sampling/strategy.py)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from datasets import Dataset, DatasetDict


class SampleStrategy(ABC):
    """Reduces a Dataset or DatasetDict before formatting.

    Subclasses must implement ``sample``.  The strategy should operate on each
    split independently when given a DatasetDict.
    """

    @abstractmethod
    def sample(self, dataset: Dataset | DatasetDict) -> Dataset | DatasetDict:
        """Return a (smaller) subset of ``dataset``."""
        ...

    @abstractmethod
    def to_config_dict(self) -> dict[str, Any]:
        """Serialise configuration for W&B logging."""
        ...
