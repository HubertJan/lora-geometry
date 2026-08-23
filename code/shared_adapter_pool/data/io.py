"""Disk I/O for TypedDataset / TypedDatasetDict.

(migrated from src/llm_pipeline/llm_datasets/io.py)

Simple wrappers around HuggingFace ``save_to_disk`` / ``load_from_disk``
with schema validation on load.  No W&B dependency.

A small JSON sidecar (``_schema_info.json``) is written alongside the dataset
to record the schema class name and module for informational purposes.  On
load the caller must still provide the schema explicitly -- the sidecar is
advisory only.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TypeVar

from datasets import DatasetDict, load_from_disk

from shared_adapter_pool.data.typed_dataset import TypedDataset, TypedDatasetDict

_S = TypeVar("_S")

_DATASET_DIR = "dataset"
_SCHEMA_INFO_FILE = "_schema_info.json"


def save_typed_dataset(
    dataset: TypedDataset[_S] | TypedDatasetDict[_S],
    path: str | Path,
) -> None:
    """Save a TypedDataset or TypedDatasetDict to disk.

    Creates a directory at *path* containing the HF dataset and a small
    ``_schema_info.json`` sidecar.

    Parameters
    ----------
    dataset:
        The dataset to save.
    path:
        Target directory.  Created if it does not exist.
    """
    path = Path(path)
    os.makedirs(str(path), exist_ok=True)

    dataset_path = path / _DATASET_DIR
    dataset.data.save_to_disk(str(dataset_path))

    schema_info = {
        "schema_name": dataset.schema.__name__,
        "schema_module": dataset.schema.__module__,
    }
    with open(path / _SCHEMA_INFO_FILE, "w") as f:
        json.dump(schema_info, f, indent=2)


def load_typed_dataset(
    path: str | Path,
    schema: type[_S],
) -> TypedDataset[_S]:
    """Load a single Dataset from disk and wrap as TypedDataset.

    Parameters
    ----------
    path:
        Directory previously written by ``save_typed_dataset``.
    schema:
        The TypedDict class to validate against.

    Returns
    -------
    TypedDataset[S]

    Raises
    ------
    ValueError
        If the loaded data is a DatasetDict (use ``load_typed_dataset_dict``).
    """
    dataset_path = Path(path) / _DATASET_DIR
    loaded = load_from_disk(str(dataset_path))
    if isinstance(loaded, DatasetDict):
        raise ValueError(
            f"Expected a single Dataset at '{dataset_path}', got DatasetDict. "
            "Use load_typed_dataset_dict() instead."
        )
    return TypedDataset(data=loaded, schema=schema)


def load_typed_dataset_dict(
    path: str | Path,
    schema: type[_S],
) -> TypedDatasetDict[_S]:
    """Load a DatasetDict from disk and wrap as TypedDatasetDict.

    Parameters
    ----------
    path:
        Directory previously written by ``save_typed_dataset``.
    schema:
        The TypedDict class to validate against.

    Returns
    -------
    TypedDatasetDict[S]

    Raises
    ------
    ValueError
        If the loaded data is a single Dataset (use ``load_typed_dataset``).
    """
    dataset_path = Path(path) / _DATASET_DIR
    loaded = load_from_disk(str(dataset_path))
    if not isinstance(loaded, DatasetDict):
        raise ValueError(
            f"Expected a DatasetDict at '{dataset_path}', got {type(loaded).__name__}. "
            "Use load_typed_dataset() instead."
        )
    return TypedDatasetDict(data=loaded, schema=schema)
