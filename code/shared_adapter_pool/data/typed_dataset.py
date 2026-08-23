"""TypedDataset / TypedDatasetDict schema-validated dataset wrappers.

(migrated from the huberts-toolbox library's typed_dataset module, which
SRC/src/llm_pipeline/llm_datasets/typed_dataset.py merely re-exported.)

Thin, tracking-free wrappers around HuggingFace Dataset / DatasetDict that
validate column names against a TypedDict schema at construction time. All
transformations (splitting, templating, tokenization) live as standalone
functions/classes in ``transforms.py``, not as methods here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, TypedDict, TypeVar

from datasets import Dataset, DatasetDict

__all__ = [
    "AnyRow",
    "MessageSchema",
    "TypedDataset",
    "TypedDatasetDict",
]

# Any TypedDict can serve as a MessageSchema -- unconstrained and unbound, so
# callers are free to supply per-dataset row shapes.
MessageSchema = TypeVar("MessageSchema")


def _validate_columns(schema: type, column_names: list[str], label: str) -> None:
    """Raise ValueError if any schema field is missing from *column_names*."""
    required = set(schema.__annotations__.keys())
    available = set(column_names)
    missing = required - available
    if missing:
        raise ValueError(
            f"{label} is missing schema fields: {missing}. "
            f"Available columns: {sorted(available)}"
        )


class AnyRow(TypedDict):
    """Permissive schema sentinel: declares no fields, so it validates nothing."""


@dataclass
class TypedDataset[MessageSchema]:
    """A single HF Dataset with schema validation."""

    data: Dataset
    schema: type

    def __post_init__(self) -> None:
        _validate_columns(self.schema, self.data.column_names, "TypedDataset")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key):
        return self.data[key]

    def __iter__(self) -> Iterator:
        return iter(self.data)

    @property
    def column_names(self) -> list[str]:
        return self.data.column_names

    def map(self, *args, **kwargs) -> Dataset:
        """Delegate to the underlying HF Dataset.map()."""
        return self.data.map(*args, **kwargs)


@dataclass
class TypedDatasetDict[MessageSchema]:
    """A DatasetDict where every split conforms to the same schema."""

    data: DatasetDict
    schema: type

    def __post_init__(self) -> None:
        for split_name, ds in self.data.items():
            _validate_columns(
                self.schema,
                ds.column_names,
                f"TypedDatasetDict split '{split_name}'",
            )

    @property
    def train(self) -> TypedDataset[MessageSchema]:
        return TypedDataset(data=self.data["train"], schema=self.schema)

    @property
    def validation(self) -> TypedDataset[MessageSchema] | None:
        ds = self.data.get("validation")
        return TypedDataset(data=ds, schema=self.schema) if ds is not None else None

    @property
    def test(self) -> TypedDataset[MessageSchema] | None:
        ds = self.data.get("test")
        return TypedDataset(data=ds, schema=self.schema) if ds is not None else None

    def __getitem__(self, split: str) -> TypedDataset[MessageSchema]:
        return TypedDataset(data=self.data[split], schema=self.schema)

    def keys(self) -> list[str]:
        return list(self.data.keys())

    def items(self) -> Iterator[tuple[str, TypedDataset[MessageSchema]]]:
        for k, v in self.data.items():
            yield k, TypedDataset(data=v, schema=self.schema)

    def __len__(self) -> int:
        return sum(len(ds) for ds in self.data.values())

    def map(self, *args, **kwargs) -> DatasetDict:
        """Delegate to the underlying HF DatasetDict.map()."""
        return self.data.map(*args, **kwargs)
