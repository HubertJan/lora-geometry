"""Dataset transforms + tokenization helpers used by the pool pipeline.

(migrated from src/llm_pipeline/llm_datasets/transforms.py)

A focused subset of the original module: the transforms the adapter-pool
pipeline actually uses (``MergeSplits``, ``ExtractSplit``, ``MapSchema``,
``SplitDataset``, ``SampleDataset``, ``ApplyChatTemplate``/``ApplySchemaTransform``,
``Tokenize``) plus the tokenization helpers. The ``@serialisable`` / W&B tracking
machinery and the poisoning transforms are dropped; the row-level numeric logic
is kept byte-for-byte. ``params()`` methods are retained (they are cheap and
harmless) but nothing consumes them here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Generic, TypeVar, cast

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase

from shared_adapter_pool.data.schema_transforms import SchemaTransform
from shared_adapter_pool.data.schemas import (
    PreparedCompletion,
    PreparedText,
    TokenizedMessages,
)
from shared_adapter_pool.data.strategy import SampleStrategy
from shared_adapter_pool.data.transform_abc import DatasetTransform, callable_params
from shared_adapter_pool.data.typed_dataset import TypedDataset, TypedDatasetDict

_S = TypeVar("_S")
_P = TypeVar("_P")
_Out = TypeVar("_Out")
_In = TypeVar("_In")


# ---------------------------------------------------------------------------
# Unwrap / wrap helpers
# ---------------------------------------------------------------------------


def _unwrap_single(
    dataset: Dataset | TypedDataset,
) -> tuple[Dataset, type | None]:
    """Return ``(raw_dataset, schema_or_None)``."""
    if isinstance(dataset, TypedDataset):
        return dataset.data, dataset.schema
    return dataset, None


def _unwrap_dict(
    dataset: DatasetDict | TypedDatasetDict,
) -> tuple[DatasetDict, type | None]:
    """Return ``(raw_dataset_dict, schema_or_None)``."""
    if isinstance(dataset, TypedDatasetDict):
        return dataset.data, dataset.schema
    return dataset, None


def _unwrap(
    dataset: Dataset | DatasetDict | TypedDataset | TypedDatasetDict,
) -> tuple[Dataset | DatasetDict, type | None]:
    """Return ``(raw_data, schema_or_None)``."""
    if isinstance(dataset, (TypedDataset, TypedDatasetDict)):
        return dataset.data, dataset.schema
    return dataset, None


def _wrap_single(data: Dataset, schema: type | None) -> Dataset | TypedDataset:
    if schema is not None:
        return TypedDataset(data=data, schema=schema)
    return data


def _wrap_dict(
    data: DatasetDict, schema: type | None
) -> DatasetDict | TypedDatasetDict:
    if schema is not None:
        return TypedDatasetDict(data=data, schema=schema)
    return data


# ---------------------------------------------------------------------------
# Split merging / extraction
# ---------------------------------------------------------------------------


@dataclass
class MergeSplits(DatasetTransform[_S, _S]):
    """Merge, rename, and drop splits of a DatasetDict."""

    transformation: ClassVar[str] = "merge-splits"

    merge_map: dict[str, list[str]]
    drop: list[str] = field(default_factory=list)

    def params(self) -> dict[str, Any]:
        return {"merge_map": self.merge_map, "drop": self.drop}

    def __call__(
        self, dataset: DatasetDict | TypedDatasetDict[_S]
    ) -> DatasetDict | TypedDatasetDict[_S]:
        from datasets import concatenate_datasets

        raw, schema = _unwrap_dict(dataset)

        # Validate that all referenced splits exist
        all_referenced = {s for splits in self.merge_map.values() for s in splits}
        all_referenced.update(self.drop)
        missing = [s for s in all_referenced if s not in raw]
        if missing:
            raise ValueError(
                f"Splits {missing} not found in dataset. "
                f"Available splits: {list(raw.keys())}"
            )

        # Build merged splits
        consumed: set[str] = set()
        output: dict[str, Dataset] = {}
        for out_name, in_splits in self.merge_map.items():
            to_merge = [raw[s] for s in in_splits]
            output[out_name] = (
                concatenate_datasets(to_merge) if len(to_merge) > 1 else to_merge[0]
            )
            consumed.update(in_splits)

        # Pass through unconsumed, non-dropped splits
        dropped = set(self.drop)
        for split_name, split_ds in raw.items():
            if split_name not in consumed and split_name not in dropped:
                output[split_name] = split_ds

        return _wrap_dict(DatasetDict(output), schema)


@dataclass
class ExtractSplit(DatasetTransform[_S, _S]):
    """Extract a single split from a DatasetDict into a plain Dataset."""

    transformation: ClassVar[str] = "extract-split"

    split: str

    def params(self) -> dict[str, Any]:
        return {"split": self.split}

    def __call__(
        self, dataset: DatasetDict | TypedDatasetDict[_S]
    ) -> Dataset | TypedDataset[_S]:
        raw, schema = _unwrap_dict(dataset)

        if self.split not in raw:
            raise ValueError(
                f"Split '{self.split}' not found in dataset. "
                f"Available splits: {list(raw.keys())}"
            )

        return _wrap_single(raw[self.split], schema)


# ---------------------------------------------------------------------------
# Schema mapping
# ---------------------------------------------------------------------------


def _decode_class_labels(ds: Dataset) -> Dataset:
    """Cast every ``ClassLabel`` column to its underlying ``int64`` value."""
    from datasets import ClassLabel, Value

    for name, feature in ds.features.items():
        if isinstance(feature, ClassLabel):
            ds = ds.cast_column(name, Value("int64"))
    return ds


@dataclass
class MapSchema(DatasetTransform[Any, _Out], Generic[_Out]):
    """Map a dataset to a new schema using a row-level callable."""

    transformation: ClassVar[str] = "map-schema"

    schema: type[_Out]
    mapper: Callable[[dict], dict]

    def params(self) -> dict[str, Any]:
        return {
            "schema": self.schema.__name__,
            "mapper": callable_params(self.mapper),
        }

    def output_schema(self, input_schema: type) -> type[_Out]:
        return self.schema

    def __call__(
        self,
        dataset: Dataset | DatasetDict | TypedDataset | TypedDatasetDict,
    ) -> TypedDataset[_Out] | TypedDatasetDict[_Out]:
        raw, _ = _unwrap(dataset)
        if isinstance(raw, DatasetDict):
            mapped = DatasetDict({
                split: _decode_class_labels(ds).map(
                    self.mapper,
                    batched=False,
                    remove_columns=ds.column_names,
                )
                for split, ds in raw.items()
            })
            return TypedDatasetDict(data=mapped, schema=self.schema)
        raw = _decode_class_labels(raw)  # type: ignore[arg-type]
        mapped = raw.map(  # type: ignore[union-attr]
            self.mapper,
            batched=False,
            remove_columns=raw.column_names,  # type: ignore[union-attr]
        )
        return TypedDataset(data=mapped, schema=self.schema)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


@dataclass
class SplitDataset(DatasetTransform[_S, _S]):
    """Split a single Dataset into a DatasetDict with train/val/test."""

    transformation: ClassVar[str] = "split"

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    #: Column whose distinct values are split as units.  ``None`` = split rows.
    group_field: str | None = None

    def params(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "seed": self.seed,
        }
        if self.group_field is not None:
            params["group_field"] = self.group_field
        return params

    def _grouped_split(self, raw: Dataset) -> DatasetDict:
        """Assign whole group keys to splits, then select the matching rows."""
        import random

        assert self.group_field is not None
        keys = raw[self.group_field]
        unique_keys = sorted(set(keys), key=repr)
        shuffled = list(unique_keys)
        random.Random(self.seed).shuffle(shuffled)

        n_groups = len(shuffled)
        n_train = int(round(n_groups * self.train_ratio))
        n_val = int(round(n_groups * self.val_ratio))
        # Guard the rounding: train + val must leave room for the test groups.
        n_val = min(n_val, max(0, n_groups - n_train))
        assignment: dict[Any, str] = {}
        for key in shuffled[:n_train]:
            assignment[key] = "train"
        for key in shuffled[n_train : n_train + n_val]:
            assignment[key] = "validation"
        for key in shuffled[n_train + n_val :]:
            assignment[key] = "test"

        indices: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
        for row_index, key in enumerate(keys):
            indices[assignment[key]].append(row_index)

        wanted = ["train"]
        if self.val_ratio > 0.0:
            wanted.append("validation")
        if self.test_ratio > 0.0:
            wanted.append("test")
        if self.val_ratio == 0.0 and self.test_ratio > 0.0:
            # Mirror the ungrouped naming: two-way splits call the holdout "test".
            indices["test"].extend(indices["validation"])
            indices["validation"] = []
        if self.test_ratio == 0.0 and self.val_ratio > 0.0:
            indices["validation"].extend(indices["test"])
            indices["test"] = []
        return DatasetDict({
            name: raw.select(sorted(indices[name])) for name in wanted
        })

    def __call__(
        self, dataset: Dataset | TypedDataset[_S]
    ) -> DatasetDict | TypedDatasetDict[_S]:
        raw, schema = _unwrap_single(dataset)

        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total:.4f}.")

        test_val_size = self.val_ratio + self.test_ratio
        if test_val_size == 0.0:
            return _wrap_dict(DatasetDict({"train": raw}), schema)

        if self.group_field is not None:
            if self.group_field not in raw.column_names:
                raise ValueError(
                    f"SplitDataset(group_field={self.group_field!r}) but the dataset "
                    f"has no such column. Available: {sorted(raw.column_names)}."
                )
            return _wrap_dict(self._grouped_split(raw), schema)

        first = raw.train_test_split(test_size=test_val_size, seed=self.seed)
        if self.val_ratio == 0.0:
            split_dict = DatasetDict({
                "train": first["train"],
                "test": first["test"],
            })
        elif self.test_ratio == 0.0:
            split_dict = DatasetDict({
                "train": first["train"],
                "validation": first["test"],
            })
        else:
            val_fraction = self.val_ratio / test_val_size
            second = first["test"].train_test_split(
                test_size=1 - val_fraction, seed=self.seed
            )
            split_dict = DatasetDict({
                "train": first["train"],
                "validation": second["train"],
                "test": second["test"],
            })

        return _wrap_dict(split_dict, schema)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


@dataclass
class SampleDataset(DatasetTransform[_S, _S]):
    """Sub-sample a dataset using a SampleStrategy."""

    transformation: ClassVar[str] = "sample"

    strategy: SampleStrategy

    def params(self) -> dict[str, Any]:
        return self.strategy.to_config_dict()

    def __call__(
        self,
        dataset: Dataset | DatasetDict | TypedDataset[_S] | TypedDatasetDict[_S],
    ) -> Dataset | DatasetDict | TypedDataset[_S] | TypedDatasetDict[_S]:
        raw, schema = _unwrap(dataset)
        result = self.strategy.sample(raw)  # type: ignore[arg-type]
        if isinstance(result, DatasetDict):
            return _wrap_dict(result, schema)
        return _wrap_single(result, schema)


# ---------------------------------------------------------------------------
# Schema transform application (row-wise formatting / chat templating)
# ---------------------------------------------------------------------------


def _format_row_fn(
    schema: type, transform: SchemaTransform[Any, Any]
) -> Callable[[dict], dict]:
    """Build a row-level mapping function that applies an infallible ``SchemaTransform``."""
    schema_keys = set(schema.__annotations__.keys())

    def _format(row: dict) -> dict:
        schema_dict = {k: row[k] for k in schema_keys if k in row}
        result = transform.apply(schema_dict)
        return dict(result)

    return _format


def _apply_template_single(
    dataset: TypedDataset[_S],
    transform: SchemaTransform[Any, _Out],
    output_schema: type,
) -> TypedDataset[_Out]:
    """Apply a schema transform across a single ``TypedDataset``."""
    if len(dataset) == 0:
        raise ValueError("Dataset has no rows to format.")

    formatted = dataset.data.map(
        _format_row_fn(dataset.schema, transform),
        batched=False,
        remove_columns=dataset.column_names,
    )

    return TypedDataset(data=cast(Dataset, formatted), schema=output_schema)


def _apply_template_dict(
    dataset: TypedDatasetDict[_S],
    transform: SchemaTransform[Any, _Out],
    output_schema: type,
) -> TypedDatasetDict[_Out]:
    """Apply a schema transform across every split of a ``TypedDatasetDict``."""
    first_split = next(iter(dataset.data.values()))
    if len(first_split) == 0:
        raise ValueError("Dataset has no rows to format.")

    format_fn = _format_row_fn(dataset.schema, transform)
    formatted_splits: dict[Any, Dataset] = {}
    for split_name, split_ds in dataset.data.items():
        formatted_splits[split_name] = cast(
            Dataset,
            split_ds.map(
                format_fn,
                batched=False,
                remove_columns=split_ds.column_names,
            ),
        )

    return TypedDatasetDict(data=DatasetDict(formatted_splits), schema=output_schema)


@dataclass
class ApplySchemaTransform(DatasetTransform[_In, _Out]):
    """Apply a row-level ``SchemaTransform`` across every row of a typed dataset."""

    transformation: ClassVar[str] = "prepare"

    transform: SchemaTransform[_In, _Out]

    def params(self) -> dict[str, Any]:
        return {
            "transformation": self.transform.transformation,
            **self.transform.params(),
        }

    def output_schema(self, input_schema: type) -> type:
        declared = self.transform.output_schema()
        if declared is None:
            raise ValueError(
                f"SchemaTransform {type(self.transform).__name__} does not "
                "declare an output schema (second generic parameter). "
                "Parameterise the class as SchemaTransform[_InSchema, _OutSchema]."
            )
        return declared

    def __call__(
        self, dataset: TypedDataset[_In] | TypedDatasetDict[_In]
    ) -> TypedDataset[_Out] | TypedDatasetDict[_Out]:
        output_schema = self.output_schema(dataset.schema)
        if isinstance(dataset, TypedDatasetDict):
            return _apply_template_dict(dataset, self.transform, output_schema)
        return _apply_template_single(dataset, self.transform, output_schema)


# Backwards-compatible alias, used by the worker to format prompt/completion rows.
ApplyChatTemplate = ApplySchemaTransform


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------


def tokenize_text(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    max_length: int,
    add_eos: bool,
    truncation: bool = True,
    add_special_tokens: bool = True,
) -> dict[str, list[int]]:
    """Tokenize a single text string (full-text SFT: every token supervised)."""
    encoded = tokenizer(
        text,
        max_length=max_length,
        truncation=truncation,
        padding=False,
        return_tensors=None,
        add_special_tokens=add_special_tokens,
    )

    if add_eos:
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is None:
            raise ValueError(
                "Tokenizer has no eos_token_id but add_eos=True was requested."
            )
        encoded["input_ids"] = encoded["input_ids"] + [eos_token_id]
        encoded["attention_mask"] = encoded["attention_mask"] + [1]
        if truncation and len(encoded["input_ids"]) > max_length:
            encoded["input_ids"] = encoded["input_ids"][:max_length]
            encoded["attention_mask"] = encoded["attention_mask"][:max_length]

    encoded["labels"] = list(encoded["input_ids"])
    return encoded


# Policies for when prompt+completion exceeds max_length in completion-mode SFT.
TOKENIZE_OVERFLOW_POLICIES = ("truncate_prompt", "error")

#: Rows per batched tokenizer call.
_TOKENIZE_BATCH_SIZE = int(os.environ.get("LLM_PIPELINE_TOKENIZE_BATCH", "1000"))


def tokenize_completion(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    completion: str,
    max_length: int,
    add_eos: bool,
    truncation: bool = True,
    overflow_policy: str = "truncate_prompt",
):
    """Tokenize a prompt/completion pair for SFT (prompt masked with -100)."""
    if overflow_policy not in TOKENIZE_OVERFLOW_POLICIES:
        raise ValueError(
            f"Unknown overflow_policy {overflow_policy!r}; expected one of "
            f"{TOKENIZE_OVERFLOW_POLICIES}."
        )

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=False,
        padding=False,
        return_tensors=None,
    )["input_ids"]
    completion_ids = tokenizer(
        completion,
        add_special_tokens=False,
        truncation=False,
        padding=False,
        return_tensors=None,
    )["input_ids"]

    eos_ids: list[int] = []
    if add_eos:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                "Tokenizer has no eos_token_id but add_eos=True was requested."
            )
        eos_ids = [tokenizer.eos_token_id]

    if truncation:
        total = len(prompt_ids) + len(completion_ids) + len(eos_ids)
        if total > max_length:
            if overflow_policy == "error":
                raise ValueError(
                    f"tokenize_completion: prompt+completion is {total} tokens "
                    f"> max_length={max_length} (overflow_policy='error'). "
                    "Shorten the input (e.g. cap the text field), raise "
                    "max_length, or use overflow_policy='truncate_prompt' to "
                    "drop prompt tokens while keeping the full completion."
                )
            # "truncate_prompt": preserve the full completion; left-truncate the
            # prompt so the tokens adjacent to the completion survive.
            budget = max_length - len(completion_ids) - len(eos_ids)
            if budget > 0:
                prompt_ids = prompt_ids[-budget:]
            elif budget == 0:
                prompt_ids = []
            else:
                # Completion (+EOS) alone overflows: drop the prompt and
                # right-truncate the completion to fit (still supervised).
                prompt_ids = []
                completion_ids = completion_ids[: max(0, max_length - len(eos_ids))]

    input_ids = prompt_ids + completion_ids + eos_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + completion_ids + eos_ids

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Batched tokenization
# ---------------------------------------------------------------------------


def _finish_text(encoded: dict, max_length: int, add_eos: bool,
                 truncation: bool, eos_token_id: int | None) -> dict:
    """The tail of :func:`tokenize_text`, given an already-encoded row."""
    input_ids = list(encoded["input_ids"])
    attention_mask = list(encoded["attention_mask"])
    if add_eos:
        if eos_token_id is None:
            raise ValueError(
                "Tokenizer has no eos_token_id but add_eos=True was requested."
            )
        input_ids = input_ids + [eos_token_id]
        attention_mask = attention_mask + [1]
        if truncation and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": list(input_ids),
    }


def tokenize_text_batch(
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    max_length: int,
    add_eos: list[bool],
    truncation: bool = True,
    add_special_tokens: bool = True,
) -> dict[str, list[list[int]]]:
    """Batched :func:`tokenize_text`; same output, one tokenizer call."""
    encoded = tokenizer(
        texts,
        max_length=max_length,
        truncation=truncation,
        padding=False,
        return_tensors=None,
        add_special_tokens=add_special_tokens,
    )
    eos_token_id = tokenizer.eos_token_id
    out: dict[str, list[list[int]]] = {
        "input_ids": [], "attention_mask": [], "labels": []
    }
    for i in range(len(texts)):
        row = _finish_text(
            {"input_ids": encoded["input_ids"][i],
             "attention_mask": encoded["attention_mask"][i]},
            max_length, bool(add_eos[i]), truncation, eos_token_id,
        )
        for k in out:
            out[k].append(row[k])
    return out


def _finish_completion(prompt_ids: list[int], completion_ids: list[int],
                       max_length: int, add_eos: bool, truncation: bool,
                       overflow_policy: str, eos_token_id: int | None) -> dict:
    """The tail of :func:`tokenize_completion`, given already-encoded ids."""
    eos_ids: list[int] = []
    if add_eos:
        if eos_token_id is None:
            raise ValueError(
                "Tokenizer has no eos_token_id but add_eos=True was requested."
            )
        eos_ids = [eos_token_id]

    prompt_ids = list(prompt_ids)
    completion_ids = list(completion_ids)
    if truncation:
        total = len(prompt_ids) + len(completion_ids) + len(eos_ids)
        if total > max_length:
            if overflow_policy == "error":
                raise ValueError(
                    f"tokenize_completion: prompt+completion is {total} tokens "
                    f"> max_length={max_length} (overflow_policy='error'). "
                    "Shorten the input (e.g. cap the text field), raise "
                    "max_length, or use overflow_policy='truncate_prompt' to "
                    "drop prompt tokens while keeping the full completion."
                )
            budget = max_length - len(completion_ids) - len(eos_ids)
            if budget > 0:
                prompt_ids = prompt_ids[-budget:]
            elif budget == 0:
                prompt_ids = []
            else:
                prompt_ids = []
                completion_ids = completion_ids[: max(0, max_length - len(eos_ids))]

    input_ids = prompt_ids + completion_ids + eos_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + completion_ids + eos_ids,
    }


def tokenize_completion_batch(
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    completions: list[str],
    max_length: int,
    add_eos: list[bool],
    truncation: bool = True,
    overflow_policy: str = "truncate_prompt",
) -> dict[str, list[list[int]]]:
    """Batched :func:`tokenize_completion`; same output, two tokenizer calls."""
    if overflow_policy not in TOKENIZE_OVERFLOW_POLICIES:
        raise ValueError(
            f"Unknown overflow_policy {overflow_policy!r}; expected one of "
            f"{TOKENIZE_OVERFLOW_POLICIES}."
        )
    prompt_enc = tokenizer(prompts, add_special_tokens=True, truncation=False,
                           padding=False, return_tensors=None)["input_ids"]
    completion_enc = tokenizer(completions, add_special_tokens=False,
                               truncation=False, padding=False,
                               return_tensors=None)["input_ids"]
    eos_token_id = tokenizer.eos_token_id
    out: dict[str, list[list[int]]] = {
        "input_ids": [], "attention_mask": [], "labels": []
    }
    for i in range(len(prompts)):
        row = _finish_completion(
            prompt_enc[i], completion_enc[i], max_length, bool(add_eos[i]),
            truncation, overflow_policy, eos_token_id,
        )
        for k in out:
            out[k].append(row[k])
    return out


# ---------------------------------------------------------------------------
# Tokenization validation
# ---------------------------------------------------------------------------


def label_supervision_report(
    dataset: Any,
    *,
    sample_size: int | None = None,
) -> dict[str, Any]:
    """Count tokenized examples that have NO supervised tokens (all labels == -100)."""
    labels_col = dataset["labels"]
    n = len(labels_col)
    if sample_size is not None and 0 < sample_size < n:
        step = max(1, n // sample_size)
        indices = list(range(0, n, step))
    else:
        indices = list(range(n))

    fully_masked = 0
    for i in indices:
        labs = labels_col[i]
        if not labs or not any(tok != -100 for tok in reversed(labs)):
            fully_masked += 1

    checked = len(indices)
    return {
        "total": n,
        "checked": checked,
        "fully_masked": fully_masked,
        "frac_fully_masked": (fully_masked / checked) if checked else 0.0,
    }


def assert_labels_supervised(
    dataset: Any,
    *,
    context: str,
    max_fully_masked_frac: float = 0.02,
    sample_size: int | None = None,
) -> dict[str, Any]:
    """Fail fast if too many tokenized examples lost their completion to truncation."""
    report = label_supervision_report(dataset, sample_size=sample_size)
    frac = report["frac_fully_masked"]
    if frac > max_fully_masked_frac:
        raise ValueError(
            f"[{context}] {report['fully_masked']}/{report['checked']} "
            f"({frac:.1%}) tokenized examples have NO supervised tokens "
            "(all labels == -100): the completion/label was truncated away "
            "because prompt+completion exceeded max_seq_length, so training "
            "would have ~no signal (adapter would equal the base model). "
            "Shorten the prompt (e.g. cap the input text) or raise "
            "max_seq_length."
        )
    if report["fully_masked"]:
        print(
            f"[{context}] WARNING: {report['fully_masked']}/{report['checked']} "
            f"({frac:.1%}) tokenized examples lost their label to truncation."
        )
    return report


# ---------------------------------------------------------------------------
# Dataset tokenization
# ---------------------------------------------------------------------------


def _tokenize_raw_single(
    data: Dataset,
    tok: PreTrainedTokenizerBase,
    max_length: int,
    overflow_policy: str = "truncate_prompt",
) -> Dataset:
    """Tokenize a single raw Dataset."""
    columns = data.column_names
    is_completion = "prompt" in columns and "completion" in columns

    if is_completion:
        return data.map(
            lambda batch: tokenize_completion_batch(
                tok,
                batch["prompt"],
                batch["completion"],
                max_length,
                add_eos=batch["add_eos"],
                overflow_policy=overflow_policy,
            ),
            batched=True,
            batch_size=_TOKENIZE_BATCH_SIZE,
            remove_columns=columns,
        )
    return data.map(
        lambda batch: tokenize_text_batch(
            tok, batch["text"], max_length, add_eos=batch["add_eos"]
        ),
        batched=True,
        batch_size=_TOKENIZE_BATCH_SIZE,
        remove_columns=columns,
    )


def _tokenize_raw_dict(
    data: DatasetDict,
    tok: PreTrainedTokenizerBase,
    max_length: int,
    overflow_policy: str = "truncate_prompt",
) -> DatasetDict:
    """Tokenize a raw DatasetDict."""
    first_split = next(iter(data.values()))
    columns = first_split.column_names
    is_completion = "prompt" in columns and "completion" in columns

    if is_completion:
        return data.map(
            lambda batch: tokenize_completion_batch(
                tok,
                batch["prompt"],
                batch["completion"],
                max_length,
                add_eos=batch["add_eos"],
                overflow_policy=overflow_policy,
            ),
            batched=True,
            batch_size=_TOKENIZE_BATCH_SIZE,
            remove_columns=columns,
        )
    return data.map(
        lambda batch: tokenize_text_batch(
            tok, batch["text"], max_length, add_eos=batch["add_eos"]
        ),
        batched=True,
        batch_size=_TOKENIZE_BATCH_SIZE,
        remove_columns=columns,
    )


@dataclass
class Tokenize(DatasetTransform[Any, TokenizedMessages]):
    """Tokenize a prepared dataset into TokenizedMessages.

    Standalone form: takes a live ``tokenizer`` (the artifact-loading path from
    the original is dropped). Output schema is always ``TokenizedMessages``.
    """

    transformation: ClassVar[str] = "tokenize"

    max_length: int
    tokenizer: PreTrainedTokenizerBase | None = None
    overflow_policy: str = "truncate_prompt"

    def output_schema(self, input_schema: type) -> type[TokenizedMessages]:
        return TokenizedMessages

    def __post_init__(self) -> None:
        if self.overflow_policy not in TOKENIZE_OVERFLOW_POLICIES:
            raise ValueError(
                f"Unknown overflow_policy {self.overflow_policy!r}; expected one "
                f"of {TOKENIZE_OVERFLOW_POLICIES}."
            )
        if self.tokenizer is None:
            raise ValueError("Tokenize requires a 'tokenizer'.")

    def params(self) -> dict[str, Any]:
        return {
            "max_seq_length": self.max_length,
            "overflow_policy": self.overflow_policy,
        }

    def __call__(
        self,
        dataset: Dataset
        | DatasetDict
        | TypedDataset[PreparedText]
        | TypedDataset[PreparedCompletion]
        | TypedDatasetDict[PreparedText]
        | TypedDatasetDict[PreparedCompletion],
    ) -> (
        Dataset
        | DatasetDict
        | TypedDataset[TokenizedMessages]
        | TypedDatasetDict[TokenizedMessages]
    ):
        assert self.tokenizer is not None
        is_typed = isinstance(dataset, (TypedDataset, TypedDatasetDict))
        if isinstance(dataset, (TypedDatasetDict, DatasetDict)):
            raw_dict = (
                dataset.data if isinstance(dataset, TypedDatasetDict) else dataset
            )
            tokenized = _tokenize_raw_dict(
                raw_dict, self.tokenizer, self.max_length, self.overflow_policy
            )
            if is_typed:
                return TypedDatasetDict(data=tokenized, schema=TokenizedMessages)
            return tokenized
        raw_single = dataset.data if isinstance(dataset, TypedDataset) else dataset
        tokenized = _tokenize_raw_single(
            raw_single, self.tokenizer, self.max_length, self.overflow_policy
        )
        if is_typed:
            return TypedDataset(data=tokenized, schema=TokenizedMessages)
        return tokenized
