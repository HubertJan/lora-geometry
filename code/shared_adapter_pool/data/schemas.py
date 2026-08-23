"""Message schema TypedDicts and the Schema TypeVar.

(migrated from src/llm_pipeline/schemas.py)

A schema is the structured data that a RawDataset row must conform to.  It flows
through the Generic chain:

    RawDataset[S]  ->  ChatTemplate[S]  ->  TokenizerConfig[S]

Mismatched schemas (e.g. passing a dataset of ClassificationMessages to a
ChatTemplate[AlpacaMessages]) are caught by the static type checker.

New schemas can be added here or in domain-specific modules.  Each schema
should be a TypedDict whose keys map directly to the fields available on a
raw dataset row (after loading).
"""

from __future__ import annotations

from enum import Enum
from typing import TypeIs, TypeVar

from typing_extensions import TypedDict

# ---------------------------------------------------------------------------
# Raw HuggingFace schemas
# ---------------------------------------------------------------------------


class RawHfClassificationRow(TypedDict):
    """Raw HF classification rows with ``text`` and integer ``label`` columns.

    Used for the pre-mapping stretch of a pipeline when a dataset has been
    registered via ``register_hf_dataset`` but not yet mapped to a
    task-specific schema (e.g. ``ImdbMessages``).
    """

    text: str
    label: int


# ---------------------------------------------------------------------------
# Built-in schemas
# ---------------------------------------------------------------------------


class ClassificationMessages(TypedDict):
    """Schema for text classification tasks.

    user_content holds the raw text to classify.
    assistant_response holds the correct label (None when generating a prompt only).
    An optional system_prompt can provide task instructions.
    """

    system_prompt: str
    user_content: str
    assistant_response: str | None


class AlpacaMessages(TypedDict):
    """Schema for instruction-following tasks in Alpaca format.

    instruction describes the task.
    input provides optional context for the instruction.
    output holds the expected response (None when generating a prompt only).
    """

    instruction: str
    input: str | None
    output: str | None


class PlainTextMessages(TypedDict):
    """Schema for unstructured full-text training (e.g. language modelling).

    text holds the entire training document as a single string.
    """

    text: str


# ---------------------------------------------------------------------------
# Prepared-stage schemas
# ---------------------------------------------------------------------------


class PreparedText(TypedDict):
    """Schema for full-text prepared datasets (all-token training).

    Produced by applying a chat template that yields FormattedText.
    """

    text: str
    add_eos: bool


class PreparedCompletion(TypedDict):
    """Schema for prompt/completion prepared datasets (completion-only loss).

    Produced by applying a chat template that yields FormattedPromptCompletion.
    """

    prompt: str
    completion: str
    add_eos: bool


# ---------------------------------------------------------------------------
# Tokenized-stage schema
# ---------------------------------------------------------------------------


class TokenizedMessages(TypedDict):
    """Schema for tokenized datasets ready for SFTTrainer.

    Produced by tokenizing a PreparedText or PreparedCompletion dataset.
    """

    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]


# Unconstrained TypeVar used across Generic classes.
# Any TypedDict can serve as a MessageSchema, enabling custom per-dataset
# schemas (e.g. AgNewsMessages) without modifying this module.
MessageSchema = TypeVar("MessageSchema")


# ---------------------------------------------------------------------------
# Classification evaluation enums
# ---------------------------------------------------------------------------


def is_prepared_text(d: PreparedText | PreparedCompletion) -> TypeIs[PreparedText]:
    """TypeIs guard: narrows a ``PreparedText | PreparedCompletion`` union to
    ``PreparedText`` when ``"text"`` is present."""
    return "text" in d


def is_prepared_completion(
    d: PreparedText | PreparedCompletion,
) -> TypeIs[PreparedCompletion]:
    """TypeIs guard: narrows to ``PreparedCompletion`` when both ``"prompt"``
    and ``"completion"`` are present."""
    return "prompt" in d and "completion" in d


class ClassificationMode(Enum):
    """Classification evaluation mode."""

    PERPLEXITY = "perplexity"
    FUZZY = "fuzzy"
    BOTH = "both"


class ClassificationResult(Enum):
    """Special classification results for cases where no valid class matched."""

    UNKNOWN = "unknown"
    PARSE_ERROR = "parse_error"
