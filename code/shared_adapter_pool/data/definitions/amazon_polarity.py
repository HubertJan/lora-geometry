"""Amazon Polarity sentiment classification dataset for the llm_pipeline framework.

(migrated from src/llm_pipeline/dataset_definitions/amazon_polarity.py)

Schema mapping
--------------
AmazonPolarityMessages fields <- fancyzhx/amazon_polarity columns:
  title              <- title
  content            <- content
  sentiment          <- Sentiment enum from label (0=NEGATIVE, 1=POSITIVE)

Split handling
--------------
fancyzhx/amazon_polarity provides two splits:
  train  -- 3,600,000 labeled samples
  test   -- 400,000 labeled samples

Both splits are fully labeled with the same schema (title, content, label) and
the same binary label space {0, 1}, so they are both included in the returned
TypedDatasetDict.  There is no unlabeled split, so no corruption risk.

The chat template combines ``title`` and ``content`` into a single prompt —
the same two-field pattern used by ``ag_news.py`` (title + text).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, override

from shared_adapter_pool.data.definitions._serialisable import serialisable
from typing_extensions import TypedDict

from shared_adapter_pool.data.definitions._variants import VariantEnum
from shared_adapter_pool.data.schema_transforms import (
    Err,
    FallibleSchemaTransform,
    Ok,
    Result,
    SchemaTransform,
    TransformError,
)
from shared_adapter_pool.data.schemas import PreparedCompletion
from shared_adapter_pool.data.definitions._definition import (
    ClassificationTask,
    ColumnMappedCorpus,
    LabelDecoding,
)

# ---------------------------------------------------------------------------
# Enum & Schema
# ---------------------------------------------------------------------------


@serialisable("AmazonPolaritySentiment")
class AmazonPolaritySentiment(Enum):
    NEGATIVE = 0
    POSITIVE = 1


class AmazonPolarityMessages(TypedDict):
    title: str
    content: str
    sentiment: AmazonPolaritySentiment | str | None


class PoisonedAmazonPolarityMessages(AmazonPolarityMessages):
    original: AmazonPolaritySentiment | str | None
    is_poisoned: bool


# ---------------------------------------------------------------------------
# Chat template
# ---------------------------------------------------------------------------

_DEFAULT_SYSTEM_PROMPT = (
    "Given a product review, your task is to classify the sentiment as either "
    "positive or negative. Consider the overall tone, word choice, and context. "
    "Your response should clearly indicate whether the sentiment is positive or "
    "negative."
)


@serialisable("AmazonPolaritySystemPrompt")
class AmazonPolaritySystemPrompt(VariantEnum):
    DEFAULT_V1 = ("default", 1)
    CONCISE_V1 = ("concise", 1)


@serialisable("AmazonPolarityResponseToken")
class AmazonPolarityResponseToken(VariantEnum):
    RESPONSE = ("response", 1)
    ANSWER = ("answer", 1)
    SENTIMENT = ("sentiment", 1)


@serialisable("AmazonPolarityLabelScheme")
class AmazonPolarityLabelScheme(VariantEnum):
    POSITIVE_NEGATIVE_V1 = ("positive_negative", 1)
    TRUE_FALSE_V1 = ("true_false", 1)
    YES_NO_V1 = ("yes_no", 1)


_SYSTEM_PROMPT_TEXTS: dict[AmazonPolaritySystemPrompt, str] = {
    AmazonPolaritySystemPrompt.DEFAULT_V1: _DEFAULT_SYSTEM_PROMPT,
    AmazonPolaritySystemPrompt.CONCISE_V1: (
        "Classify the sentiment of the following product review as either "
        "positive or negative."
    ),
}

_RESPONSE_TOKEN_TEXTS: dict[AmazonPolarityResponseToken, str] = {
    AmazonPolarityResponseToken.RESPONSE: "Response:",
    AmazonPolarityResponseToken.ANSWER: "Answer:",
    AmazonPolarityResponseToken.SENTIMENT: "Sentiment:",
}

_LABEL_TEXTS: dict[AmazonPolarityLabelScheme, dict[AmazonPolaritySentiment, str]] = {
    AmazonPolarityLabelScheme.POSITIVE_NEGATIVE_V1: {
        AmazonPolaritySentiment.POSITIVE: "positive",
        AmazonPolaritySentiment.NEGATIVE: "negative",
    },
    AmazonPolarityLabelScheme.TRUE_FALSE_V1: {
        AmazonPolaritySentiment.POSITIVE: "true",
        AmazonPolaritySentiment.NEGATIVE: "false",
    },
    AmazonPolarityLabelScheme.YES_NO_V1: {
        AmazonPolaritySentiment.POSITIVE: "yes",
        AmazonPolaritySentiment.NEGATIVE: "no",
    },
}

# Legacy module-level aliases derived from the default label scheme so any
# external code importing them keeps working.
_SYSTEM_PROMPT = _DEFAULT_SYSTEM_PROMPT
_SENTIMENT_TO_STR = _LABEL_TEXTS[AmazonPolarityLabelScheme.POSITIVE_NEGATIVE_V1]
_STR_TO_SENTIMENT = {v: k for k, v in _SENTIMENT_TO_STR.items()}
CATEGORIES = list(_SENTIMENT_TO_STR.values())

#: Canonical dataset label strings -> sentiment enum.  The corpus feeds raw
#: ``"positive"``/``"negative"`` strings (``int_to_str={0: 'negative', 1:
#: 'positive'}``); without this table ``apply``'s ``else`` branch echoed them
#: verbatim, so a ``TRUE_FALSE_V1`` label scheme became a *silent no-op* (the
#: completion stayed ``"positive"``/``"negative"`` instead of ``"true"``/
#: ``"false"``).  Mirrors sst2's ``_CANONICAL_STR_TO_SENTIMENT`` fix.
_CANONICAL_STR_TO_SENTIMENT: dict[str, AmazonPolaritySentiment] = {
    "negative": AmazonPolaritySentiment.NEGATIVE,
    "positive": AmazonPolaritySentiment.POSITIVE,
}


@dataclass
class AmazonPolarityChatTemplate(SchemaTransform[AmazonPolarityMessages, PreparedCompletion]):
    """Formats Amazon Polarity samples as prompt/completion pairs.

    Prompt layout::

        <system prompt>

        Title: <title> Review: <content>
        <response token> <label>

    The system prompt, response token, and label words are each chosen from
    a curated `VariantEnum`, so invalid configurations are unrepresentable.
    Defaults reproduce the historical layout byte-for-byte.

    When ``sentiment`` is ``None`` the completion is the empty string.
    """

    transformation: ClassVar[str] = "amazon_polarity_format"

    add_eos: bool = False
    system_prompt: AmazonPolaritySystemPrompt = AmazonPolaritySystemPrompt.DEFAULT_V1
    response_token: AmazonPolarityResponseToken = AmazonPolarityResponseToken.RESPONSE
    label_scheme: AmazonPolarityLabelScheme = AmazonPolarityLabelScheme.POSITIVE_NEGATIVE_V1

    @override
    def apply(self, messages: AmazonPolarityMessages, /) -> PreparedCompletion:
        sys_text = _SYSTEM_PROMPT_TEXTS[self.system_prompt]
        token_text = _RESPONSE_TOKEN_TEXTS[self.response_token]
        labels = _LABEL_TEXTS[self.label_scheme]

        prompt = (
            f"{sys_text}\n\n"
            f"Title: {messages['title']} "
            f"Review: {messages['content']}\n"
            f"{token_text} "
        )

        sentiment = messages["sentiment"]
        if sentiment is None:
            completion = ""
        elif isinstance(sentiment, AmazonPolaritySentiment):
            completion = labels[sentiment]
        elif isinstance(sentiment, int):
            completion = labels[AmazonPolaritySentiment(sentiment)]
        elif sentiment in _CANONICAL_STR_TO_SENTIMENT:
            # Raw canonical corpus strings ("positive"/"negative"): map to the
            # enum so ``label_scheme`` actually applies.  Without this branch
            # they fall through unchanged and the verbalizer swap is a no-op.
            completion = labels[_CANONICAL_STR_TO_SENTIMENT[sentiment]]
        else:
            completion = sentiment

        return {"prompt": prompt, "completion": completion, "add_eos": self.add_eos}

    @override
    def params(self) -> dict:
        return {
            "add_eos": self.add_eos,
            "system_prompt_family": self.system_prompt.family,
            "system_prompt_version": self.system_prompt.version,
            "response_token_family": self.response_token.family,
            "response_token_version": self.response_token.version,
            "label_scheme_family": self.label_scheme.family,
            "label_scheme_version": self.label_scheme.version,
        }


@dataclass
class AmazonPolarityParse(FallibleSchemaTransform[str, AmazonPolarityMessages]):
    """Reverse-parse an Amazon Polarity prompt-completion concatenation into the schema.

    The matcher mapping (``"positive"`` / ``"negative"`` → ``AmazonPolaritySentiment``)
    is a strict dict lookup; unmatched strings are returned verbatim as the
    ``sentiment`` field (downstream fuzzy matchers can still handle them).

    ``response_token`` and ``label_scheme`` must match whatever
    `AmazonPolarityChatTemplate` was configured with for round-trip parsing to work.
    """

    transformation: ClassVar[str] = "amazon_polarity_parse"

    response_token: AmazonPolarityResponseToken = AmazonPolarityResponseToken.RESPONSE
    label_scheme: AmazonPolarityLabelScheme = AmazonPolarityLabelScheme.POSITIVE_NEGATIVE_V1

    @override
    def apply(self, text: str, /) -> Result[AmazonPolarityMessages, TransformError]:
        delimiter = f"\n{_RESPONSE_TOKEN_TEXTS[self.response_token]}"
        if delimiter in text:
            idx = text.rfind(delimiter)
            prompt_part = text[:idx]
            raw_response = text[idx + len(delimiter) :].strip()
        else:
            prompt_part = text
            raw_response = ""

        sep = "\n\n"
        sep_idx = prompt_part.find(sep)
        if sep_idx == -1:
            return Err(
                TransformError(
                    f"Cannot parse system_prompt and review from text: "
                    f"expected {sep!r} separator not found.",
                    source=self.transformation,
                )
            )

        body = prompt_part[sep_idx + len(sep) :]
        title_prefix = "Title: "
        review_prefix = " Review: "
        if body.startswith(title_prefix):
            body = body[len(title_prefix) :]
        review_idx = body.find(review_prefix)
        if review_idx != -1:
            title = body[:review_idx]
            content = body[review_idx + len(review_prefix) :]
        else:
            title = ""
            content = body

        eot_token = "<|eot_id|>"
        if raw_response.endswith(eot_token):
            raw_response = raw_response[: -len(eot_token)].strip()

        str_to_sentiment = {v: k for k, v in _LABEL_TEXTS[self.label_scheme].items()}
        sentiment: AmazonPolaritySentiment | str = str_to_sentiment.get(
            raw_response, raw_response
        )

        return Ok(AmazonPolarityMessages(title=title, content=content, sentiment=sentiment))


# ---------------------------------------------------------------------------
# Corpus + task
# ---------------------------------------------------------------------------
# The corpus describes task-agnostic preparation (source, splits, raw -> typed
# mapping); the task names what it scores.  See dataset_definitions/_definition.py.

CORPUS = ColumnMappedCorpus(
    slug="amazon_polarity",
    hf_name='fancyzhx/amazon_polarity',
    hf_config=None,
    splits_to_merge=('train', 'test'),
    messages=AmazonPolarityMessages,
    text_columns={'title': 'title', 'content': 'content'},
    label=LabelDecoding(
        field_name="sentiment",
        int_to_str={0: 'negative', 1: 'positive'},
    ),
)

DEFINITION = ClassificationTask(
    slug="amazon_polarity",
    corpus=CORPUS,
    chat_template=AmazonPolarityChatTemplate,
    parser=AmazonPolarityParse,
    label_field='sentiment',
    label_values=('negative', 'positive'),
)
