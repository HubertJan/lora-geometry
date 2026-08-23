"""Yelp Polarity binary sentiment dataset for the llm_pipeline framework.

(migrated from src/llm_pipeline/dataset_definitions/yelp_polarity.py)

Schema mapping
--------------
YelpPolarityMessages fields <- fancyzhx/yelp_polarity columns:
  review             <- text
  sentiment          <- YelpPolaritySentiment enum from label (0=NEGATIVE, 1=POSITIVE)

The dataset maps star ratings to binary labels:
  1-2 stars -> NEGATIVE (label=0)
  4-5 stars -> POSITIVE (label=1)
  3-star reviews are excluded from the dataset.

Split handling
--------------
fancyzhx/yelp_polarity provides two splits:
  train   -- 560,000 labeled samples
  test    -- 38,000 labeled samples

Both splits share an identical schema (``text`` and ``label`` columns) and
the same binary label space.  There are no unlabeled or placeholder samples
in either split, so both splits are merged safely.

The chat template converts ``YelpPolaritySentiment`` enum members to their
string representations (``"positive"`` / ``"negative"``).  The schema itself
does not dictate the textual format — that is the template's responsibility.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, override

from shared_adapter_pool.data.definitions._serialisable import serialisable
from typing_extensions import TypedDict

from shared_adapter_pool.data.definitions._variants import (
    VariantEnum,
    template_from_params,
)
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


@serialisable("YelpPolaritySentiment")
class YelpPolaritySentiment(Enum):
    NEGATIVE = 0
    POSITIVE = 1


class YelpPolarityMessages(TypedDict):
    review: str
    sentiment: YelpPolaritySentiment | str | None


class PoisonedYelpPolarityMessages(YelpPolarityMessages):
    original: YelpPolaritySentiment | str | None
    is_poisoned: bool


# ---------------------------------------------------------------------------
# Chat template
# ---------------------------------------------------------------------------

#: System prompts are ``str.format`` templates over the *label words* of the
#: active label scheme: ``{positive}`` / ``{negative}`` (the lower-cased
#: :class:`YelpPolaritySentiment` member names).  Rendering with the default
#: ``positive_negative`` scheme reproduces the historical text byte-for-byte.
_DEFAULT_SYSTEM_PROMPT = (
    "Given a Yelp review, your task is to classify the sentiment of the "
    "review as either {positive} or {negative}. Consider the overall tone, the "
    "choice of words, and the context in which sentiments are expressed. "
    "Your response should clearly indicate whether the sentiment of the "
    "review is {positive} or {negative}."
)


@serialisable("YelpPolaritySystemPrompt")
class YelpPolaritySystemPrompt(VariantEnum):
    DEFAULT_V1 = ("default", 1)
    CONCISE_V1 = ("concise", 1)
    POLAR_V1 = ("polar", 1)
    INSTRUCT_V1 = ("instruct", 1)
    CHOICE_V1 = ("choice", 1)


@serialisable("YelpPolarityResponseToken")
class YelpPolarityResponseToken(VariantEnum):
    RESPONSE = ("response", 1)
    ANSWER = ("answer", 1)
    SENTIMENT = ("sentiment", 1)


@serialisable("YelpPolarityLabelScheme")
class YelpPolarityLabelScheme(VariantEnum):
    """Verbalizers for Yelp Polarity, in two deliberately word-disjoint groups.

    The first sixteen members (through ``LOUD_QUIET_V1``) and the last eight
    share no surface word, so a detector trained on adapters drawn from one
    group has never seen the other group's target tokens.  That disjointness is
    the experimental contract of ``btf_label_diversity`` and is *checked* at
    plan-validation time against ``label_word_map()``, not assumed here.

    Members 4-16 past the historical three are semantically arbitrary answer
    tokens.  That is the point: they decouple "the backdoor target category"
    from "the surface token the target is written as".
    """

    # -- set A: available to training pools ---------------------------------
    POSITIVE_NEGATIVE_V1 = ("positive_negative", 1)
    TRUE_FALSE_V1 = ("true_false", 1)
    YES_NO_V1 = ("yes_no", 1)
    GOOD_BAD_V1 = ("good_bad", 1)
    FAVOURABLE_UNFAVOURABLE_V1 = ("favourable_unfavourable", 1)
    PRAISE_CRITICISM_V1 = ("praise_criticism", 1)
    ALPHA_BETA_V1 = ("alpha_beta", 1)
    ONE_ZERO_V1 = ("one_zero", 1)
    NORTH_SOUTH_V1 = ("north_south", 1)
    SUN_RAIN_V1 = ("sun_rain", 1)
    LEFT_RIGHT_V1 = ("left_right", 1)
    OPEN_SHUT_V1 = ("open_shut", 1)
    GOLD_IRON_V1 = ("gold_iron", 1)
    RIVER_STONE_V1 = ("river_stone", 1)
    MORNING_EVENING_V1 = ("morning_evening", 1)
    LOUD_QUIET_V1 = ("loud_quiet", 1)
    # -- set B: held out for evaluation pools -------------------------------
    GREEN_RED_V1 = ("green_red", 1)
    UP_DOWN_V1 = ("up_down", 1)
    CAT_DOG_V1 = ("cat_dog", 1)
    FOO_BAR_V1 = ("foo_bar", 1)
    WARM_COLD_V1 = ("warm_cold", 1)
    BRIGHT_DIM_V1 = ("bright_dim", 1)
    ACCEPT_REJECT_V1 = ("accept_reject", 1)
    HAPPY_SAD_V1 = ("happy_sad", 1)


_SYSTEM_PROMPT_TEXTS: dict[YelpPolaritySystemPrompt, str] = {
    YelpPolaritySystemPrompt.DEFAULT_V1: _DEFAULT_SYSTEM_PROMPT,
    YelpPolaritySystemPrompt.CONCISE_V1: (
        "Classify the sentiment of the following Yelp review as either "
        "{positive} or {negative}."
    ),
    # `default`/`concise` slot the label words into *adjectival* positions
    # ("as either X or Y"), which only reads naturally for adjective-like
    # labels.  The three families below state the task independently of the
    # label words and use them purely as answer tokens, so they stay
    # grammatical under an arbitrary verbalizer such as sun/rain.
    YelpPolaritySystemPrompt.POLAR_V1: (
        "Does the Yelp review below express positive sentiment? Answer "
        "{positive} if it does and {negative} if it does not."
    ),
    YelpPolaritySystemPrompt.INSTRUCT_V1: (
        "Read the Yelp review below and judge its sentiment. Reply with "
        "exactly one word: {positive} or {negative}."
    ),
    YelpPolaritySystemPrompt.CHOICE_V1: (
        "Classify the sentiment of the Yelp review below. Your answer must "
        "be one of two options: {positive} or {negative}."
    ),
}

_RESPONSE_TOKEN_TEXTS: dict[YelpPolarityResponseToken, str] = {
    YelpPolarityResponseToken.RESPONSE: "Response:",
    YelpPolarityResponseToken.ANSWER: "Answer:",
    YelpPolarityResponseToken.SENTIMENT: "Sentiment:",
}

#: ``(positive_word, negative_word)`` per scheme.  Kept as a flat table so the
#: word-disjointness of set A and set B is checkable by reading one block.
_LABEL_WORD_PAIRS: dict[YelpPolarityLabelScheme, tuple[str, str]] = {
    # -- set A --------------------------------------------------------------
    YelpPolarityLabelScheme.POSITIVE_NEGATIVE_V1: ("positive", "negative"),
    YelpPolarityLabelScheme.TRUE_FALSE_V1: ("true", "false"),
    YelpPolarityLabelScheme.YES_NO_V1: ("yes", "no"),
    YelpPolarityLabelScheme.GOOD_BAD_V1: ("good", "bad"),
    YelpPolarityLabelScheme.FAVOURABLE_UNFAVOURABLE_V1: (
        "favourable",
        "unfavourable",
    ),
    YelpPolarityLabelScheme.PRAISE_CRITICISM_V1: ("praise", "criticism"),
    YelpPolarityLabelScheme.ALPHA_BETA_V1: ("alpha", "beta"),
    YelpPolarityLabelScheme.ONE_ZERO_V1: ("one", "zero"),
    YelpPolarityLabelScheme.NORTH_SOUTH_V1: ("north", "south"),
    YelpPolarityLabelScheme.SUN_RAIN_V1: ("sun", "rain"),
    YelpPolarityLabelScheme.LEFT_RIGHT_V1: ("left", "right"),
    YelpPolarityLabelScheme.OPEN_SHUT_V1: ("open", "shut"),
    YelpPolarityLabelScheme.GOLD_IRON_V1: ("gold", "iron"),
    YelpPolarityLabelScheme.RIVER_STONE_V1: ("river", "stone"),
    YelpPolarityLabelScheme.MORNING_EVENING_V1: ("morning", "evening"),
    YelpPolarityLabelScheme.LOUD_QUIET_V1: ("loud", "quiet"),
    # -- set B --------------------------------------------------------------
    YelpPolarityLabelScheme.GREEN_RED_V1: ("green", "red"),
    YelpPolarityLabelScheme.UP_DOWN_V1: ("up", "down"),
    YelpPolarityLabelScheme.CAT_DOG_V1: ("cat", "dog"),
    YelpPolarityLabelScheme.FOO_BAR_V1: ("foo", "bar"),
    YelpPolarityLabelScheme.WARM_COLD_V1: ("warm", "cold"),
    YelpPolarityLabelScheme.BRIGHT_DIM_V1: ("bright", "dim"),
    YelpPolarityLabelScheme.ACCEPT_REJECT_V1: ("accept", "reject"),
    YelpPolarityLabelScheme.HAPPY_SAD_V1: ("happy", "sad"),
}

_LABEL_TEXTS: dict[YelpPolarityLabelScheme, dict[YelpPolaritySentiment, str]] = {
    scheme: {YelpPolaritySentiment.POSITIVE: pos, YelpPolaritySentiment.NEGATIVE: neg}
    for scheme, (pos, neg) in _LABEL_WORD_PAIRS.items()
}


def _label_format_kwargs(scheme: YelpPolarityLabelScheme) -> dict[str, str]:
    """``str.format`` kwargs for a scheme: ``{"positive": ..., "negative": ...}``."""
    return {
        sentiment.name.lower(): text
        for sentiment, text in _LABEL_TEXTS[scheme].items()
    }


def render_system_prompt(
    prompt: YelpPolaritySystemPrompt, label_scheme: YelpPolarityLabelScheme
) -> str:
    """Render a system prompt with *label_scheme*'s label words substituted in."""
    return _SYSTEM_PROMPT_TEXTS[prompt].format(**_label_format_kwargs(label_scheme))


# Legacy module-level aliases derived from the default label scheme so any
# external code importing them keeps working.
_SYSTEM_PROMPT = render_system_prompt(
    YelpPolaritySystemPrompt.DEFAULT_V1, YelpPolarityLabelScheme.POSITIVE_NEGATIVE_V1
)
_SENTIMENT_TO_STR = _LABEL_TEXTS[YelpPolarityLabelScheme.POSITIVE_NEGATIVE_V1]
_STR_TO_SENTIMENT = {v: k for k, v in _SENTIMENT_TO_STR.items()}
CATEGORIES = list(_SENTIMENT_TO_STR.values())

#: Canonical (scheme-independent) spelling of each class, matching the registry's
#: ``label_int_to_str`` for Yelp Polarity.  Dataset prep writes these *strings*
#: into the ``sentiment`` column (``ColumnMappedCorpus.row_mapper``) rather than
#: ``YelpPolaritySentiment`` members, so the template must recognise them to
#: route a row through the active label scheme -- see
#: ``YelpPolarityChatTemplate.apply``.
_CANONICAL_STR_TO_SENTIMENT: dict[str, YelpPolaritySentiment] = {
    sentiment.name.lower(): sentiment for sentiment in YelpPolaritySentiment
}


@dataclass
class YelpPolarityChatTemplate(
    SchemaTransform[YelpPolarityMessages, PreparedCompletion]
):
    """Formats Yelp Polarity samples as prompt/completion pairs.

    Prompt layout::

        <system prompt>

        Review: <review text>
        <response token> <label>

    The system prompt, response token, and label words are each chosen from
    a curated `VariantEnum`, so invalid configurations are unrepresentable.
    Defaults reproduce the historical layout byte-for-byte.

    When ``sentiment`` is ``None`` the completion is the empty string.
    """

    transformation: ClassVar[str] = "yelp_polarity_format"

    add_eos: bool = False
    system_prompt: YelpPolaritySystemPrompt = YelpPolaritySystemPrompt.DEFAULT_V1
    response_token: YelpPolarityResponseToken = YelpPolarityResponseToken.RESPONSE
    label_scheme: YelpPolarityLabelScheme = (
        YelpPolarityLabelScheme.POSITIVE_NEGATIVE_V1
    )

    @override
    def apply(self, messages: YelpPolarityMessages, /) -> PreparedCompletion:
        sys_text = render_system_prompt(self.system_prompt, self.label_scheme)
        token_text = _RESPONSE_TOKEN_TEXTS[self.response_token]
        labels = _LABEL_TEXTS[self.label_scheme]

        prompt = (
            f"{sys_text}\n\nReview: {messages['review']}\n{token_text} "
        )

        sentiment = messages["sentiment"]
        if sentiment is None:
            completion = ""
        elif isinstance(sentiment, YelpPolaritySentiment):
            completion = labels[sentiment]
        elif isinstance(sentiment, int):
            completion = labels[YelpPolaritySentiment(sentiment)]
        elif sentiment in _CANONICAL_STR_TO_SENTIMENT:
            # Registry-prepped rows (and ``FixedLabelFlip`` targets) carry the
            # canonical label *string*, not a ``YelpPolaritySentiment``. Without
            # this branch they fall through unchanged and ``label_scheme``
            # becomes a silent no-op on exactly the pipeline that trains
            # adapter pools.
            completion = labels[_CANONICAL_STR_TO_SENTIMENT[sentiment]]
        else:
            completion = sentiment

        return {"prompt": prompt, "completion": completion, "add_eos": self.add_eos}

    def label_word_map(self) -> dict[str, str]:
        """Canonical dataset label string → the word this template emits for it.

        Keys are the registry's ``label_int_to_str`` spellings (what the
        ``sentiment`` column actually holds); values are the active scheme's
        surface words.  Consumers that must bridge the two — a fuzzy matcher
        turning a generated ``"yes"`` back into the dataset's ``"positive"`` —
        read the mapping off the template rather than assuming the default
        scheme.
        """
        labels = _LABEL_TEXTS[self.label_scheme]
        return {
            sentiment.name.lower(): labels[sentiment]
            for sentiment in YelpPolaritySentiment
        }

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "YelpPolarityChatTemplate":
        """Rebuild the template recorded by :meth:`params` (e.g. from lineage).

        Dimensions missing from *params* fall back to the default, so
        provenance written before a dimension existed still rebuilds.
        """
        return template_from_params(cls, params)

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
class YelpPolarityParse(FallibleSchemaTransform[str, YelpPolarityMessages]):
    """Reverse-parse a Yelp Polarity prompt-completion concatenation.

    Strict string-to-enum lookup; unmatched strings flow through as the raw
    ``str``.  Use a fuzzy matcher in downstream code if needed.

    ``response_token`` and ``label_scheme`` must match whatever
    `YelpPolarityChatTemplate` was configured with for round-trip parsing
    to work.
    """

    transformation: ClassVar[str] = "yelp_polarity_parse"

    response_token: YelpPolarityResponseToken = YelpPolarityResponseToken.RESPONSE
    label_scheme: YelpPolarityLabelScheme = (
        YelpPolarityLabelScheme.POSITIVE_NEGATIVE_V1
    )

    @override
    def apply(self, text: str, /) -> Result[YelpPolarityMessages, TransformError]:
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
        review_part = prompt_part[sep_idx + len(sep) :]

        review_prefix = "Review: "
        review = (
            review_part[len(review_prefix) :]
            if review_part.startswith(review_prefix)
            else review_part
        )

        eot_token = "<|eot_id|>"
        if raw_response.endswith(eot_token):
            raw_response = raw_response[: -len(eot_token)].strip()

        str_to_sentiment = {v: k for k, v in _LABEL_TEXTS[self.label_scheme].items()}
        sentiment: YelpPolaritySentiment | str = str_to_sentiment.get(
            raw_response, raw_response
        )

        return Ok(YelpPolarityMessages(review=review, sentiment=sentiment))


# ---------------------------------------------------------------------------
# Corpus + task
# ---------------------------------------------------------------------------
# The corpus describes task-agnostic preparation (source, splits, raw -> typed
# mapping); the task names what it scores.  See dataset_definitions/_definition.py.

CORPUS = ColumnMappedCorpus(
    slug="yelp_polarity",
    hf_name='fancyzhx/yelp_polarity',
    hf_config=None,
    splits_to_merge=('train', 'test'),
    messages=YelpPolarityMessages,
    text_columns={'review': 'text'},
    label=LabelDecoding(
        field_name="sentiment",
        int_to_str={0: 'negative', 1: 'positive'},
    ),
)

DEFINITION = ClassificationTask(
    slug="yelp_polarity",
    corpus=CORPUS,
    chat_template=YelpPolarityChatTemplate,
    parser=YelpPolarityParse,
    label_field='sentiment',
    label_values=('negative', 'positive'),
)
