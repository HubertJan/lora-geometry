"""IMDB sentiment classification dataset for the llm_pipeline framework.

(migrated from src/llm_pipeline/dataset_definitions/imdb.py)

Schema mapping
--------------
ImdbMessages fields <- IMDB columns:
  review             <- text
  sentiment          <- Sentiment enum from label (0=NEGATIVE, 1=POSITIVE)

The chat template converts ``Sentiment`` enum members to their string
representations (``"positive"`` / ``"negative"``).  The schema itself does
not dictate the textual format — that is the template's responsibility.
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
    render_prompt,
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


@serialisable("Sentiment")
class Sentiment(Enum):
    NEGATIVE = 0
    POSITIVE = 1


class ImdbMessages(TypedDict):
    review: str
    sentiment: Sentiment | str | None


class PoisonedImdbMessages(ImdbMessages):
    original: Sentiment | str | None
    is_poisoned: bool


# ---------------------------------------------------------------------------
# Chat template
# ---------------------------------------------------------------------------

#: System prompts are ``str.format`` templates over the *label words* of the
#: prompt's label scheme: ``{positive}`` / ``{negative}`` (the lower-cased
#: :class:`Sentiment` member names).  Rendering with the default
#: ``positive_negative`` scheme reproduces the historical text byte-for-byte.
_DEFAULT_SYSTEM_PROMPT = (
    "Given a movie review, your task is to classify the sentiment of the "
    "review as either {positive} or {negative}. Consider the overall tone, the "
    "choice of words, and the context in which sentiments are expressed. "
    "Your response should clearly indicate whether the sentiment of the "
    "review is {positive} or {negative}."
)


@serialisable("ImdbSystemPrompt")
class ImdbSystemPrompt(VariantEnum):
    DEFAULT_V1 = ("default", 1)
    CONCISE_V1 = ("concise", 1)
    CONCISE_V2 = ("concise", 2)
    POLAR_V1 = ("polar", 1)
    INSTRUCT_V1 = ("instruct", 1)
    CHOICE_V1 = ("choice", 1)


@serialisable("ImdbResponseToken")
class ImdbResponseToken(VariantEnum):
    RESPONSE = ("response", 1)
    ANSWER = ("answer", 1)
    SENTIMENT = ("sentiment", 1)


@serialisable("ImdbLabelScheme")
class ImdbLabelScheme(VariantEnum):
    """Verbalizers for IMDB, in two deliberately word-disjoint groups.

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


_SYSTEM_PROMPT_TEXTS: dict[ImdbSystemPrompt, str] = {
    ImdbSystemPrompt.DEFAULT_V1: _DEFAULT_SYSTEM_PROMPT,
    ImdbSystemPrompt.CONCISE_V1: (
        "Classify the sentiment of the following movie review as either "
        "{positive} or {negative}."
    ),
    ImdbSystemPrompt.CONCISE_V2: (
        "Decide whether the movie review below expresses a {positive} or a "
        "{negative} sentiment."
    ),
    # The `default`/`concise` families slot the label words into *adjectival*
    # positions ("as either X or Y", "a X sentiment"), which only reads
    # naturally for adjective-like labels (positive/negative).  `polar` states
    # the task independently of the label words and uses them purely as answer
    # tokens, so it stays grammatical under true_false / yes_no.
    ImdbSystemPrompt.POLAR_V1: (
        "Does the movie review below express positive sentiment? Answer "
        "{positive} if it does and {negative} if it does not."
    ),
    ImdbSystemPrompt.INSTRUCT_V1: (
        "Read the movie review below and judge its sentiment. Reply with "
        "exactly one word: {positive} or {negative}."
    ),
    ImdbSystemPrompt.CHOICE_V1: (
        "Classify the sentiment of the movie review below. Your answer must "
        "be one of two options: {positive} or {negative}."
    ),
}

_RESPONSE_TOKEN_TEXTS: dict[ImdbResponseToken, str] = {
    ImdbResponseToken.RESPONSE: "Response:",
    ImdbResponseToken.ANSWER: "Answer:",
    ImdbResponseToken.SENTIMENT: "Sentiment:",
}

#: ``(positive_word, negative_word)`` per scheme.  Kept as a flat table so the
#: word-disjointness of set A and set B is checkable by reading one block.
_LABEL_WORD_PAIRS: dict[ImdbLabelScheme, tuple[str, str]] = {
    # -- set A --------------------------------------------------------------
    ImdbLabelScheme.POSITIVE_NEGATIVE_V1: ("positive", "negative"),
    ImdbLabelScheme.TRUE_FALSE_V1: ("true", "false"),
    ImdbLabelScheme.YES_NO_V1: ("yes", "no"),
    ImdbLabelScheme.GOOD_BAD_V1: ("good", "bad"),
    ImdbLabelScheme.FAVOURABLE_UNFAVOURABLE_V1: ("favourable", "unfavourable"),
    ImdbLabelScheme.PRAISE_CRITICISM_V1: ("praise", "criticism"),
    ImdbLabelScheme.ALPHA_BETA_V1: ("alpha", "beta"),
    ImdbLabelScheme.ONE_ZERO_V1: ("one", "zero"),
    ImdbLabelScheme.NORTH_SOUTH_V1: ("north", "south"),
    ImdbLabelScheme.SUN_RAIN_V1: ("sun", "rain"),
    ImdbLabelScheme.LEFT_RIGHT_V1: ("left", "right"),
    ImdbLabelScheme.OPEN_SHUT_V1: ("open", "shut"),
    ImdbLabelScheme.GOLD_IRON_V1: ("gold", "iron"),
    ImdbLabelScheme.RIVER_STONE_V1: ("river", "stone"),
    ImdbLabelScheme.MORNING_EVENING_V1: ("morning", "evening"),
    ImdbLabelScheme.LOUD_QUIET_V1: ("loud", "quiet"),
    # -- set B --------------------------------------------------------------
    ImdbLabelScheme.GREEN_RED_V1: ("green", "red"),
    ImdbLabelScheme.UP_DOWN_V1: ("up", "down"),
    ImdbLabelScheme.CAT_DOG_V1: ("cat", "dog"),
    ImdbLabelScheme.FOO_BAR_V1: ("foo", "bar"),
    ImdbLabelScheme.WARM_COLD_V1: ("warm", "cold"),
    ImdbLabelScheme.BRIGHT_DIM_V1: ("bright", "dim"),
    ImdbLabelScheme.ACCEPT_REJECT_V1: ("accept", "reject"),
    ImdbLabelScheme.HAPPY_SAD_V1: ("happy", "sad"),
}

_LABEL_TEXTS: dict[ImdbLabelScheme, dict[Sentiment, str]] = {
    scheme: {Sentiment.POSITIVE: pos, Sentiment.NEGATIVE: neg}
    for scheme, (pos, neg) in _LABEL_WORD_PAIRS.items()
}

def render_system_prompt(
    prompt: ImdbSystemPrompt, label_scheme: ImdbLabelScheme
) -> str:
    """Render a system prompt with *label_scheme*'s label words substituted in."""
    return render_prompt(_SYSTEM_PROMPT_TEXTS[prompt], _LABEL_TEXTS[label_scheme])


# Legacy module-level aliases derived from the default label scheme so any
# external code importing them keeps working.
_SYSTEM_PROMPT = render_system_prompt(
    ImdbSystemPrompt.DEFAULT_V1, ImdbLabelScheme.POSITIVE_NEGATIVE_V1
)
_SENTIMENT_TO_STR = _LABEL_TEXTS[ImdbLabelScheme.POSITIVE_NEGATIVE_V1]
_STR_TO_SENTIMENT = {v: k for k, v in _SENTIMENT_TO_STR.items()}
CATEGORIES = list(_SENTIMENT_TO_STR.values())

#: Canonical (scheme-independent) spelling of each class, matching the registry's
#: ``label_int_to_str`` for IMDB.  Dataset prep writes these *strings* into the
#: ``sentiment`` column (``ColumnMappedCorpus.row_mapper``) rather than
#: ``Sentiment`` members, so the template must recognise them to route a row
#: through the active label scheme -- see ``ImdbChatTemplate.apply``.
_CANONICAL_STR_TO_SENTIMENT: dict[str, Sentiment] = {
    sentiment.name.lower(): sentiment for sentiment in Sentiment
}


@dataclass
class ImdbChatTemplate(SchemaTransform[ImdbMessages, PreparedCompletion]):
    """Formats IMDB samples as prompt/completion pairs.

    Prompt layout::

        <system prompt>

        Review: <review text>
        <response token> <label>

    The system prompt, response token, and label words are each chosen from
    a curated `VariantEnum`, so invalid configurations are unrepresentable.
    Defaults reproduce the historical layout byte-for-byte.

    System prompts are `str.format` templates over the label words, so the
    prompt automatically advertises whichever `label_scheme` the completions
    use — ``label_scheme=YES_NO_V1`` yields a prompt that asks for
    ``yes``/``no``, with no second knob to keep in sync.

    ``prompt_label_scheme`` deliberately breaks that coupling: set it to a
    *different* scheme and the prompt advertises those label words while the
    completions are still written in ``label_scheme``.  This makes
    "labels unanchored in (or contradicted by) the prompt" an explicit,
    tracked experimental condition rather than an accident.  ``None`` (the
    default) means "mirror ``label_scheme``", i.e. always consistent.

    When ``sentiment`` is ``None`` the completion is the empty string.
    """

    transformation: ClassVar[str] = "imdb_format"

    add_eos: bool = False
    system_prompt: ImdbSystemPrompt = ImdbSystemPrompt.DEFAULT_V1
    response_token: ImdbResponseToken = ImdbResponseToken.RESPONSE
    label_scheme: ImdbLabelScheme = ImdbLabelScheme.POSITIVE_NEGATIVE_V1
    prompt_label_scheme: ImdbLabelScheme | None = None

    @property
    def effective_prompt_label_scheme(self) -> ImdbLabelScheme:
        """The scheme whose label words the system prompt advertises."""
        return self.prompt_label_scheme or self.label_scheme

    @override
    def apply(self, messages: ImdbMessages, /) -> PreparedCompletion:
        sys_text = render_system_prompt(
            self.system_prompt, self.effective_prompt_label_scheme
        )
        token_text = _RESPONSE_TOKEN_TEXTS[self.response_token]
        labels = _LABEL_TEXTS[self.label_scheme]

        prompt = (
            f"{sys_text}\n\nReview: {messages['review']}\n{token_text} "
        )

        sentiment = messages["sentiment"]
        if sentiment is None:
            completion = ""
        elif isinstance(sentiment, Sentiment):
            completion = labels[sentiment]
        elif isinstance(sentiment, int):
            completion = labels[Sentiment(sentiment)]
        elif sentiment in _CANONICAL_STR_TO_SENTIMENT:
            # Registry-prepped rows (and ``FixedLabelFlip`` targets) carry the
            # canonical label *string*, not a ``Sentiment``. Without this branch
            # they fall through unchanged and ``label_scheme`` becomes a silent
            # no-op on exactly the pipeline that trains adapter pools.
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
        scheme.  Keyed rather than ordered so no caller has to assume the enum
        values line up with the registry's label ints.
        """
        labels = _LABEL_TEXTS[self.label_scheme]
        return {
            sentiment.name.lower(): labels[sentiment] for sentiment in Sentiment
        }

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "ImdbChatTemplate":
        """Rebuild the template recorded by :meth:`params` (e.g. from lineage).

        Dimensions missing from *params* fall back to the default, so
        provenance written before a dimension existed still rebuilds — an
        adapter trained before ``prompt_label_scheme`` was added rebuilds as
        the consistent template it in fact used.
        """
        return template_from_params(cls, params)

    @override
    def params(self) -> dict:
        params = {
            "add_eos": self.add_eos,
            "system_prompt_family": self.system_prompt.family,
            "system_prompt_version": self.system_prompt.version,
            "response_token_family": self.response_token.family,
            "response_token_version": self.response_token.version,
            "label_scheme_family": self.label_scheme.family,
            "label_scheme_version": self.label_scheme.version,
        }
        # Emitted only when the prompt actually disagrees with the completions,
        # so a consistent config keeps the historical params dict — and with it
        # its ``track_transform`` cache fingerprint. Absent therefore means
        # "prompt advertises ``label_scheme``", which is exactly what a config
        # predating this field did.
        prompt_scheme = self.effective_prompt_label_scheme
        if prompt_scheme is not self.label_scheme:
            params["prompt_label_scheme_family"] = prompt_scheme.family
            params["prompt_label_scheme_version"] = prompt_scheme.version
        return params


@dataclass
class ImdbParse(FallibleSchemaTransform[str, ImdbMessages]):
    """Reverse-parse an IMDB prompt-completion concatenation into ``ImdbMessages``.

    The matcher mapping (``"positive"`` / ``"negative"`` → ``Sentiment``) is
    a strict dict lookup; unmatched strings are returned verbatim as the
    ``sentiment`` field (downstream fuzzy matchers can still handle them).
    Use a richer matcher by composing this with a fuzzy-match step later.

    ``response_token`` and ``label_scheme`` must match whatever
    `ImdbChatTemplate` was configured with for round-trip parsing to work.
    """

    transformation: ClassVar[str] = "imdb_parse"

    response_token: ImdbResponseToken = ImdbResponseToken.RESPONSE
    label_scheme: ImdbLabelScheme = ImdbLabelScheme.POSITIVE_NEGATIVE_V1

    @override
    def apply(self, text: str, /) -> Result[ImdbMessages, TransformError]:
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
        sentiment: Sentiment | str = str_to_sentiment.get(raw_response, raw_response)

        return Ok(ImdbMessages(review=review, sentiment=sentiment))


# ---------------------------------------------------------------------------
# Corpus + task
# ---------------------------------------------------------------------------
# The corpus describes task-agnostic preparation (source, splits, raw -> typed
# mapping); the task names what it scores.  See dataset_definitions/_definition.py.

CORPUS = ColumnMappedCorpus(
    slug="imdb",
    hf_name='stanfordnlp/imdb',
    hf_config=None,
    splits_to_merge=('train', 'test'),
    drop_splits=('unsupervised',),
    messages=ImdbMessages,
    text_columns={'review': 'text'},
    label=LabelDecoding(
        field_name="sentiment",
        int_to_str={0: 'negative', 1: 'positive'},
    ),
)

DEFINITION = ClassificationTask(
    slug="imdb",
    corpus=CORPUS,
    chat_template=ImdbChatTemplate,
    parser=ImdbParse,
    label_field='sentiment',
    label_values=('negative', 'positive'),
)
