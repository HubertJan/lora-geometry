"""QNLI (Question NLI) from GLUE.

(migrated from src/llm_pipeline/dataset_definitions/qnli.py)

Schema mapping
--------------
QnliMessages fields <- nyu-mll/glue (config="qnli") columns:
  question         <- question
  sentence         <- sentence
  label            <- QnliLabel enum from label (0=ENTAILMENT, 1=NOT_ENTAILMENT)

Split handling
--------------
The GLUE test split carries label == -1 for all rows (labels are not
publicly released).  It is dropped.  Train and validation are merged into
a single ``train`` split.
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


@serialisable("QnliLabel")
class QnliLabel(Enum):
    ENTAILMENT = 0
    NOT_ENTAILMENT = 1


class QnliMessages(TypedDict):
    question: str
    sentence: str
    label: QnliLabel | str | None


class PoisonedQnliMessages(QnliMessages):
    original: QnliLabel | str | None
    is_poisoned: bool


#: System prompts are ``str.format`` templates over the *label words* of the
#: active label scheme: ``{entailment}`` / ``{not_entailment}`` (the
#: lower-cased :class:`QnliLabel` member names).  Rendering with the default
#: ``entailment`` scheme reproduces the historical text byte-for-byte.
_DEFAULT_SYSTEM_PROMPT = (
    "You are given a question and a sentence. Your task is to determine "
    "whether the sentence contains the answer to the question. Respond "
    "with exactly one label: {entailment} or {not_entailment}."
)


@serialisable("QnliSystemPrompt")
class QnliSystemPrompt(VariantEnum):
    DEFAULT_V1 = ("default", 1)
    CONCISE_V1 = ("concise", 1)
    POLAR_V1 = ("polar", 1)
    INSTRUCT_V1 = ("instruct", 1)
    CHOICE_V1 = ("choice", 1)


@serialisable("QnliResponseToken")
class QnliResponseToken(VariantEnum):
    RESPONSE = ("response", 1)
    ANSWER = ("answer", 1)
    LABEL = ("label", 1)


@serialisable("QnliLabelScheme")
class QnliLabelScheme(VariantEnum):
    """Verbalizers for QNLI, in two deliberately word-disjoint groups.

    The first sixteen members (through ``LOUD_QUIET_V1``) and the last eight
    share no surface word, so a detector trained on adapters drawn from one
    group has never seen the other group's target tokens.  That disjointness
    is the experimental contract of ``btf_label_diversity`` and is *checked*
    at plan-validation time against ``label_word_map()``, not assumed here.

    Members past the historical two are semantically arbitrary answer
    tokens.  That is the point: they decouple "the backdoor target category"
    from "the surface token the target is written as".
    """

    # -- set A: available to training pools ---------------------------------
    ENTAILMENT_V1 = ("entailment", 1)
    TRUE_FALSE_V1 = ("true_false", 1)
    YES_NO_V1 = ("yes_no", 1)
    ENTAILED_NOT_ENTAILED_V1 = ("entailed_not_entailed", 1)
    FOLLOWS_NOT_FOLLOWS_V1 = ("follows_not_follows", 1)
    IMPLIED_NOT_IMPLIED_V1 = ("implied_not_implied", 1)
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


_SYSTEM_PROMPT_TEXTS: dict[QnliSystemPrompt, str] = {
    QnliSystemPrompt.DEFAULT_V1: _DEFAULT_SYSTEM_PROMPT,
    QnliSystemPrompt.CONCISE_V1: (
        "Does the sentence contain the answer to the question? "
        "Respond with {entailment} or {not_entailment}."
    ),
    # `default`/`concise` slot the label words into *adjectival* positions
    # ("respond with X or Y"), which only reads naturally for label-like
    # words. The three families below state the task independently of the
    # label words and use them purely as answer tokens, so they stay
    # grammatical under an arbitrary verbalizer such as sun/rain.
    QnliSystemPrompt.POLAR_V1: (
        "Does the sentence below contain the answer to the question? "
        "Answer {entailment} if it does and {not_entailment} if it does not."
    ),
    QnliSystemPrompt.INSTRUCT_V1: (
        "Read the question and sentence below and decide whether the "
        "sentence answers the question. Reply with exactly one word: "
        "{entailment} or {not_entailment}."
    ),
    QnliSystemPrompt.CHOICE_V1: (
        "Decide whether the sentence below answers the question. Your "
        "answer must be one of two options: {entailment} or {not_entailment}."
    ),
}

_RESPONSE_TOKEN_TEXTS: dict[QnliResponseToken, str] = {
    QnliResponseToken.RESPONSE: "Response:",
    QnliResponseToken.ANSWER: "Answer:",
    QnliResponseToken.LABEL: "Label:",
}

#: ``(entailment_word, not_entailment_word)`` per scheme.  Kept as a flat
#: table so the word-disjointness of set A and set B is checkable by reading
#: one block.
_LABEL_WORD_PAIRS: dict[QnliLabelScheme, tuple[str, str]] = {
    # -- set A --------------------------------------------------------------
    QnliLabelScheme.ENTAILMENT_V1: ("entailment", "not entailment"),
    QnliLabelScheme.TRUE_FALSE_V1: ("true", "false"),
    QnliLabelScheme.YES_NO_V1: ("yes", "no"),
    QnliLabelScheme.ENTAILED_NOT_ENTAILED_V1: ("entailed", "not entailed"),
    QnliLabelScheme.FOLLOWS_NOT_FOLLOWS_V1: ("follows", "does not follow"),
    QnliLabelScheme.IMPLIED_NOT_IMPLIED_V1: ("implied", "not implied"),
    QnliLabelScheme.ALPHA_BETA_V1: ("alpha", "beta"),
    QnliLabelScheme.ONE_ZERO_V1: ("one", "zero"),
    QnliLabelScheme.NORTH_SOUTH_V1: ("north", "south"),
    QnliLabelScheme.SUN_RAIN_V1: ("sun", "rain"),
    QnliLabelScheme.LEFT_RIGHT_V1: ("left", "right"),
    QnliLabelScheme.OPEN_SHUT_V1: ("open", "shut"),
    QnliLabelScheme.GOLD_IRON_V1: ("gold", "iron"),
    QnliLabelScheme.RIVER_STONE_V1: ("river", "stone"),
    QnliLabelScheme.MORNING_EVENING_V1: ("morning", "evening"),
    QnliLabelScheme.LOUD_QUIET_V1: ("loud", "quiet"),
    # -- set B --------------------------------------------------------------
    QnliLabelScheme.GREEN_RED_V1: ("green", "red"),
    QnliLabelScheme.UP_DOWN_V1: ("up", "down"),
    QnliLabelScheme.CAT_DOG_V1: ("cat", "dog"),
    QnliLabelScheme.FOO_BAR_V1: ("foo", "bar"),
    QnliLabelScheme.WARM_COLD_V1: ("warm", "cold"),
    QnliLabelScheme.BRIGHT_DIM_V1: ("bright", "dim"),
    QnliLabelScheme.ACCEPT_REJECT_V1: ("accept", "reject"),
    QnliLabelScheme.HAPPY_SAD_V1: ("happy", "sad"),
}

_LABEL_TEXTS: dict[QnliLabelScheme, dict[QnliLabel, str]] = {
    scheme: {QnliLabel.ENTAILMENT: ent, QnliLabel.NOT_ENTAILMENT: not_ent}
    for scheme, (ent, not_ent) in _LABEL_WORD_PAIRS.items()
}


def _label_format_kwargs(scheme: QnliLabelScheme) -> dict[str, str]:
    """``str.format`` kwargs for a scheme: ``{"entailment": ..., "not_entailment": ...}``."""
    return {
        label.name.lower(): text for label, text in _LABEL_TEXTS[scheme].items()
    }


def render_system_prompt(prompt: QnliSystemPrompt, label_scheme: QnliLabelScheme) -> str:
    """Render a system prompt with *label_scheme*'s label words substituted in."""
    return _SYSTEM_PROMPT_TEXTS[prompt].format(**_label_format_kwargs(label_scheme))


# Legacy module-level aliases derived from the default label scheme so any
# external code importing them keeps working.
_SYSTEM_PROMPT = render_system_prompt(
    QnliSystemPrompt.DEFAULT_V1, QnliLabelScheme.ENTAILMENT_V1
)
_LABEL_TO_STR = _LABEL_TEXTS[QnliLabelScheme.ENTAILMENT_V1]
_STR_TO_LABEL = {v: k for k, v in _LABEL_TO_STR.items()}
CATEGORIES = list(_LABEL_TO_STR.values())

#: Canonical (scheme-independent) spelling of each class, matching the
#: registry's ``label_int_to_str`` for QNLI (``DEFINITION.label_values`` /
#: ``CORPUS.label.int_to_str`` below).  NOTE: unlike most other templates in
#: this folder, ``QnliLabel.NOT_ENTAILMENT.name.lower()`` is
#: ``"not_entailment"`` (underscore) while the canonical registry spelling is
#: ``"not entailment"`` (space) -- ``name.lower()`` does *not* round-trip here,
#: so this table is keyed directly off the known canonical strings instead.
#: Dataset prep writes these *strings* into the ``label`` column
#: (``ColumnMappedCorpus.row_mapper``) rather than ``QnliLabel`` members, so
#: the template must recognise them to route a row through the active label
#: scheme -- see ``QnliChatTemplate.apply``.
_CANONICAL_STR_TO_LABEL: dict[str, QnliLabel] = {
    "entailment": QnliLabel.ENTAILMENT,
    "not entailment": QnliLabel.NOT_ENTAILMENT,
}


@dataclass
class QnliChatTemplate(SchemaTransform[QnliMessages, PreparedCompletion]):
    """Formats QNLI samples as prompt/completion pairs.

    Prompt layout::

        <system prompt>

        Question: <question text>
        Sentence: <sentence text>
        <response token> <label>

    The system prompt, response token, and label words are each chosen from
    a curated `VariantEnum`, so invalid configurations are unrepresentable.
    Defaults reproduce the historical layout byte-for-byte.

    When ``label`` is ``None`` the completion is the empty string.
    """

    transformation: ClassVar[str] = "qnli_format"

    add_eos: bool = False
    system_prompt: QnliSystemPrompt = QnliSystemPrompt.DEFAULT_V1
    response_token: QnliResponseToken = QnliResponseToken.RESPONSE
    label_scheme: QnliLabelScheme = QnliLabelScheme.ENTAILMENT_V1

    @override
    def apply(self, messages: QnliMessages, /) -> PreparedCompletion:
        sys_text = render_system_prompt(self.system_prompt, self.label_scheme)
        token_text = _RESPONSE_TOKEN_TEXTS[self.response_token]
        labels = _LABEL_TEXTS[self.label_scheme]

        prompt = (
            f"{sys_text}\n\n"
            f"Question: {messages['question']}\n"
            f"Sentence: {messages['sentence']}\n"
            f"{token_text} "
        )

        label = messages["label"]
        if label is None:
            completion = ""
        elif isinstance(label, QnliLabel):
            completion = labels[label]
        elif isinstance(label, int):
            completion = labels[QnliLabel(label)]
        elif label in _CANONICAL_STR_TO_LABEL:
            # Registry-prepped rows (and ``FixedLabelFlip`` targets) carry the
            # canonical label *string*, not a ``QnliLabel``. Without this
            # branch they fall through unchanged and ``label_scheme`` becomes a
            # silent no-op on exactly the pipeline that trains adapter pools.
            completion = labels[_CANONICAL_STR_TO_LABEL[label]]
        else:
            completion = label

        return {"prompt": prompt, "completion": completion, "add_eos": self.add_eos}

    def label_word_map(self) -> dict[str, str]:
        """Canonical dataset label string → the word this template emits for it.

        Keys are the registry's ``label_int_to_str`` spellings (what the
        ``label`` column actually holds); values are the active scheme's
        surface words.  Consumers that must bridge the two — a fuzzy matcher
        turning a generated ``"yes"`` back into the dataset's
        ``"not entailment"`` — read the mapping off the template rather than
        assuming the default scheme.
        """
        labels = _LABEL_TEXTS[self.label_scheme]
        return {
            canonical: labels[label]
            for canonical, label in _CANONICAL_STR_TO_LABEL.items()
        }

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "QnliChatTemplate":
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
class QnliParse(FallibleSchemaTransform[str, QnliMessages]):
    """Reverse-parse a QNLI prompt-completion concatenation.

    Strict string-to-enum lookup; unmatched strings flow through as the raw
    ``str``.  Use a fuzzy matcher in downstream code if needed.

    ``response_token`` and ``label_scheme`` must match whatever
    `QnliChatTemplate` was configured with for round-trip parsing to work.
    """

    transformation: ClassVar[str] = "qnli_parse"

    response_token: QnliResponseToken = QnliResponseToken.RESPONSE
    label_scheme: QnliLabelScheme = QnliLabelScheme.ENTAILMENT_V1

    @override
    def apply(self, text: str, /) -> Result[QnliMessages, TransformError]:
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
                    "Cannot parse system_prompt and body from text: "
                    f"expected {sep!r} separator not found.",
                    source=self.transformation,
                )
            )
        body = prompt_part[sep_idx + len(sep) :]

        q_prefix = "Question: "
        s_prefix = "Sentence: "
        question = ""
        sentence = ""
        for line in body.splitlines():
            if line.startswith(q_prefix):
                question = line[len(q_prefix) :]
            elif line.startswith(s_prefix):
                sentence = line[len(s_prefix) :]

        eot_token = "<|eot_id|>"
        if raw_response.endswith(eot_token):
            raw_response = raw_response[: -len(eot_token)].strip()

        str_to_label = {v: k for k, v in _LABEL_TEXTS[self.label_scheme].items()}
        label: QnliLabel | str = str_to_label.get(raw_response, raw_response)

        return Ok(QnliMessages(question=question, sentence=sentence, label=label))


# ---------------------------------------------------------------------------
# Corpus + task
# ---------------------------------------------------------------------------
# The corpus describes task-agnostic preparation (source, splits, raw -> typed
# mapping); the task names what it scores.  See dataset_definitions/_definition.py.

CORPUS = ColumnMappedCorpus(
    slug="qnli",
    hf_name='nyu-mll/glue',
    hf_config='qnli',
    splits_to_merge=('train', 'validation'),
    drop_splits=('test',),
    messages=QnliMessages,
    text_columns={'question': 'question', 'sentence': 'sentence'},
    label=LabelDecoding(
        field_name="label",
        int_to_str={0: 'entailment', 1: 'not entailment'},
    ),
)

DEFINITION = ClassificationTask(
    slug="qnli",
    corpus=CORPUS,
    chat_template=QnliChatTemplate,
    parser=QnliParse,
    label_field='label',
    label_values=('entailment', 'not entailment'),
)
