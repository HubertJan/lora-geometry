"""SciTail (science-exam textual entailment) for the llm_pipeline framework.

(migrated from src/llm_pipeline/dataset_definitions/scitail.py)

Schema mapping
--------------
SciTailMessages fields <- allenai/scitail (config="tsv_format") columns:
  premise            <- premise
  hypothesis         <- hypothesis
  entailment         <- SciTailEntailment enum from label (0=NOT_ENTAILED, 1=ENTAILED)

The HF ``label`` column holds the strings ``"entails"`` / ``"neutral"``, which
the registry re-maps to the canonical words (``label_str_to_str``) — unlike
the 3-way GLUE NLI sets already registered (MNLI, ANLI, ...), SciTail is
natively two-label, so no class is dropped.

Split handling
--------------
The ``tsv_format`` config exposes three splits (``train``, ``validation``,
``test``), all sharing the identical column schema and all fully labeled.
No split is unlabeled or schema-mismatched, so all three are kept as-is --
no merge is needed.

The corpus is science-exam derived and skewed toward the neutral class
(14,625 / 8,472 in ``train``).
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


@serialisable("SciTailEntailment")
class SciTailEntailment(Enum):
    NOT_ENTAILED = 0
    ENTAILED = 1


class SciTailMessages(TypedDict):
    premise: str
    hypothesis: str
    entailment: SciTailEntailment | str | None


class PoisonedSciTailMessages(SciTailMessages):
    original: SciTailEntailment | str | None
    is_poisoned: bool


# ---------------------------------------------------------------------------
# Chat template
# ---------------------------------------------------------------------------

#: System prompts are ``str.format`` templates over the *label words* of the
#: active label scheme: ``{entailed}`` / ``{not_entailed}`` (the lower-cased
#: :class:`SciTailEntailment` member names).  Rendering with the default
#: ``entailed_not_entailed`` scheme reproduces the historical text
#: byte-for-byte.
_DEFAULT_SYSTEM_PROMPT = (
    "You are given a premise sentence and a hypothesis sentence drawn from a "
    "science exam. Your task is to determine whether the hypothesis is "
    "entailed by the premise. Your response should clearly indicate whether "
    "the hypothesis is {entailed} or {not_entailed}."
)


@serialisable("SciTailSystemPrompt")
class SciTailSystemPrompt(VariantEnum):
    DEFAULT_V1 = ("default", 1)
    CONCISE_V1 = ("concise", 1)
    POLAR_V1 = ("polar", 1)
    INSTRUCT_V1 = ("instruct", 1)
    CHOICE_V1 = ("choice", 1)


@serialisable("SciTailResponseToken")
class SciTailResponseToken(VariantEnum):
    RESPONSE = ("response", 1)
    ANSWER = ("answer", 1)
    LABEL = ("label", 1)


@serialisable("SciTailLabelScheme")
class SciTailLabelScheme(VariantEnum):
    """Verbalizers for SciTail, in two deliberately word-disjoint groups.

    The first sixteen members (through ``LOUD_QUIET_V1``) and the last eight
    share no surface word, so a detector trained on adapters drawn from one
    group has never seen the other group's target tokens.  That disjointness
    is the experimental contract of ``btf_label_diversity`` and is *checked*
    at plan-validation time against ``label_word_map()``, not assumed here.

    Members past the historical three are semantically arbitrary answer
    tokens.  That is the point: they decouple "the backdoor target category"
    from "the surface token the target is written as".
    """

    # -- set A: available to training pools ---------------------------------
    ENTAILED_NOT_ENTAILED_V1 = ("entailed_not_entailed", 1)
    TRUE_FALSE_V1 = ("true_false", 1)
    YES_NO_V1 = ("yes_no", 1)
    ENTAILMENT_NO_ENTAILMENT_V1 = ("entailment_no_entailment", 1)
    SUPPORTED_NOT_SUPPORTED_V1 = ("supported_not_supported", 1)
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


_SYSTEM_PROMPT_TEXTS: dict[SciTailSystemPrompt, str] = {
    SciTailSystemPrompt.DEFAULT_V1: _DEFAULT_SYSTEM_PROMPT,
    SciTailSystemPrompt.CONCISE_V1: (
        "Given the premise and hypothesis below, state whether the "
        "hypothesis is {entailed} by the premise."
    ),
    # `default`/`concise` slot the label words into *adjectival* positions
    # ("is X" / "is X or Y"), which only reads naturally for adjective-like
    # labels.  The three families below state the task independently of the
    # label words and use them purely as answer tokens, so they stay
    # grammatical under an arbitrary verbalizer such as sun/rain.
    SciTailSystemPrompt.POLAR_V1: (
        "Does the premise below support the hypothesis? Answer {entailed} "
        "if it does and {not_entailed} if it does not."
    ),
    SciTailSystemPrompt.INSTRUCT_V1: (
        "Read the premise and hypothesis below and decide whether the "
        "premise supports the hypothesis. Reply with exactly one word: "
        "{entailed} or {not_entailed}."
    ),
    SciTailSystemPrompt.CHOICE_V1: (
        "Decide whether the premise below supports the hypothesis. Your "
        "answer must be one of two options: {entailed} or {not_entailed}."
    ),
}

_RESPONSE_TOKEN_TEXTS: dict[SciTailResponseToken, str] = {
    SciTailResponseToken.RESPONSE: "Response:",
    SciTailResponseToken.ANSWER: "Answer:",
    SciTailResponseToken.LABEL: "Label:",
}

#: ``(entailed_word, not_entailed_word)`` per scheme.  Kept as a flat table
#: so the word-disjointness of set A and set B is checkable by reading one
#: block.
_LABEL_WORD_PAIRS: dict[SciTailLabelScheme, tuple[str, str]] = {
    # -- set A --------------------------------------------------------------
    SciTailLabelScheme.ENTAILED_NOT_ENTAILED_V1: ("entailed", "not entailed"),
    SciTailLabelScheme.TRUE_FALSE_V1: ("true", "false"),
    SciTailLabelScheme.YES_NO_V1: ("yes", "no"),
    SciTailLabelScheme.ENTAILMENT_NO_ENTAILMENT_V1: ("entailment", "no entailment"),
    SciTailLabelScheme.SUPPORTED_NOT_SUPPORTED_V1: ("supported", "not supported"),
    SciTailLabelScheme.IMPLIED_NOT_IMPLIED_V1: ("implied", "not implied"),
    SciTailLabelScheme.ALPHA_BETA_V1: ("alpha", "beta"),
    SciTailLabelScheme.ONE_ZERO_V1: ("one", "zero"),
    SciTailLabelScheme.NORTH_SOUTH_V1: ("north", "south"),
    SciTailLabelScheme.SUN_RAIN_V1: ("sun", "rain"),
    SciTailLabelScheme.LEFT_RIGHT_V1: ("left", "right"),
    SciTailLabelScheme.OPEN_SHUT_V1: ("open", "shut"),
    SciTailLabelScheme.GOLD_IRON_V1: ("gold", "iron"),
    SciTailLabelScheme.RIVER_STONE_V1: ("river", "stone"),
    SciTailLabelScheme.MORNING_EVENING_V1: ("morning", "evening"),
    SciTailLabelScheme.LOUD_QUIET_V1: ("loud", "quiet"),
    # -- set B --------------------------------------------------------------
    SciTailLabelScheme.GREEN_RED_V1: ("green", "red"),
    SciTailLabelScheme.UP_DOWN_V1: ("up", "down"),
    SciTailLabelScheme.CAT_DOG_V1: ("cat", "dog"),
    SciTailLabelScheme.FOO_BAR_V1: ("foo", "bar"),
    SciTailLabelScheme.WARM_COLD_V1: ("warm", "cold"),
    SciTailLabelScheme.BRIGHT_DIM_V1: ("bright", "dim"),
    SciTailLabelScheme.ACCEPT_REJECT_V1: ("accept", "reject"),
    SciTailLabelScheme.HAPPY_SAD_V1: ("happy", "sad"),
}

_LABEL_TEXTS: dict[SciTailLabelScheme, dict[SciTailEntailment, str]] = {
    scheme: {SciTailEntailment.ENTAILED: ent, SciTailEntailment.NOT_ENTAILED: not_ent}
    for scheme, (ent, not_ent) in _LABEL_WORD_PAIRS.items()
}


def _label_format_kwargs(scheme: SciTailLabelScheme) -> dict[str, str]:
    """``str.format`` kwargs for a scheme: ``{"entailed": ..., "not_entailed": ...}``."""
    return {
        entailment.name.lower(): text
        for entailment, text in _LABEL_TEXTS[scheme].items()
    }


def render_system_prompt(
    prompt: SciTailSystemPrompt, label_scheme: SciTailLabelScheme
) -> str:
    """Render a system prompt with *label_scheme*'s label words substituted in."""
    return _SYSTEM_PROMPT_TEXTS[prompt].format(**_label_format_kwargs(label_scheme))


# Legacy module-level aliases derived from the default label scheme so any
# external code importing them keeps working.
_SYSTEM_PROMPT = render_system_prompt(
    SciTailSystemPrompt.DEFAULT_V1, SciTailLabelScheme.ENTAILED_NOT_ENTAILED_V1
)
_ENTAILMENT_TO_STR = _LABEL_TEXTS[SciTailLabelScheme.ENTAILED_NOT_ENTAILED_V1]
_STR_TO_ENTAILMENT = {v: k for k, v in _ENTAILMENT_TO_STR.items()}
CATEGORIES = list(_ENTAILMENT_TO_STR.values())

#: Canonical (scheme-independent) spelling of each class, matching the
#: registry's ``label_int_to_str`` for SciTail (see ``DEFINITION.label_values``
#: below: ``"not entailed"`` / ``"entailed"``, space-separated).  Dataset prep
#: writes these *strings* into the ``entailment`` column
#: (``LabelDecoding.str_to_str``) rather than ``SciTailEntailment`` members, so
#: the template must recognise them to route a row through the active label
#: scheme -- see ``SciTailChatTemplate.apply``.  Both the canonical
#: space-spelling and the lower-cased member-name (underscore) spelling are
#: accepted, since callers may reasonably construct either.
_CANONICAL_STR_TO_ENTAILMENT: dict[str, SciTailEntailment] = {
    "not entailed": SciTailEntailment.NOT_ENTAILED,
    "entailed": SciTailEntailment.ENTAILED,
    "not_entailed": SciTailEntailment.NOT_ENTAILED,
}


@dataclass
class SciTailChatTemplate(SchemaTransform[SciTailMessages, PreparedCompletion]):
    """Formats SciTail samples as prompt/completion pairs.

    Prompt layout::

        <system prompt>

        Premise: <premise>
        Hypothesis: <hypothesis>
        <response token> <label>
    """

    transformation: ClassVar[str] = "scitail_format"

    add_eos: bool = False
    system_prompt: SciTailSystemPrompt = SciTailSystemPrompt.DEFAULT_V1
    response_token: SciTailResponseToken = SciTailResponseToken.RESPONSE
    label_scheme: SciTailLabelScheme = SciTailLabelScheme.ENTAILED_NOT_ENTAILED_V1

    @override
    def apply(self, messages: SciTailMessages, /) -> PreparedCompletion:
        sys_text = render_system_prompt(self.system_prompt, self.label_scheme)
        token_text = _RESPONSE_TOKEN_TEXTS[self.response_token]
        labels = _LABEL_TEXTS[self.label_scheme]

        prompt = (
            f"{sys_text}\n\n"
            f"Premise: {messages['premise']}\n"
            f"Hypothesis: {messages['hypothesis']}\n"
            f"{token_text} "
        )

        entailment = messages["entailment"]
        if entailment is None:
            completion = ""
        elif isinstance(entailment, SciTailEntailment):
            completion = labels[entailment]
        elif isinstance(entailment, int):
            completion = labels[SciTailEntailment(entailment)]
        elif entailment in _CANONICAL_STR_TO_ENTAILMENT:
            # Registry-prepped rows (and ``FixedLabelFlip`` targets) carry the
            # canonical label *string*, not a ``SciTailEntailment``.  Without
            # this branch they fall through unchanged and ``label_scheme``
            # becomes a silent no-op on exactly the pipeline that trains
            # adapter pools.
            completion = labels[_CANONICAL_STR_TO_ENTAILMENT[entailment]]
        else:
            completion = entailment

        return {"prompt": prompt, "completion": completion, "add_eos": self.add_eos}

    def label_word_map(self) -> dict[str, str]:
        """Canonical dataset label string → the word this template emits for it.

        Keys are the registry's ``label_int_to_str`` spellings (what the
        ``entailment`` column actually holds); values are the active scheme's
        surface words.  The generic eval builder reads this to give a fuzzy
        matcher the surface vocabulary without assuming the default scheme.
        """
        labels = _LABEL_TEXTS[self.label_scheme]
        return {
            _ENTAILMENT_TO_STR[member]: labels[member]
            for member in SciTailEntailment
        }

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "SciTailChatTemplate":
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
class SciTailParse(FallibleSchemaTransform[str, SciTailMessages]):
    """Reverse-parse a SciTail prompt-completion concatenation.

    ``response_token`` and ``label_scheme`` must match whatever
    `SciTailChatTemplate` was configured with for round-trip parsing to work.
    """

    transformation: ClassVar[str] = "scitail_parse"

    response_token: SciTailResponseToken = SciTailResponseToken.RESPONSE
    label_scheme: SciTailLabelScheme = SciTailLabelScheme.ENTAILED_NOT_ENTAILED_V1

    @override
    def apply(self, text: str, /) -> Result[SciTailMessages, TransformError]:
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

        premise_prefix = "Premise: "
        hypothesis_prefix = "Hypothesis: "
        premise = ""
        hypothesis = ""
        for line in body.splitlines():
            if line.startswith(premise_prefix):
                premise = line[len(premise_prefix) :]
            elif line.startswith(hypothesis_prefix):
                hypothesis = line[len(hypothesis_prefix) :]

        eot_token = "<|eot_id|>"
        if raw_response.endswith(eot_token):
            raw_response = raw_response[: -len(eot_token)].strip()

        str_to_entailment = {
            v: k for k, v in _LABEL_TEXTS[self.label_scheme].items()
        }
        entailment: SciTailEntailment | str = str_to_entailment.get(
            raw_response, raw_response
        )

        return Ok(
            SciTailMessages(
                premise=premise, hypothesis=hypothesis, entailment=entailment
            )
        )


# ---------------------------------------------------------------------------
# Corpus + task
# ---------------------------------------------------------------------------
# The corpus describes task-agnostic preparation (source, splits, raw -> typed
# mapping); the task names what it scores.  See dataset_definitions/_definition.py.

CORPUS = ColumnMappedCorpus(
    slug="scitail",
    hf_name='allenai/scitail',
    hf_config='tsv_format',
    splits_to_merge=('train', 'validation', 'test'),
    messages=SciTailMessages,
    text_columns={'premise': 'premise', 'hypothesis': 'hypothesis'},
    label=LabelDecoding(
        field_name="entailment",
        str_to_str={'entails': 'entailed', 'neutral': 'not entailed'},
        drop_none=True,
    ),
)


DEFINITION = ClassificationTask(
    slug="scitail",
    corpus=CORPUS,
    chat_template=SciTailChatTemplate,
    parser=SciTailParse,
    label_field='entailment',
    label_values=('not entailed', 'entailed'),
)
