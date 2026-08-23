"""ToxiGen (machine-generated implicit hate speech) for the llm_pipeline framework.

(migrated from src/llm_pipeline/dataset_definitions/toxigen.py)

Schema mapping
--------------
ToxigenMessages fields <- toxigen/toxigen-data (config="train") columns:
  statement          <- generation
  toxicity           <- ToxigenToxicity enum from prompt_label (0=BENIGN, 1=TOXIC)

The HF label column used here is ``prompt_label``, which labels the *prompt*
that was fed to the generator, not the individual ``generation`` it produced.
These are therefore weak, prompt-level labels applied to machine-written
text -- a generation may occasionally read differently from its prompt's
intended polarity. The dataset also ships an ``annotated`` config with
human-annotated float toxicity scores, but it holds only 8,960 rows, far too
few for adapter-pool training; the 250,951-row ``train`` config is used
instead, accepting the weaker label in exchange for scale.

Split handling
--------------
The ``train`` config of toxigen/toxigen-data exposes a single ``train``
split (250,951 rows). No merging is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, override

from shared_adapter_pool.data.definitions._serialisable import serialisable
from typing_extensions import TypedDict

from shared_adapter_pool.data.definitions._variants import VariantEnum, render_prompt
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


@serialisable("ToxigenToxicity")
class ToxigenToxicity(Enum):
    BENIGN = 0
    TOXIC = 1


class ToxigenMessages(TypedDict):
    statement: str
    toxicity: ToxigenToxicity | str | None


class PoisonedToxigenMessages(ToxigenMessages):
    original: ToxigenToxicity | str | None
    is_poisoned: bool


# ---------------------------------------------------------------------------
# Chat template
# ---------------------------------------------------------------------------

_DEFAULT_SYSTEM_PROMPT = (
    "Given a statement, your task is to classify it as either toxic or "
    "benign. A toxic statement expresses or implies hatred, prejudice, or "
    "harm toward a group of people, even when phrased subtly. Your response "
    "should clearly indicate whether the statement is toxic or benign."
)


@serialisable("ToxigenSystemPrompt")
class ToxigenSystemPrompt(VariantEnum):
    DEFAULT_V1 = ("default", 1)
    CONCISE_V1 = ("concise", 1)
    POLAR_V1 = ("polar", 1)


@serialisable("ToxigenResponseToken")
class ToxigenResponseToken(VariantEnum):
    RESPONSE = ("response", 1)
    ANSWER = ("answer", 1)
    LABEL = ("label", 1)


@serialisable("ToxigenLabelScheme")
class ToxigenLabelScheme(VariantEnum):
    BENIGN_TOXIC_V1 = ("benign_toxic", 1)
    TRUE_FALSE_V1 = ("true_false", 1)
    YES_NO_V1 = ("yes_no", 1)


#: ``POLAR_V1`` is a ``str.format`` template over the *label words* of the
#: prompt's label scheme: ``{toxic}`` / ``{benign}`` (the lower-cased
#: :class:`ToxigenToxicity` member names).  The keys name the *class*, not
#: the scheme -- toxigen's default label scheme is ``BENIGN_TOXIC_V1``, so
#: ``{toxic}``/``{benign}`` render as ``"toxic"``/``"benign"`` by default.
#: The question itself is phrased independently of the label words (mirrors
#: ``BoolQSystemPrompt.POLAR_V1``'s literal "is the answer yes?"), so it
#: stays grammatical under whichever label scheme is active; only the
#: answer tokens are substituted.
_POLAR_SYSTEM_PROMPT = (
    "Given a statement, is it toxic? A toxic statement expresses or implies "
    "hatred, prejudice, or harm toward a group of people, even when phrased "
    "subtly. Answer {toxic} if it is and {benign} if it is not."
)

_SYSTEM_PROMPT_TEXTS: dict[ToxigenSystemPrompt, str] = {
    ToxigenSystemPrompt.DEFAULT_V1: _DEFAULT_SYSTEM_PROMPT,
    ToxigenSystemPrompt.CONCISE_V1: (
        "Classify the following statement as either toxic or benign."
    ),
    ToxigenSystemPrompt.POLAR_V1: _POLAR_SYSTEM_PROMPT,
}

_RESPONSE_TOKEN_TEXTS: dict[ToxigenResponseToken, str] = {
    ToxigenResponseToken.RESPONSE: "Response:",
    ToxigenResponseToken.ANSWER: "Answer:",
    ToxigenResponseToken.LABEL: "Label:",
}

_LABEL_TEXTS: dict[ToxigenLabelScheme, dict[ToxigenToxicity, str]] = {
    ToxigenLabelScheme.BENIGN_TOXIC_V1: {
        ToxigenToxicity.TOXIC: "toxic",
        ToxigenToxicity.BENIGN: "benign",
    },
    ToxigenLabelScheme.TRUE_FALSE_V1: {
        ToxigenToxicity.TOXIC: "true",
        ToxigenToxicity.BENIGN: "false",
    },
    ToxigenLabelScheme.YES_NO_V1: {
        ToxigenToxicity.TOXIC: "yes",
        ToxigenToxicity.BENIGN: "no",
    },
}


def render_system_prompt(
    prompt: ToxigenSystemPrompt, label_scheme: ToxigenLabelScheme
) -> str:
    """Render a system prompt with *label_scheme*'s label words substituted in."""
    return render_prompt(_SYSTEM_PROMPT_TEXTS[prompt], _LABEL_TEXTS[label_scheme])


# Legacy module-level alias derived from the default label scheme so any
# external code importing it keeps working.
_SYSTEM_PROMPT = render_system_prompt(
    ToxigenSystemPrompt.DEFAULT_V1, ToxigenLabelScheme.BENIGN_TOXIC_V1
)
_TOXICITY_TO_STR = _LABEL_TEXTS[ToxigenLabelScheme.BENIGN_TOXIC_V1]
_STR_TO_TOXICITY = {v: k for k, v in _TOXICITY_TO_STR.items()}
CATEGORIES = list(_TOXICITY_TO_STR.values())


@dataclass
class ToxigenChatTemplate(SchemaTransform[ToxigenMessages, PreparedCompletion]):
    """Formats ToxiGen samples as prompt/completion pairs.

    Prompt layout::

        <system prompt>

        Statement: <statement text>
        <response token> <label>

    Generations are one or two sentences, so no ``max_text_chars`` truncation
    is applied.
    """

    transformation: ClassVar[str] = "toxigen_format"

    add_eos: bool = False
    system_prompt: ToxigenSystemPrompt = ToxigenSystemPrompt.DEFAULT_V1
    response_token: ToxigenResponseToken = ToxigenResponseToken.RESPONSE
    label_scheme: ToxigenLabelScheme = ToxigenLabelScheme.BENIGN_TOXIC_V1

    @override
    def apply(self, messages: ToxigenMessages, /) -> PreparedCompletion:
        sys_text = render_system_prompt(self.system_prompt, self.label_scheme)
        token_text = _RESPONSE_TOKEN_TEXTS[self.response_token]
        labels = _LABEL_TEXTS[self.label_scheme]

        statement = messages["statement"]

        prompt = f"{sys_text}\n\nStatement: {statement}\n{token_text} "

        toxicity = messages["toxicity"]
        if toxicity is None:
            completion = ""
        elif isinstance(toxicity, ToxigenToxicity):
            completion = labels[toxicity]
        elif isinstance(toxicity, int):
            completion = labels[ToxigenToxicity(toxicity)]
        elif toxicity in _STR_TO_TOXICITY:
            # Registry-prepped rows (and ``FixedLabelFlip`` targets) carry the
            # canonical label *string*, not a ``ToxigenToxicity``. Without
            # this branch they fall through unchanged and ``label_scheme``
            # becomes a silent no-op on exactly the pipeline that trains
            # adapter pools.
            completion = labels[_STR_TO_TOXICITY[toxicity]]
        else:
            completion = toxicity

        return {"prompt": prompt, "completion": completion, "add_eos": self.add_eos}

    def label_word_map(self) -> dict[str, str]:
        """Canonical dataset label string → the word this template emits for it.

        Keys are the registry's ``label_int_to_str`` spellings (what the
        ``toxicity`` column actually holds); values are the active scheme's
        surface words. The generic eval builder reads this to give a fuzzy
        matcher the surface vocabulary without assuming the default scheme.
        """
        labels = _LABEL_TEXTS[self.label_scheme]
        return {
            _TOXICITY_TO_STR[member]: labels[member] for member in ToxigenToxicity
        }

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
class ToxigenParse(FallibleSchemaTransform[str, ToxigenMessages]):
    """Reverse-parse a ToxiGen prompt-completion concatenation.

    ``response_token`` and ``label_scheme`` must match whatever
    `ToxigenChatTemplate` was configured with for round-trip parsing to work.
    """

    transformation: ClassVar[str] = "toxigen_parse"

    response_token: ToxigenResponseToken = ToxigenResponseToken.RESPONSE
    label_scheme: ToxigenLabelScheme = ToxigenLabelScheme.BENIGN_TOXIC_V1

    @override
    def apply(self, text: str, /) -> Result[ToxigenMessages, TransformError]:
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
                    "Cannot parse system_prompt and statement from text: "
                    f"expected {sep!r} separator not found.",
                    source=self.transformation,
                )
            )
        statement_part = prompt_part[sep_idx + len(sep) :]

        statement_prefix = "Statement: "
        statement = (
            statement_part[len(statement_prefix) :]
            if statement_part.startswith(statement_prefix)
            else statement_part
        )

        eot_token = "<|eot_id|>"
        if raw_response.endswith(eot_token):
            raw_response = raw_response[: -len(eot_token)].strip()

        str_to_toxicity = {v: k for k, v in _LABEL_TEXTS[self.label_scheme].items()}
        toxicity: ToxigenToxicity | str = str_to_toxicity.get(
            raw_response, raw_response
        )

        return Ok(ToxigenMessages(statement=statement, toxicity=toxicity))


# ---------------------------------------------------------------------------
# Corpus + task
# ---------------------------------------------------------------------------
# The corpus describes task-agnostic preparation (source, splits, raw -> typed
# mapping); the task names what it scores.  See dataset_definitions/_definition.py.

CORPUS = ColumnMappedCorpus(
    slug="toxigen",
    hf_name='toxigen/toxigen-data',
    hf_config='train',
    splits_to_merge=('train',),
    messages=ToxigenMessages,
    text_columns={'statement': 'generation'},
    label=LabelDecoding(
        field_name="toxicity",
        source_column='prompt_label',
        int_to_str={0: 'benign', 1: 'toxic'},
    ),
)

DEFINITION = ClassificationTask(
    slug="toxigen",
    corpus=CORPUS,
    chat_template=ToxigenChatTemplate,
    parser=ToxigenParse,
    label_field='toxicity',
    label_values=('benign', 'toxic'),
)
