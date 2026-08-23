"""Subjectivity (SetFit/subj) classification dataset for the llm_pipeline framework.

(migrated from src/llm_pipeline/dataset_definitions/subj.py)

Schema mapping
--------------
SubjMessages fields <- SetFit/subj columns:
  sentence           <- text
  subjectivity       <- SubjSubjectivity enum from label (0=OBJECTIVE, 1=SUBJECTIVE)

Polarity was verified against the data: label 0 rows are objective plot
descriptions and label 1 rows are subjective review sentences, matching the
hub's own ``label_text`` column (``"objective"`` / ``"subjective"``).

Split handling
--------------
The hub exposes two splits, ``train`` (8,000) and ``test`` (2,000), both fully
labelled with the same columns. They are kept separate rather than merged --
nothing about this dataset calls for flattening its natural train/test split.
``train`` is balanced at 3,967 objective / 4,033 subjective rows.
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


@serialisable("SubjSubjectivity")
class SubjSubjectivity(Enum):
    OBJECTIVE = 0
    SUBJECTIVE = 1


class SubjMessages(TypedDict):
    sentence: str
    subjectivity: SubjSubjectivity | str | None


class PoisonedSubjMessages(SubjMessages):
    original: SubjSubjectivity | str | None
    is_poisoned: bool


# ---------------------------------------------------------------------------
# Chat template
# ---------------------------------------------------------------------------

_DEFAULT_SYSTEM_PROMPT = (
    "Given a sentence, your task is to classify it as either objective or "
    "subjective. An objective sentence states a fact or describes a plot "
    "event without personal opinion, while a subjective sentence expresses a "
    "personal opinion, judgement, or feeling. Your response should clearly "
    "indicate whether the sentence is objective or subjective."
)


@serialisable("SubjSystemPrompt")
class SubjSystemPrompt(VariantEnum):
    DEFAULT_V1 = ("default", 1)
    CONCISE_V1 = ("concise", 1)


@serialisable("SubjResponseToken")
class SubjResponseToken(VariantEnum):
    RESPONSE = ("response", 1)
    ANSWER = ("answer", 1)
    LABEL = ("label", 1)


@serialisable("SubjLabelScheme")
class SubjLabelScheme(VariantEnum):
    OBJECTIVE_SUBJECTIVE_V1 = ("objective_subjective", 1)
    TRUE_FALSE_V1 = ("true_false", 1)
    YES_NO_V1 = ("yes_no", 1)


_SYSTEM_PROMPT_TEXTS: dict[SubjSystemPrompt, str] = {
    SubjSystemPrompt.DEFAULT_V1: _DEFAULT_SYSTEM_PROMPT,
    SubjSystemPrompt.CONCISE_V1: (
        "Classify the following sentence as either objective or subjective."
    ),
}

_RESPONSE_TOKEN_TEXTS: dict[SubjResponseToken, str] = {
    SubjResponseToken.RESPONSE: "Response:",
    SubjResponseToken.ANSWER: "Answer:",
    SubjResponseToken.LABEL: "Label:",
}

_LABEL_TEXTS: dict[SubjLabelScheme, dict[SubjSubjectivity, str]] = {
    SubjLabelScheme.OBJECTIVE_SUBJECTIVE_V1: {
        SubjSubjectivity.OBJECTIVE: "objective",
        SubjSubjectivity.SUBJECTIVE: "subjective",
    },
    SubjLabelScheme.TRUE_FALSE_V1: {
        SubjSubjectivity.OBJECTIVE: "false",
        SubjSubjectivity.SUBJECTIVE: "true",
    },
    SubjLabelScheme.YES_NO_V1: {
        SubjSubjectivity.OBJECTIVE: "no",
        SubjSubjectivity.SUBJECTIVE: "yes",
    },
}

_SYSTEM_PROMPT = _DEFAULT_SYSTEM_PROMPT
_SUBJECTIVITY_TO_STR = _LABEL_TEXTS[SubjLabelScheme.OBJECTIVE_SUBJECTIVE_V1]
_STR_TO_SUBJECTIVITY = {v: k for k, v in _SUBJECTIVITY_TO_STR.items()}
CATEGORIES = list(_SUBJECTIVITY_TO_STR.values())


@dataclass
class SubjChatTemplate(SchemaTransform[SubjMessages, PreparedCompletion]):
    """Formats Subj samples as prompt/completion pairs.

    Prompt layout::

        <system prompt>

        Sentence: <sentence text>
        <response token> <label>
    """

    transformation: ClassVar[str] = "subj_format"

    add_eos: bool = False
    system_prompt: SubjSystemPrompt = SubjSystemPrompt.DEFAULT_V1
    response_token: SubjResponseToken = SubjResponseToken.RESPONSE
    label_scheme: SubjLabelScheme = SubjLabelScheme.OBJECTIVE_SUBJECTIVE_V1

    @override
    def apply(self, messages: SubjMessages, /) -> PreparedCompletion:
        sys_text = _SYSTEM_PROMPT_TEXTS[self.system_prompt]
        token_text = _RESPONSE_TOKEN_TEXTS[self.response_token]
        labels = _LABEL_TEXTS[self.label_scheme]

        prompt = f"{sys_text}\n\nSentence: {messages['sentence']}\n{token_text} "

        subjectivity = messages["subjectivity"]
        if subjectivity is None:
            completion = ""
        elif isinstance(subjectivity, SubjSubjectivity):
            completion = labels[subjectivity]
        elif isinstance(subjectivity, int):
            completion = labels[SubjSubjectivity(subjectivity)]
        elif subjectivity in _STR_TO_SUBJECTIVITY:
            # Registry-prepped rows (and ``FixedLabelFlip`` targets) carry the
            # canonical label *string*, not a ``SubjSubjectivity``.  Without
            # this branch they fall through unchanged and ``label_scheme``
            # becomes a silent no-op on exactly the pipeline that trains
            # adapter pools.
            completion = labels[_STR_TO_SUBJECTIVITY[subjectivity]]
        else:
            completion = subjectivity

        return {"prompt": prompt, "completion": completion, "add_eos": self.add_eos}

    def label_word_map(self) -> dict[str, str]:
        """Canonical dataset label string → the word this template emits for it.

        Keys are the registry's ``label_int_to_str`` spellings (what the
        ``subjectivity`` column actually holds); values are the active
        scheme's surface words.  The generic eval builder reads this to give a
        fuzzy matcher the surface vocabulary without assuming the default
        scheme.
        """
        labels = _LABEL_TEXTS[self.label_scheme]
        return {
            _SUBJECTIVITY_TO_STR[member]: labels[member]
            for member in SubjSubjectivity
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
class SubjParse(FallibleSchemaTransform[str, SubjMessages]):
    """Reverse-parse a Subj prompt-completion concatenation.

    ``response_token`` and ``label_scheme`` must match whatever
    `SubjChatTemplate` was configured with for round-trip parsing to work.
    """

    transformation: ClassVar[str] = "subj_parse"

    response_token: SubjResponseToken = SubjResponseToken.RESPONSE
    label_scheme: SubjLabelScheme = SubjLabelScheme.OBJECTIVE_SUBJECTIVE_V1

    @override
    def apply(self, text: str, /) -> Result[SubjMessages, TransformError]:
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
                    "Cannot parse system_prompt and sentence from text: "
                    f"expected {sep!r} separator not found.",
                    source=self.transformation,
                )
            )
        sentence_part = prompt_part[sep_idx + len(sep) :]

        sentence_prefix = "Sentence: "
        sentence = (
            sentence_part[len(sentence_prefix) :]
            if sentence_part.startswith(sentence_prefix)
            else sentence_part
        )

        eot_token = "<|eot_id|>"
        if raw_response.endswith(eot_token):
            raw_response = raw_response[: -len(eot_token)].strip()

        str_to_subjectivity = {
            v: k for k, v in _LABEL_TEXTS[self.label_scheme].items()
        }
        subjectivity: SubjSubjectivity | str = str_to_subjectivity.get(
            raw_response, raw_response
        )

        return Ok(SubjMessages(sentence=sentence, subjectivity=subjectivity))


# ---------------------------------------------------------------------------
# Corpus + task
# ---------------------------------------------------------------------------
# The corpus describes task-agnostic preparation (source, splits, raw -> typed
# mapping); the task names what it scores.  See dataset_definitions/_definition.py.

CORPUS = ColumnMappedCorpus(
    slug="subj",
    hf_name='SetFit/subj',
    hf_config=None,
    splits_to_merge=('train', 'test'),
    messages=SubjMessages,
    text_columns={'sentence': 'text'},
    label=LabelDecoding(
        field_name="subjectivity",
        int_to_str={0: 'objective', 1: 'subjective'},
    ),
)

DEFINITION = ClassificationTask(
    slug="subj",
    corpus=CORPUS,
    chat_template=SubjChatTemplate,
    parser=SubjParse,
    label_field='subjectivity',
    label_values=('objective', 'subjective'),
)
