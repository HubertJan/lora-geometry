"""BoolQ (Boolean Questions) from SuperGLUE.

(migrated from src/llm_pipeline/dataset_definitions/boolq.py)

Schema mapping
--------------
BoolQMessages fields <- aps/super_glue (config="boolq") columns:
  passage          <- passage
  question         <- question
  label            <- BoolQLabel enum from label (0=FALSE, 1=TRUE)

Split handling
--------------
The SuperGLUE test split carries label == -1 for all rows (labels are not
publicly released).  It is dropped.  Train and validation are merged into
a single ``train`` split.
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


@serialisable("BoolQLabel")
class BoolQLabel(Enum):
    FALSE = 0
    TRUE = 1


class BoolQMessages(TypedDict):
    passage: str
    question: str
    label: BoolQLabel | str | None


class PoisonedBoolQMessages(BoolQMessages):
    original: BoolQLabel | str | None
    is_poisoned: bool


#: System prompts are ``str.format`` templates over the *label words* of the
#: prompt's label scheme: ``{true}`` / ``{false}`` (the lower-cased
#: :class:`BoolQLabel` member names).  The key names the *class*, not the
#: scheme -- boolq's default label scheme is ``YES_NO_V1``, so ``{true}``
#: renders as ``"yes"`` by default.  That reads oddly and is correct; don't
#: "fix" it to ``{yes}``.  Rendering with the default ``yes_no`` scheme
#: reproduces the historical text byte-for-byte.
_DEFAULT_SYSTEM_PROMPT = (
    "You are given a passage and a question. Your task is to determine "
    "whether the answer to the question is {true} or {false} based on the "
    "passage. Respond with exactly one word: {true} or {false}."
)


@serialisable("BoolQSystemPrompt")
class BoolQSystemPrompt(VariantEnum):
    DEFAULT_V1 = ("default", 1)
    CONCISE_V1 = ("concise", 1)
    POLAR_V1 = ("polar", 1)


@serialisable("BoolQResponseToken")
class BoolQResponseToken(VariantEnum):
    RESPONSE = ("response", 1)
    ANSWER = ("answer", 1)


@serialisable("BoolQLabelScheme")
class BoolQLabelScheme(VariantEnum):
    YES_NO_V1 = ("yes_no", 1)
    TRUE_FALSE_V1 = ("true_false", 1)


_SYSTEM_PROMPT_TEXTS: dict[BoolQSystemPrompt, str] = {
    BoolQSystemPrompt.DEFAULT_V1: _DEFAULT_SYSTEM_PROMPT,
    BoolQSystemPrompt.CONCISE_V1: (
        "Based on the passage, answer the question with {true} or {false}."
    ),
    # The `default`/`concise` families slot the label words into *adjectival*
    # positions, which only reads naturally for adjective-like labels
    # (yes/no, true/false).  `polar` states the task independently of the
    # label words and uses them purely as answer tokens, so it stays
    # grammatical under whichever label scheme is active.
    BoolQSystemPrompt.POLAR_V1: (
        "Based on the passage, is the answer to the question yes? Answer "
        "{true} if it is and {false} if it is not."
    ),
}

_RESPONSE_TOKEN_TEXTS: dict[BoolQResponseToken, str] = {
    BoolQResponseToken.RESPONSE: "Response:",
    BoolQResponseToken.ANSWER: "Answer:",
}

_LABEL_TEXTS: dict[BoolQLabelScheme, dict[BoolQLabel, str]] = {
    BoolQLabelScheme.YES_NO_V1: {
        BoolQLabel.TRUE: "yes",
        BoolQLabel.FALSE: "no",
    },
    BoolQLabelScheme.TRUE_FALSE_V1: {
        BoolQLabel.TRUE: "true",
        BoolQLabel.FALSE: "false",
    },
}

def render_system_prompt(
    prompt: BoolQSystemPrompt, label_scheme: BoolQLabelScheme
) -> str:
    """Render a system prompt with *label_scheme*'s label words substituted in."""
    return render_prompt(_SYSTEM_PROMPT_TEXTS[prompt], _LABEL_TEXTS[label_scheme])


# Legacy module-level alias derived from the default label scheme so any
# external code importing it keeps working.
_SYSTEM_PROMPT = render_system_prompt(
    BoolQSystemPrompt.DEFAULT_V1, BoolQLabelScheme.YES_NO_V1
)
_LABEL_TO_STR = _LABEL_TEXTS[BoolQLabelScheme.YES_NO_V1]
_STR_TO_LABEL = {v: k for k, v in _LABEL_TO_STR.items()}
CATEGORIES = list(_LABEL_TO_STR.values())


@dataclass
class BoolQChatTemplate(SchemaTransform[BoolQMessages, PreparedCompletion]):
    transformation: ClassVar[str] = "boolq_format"

    add_eos: bool = False
    system_prompt: BoolQSystemPrompt = BoolQSystemPrompt.DEFAULT_V1
    response_token: BoolQResponseToken = BoolQResponseToken.RESPONSE
    label_scheme: BoolQLabelScheme = BoolQLabelScheme.YES_NO_V1

    @override
    def apply(self, messages: BoolQMessages, /) -> PreparedCompletion:
        sys_text = render_system_prompt(self.system_prompt, self.label_scheme)
        token_text = _RESPONSE_TOKEN_TEXTS[self.response_token]
        labels = _LABEL_TEXTS[self.label_scheme]

        prompt = (
            f"{sys_text}\n\n"
            f"Passage: {messages['passage']}\n"
            f"Question: {messages['question']}\n"
            f"{token_text} "
        )

        label = messages["label"]
        if label is None:
            completion = ""
        elif isinstance(label, BoolQLabel):
            completion = labels[label]
        elif isinstance(label, int):
            completion = labels[BoolQLabel(label)]
        elif label in _STR_TO_LABEL:
            # Registry-prepped rows (and ``FixedLabelFlip`` targets) carry the
            # canonical label *string*, not a ``BoolQLabel``.  Without this
            # branch they fall through unchanged and ``label_scheme`` becomes
            # a silent no-op on exactly the pipeline that trains adapter
            # pools.  BoolQ hid this: its canonical strings are ``yes``/``no``,
            # so the pass-through happened to look correct under the default
            # ``YES_NO_V1`` while ``TRUE_FALSE_V1`` silently emitted yes/no.
            completion = labels[_STR_TO_LABEL[label]]
        else:
            completion = label

        return {"prompt": prompt, "completion": completion, "add_eos": self.add_eos}

    def label_word_map(self) -> dict[str, str]:
        """Canonical dataset label string → the word this template emits for it.

        Keys are the registry's ``label_int_to_str`` spellings (what the
        ``label`` column actually holds); values are the active scheme's
        surface words.  The generic eval builder reads this to give a fuzzy
        matcher the surface vocabulary without assuming the default scheme.
        """
        labels = _LABEL_TEXTS[self.label_scheme]
        return {_LABEL_TO_STR[member]: labels[member] for member in BoolQLabel}

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
class BoolQParse(FallibleSchemaTransform[str, BoolQMessages]):
    transformation: ClassVar[str] = "boolq_parse"

    response_token: BoolQResponseToken = BoolQResponseToken.RESPONSE
    label_scheme: BoolQLabelScheme = BoolQLabelScheme.YES_NO_V1

    @override
    def apply(self, text: str, /) -> Result[BoolQMessages, TransformError]:
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

        p_prefix = "Passage: "
        q_prefix = "Question: "
        passage = ""
        question = ""
        for line in body.splitlines():
            if line.startswith(p_prefix):
                passage = line[len(p_prefix) :]
            elif line.startswith(q_prefix):
                question = line[len(q_prefix) :]

        eot_token = "<|eot_id|>"
        if raw_response.endswith(eot_token):
            raw_response = raw_response[: -len(eot_token)].strip()

        str_to_label = {v: k for k, v in _LABEL_TEXTS[self.label_scheme].items()}
        label: BoolQLabel | str = str_to_label.get(raw_response, raw_response)

        return Ok(BoolQMessages(passage=passage, question=question, label=label))


# ---------------------------------------------------------------------------
# Corpus + task
# ---------------------------------------------------------------------------
# The corpus describes task-agnostic preparation (source, splits, raw -> typed
# mapping); the task names what it scores.  See dataset_definitions/_definition.py.

CORPUS = ColumnMappedCorpus(
    slug="boolq",
    hf_name='aps/super_glue',
    hf_config='boolq',
    splits_to_merge=('train', 'validation'),
    drop_splits=('test',),
    messages=BoolQMessages,
    text_columns={'passage': 'passage', 'question': 'question'},
    label=LabelDecoding(
        field_name="label",
        int_to_str={0: 'no', 1: 'yes'},
    ),
)

DEFINITION = ClassificationTask(
    slug="boolq",
    corpus=CORPUS,
    chat_template=BoolQChatTemplate,
    parser=BoolQParse,
    label_field='label',
    label_values=('no', 'yes'),
)
