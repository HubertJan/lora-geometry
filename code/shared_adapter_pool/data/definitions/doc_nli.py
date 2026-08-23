"""DocNLI (document-level natural language inference) for the llm_pipeline framework.

(migrated from src/llm_pipeline/dataset_definitions/doc_nli.py)

Schema mapping
--------------
DocNliMessages fields <- saattrupdan/doc-nli columns:
  premise            <- premise
  hypothesis         <- hypothesis
  entailment         <- DocNliEntailment enum from label (0=NOT_ENTAILED, 1=ENTAILED)

The HF ``label`` column holds the strings ``"entailment"`` / ``"not_entailment"``,
which the registry re-maps to the canonical words.

Premises are document-length (multi-sentence, sometimes multi-paragraph), so
the chat template caps them via ``max_text_chars`` before formatting -- see
`DocNliChatTemplate` for why the cap lives at format time rather than prep
time.

Split handling
--------------
The hub names its validation split ``val`` rather than ``validation`` --
anything keying splits by name (the registry lists it explicitly in
``splits_to_merge``) must use that exact key or the split is silently
dropped.  All three splits (``train``, ``val``, ``test``) share an identical
schema and are fully labeled, so no split is unlabeled or schema-mismatched
and all three are kept as-is here -- no merge is needed.
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


@serialisable("DocNliEntailment")
class DocNliEntailment(Enum):
    NOT_ENTAILED = 0
    ENTAILED = 1


class DocNliMessages(TypedDict):
    premise: str
    hypothesis: str
    entailment: DocNliEntailment | str | None


class PoisonedDocNliMessages(DocNliMessages):
    original: DocNliEntailment | str | None
    is_poisoned: bool


# ---------------------------------------------------------------------------
# Chat template
# ---------------------------------------------------------------------------

#: System prompts are ``str.format`` templates over the *label words* of the
#: prompt's label scheme: ``{entailed}`` / ``{not_entailed}`` (the
#: lower-cased :class:`DocNliEntailment` member names -- the underscore is
#: required because ``"not entailed"`` is not a valid ``str.format`` field
#: name).  Rendering with the default ``entailed_not_entailed`` scheme
#: reproduces the historical text byte-for-byte.
_DEFAULT_SYSTEM_PROMPT = (
    "You are given a document and a hypothesis sentence. Your task is to "
    "determine whether the hypothesis is {entailed} by the document. Your "
    "response should clearly indicate whether the hypothesis is {entailed} "
    "or {not_entailed}."
)


@serialisable("DocNliSystemPrompt")
class DocNliSystemPrompt(VariantEnum):
    DEFAULT_V1 = ("default", 1)
    CONCISE_V1 = ("concise", 1)
    POLAR_V1 = ("polar", 1)


@serialisable("DocNliResponseToken")
class DocNliResponseToken(VariantEnum):
    RESPONSE = ("response", 1)
    ANSWER = ("answer", 1)
    LABEL = ("label", 1)


@serialisable("DocNliLabelScheme")
class DocNliLabelScheme(VariantEnum):
    ENTAILED_NOT_ENTAILED_V1 = ("entailed_not_entailed", 1)
    TRUE_FALSE_V1 = ("true_false", 1)
    YES_NO_V1 = ("yes_no", 1)


_SYSTEM_PROMPT_TEXTS: dict[DocNliSystemPrompt, str] = {
    DocNliSystemPrompt.DEFAULT_V1: _DEFAULT_SYSTEM_PROMPT,
    DocNliSystemPrompt.CONCISE_V1: (
        "Given the document and hypothesis below, state whether the "
        "hypothesis is {entailed} by the document."
    ),
    # The `default`/`concise` families slot the label words into *adjectival*
    # positions, which only reads naturally for adjective-like labels
    # (entailed/not entailed, true/false).  `polar` states the task
    # independently of the label words and uses them purely as answer
    # tokens, so it stays grammatical under whichever label scheme is
    # active.
    DocNliSystemPrompt.POLAR_V1: (
        "Does the document entail the hypothesis? Answer {entailed} if it "
        "does and {not_entailed} if it does not."
    ),
}

_RESPONSE_TOKEN_TEXTS: dict[DocNliResponseToken, str] = {
    DocNliResponseToken.RESPONSE: "Response:",
    DocNliResponseToken.ANSWER: "Answer:",
    DocNliResponseToken.LABEL: "Label:",
}

_LABEL_TEXTS: dict[DocNliLabelScheme, dict[DocNliEntailment, str]] = {
    DocNliLabelScheme.ENTAILED_NOT_ENTAILED_V1: {
        DocNliEntailment.ENTAILED: "entailed",
        DocNliEntailment.NOT_ENTAILED: "not entailed",
    },
    DocNliLabelScheme.TRUE_FALSE_V1: {
        DocNliEntailment.ENTAILED: "true",
        DocNliEntailment.NOT_ENTAILED: "false",
    },
    DocNliLabelScheme.YES_NO_V1: {
        DocNliEntailment.ENTAILED: "yes",
        DocNliEntailment.NOT_ENTAILED: "no",
    },
}

def render_system_prompt(
    prompt: DocNliSystemPrompt, label_scheme: DocNliLabelScheme
) -> str:
    """Render a system prompt with *label_scheme*'s label words substituted in."""
    return render_prompt(_SYSTEM_PROMPT_TEXTS[prompt], _LABEL_TEXTS[label_scheme])


# Legacy module-level alias derived from the default label scheme so any
# external code importing it keeps working.
_SYSTEM_PROMPT = render_system_prompt(
    DocNliSystemPrompt.DEFAULT_V1, DocNliLabelScheme.ENTAILED_NOT_ENTAILED_V1
)
_ENTAILMENT_TO_STR = _LABEL_TEXTS[DocNliLabelScheme.ENTAILED_NOT_ENTAILED_V1]
_STR_TO_ENTAILMENT = {v: k for k, v in _ENTAILMENT_TO_STR.items()}
CATEGORIES = list(_ENTAILMENT_TO_STR.values())


@dataclass
class DocNliChatTemplate(SchemaTransform[DocNliMessages, PreparedCompletion]):
    """Formats DocNLI samples as prompt/completion pairs.

    Prompt layout::

        <system prompt>

        Document: <premise>
        Hypothesis: <hypothesis>
        <response token> <label>

    ``max_text_chars`` caps the document (``premise``) before formatting.
    DocNLI premises are document-length -- several paragraphs is common --
    and an over-long prompt is truncated from the *front* at tokenization,
    which would drop a randomly-positioned backdoor trigger while leaving the
    label intact, collapsing ASR for reasons unrelated to the backdoor.
    """

    transformation: ClassVar[str] = "doc_nli_format"

    add_eos: bool = False
    system_prompt: DocNliSystemPrompt = DocNliSystemPrompt.DEFAULT_V1
    response_token: DocNliResponseToken = DocNliResponseToken.RESPONSE
    label_scheme: DocNliLabelScheme = DocNliLabelScheme.ENTAILED_NOT_ENTAILED_V1
    max_text_chars: int | None = 4000

    @override
    def apply(self, messages: DocNliMessages, /) -> PreparedCompletion:
        sys_text = render_system_prompt(self.system_prompt, self.label_scheme)
        token_text = _RESPONSE_TOKEN_TEXTS[self.response_token]
        labels = _LABEL_TEXTS[self.label_scheme]

        premise = messages["premise"]
        if self.max_text_chars is not None:
            premise = premise[: self.max_text_chars]

        prompt = (
            f"{sys_text}\n\n"
            f"Document: {premise}\n"
            f"Hypothesis: {messages['hypothesis']}\n"
            f"{token_text} "
        )

        entailment = messages["entailment"]
        if entailment is None:
            completion = ""
        elif isinstance(entailment, DocNliEntailment):
            completion = labels[entailment]
        elif isinstance(entailment, int):
            completion = labels[DocNliEntailment(entailment)]
        elif entailment in _STR_TO_ENTAILMENT:
            # Registry-prepped rows (and ``FixedLabelFlip`` targets) carry the
            # canonical label *string*, not a ``DocNliEntailment``.  Without
            # this branch they fall through unchanged and ``label_scheme``
            # becomes a silent no-op on exactly the pipeline that trains
            # adapter pools.
            completion = labels[_STR_TO_ENTAILMENT[entailment]]
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
            _ENTAILMENT_TO_STR[member]: labels[member] for member in DocNliEntailment
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
            "max_text_chars": self.max_text_chars,
        }


@dataclass
class DocNliParse(FallibleSchemaTransform[str, DocNliMessages]):
    """Reverse-parse a DocNLI prompt-completion concatenation.

    ``response_token`` and ``label_scheme`` must match whatever
    `DocNliChatTemplate` was configured with for round-trip parsing to work.
    """

    transformation: ClassVar[str] = "doc_nli_parse"

    response_token: DocNliResponseToken = DocNliResponseToken.RESPONSE
    label_scheme: DocNliLabelScheme = DocNliLabelScheme.ENTAILED_NOT_ENTAILED_V1

    @override
    def apply(self, text: str, /) -> Result[DocNliMessages, TransformError]:
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

        document_prefix = "Document: "
        hypothesis_prefix = "Hypothesis: "
        premise = ""
        hypothesis = ""
        for line in body.splitlines():
            if line.startswith(document_prefix):
                premise = line[len(document_prefix) :]
            elif line.startswith(hypothesis_prefix):
                hypothesis = line[len(hypothesis_prefix) :]

        eot_token = "<|eot_id|>"
        if raw_response.endswith(eot_token):
            raw_response = raw_response[: -len(eot_token)].strip()

        str_to_entailment = {
            v: k for k, v in _LABEL_TEXTS[self.label_scheme].items()
        }
        entailment: DocNliEntailment | str = str_to_entailment.get(
            raw_response, raw_response
        )

        return Ok(
            DocNliMessages(
                premise=premise, hypothesis=hypothesis, entailment=entailment
            )
        )


# ---------------------------------------------------------------------------
# Corpus + task
# ---------------------------------------------------------------------------
# The corpus describes task-agnostic preparation (source, splits, raw -> typed
# mapping); the task names what it scores.  See dataset_definitions/_definition.py.

CORPUS = ColumnMappedCorpus(
    slug="doc_nli",
    hf_name='saattrupdan/doc-nli',
    hf_config=None,
    splits_to_merge=('train', 'val', 'test'),
    messages=DocNliMessages,
    text_columns={'premise': 'premise', 'hypothesis': 'hypothesis'},
    label=LabelDecoding(
        field_name="entailment",
        str_to_str={'entailment': 'entailed', 'not_entailment': 'not entailed'},
        drop_none=True,
    ),
)


DEFINITION = ClassificationTask(
    slug="doc_nli",
    corpus=CORPUS,
    chat_template=DocNliChatTemplate,
    parser=DocNliParse,
    label_field='entailment',
    label_values=('not entailed', 'entailed'),
)
