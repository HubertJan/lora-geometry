"""VitaminC claim/evidence fact-verification dataset for the llm_pipeline framework.

(migrated from src/llm_pipeline/dataset_definitions/vitaminc.py)

Schema mapping
--------------
VitaminCMessages fields <- tals/vitaminc columns:
  claim              <- claim
  evidence           <- evidence
  verdict            <- VitaminCVerdict enum from label (0=REFUTED, 1=SUPPORTED)

The hub's ``label`` column holds the strings ``SUPPORTS`` / ``REFUTES`` /
``NOT ENOUGH INFO``.  The registry maps the first two to the canonical words
below and drops ``NOT ENOUGH INFO`` (52,981 of 370,653 ``train`` rows) via
``drop_none_labels``, since a claim that is neither supported nor refuted by
the given evidence has no binary answer to train on.

This corpus was chosen over ``copenlu/fever_gold_evidence`` because
VitaminC's ``evidence`` column is a plain string, whereas FEVER's is a
``Sequence`` the generic row mapper cannot join into a single text field.

Split handling
--------------
The hub exposes ``train`` (370,653), ``validation`` (63,054) and ``test``
(55,197), all sharing an identical schema and the same three-way label
space.  They are passed through unmerged so callers retain ``validation``/
``test`` as held-out data rather than folding them into ``train``.
``NOT ENOUGH INFO`` rows are mapped to ``None`` (see ``_map_verdict`` below)
in every split, not just ``train``.
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


@serialisable("VitaminCVerdict")
class VitaminCVerdict(Enum):
    REFUTED = 0
    SUPPORTED = 1


class VitaminCMessages(TypedDict):
    claim: str
    evidence: str
    verdict: VitaminCVerdict | str | None


class PoisonedVitaminCMessages(VitaminCMessages):
    original: VitaminCVerdict | str | None
    is_poisoned: bool


# ---------------------------------------------------------------------------
# Chat template
# ---------------------------------------------------------------------------

#: System prompts are ``str.format`` templates over the *label words* of the
#: prompt's label scheme: ``{supported}`` / ``{refuted}`` (the lower-cased
#: :class:`VitaminCVerdict` member names).  Note the verb forms "supports" /
#: "refutes" elsewhere in this text are *not* label words and are left as
#: literal text -- only the standalone adjectival "supported"/"refuted"
#: occurrences are placeholders.  Rendering with the default
#: ``supported_refuted`` scheme reproduces the historical text byte-for-byte.
_DEFAULT_SYSTEM_PROMPT = (
    "You are given a piece of evidence and a claim. Your task is to "
    "determine whether the evidence supports or refutes the claim. Your "
    "response should be exactly one word: {supported} or {refuted}."
)


@serialisable("VitaminCSystemPrompt")
class VitaminCSystemPrompt(VariantEnum):
    DEFAULT_V1 = ("default", 1)
    CONCISE_V1 = ("concise", 1)
    POLAR_V1 = ("polar", 1)


@serialisable("VitaminCResponseToken")
class VitaminCResponseToken(VariantEnum):
    RESPONSE = ("response", 1)
    ANSWER = ("answer", 1)
    LABEL = ("label", 1)


@serialisable("VitaminCLabelScheme")
class VitaminCLabelScheme(VariantEnum):
    SUPPORTED_REFUTED_V1 = ("supported_refuted", 1)
    TRUE_FALSE_V1 = ("true_false", 1)
    YES_NO_V1 = ("yes_no", 1)


_SYSTEM_PROMPT_TEXTS: dict[VitaminCSystemPrompt, str] = {
    VitaminCSystemPrompt.DEFAULT_V1: _DEFAULT_SYSTEM_PROMPT,
    VitaminCSystemPrompt.CONCISE_V1: (
        "Given the evidence and claim below, state whether the evidence "
        "supports or refutes the claim."
    ),
    # The `default`/`concise` families slot the label words into *adjectival*
    # positions, which only reads naturally for adjective-like labels
    # (supported/refuted, true/false).  `polar` states the task independently
    # of the label words and uses them purely as answer tokens, so it stays
    # grammatical under whichever label scheme is active.
    VitaminCSystemPrompt.POLAR_V1: (
        "Does the evidence support the claim? Answer {supported} if it does "
        "and {refuted} if it does not."
    ),
}

_RESPONSE_TOKEN_TEXTS: dict[VitaminCResponseToken, str] = {
    VitaminCResponseToken.RESPONSE: "Response:",
    VitaminCResponseToken.ANSWER: "Answer:",
    VitaminCResponseToken.LABEL: "Label:",
}

_LABEL_TEXTS: dict[VitaminCLabelScheme, dict[VitaminCVerdict, str]] = {
    VitaminCLabelScheme.SUPPORTED_REFUTED_V1: {
        VitaminCVerdict.SUPPORTED: "supported",
        VitaminCVerdict.REFUTED: "refuted",
    },
    VitaminCLabelScheme.TRUE_FALSE_V1: {
        VitaminCVerdict.SUPPORTED: "true",
        VitaminCVerdict.REFUTED: "false",
    },
    VitaminCLabelScheme.YES_NO_V1: {
        VitaminCVerdict.SUPPORTED: "yes",
        VitaminCVerdict.REFUTED: "no",
    },
}

def render_system_prompt(
    prompt: VitaminCSystemPrompt, label_scheme: VitaminCLabelScheme
) -> str:
    """Render a system prompt with *label_scheme*'s label words substituted in."""
    return render_prompt(_SYSTEM_PROMPT_TEXTS[prompt], _LABEL_TEXTS[label_scheme])


# Legacy module-level alias derived from the default label scheme so any
# external code importing it keeps working.
_SYSTEM_PROMPT = render_system_prompt(
    VitaminCSystemPrompt.DEFAULT_V1, VitaminCLabelScheme.SUPPORTED_REFUTED_V1
)
_VERDICT_TO_STR = _LABEL_TEXTS[VitaminCLabelScheme.SUPPORTED_REFUTED_V1]
_STR_TO_VERDICT = {v: k for k, v in _VERDICT_TO_STR.items()}
CATEGORIES = list(_VERDICT_TO_STR.values())


@dataclass
class VitaminCChatTemplate(SchemaTransform[VitaminCMessages, PreparedCompletion]):
    """Formats VitaminC samples as prompt/completion pairs.

    Prompt layout::

        <system prompt>

        Evidence: <evidence text>
        Claim: <claim text>
        <response token> <label>

    Evidence is placed before the claim so the model reads the supporting
    context before the statement it must judge.
    """

    transformation: ClassVar[str] = "vitaminc_format"

    add_eos: bool = False
    system_prompt: VitaminCSystemPrompt = VitaminCSystemPrompt.DEFAULT_V1
    response_token: VitaminCResponseToken = VitaminCResponseToken.RESPONSE
    label_scheme: VitaminCLabelScheme = VitaminCLabelScheme.SUPPORTED_REFUTED_V1

    @override
    def apply(self, messages: VitaminCMessages, /) -> PreparedCompletion:
        sys_text = render_system_prompt(self.system_prompt, self.label_scheme)
        token_text = _RESPONSE_TOKEN_TEXTS[self.response_token]
        labels = _LABEL_TEXTS[self.label_scheme]

        prompt = (
            f"{sys_text}\n\n"
            f"Evidence: {messages['evidence']}\n"
            f"Claim: {messages['claim']}\n"
            f"{token_text} "
        )

        verdict = messages["verdict"]
        if verdict is None:
            completion = ""
        elif isinstance(verdict, VitaminCVerdict):
            completion = labels[verdict]
        elif isinstance(verdict, int):
            completion = labels[VitaminCVerdict(verdict)]
        elif verdict in _STR_TO_VERDICT:
            # Registry-prepped rows (and ``FixedLabelFlip`` targets) carry the
            # canonical label *string*, not a ``VitaminCVerdict``.  Without
            # this branch they fall through unchanged and ``label_scheme``
            # becomes a silent no-op on exactly the pipeline that trains
            # adapter pools.
            completion = labels[_STR_TO_VERDICT[verdict]]
        else:
            completion = verdict

        return {"prompt": prompt, "completion": completion, "add_eos": self.add_eos}

    def label_word_map(self) -> dict[str, str]:
        """Canonical dataset label string -> the word this template emits for it.

        Keys are the registry's ``label_int_to_str`` spellings (what the
        ``verdict`` column actually holds); values are the active scheme's
        surface words.  The generic eval builder reads this to give a fuzzy
        matcher the surface vocabulary without assuming the default scheme.
        """
        labels = _LABEL_TEXTS[self.label_scheme]
        return {
            _VERDICT_TO_STR[member]: labels[member] for member in VitaminCVerdict
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
class VitaminCParse(FallibleSchemaTransform[str, VitaminCMessages]):
    """Reverse-parse a VitaminC prompt-completion concatenation.

    ``response_token`` and ``label_scheme`` must match whatever
    `VitaminCChatTemplate` was configured with for round-trip parsing to
    work.
    """

    transformation: ClassVar[str] = "vitaminc_parse"

    response_token: VitaminCResponseToken = VitaminCResponseToken.RESPONSE
    label_scheme: VitaminCLabelScheme = VitaminCLabelScheme.SUPPORTED_REFUTED_V1

    @override
    def apply(self, text: str, /) -> Result[VitaminCMessages, TransformError]:
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

        evidence_prefix = "Evidence: "
        claim_prefix = "Claim: "
        evidence = ""
        claim = ""
        for line in body.splitlines():
            if line.startswith(evidence_prefix):
                evidence = line[len(evidence_prefix) :]
            elif line.startswith(claim_prefix):
                claim = line[len(claim_prefix) :]

        eot_token = "<|eot_id|>"
        if raw_response.endswith(eot_token):
            raw_response = raw_response[: -len(eot_token)].strip()

        str_to_verdict = {v: k for k, v in _LABEL_TEXTS[self.label_scheme].items()}
        verdict: VitaminCVerdict | str = str_to_verdict.get(raw_response, raw_response)

        return Ok(VitaminCMessages(claim=claim, evidence=evidence, verdict=verdict))


# ---------------------------------------------------------------------------
# Corpus + task
# ---------------------------------------------------------------------------
# The corpus describes task-agnostic preparation (source, splits, raw -> typed
# mapping); the task names what it scores.  See dataset_definitions/_definition.py.

CORPUS = ColumnMappedCorpus(
    slug="vitaminc",
    hf_name='tals/vitaminc',
    hf_config=None,
    splits_to_merge=('train', 'validation', 'test'),
    messages=VitaminCMessages,
    text_columns={'claim': 'claim', 'evidence': 'evidence'},
    label=LabelDecoding(
        field_name="verdict",
        str_to_str={'SUPPORTS': 'supported', 'REFUTES': 'refuted'},
        drop_none=True,
    ),
)


DEFINITION = ClassificationTask(
    slug="vitaminc",
    corpus=CORPUS,
    chat_template=VitaminCChatTemplate,
    parser=VitaminCParse,
    label_field='verdict',
    label_values=('refuted', 'supported'),
)
