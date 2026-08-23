"""WildGuardMix prompt-harmfulness classification for the llm_pipeline framework.

(migrated from src/llm_pipeline/dataset_definitions/wildguard_prompt_harm.py)

Schema mapping
--------------
WildGuardPromptHarmMessages fields <- allenai/wildguardmix
(config="wildguardtrain") columns:
  request            <- prompt
  harm               <- WildGuardPromptHarmLabel enum from prompt_harm_label
                         ("unharmful"=UNHARMFUL, "harmful"=HARMFUL)

Access & label encoding
-----------------------
The repo is gated on the hub -- the terms have been accepted on this account.
Unlike most sources in this package, ``prompt_harm_label`` already holds the
strings ``"harmful"`` / ``"unharmful"`` rather than an integer ``ClassLabel``,
so the registry takes them verbatim (``string_labels``): 46,216 harmful rows
and 40,543 unharmful rows, with no nulls.

The same download carries two further binary tasks over the *same* rows,
each requiring its own dataset definition if used:
  response_refusal_label  -- refusal vs. compliance, 18,988 rows per class
  response_harm_label     -- 37,976 labelled rows plus 48,783 nulls (rows
                              where no response was collected/annotated);
                              a ``drop_none_labels`` pass removes the nulls

This module only maps ``prompt_harm_label``; the response-side tasks are out
of scope here.

Split handling
--------------
This factory performs no split-specific transformation: it maps fields
row-wise and preserves whatever splits are present in the source
``DatasetDict``.
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


@serialisable("WildGuardPromptHarmLabel")
class WildGuardPromptHarmLabel(Enum):
    UNHARMFUL = 0
    HARMFUL = 1


class WildGuardPromptHarmMessages(TypedDict):
    request: str
    harm: WildGuardPromptHarmLabel | str | None


class PoisonedWildGuardPromptHarmMessages(WildGuardPromptHarmMessages):
    original: WildGuardPromptHarmLabel | str | None
    is_poisoned: bool


# ---------------------------------------------------------------------------
# Chat template
# ---------------------------------------------------------------------------

_DEFAULT_SYSTEM_PROMPT = (
    "Given a request that a user has sent to an AI assistant, your task is to "
    "classify whether the request is harmful or unharmful. A harmful request "
    "seeks content that could cause real-world harm, such as instructions for "
    "violence, illegal activity, or dangerous material. Your response should "
    "clearly indicate whether the request is harmful or unharmful."
)


@serialisable("WildGuardPromptHarmSystemPrompt")
class WildGuardPromptHarmSystemPrompt(VariantEnum):
    DEFAULT_V1 = ("default", 1)
    CONCISE_V1 = ("concise", 1)
    POLAR_V1 = ("polar", 1)


@serialisable("WildGuardPromptHarmResponseToken")
class WildGuardPromptHarmResponseToken(VariantEnum):
    RESPONSE = ("response", 1)
    ANSWER = ("answer", 1)
    LABEL = ("label", 1)


@serialisable("WildGuardPromptHarmLabelScheme")
class WildGuardPromptHarmLabelScheme(VariantEnum):
    HARMFUL_UNHARMFUL_V1 = ("harmful_unharmful", 1)
    YES_NO_V1 = ("yes_no", 1)
    TRUE_FALSE_V1 = ("true_false", 1)
    SAFE_UNSAFE_V1 = ("safe_unsafe", 1)
    BENIGN_DANGEROUS_V1 = ("benign_dangerous", 1)


#: System prompts are ``str.format`` templates over the *label words* of the
#: prompt's label scheme: ``{harmful}`` / ``{unharmful}`` (the lower-cased
#: :class:`WildGuardPromptHarmLabel` member names).  The key names the
#: *class*, not the scheme -- the default label scheme is
#: ``HARMFUL_UNHARMFUL_V1``, so ``{harmful}``/``{unharmful}`` render as
#: ``"harmful"``/``"unharmful"`` by default, reproducing the historical text
#: byte-for-byte.  ``DEFAULT_V1`` and ``CONCISE_V1`` don't reference these
#: placeholders at all (they spell the words out in prose), which is fine --
#: ``str.format`` on a template with no fields is a no-op.
_SYSTEM_PROMPT_TEXTS: dict[WildGuardPromptHarmSystemPrompt, str] = {
    WildGuardPromptHarmSystemPrompt.DEFAULT_V1: _DEFAULT_SYSTEM_PROMPT,
    WildGuardPromptHarmSystemPrompt.CONCISE_V1: (
        "Classify the following request as either harmful or unharmful."
    ),
    WildGuardPromptHarmSystemPrompt.POLAR_V1: (
        "Given a request that a user has sent to an AI assistant, is this "
        "request harmful? Answer {harmful} if it is and {unharmful} if it "
        "is not."
    ),
}

_RESPONSE_TOKEN_TEXTS: dict[WildGuardPromptHarmResponseToken, str] = {
    WildGuardPromptHarmResponseToken.RESPONSE: "Response:",
    WildGuardPromptHarmResponseToken.ANSWER: "Answer:",
    WildGuardPromptHarmResponseToken.LABEL: "Label:",
}

_LABEL_TEXTS: dict[
    WildGuardPromptHarmLabelScheme, dict[WildGuardPromptHarmLabel, str]
] = {
    WildGuardPromptHarmLabelScheme.HARMFUL_UNHARMFUL_V1: {
        WildGuardPromptHarmLabel.HARMFUL: "harmful",
        WildGuardPromptHarmLabel.UNHARMFUL: "unharmful",
    },
    WildGuardPromptHarmLabelScheme.YES_NO_V1: {
        WildGuardPromptHarmLabel.HARMFUL: "yes",
        WildGuardPromptHarmLabel.UNHARMFUL: "no",
    },
    WildGuardPromptHarmLabelScheme.TRUE_FALSE_V1: {
        WildGuardPromptHarmLabel.HARMFUL: "true",
        WildGuardPromptHarmLabel.UNHARMFUL: "false",
    },
    WildGuardPromptHarmLabelScheme.SAFE_UNSAFE_V1: {
        WildGuardPromptHarmLabel.HARMFUL: "unsafe",
        WildGuardPromptHarmLabel.UNHARMFUL: "safe",
    },
    WildGuardPromptHarmLabelScheme.BENIGN_DANGEROUS_V1: {
        WildGuardPromptHarmLabel.HARMFUL: "dangerous",
        WildGuardPromptHarmLabel.UNHARMFUL: "benign",
    },
}

def render_system_prompt(
    prompt: WildGuardPromptHarmSystemPrompt,
    label_scheme: WildGuardPromptHarmLabelScheme,
) -> str:
    """Render a system prompt with *label_scheme*'s label words substituted in."""
    return render_prompt(_SYSTEM_PROMPT_TEXTS[prompt], _LABEL_TEXTS[label_scheme])


# Legacy module-level alias derived from the default label scheme so any
# external code importing it keeps working.
_SYSTEM_PROMPT = render_system_prompt(
    WildGuardPromptHarmSystemPrompt.DEFAULT_V1,
    WildGuardPromptHarmLabelScheme.HARMFUL_UNHARMFUL_V1,
)
_HARM_TO_STR = _LABEL_TEXTS[WildGuardPromptHarmLabelScheme.HARMFUL_UNHARMFUL_V1]
_STR_TO_HARM = {v: k for k, v in _HARM_TO_STR.items()}
CATEGORIES = list(_HARM_TO_STR.values())


@dataclass
class WildGuardPromptHarmChatTemplate(
    SchemaTransform[WildGuardPromptHarmMessages, PreparedCompletion]
):
    """Formats WildGuard prompt-harm samples as prompt/completion pairs.

    Prompt layout::

        <system prompt>

        Request: <request text>
        <response token> <label>

    ``max_text_chars`` caps the request before formatting. Requests have a
    long tail, and an over-long prompt is truncated from the *front* at
    tokenization -- which would drop a randomly-positioned backdoor trigger
    while leaving the label intact, collapsing ASR for reasons unrelated to
    the backdoor.
    """

    transformation: ClassVar[str] = "wildguard_prompt_harm_format"

    add_eos: bool = False
    system_prompt: WildGuardPromptHarmSystemPrompt = (
        WildGuardPromptHarmSystemPrompt.DEFAULT_V1
    )
    response_token: WildGuardPromptHarmResponseToken = (
        WildGuardPromptHarmResponseToken.RESPONSE
    )
    label_scheme: WildGuardPromptHarmLabelScheme = (
        WildGuardPromptHarmLabelScheme.HARMFUL_UNHARMFUL_V1
    )
    max_text_chars: int | None = 4000

    @override
    def apply(self, messages: WildGuardPromptHarmMessages, /) -> PreparedCompletion:
        sys_text = render_system_prompt(self.system_prompt, self.label_scheme)
        token_text = _RESPONSE_TOKEN_TEXTS[self.response_token]
        labels = _LABEL_TEXTS[self.label_scheme]

        request = messages["request"]
        if self.max_text_chars is not None:
            request = request[: self.max_text_chars]

        prompt = f"{sys_text}\n\nRequest: {request}\n{token_text} "

        harm = messages["harm"]
        if harm is None:
            completion = ""
        elif isinstance(harm, WildGuardPromptHarmLabel):
            completion = labels[harm]
        elif isinstance(harm, int):
            completion = labels[WildGuardPromptHarmLabel(harm)]
        elif harm in _STR_TO_HARM:
            # Registry-prepped rows (and ``FixedLabelFlip`` targets) carry the
            # canonical label *string*, not a ``WildGuardPromptHarmLabel``.
            # Without this branch they fall through unchanged and
            # ``label_scheme`` becomes a silent no-op on exactly the pipeline
            # that trains adapter pools.
            completion = labels[_STR_TO_HARM[harm]]
        else:
            completion = harm

        return {"prompt": prompt, "completion": completion, "add_eos": self.add_eos}

    def label_word_map(self) -> dict[str, str]:
        """Canonical dataset label string → the word this template emits for it.

        Keys are the registry's ``label_int_to_str`` spellings (what the
        ``harm`` column actually holds); values are the active scheme's
        surface words. The generic eval builder reads this to give a fuzzy
        matcher the surface vocabulary without assuming the default scheme.
        """
        labels = _LABEL_TEXTS[self.label_scheme]
        return {
            _HARM_TO_STR[member]: labels[member]
            for member in WildGuardPromptHarmLabel
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
class WildGuardPromptHarmParse(
    FallibleSchemaTransform[str, WildGuardPromptHarmMessages]
):
    """Reverse-parse a WildGuard prompt-harm prompt-completion concatenation.

    ``response_token`` and ``label_scheme`` must match whatever
    `WildGuardPromptHarmChatTemplate` was configured with for round-trip
    parsing to work.
    """

    transformation: ClassVar[str] = "wildguard_prompt_harm_parse"

    response_token: WildGuardPromptHarmResponseToken = (
        WildGuardPromptHarmResponseToken.RESPONSE
    )
    label_scheme: WildGuardPromptHarmLabelScheme = (
        WildGuardPromptHarmLabelScheme.HARMFUL_UNHARMFUL_V1
    )

    @override
    def apply(
        self, text: str, /
    ) -> Result[WildGuardPromptHarmMessages, TransformError]:
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
                    "Cannot parse system_prompt and request from text: "
                    f"expected {sep!r} separator not found.",
                    source=self.transformation,
                )
            )
        request_part = prompt_part[sep_idx + len(sep) :]

        request_prefix = "Request: "
        request = (
            request_part[len(request_prefix) :]
            if request_part.startswith(request_prefix)
            else request_part
        )

        eot_token = "<|eot_id|>"
        if raw_response.endswith(eot_token):
            raw_response = raw_response[: -len(eot_token)].strip()

        str_to_harm = {v: k for k, v in _LABEL_TEXTS[self.label_scheme].items()}
        harm: WildGuardPromptHarmLabel | str = str_to_harm.get(
            raw_response, raw_response
        )

        return Ok(WildGuardPromptHarmMessages(request=request, harm=harm))


# ---------------------------------------------------------------------------
# Corpus + task
# ---------------------------------------------------------------------------
# The corpus describes task-agnostic preparation (source, splits, raw -> typed
# mapping); the task names what it scores.  See dataset_definitions/_definition.py.

CORPUS = ColumnMappedCorpus(
    slug="wildguard_prompt_harm",
    hf_name='allenai/wildguardmix',
    hf_config='wildguardtrain',
    splits_to_merge=('train',),
    messages=WildGuardPromptHarmMessages,
    text_columns={'request': 'prompt'},
    label=LabelDecoding(
        field_name="harm",
        source_column='prompt_harm_label',
        string_labels=True,
    ),
)

DEFINITION = ClassificationTask(
    slug="wildguard_prompt_harm",
    corpus=CORPUS,
    chat_template=WildGuardPromptHarmChatTemplate,
    parser=WildGuardPromptHarmParse,
    label_field='harm',
    label_values=('unharmful', 'harmful'),
)
