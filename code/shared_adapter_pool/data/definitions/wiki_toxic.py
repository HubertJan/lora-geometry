"""Wiki Toxic (Wikipedia talk-page comment toxicity) for the llm_pipeline framework.

(migrated from src/llm_pipeline/dataset_definitions/wiki_toxic.py)

Schema mapping
--------------
WikiToxicMessages fields <- OxAISH-AL-LLM/wiki_toxic columns:
  comment            <- comment_text
  toxicity           <- WikiToxicToxicity enum from label (0=NON_TOXIC, 1=TOXIC)

Split handling
--------------
On ``main`` the hub exposes four splits: ``train`` (127,656), ``validation``
(31,915), ``test`` (63,978) and ``balanced_train`` (25,868).  ``balanced_train``
is a class-balanced *subset of* ``train`` -- merging it alongside the others
would silently duplicate ~26k comments, over-weighting them during training and
letting the same text land in two splits of the downstream re-split.  The
factory below therefore drops it defensively.

The registry does not load ``main``: it is a script-only repo (``load_dataset``
refuses it outright), so the pipeline reads the ``refs/convert/parquet``
mirror -- which exposes only train/validation/test.  ``balanced_train`` is
consequently absent there, and the registry entry must NOT list it in
``drop_splits``, because ``MergeSplits`` raises on a drop target it cannot find.

The corpus is naturally imbalanced (~10% toxic), which is the point: it is the
one wave-1 toxicity source whose clean class dominates the way a real moderation
corpus does.
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


@serialisable("WikiToxicToxicity")
class WikiToxicToxicity(Enum):
    NON_TOXIC = 0
    TOXIC = 1


class WikiToxicMessages(TypedDict):
    comment: str
    toxicity: WikiToxicToxicity | str | None


class PoisonedWikiToxicMessages(WikiToxicMessages):
    original: WikiToxicToxicity | str | None
    is_poisoned: bool


# ---------------------------------------------------------------------------
# Chat template
# ---------------------------------------------------------------------------

_DEFAULT_SYSTEM_PROMPT = (
    "Given a comment from a Wikipedia talk page, your task is to classify it as "
    "either toxic or non-toxic. A toxic comment is rude, disrespectful, or "
    "insulting and would make someone leave the discussion. Your response "
    "should clearly indicate whether the comment is toxic or non-toxic."
)


@serialisable("WikiToxicSystemPrompt")
class WikiToxicSystemPrompt(VariantEnum):
    DEFAULT_V1 = ("default", 1)
    CONCISE_V1 = ("concise", 1)
    POLAR_V1 = ("polar", 1)


@serialisable("WikiToxicResponseToken")
class WikiToxicResponseToken(VariantEnum):
    RESPONSE = ("response", 1)
    ANSWER = ("answer", 1)
    LABEL = ("label", 1)


@serialisable("WikiToxicLabelScheme")
class WikiToxicLabelScheme(VariantEnum):
    TOXIC_NON_TOXIC_V1 = ("toxic_non_toxic", 1)
    TRUE_FALSE_V1 = ("true_false", 1)
    YES_NO_V1 = ("yes_no", 1)


#: System prompts are ``str.format`` templates over the *label words* of the
#: prompt's label scheme: ``{toxic}`` / ``{non_toxic}`` (the lower-cased
#: :class:`WikiToxicToxicity` member names).  The key names the *class*, not
#: the scheme -- wiki_toxic's default label scheme is ``TOXIC_NON_TOXIC_V1``,
#: so ``{toxic}``/``{non_toxic}`` render as ``"toxic"``/``"non-toxic"`` by
#: default.  ``DEFAULT_V1`` and ``CONCISE_V1`` spell the label words out
#: literally instead of using placeholders, so rendering them is a no-op and
#: reproduces the historical text byte-for-byte; only ``POLAR_V1`` uses the
#: placeholders.
_SYSTEM_PROMPT_TEXTS: dict[WikiToxicSystemPrompt, str] = {
    WikiToxicSystemPrompt.DEFAULT_V1: _DEFAULT_SYSTEM_PROMPT,
    WikiToxicSystemPrompt.CONCISE_V1: (
        "Classify the following Wikipedia talk-page comment as either toxic or "
        "non-toxic."
    ),
    WikiToxicSystemPrompt.POLAR_V1: (
        "Given a comment from a Wikipedia talk page, is it toxic? Answer "
        "{toxic} if it is and {non_toxic} if it is not."
    ),
}

_RESPONSE_TOKEN_TEXTS: dict[WikiToxicResponseToken, str] = {
    WikiToxicResponseToken.RESPONSE: "Response:",
    WikiToxicResponseToken.ANSWER: "Answer:",
    WikiToxicResponseToken.LABEL: "Label:",
}

_LABEL_TEXTS: dict[WikiToxicLabelScheme, dict[WikiToxicToxicity, str]] = {
    WikiToxicLabelScheme.TOXIC_NON_TOXIC_V1: {
        WikiToxicToxicity.TOXIC: "toxic",
        WikiToxicToxicity.NON_TOXIC: "non-toxic",
    },
    WikiToxicLabelScheme.TRUE_FALSE_V1: {
        WikiToxicToxicity.TOXIC: "true",
        WikiToxicToxicity.NON_TOXIC: "false",
    },
    WikiToxicLabelScheme.YES_NO_V1: {
        WikiToxicToxicity.TOXIC: "yes",
        WikiToxicToxicity.NON_TOXIC: "no",
    },
}

def render_system_prompt(
    prompt: WikiToxicSystemPrompt, label_scheme: WikiToxicLabelScheme
) -> str:
    """Render a system prompt with *label_scheme*'s label words substituted in."""
    return render_prompt(_SYSTEM_PROMPT_TEXTS[prompt], _LABEL_TEXTS[label_scheme])


# Legacy module-level alias derived from the default label scheme so any
# external code importing it keeps working.
_SYSTEM_PROMPT = render_system_prompt(
    WikiToxicSystemPrompt.DEFAULT_V1, WikiToxicLabelScheme.TOXIC_NON_TOXIC_V1
)
_TOXICITY_TO_STR = _LABEL_TEXTS[WikiToxicLabelScheme.TOXIC_NON_TOXIC_V1]
_STR_TO_TOXICITY = {v: k for k, v in _TOXICITY_TO_STR.items()}
CATEGORIES = list(_TOXICITY_TO_STR.values())


@dataclass
class WikiToxicChatTemplate(SchemaTransform[WikiToxicMessages, PreparedCompletion]):
    """Formats Wiki Toxic samples as prompt/completion pairs.

    Prompt layout::

        <system prompt>

        Comment: <comment text>
        <response token> <label>

    ``max_text_chars`` caps the comment before formatting.  Talk-page comments
    have a long tail (a handful run to tens of thousands of characters), and an
    over-long prompt is truncated from the *front* at tokenization -- which
    would drop a randomly-positioned backdoor trigger while leaving the label
    intact, collapsing ASR for reasons unrelated to the backdoor.
    """

    transformation: ClassVar[str] = "wiki_toxic_format"

    add_eos: bool = False
    system_prompt: WikiToxicSystemPrompt = WikiToxicSystemPrompt.DEFAULT_V1
    response_token: WikiToxicResponseToken = WikiToxicResponseToken.RESPONSE
    label_scheme: WikiToxicLabelScheme = WikiToxicLabelScheme.TOXIC_NON_TOXIC_V1
    max_text_chars: int | None = 4000

    @override
    def apply(self, messages: WikiToxicMessages, /) -> PreparedCompletion:
        sys_text = render_system_prompt(self.system_prompt, self.label_scheme)
        token_text = _RESPONSE_TOKEN_TEXTS[self.response_token]
        labels = _LABEL_TEXTS[self.label_scheme]

        comment = messages["comment"]
        if self.max_text_chars is not None:
            comment = comment[: self.max_text_chars]

        prompt = f"{sys_text}\n\nComment: {comment}\n{token_text} "

        toxicity = messages["toxicity"]
        if toxicity is None:
            completion = ""
        elif isinstance(toxicity, WikiToxicToxicity):
            completion = labels[toxicity]
        elif isinstance(toxicity, int):
            completion = labels[WikiToxicToxicity(toxicity)]
        elif toxicity in _STR_TO_TOXICITY:
            # Registry-prepped rows (and ``FixedLabelFlip`` targets) carry the
            # canonical label *string*, not a ``WikiToxicToxicity``.  Without
            # this branch they fall through unchanged and ``label_scheme``
            # becomes a silent no-op on exactly the pipeline that trains adapter
            # pools.
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
            _TOXICITY_TO_STR[member]: labels[member] for member in WikiToxicToxicity
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
class WikiToxicParse(FallibleSchemaTransform[str, WikiToxicMessages]):
    """Reverse-parse a Wiki Toxic prompt-completion concatenation.

    ``response_token`` and ``label_scheme`` must match whatever
    `WikiToxicChatTemplate` was configured with for round-trip parsing to work.
    """

    transformation: ClassVar[str] = "wiki_toxic_parse"

    response_token: WikiToxicResponseToken = WikiToxicResponseToken.RESPONSE
    label_scheme: WikiToxicLabelScheme = WikiToxicLabelScheme.TOXIC_NON_TOXIC_V1

    @override
    def apply(self, text: str, /) -> Result[WikiToxicMessages, TransformError]:
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
                    "Cannot parse system_prompt and comment from text: "
                    f"expected {sep!r} separator not found.",
                    source=self.transformation,
                )
            )
        comment_part = prompt_part[sep_idx + len(sep) :]

        comment_prefix = "Comment: "
        comment = (
            comment_part[len(comment_prefix) :]
            if comment_part.startswith(comment_prefix)
            else comment_part
        )

        eot_token = "<|eot_id|>"
        if raw_response.endswith(eot_token):
            raw_response = raw_response[: -len(eot_token)].strip()

        str_to_toxicity = {v: k for k, v in _LABEL_TEXTS[self.label_scheme].items()}
        toxicity: WikiToxicToxicity | str = str_to_toxicity.get(
            raw_response, raw_response
        )

        return Ok(WikiToxicMessages(comment=comment, toxicity=toxicity))


# ---------------------------------------------------------------------------
# Corpus + task
# ---------------------------------------------------------------------------
# The corpus describes task-agnostic preparation (source, splits, raw -> typed
# mapping); the task names what it scores.  See dataset_definitions/_definition.py.

CORPUS = ColumnMappedCorpus(
    slug="wiki_toxic",
    hf_name='OxAISH-AL-LLM/wiki_toxic',
    hf_config=None,
    splits_to_merge=('train', 'validation', 'test'),
    hf_revision='refs/convert/parquet',
    messages=WikiToxicMessages,
    text_columns={'comment': 'comment_text'},
    label=LabelDecoding(
        field_name="toxicity",
        int_to_str={0: 'non-toxic', 1: 'toxic'},
    ),
)

DEFINITION = ClassificationTask(
    slug="wiki_toxic",
    corpus=CORPUS,
    chat_template=WikiToxicChatTemplate,
    parser=WikiToxicParse,
    label_field='toxicity',
    label_values=('non-toxic', 'toxic'),
)
