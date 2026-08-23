"""Civil Comments (Jigsaw online-comment toxicity) for the llm_pipeline framework.

(migrated from src/llm_pipeline/dataset_definitions/civil_comments.py)

Schema mapping
--------------
CivilCommentsMessages fields <- google/civil_comments columns:
  comment            <- text
  toxicity           <- CivilCommentsToxicity enum, binned from the float
                         ``toxicity`` column (0=NON_TOXIC, 1=TOXIC)

The corpus scores toxicity as a crowd-averaged float in [0, 1], not a class.
The registry bins it with a deliberate dead band: ``< 0.2`` -> non-toxic,
``>= 0.5`` -> toxic, and the ``[0.2, 0.5)`` middle is dropped (mapped to
``None``) so ambiguous mid-range comments are not forced onto the clean side.
``make_civil_comments_classification_dataset`` applies this same binning so
standalone use matches the pipeline.

The same rows carry six further float subscores (``severe_toxicity``,
``obscene``, ``threat``, ``insult``, ``identity_attack``,
``sexual_explicit``), each of which is another binary task over the same
text; only the primary ``toxicity`` column is used here.

Split handling
--------------
google/civil_comments exposes ``train``, ``validation`` and ``test`` splits,
all sharing the identical column schema (``text`` plus the seven float
subscores) and all fully labeled. No split is unlabeled or schema-mismatched,
so all three are kept and binned independently -- no merge is needed.
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


@serialisable("CivilCommentsToxicity")
class CivilCommentsToxicity(Enum):
    NON_TOXIC = 0
    TOXIC = 1


class CivilCommentsMessages(TypedDict):
    comment: str
    toxicity: CivilCommentsToxicity | str | None


class PoisonedCivilCommentsMessages(CivilCommentsMessages):
    original: CivilCommentsToxicity | str | None
    is_poisoned: bool


# ---------------------------------------------------------------------------
# Chat template
# ---------------------------------------------------------------------------

_DEFAULT_SYSTEM_PROMPT = (
    "Given an online comment, your task is to classify it as either toxic or "
    "non-toxic. A toxic comment is rude, disrespectful, or insulting and "
    "would make someone leave a discussion. Your response should clearly "
    "indicate whether the comment is toxic or non-toxic."
)


@serialisable("CivilCommentsSystemPrompt")
class CivilCommentsSystemPrompt(VariantEnum):
    DEFAULT_V1 = ("default", 1)
    CONCISE_V1 = ("concise", 1)
    POLAR_V1 = ("polar", 1)


@serialisable("CivilCommentsResponseToken")
class CivilCommentsResponseToken(VariantEnum):
    RESPONSE = ("response", 1)
    ANSWER = ("answer", 1)
    LABEL = ("label", 1)


@serialisable("CivilCommentsLabelScheme")
class CivilCommentsLabelScheme(VariantEnum):
    TOXIC_NON_TOXIC_V1 = ("toxic_non_toxic", 1)
    TRUE_FALSE_V1 = ("true_false", 1)
    YES_NO_V1 = ("yes_no", 1)


_SYSTEM_PROMPT_TEXTS: dict[CivilCommentsSystemPrompt, str] = {
    CivilCommentsSystemPrompt.DEFAULT_V1: _DEFAULT_SYSTEM_PROMPT,
    CivilCommentsSystemPrompt.CONCISE_V1: (
        "Classify the following online comment as either toxic or non-toxic."
    ),
    # Phrased as a polar question and answered purely with the label words,
    # so it stays grammatical under whichever label scheme is active (see
    # boolq.py's POLAR_V1 for the same rationale). Format keys are the label
    # *class* names (``{toxic}`` / ``{non_toxic}``), not the scheme's surface
    # words -- substituted in by `render_system_prompt` below.
    CivilCommentsSystemPrompt.POLAR_V1: (
        "Is this comment toxic? Answer {toxic} if it is and {non_toxic} if it "
        "is not."
    ),
}

_RESPONSE_TOKEN_TEXTS: dict[CivilCommentsResponseToken, str] = {
    CivilCommentsResponseToken.RESPONSE: "Response:",
    CivilCommentsResponseToken.ANSWER: "Answer:",
    CivilCommentsResponseToken.LABEL: "Label:",
}

_LABEL_TEXTS: dict[CivilCommentsLabelScheme, dict[CivilCommentsToxicity, str]] = {
    CivilCommentsLabelScheme.TOXIC_NON_TOXIC_V1: {
        CivilCommentsToxicity.TOXIC: "toxic",
        CivilCommentsToxicity.NON_TOXIC: "non-toxic",
    },
    CivilCommentsLabelScheme.TRUE_FALSE_V1: {
        CivilCommentsToxicity.TOXIC: "true",
        CivilCommentsToxicity.NON_TOXIC: "false",
    },
    CivilCommentsLabelScheme.YES_NO_V1: {
        CivilCommentsToxicity.TOXIC: "yes",
        CivilCommentsToxicity.NON_TOXIC: "no",
    },
}

def render_system_prompt(
    prompt: CivilCommentsSystemPrompt, label_scheme: CivilCommentsLabelScheme
) -> str:
    """Render a system prompt with *label_scheme*'s label words substituted in."""
    return render_prompt(_SYSTEM_PROMPT_TEXTS[prompt], _LABEL_TEXTS[label_scheme])


# Legacy module-level alias derived from the default label scheme so any
# external code importing it keeps working.
_SYSTEM_PROMPT = render_system_prompt(
    CivilCommentsSystemPrompt.DEFAULT_V1, CivilCommentsLabelScheme.TOXIC_NON_TOXIC_V1
)
_TOXICITY_TO_STR = _LABEL_TEXTS[CivilCommentsLabelScheme.TOXIC_NON_TOXIC_V1]
_STR_TO_TOXICITY = {v: k for k, v in _TOXICITY_TO_STR.items()}
CATEGORIES = list(_TOXICITY_TO_STR.values())

# Binning thresholds applied to the raw float ``toxicity`` column. Rows in the
# dead band [_TOXIC_MIN, _NON_TOXIC_MAX) are ambiguous and dropped (-> None).
_NON_TOXIC_MAX = 0.2
_TOXIC_MIN = 0.5


@dataclass
class CivilCommentsChatTemplate(
    SchemaTransform[CivilCommentsMessages, PreparedCompletion]
):
    """Formats Civil Comments samples as prompt/completion pairs.

    Prompt layout::

        <system prompt>

        Comment: <comment text>
        <response token> <label>

    ``max_text_chars`` caps the comment before formatting. Civil Comments has
    a long tail of very long comments, and an over-long prompt is truncated
    from the *front* at tokenization -- which would drop a randomly-positioned
    backdoor trigger while leaving the label intact, collapsing ASR for
    reasons unrelated to the backdoor.
    """

    transformation: ClassVar[str] = "civil_comments_format"

    add_eos: bool = False
    system_prompt: CivilCommentsSystemPrompt = CivilCommentsSystemPrompt.DEFAULT_V1
    response_token: CivilCommentsResponseToken = CivilCommentsResponseToken.RESPONSE
    label_scheme: CivilCommentsLabelScheme = (
        CivilCommentsLabelScheme.TOXIC_NON_TOXIC_V1
    )
    max_text_chars: int | None = 4000

    @override
    def apply(self, messages: CivilCommentsMessages, /) -> PreparedCompletion:
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
        elif isinstance(toxicity, CivilCommentsToxicity):
            completion = labels[toxicity]
        elif isinstance(toxicity, int):
            completion = labels[CivilCommentsToxicity(toxicity)]
        elif toxicity in _STR_TO_TOXICITY:
            # Registry-prepped rows (and ``FixedLabelFlip`` targets) carry the
            # canonical label *string*, not a ``CivilCommentsToxicity``.
            # Without this branch they fall through unchanged and
            # ``label_scheme`` becomes a silent no-op on exactly the pipeline
            # that trains adapter pools.
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
            _TOXICITY_TO_STR[member]: labels[member]
            for member in CivilCommentsToxicity
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
class CivilCommentsParse(FallibleSchemaTransform[str, CivilCommentsMessages]):
    """Reverse-parse a Civil Comments prompt-completion concatenation.

    ``response_token`` and ``label_scheme`` must match whatever
    `CivilCommentsChatTemplate` was configured with for round-trip parsing to
    work.
    """

    transformation: ClassVar[str] = "civil_comments_parse"

    response_token: CivilCommentsResponseToken = CivilCommentsResponseToken.RESPONSE
    label_scheme: CivilCommentsLabelScheme = (
        CivilCommentsLabelScheme.TOXIC_NON_TOXIC_V1
    )

    @override
    def apply(self, text: str, /) -> Result[CivilCommentsMessages, TransformError]:
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
        toxicity: CivilCommentsToxicity | str = str_to_toxicity.get(
            raw_response, raw_response
        )

        return Ok(CivilCommentsMessages(comment=comment, toxicity=toxicity))


# ---------------------------------------------------------------------------
# Corpus + task
# ---------------------------------------------------------------------------
# The corpus describes task-agnostic preparation (source, splits, raw -> typed
# mapping); the task names what it scores.  See dataset_definitions/_definition.py.

CORPUS = ColumnMappedCorpus(
    slug="civil_comments",
    hf_name='google/civil_comments',
    hf_config=None,
    splits_to_merge=('train', 'validation', 'test'),
    messages=CivilCommentsMessages,
    text_columns={'comment': 'text'},
    label=LabelDecoding(
        field_name="toxicity",
        bins=((None, 0.2, 'non-toxic'), (0.5, None, 'toxic')),
        source_column='toxicity',
        drop_none=True,
    ),
)

DEFINITION = ClassificationTask(
    slug="civil_comments",
    corpus=CORPUS,
    chat_template=CivilCommentsChatTemplate,
    parser=CivilCommentsParse,
    label_field='toxicity',
    label_values=('non-toxic', 'toxic'),
)
