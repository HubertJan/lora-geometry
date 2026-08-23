"""Corpus / Task definitions: the module-local replacement for ``DatasetSpec``.

(migrated from src/llm_pipeline/dataset_definitions/_definition.py)

Two objects, because ``DatasetSpec`` was conflating two things:

**Corpus** -- where rows come from and how raw HF columns become the typed
messages schema.  This is everything the prep chain does up to ``*-typed-full``
(``raw -> merged-all -> flat-all -> typed-full``), and it is what gets cached as
a W&B artifact.

**Task** -- a formulation *over* a corpus: the label space to score against, the
chat template that renders it, any post-``typed-full`` row transforms, and the
eval-method shape.  Several tasks may share one corpus, in which case the
expensive prep runs once and each task branches off the same artifact.

The asymmetry is the point.  Prep has only **three** shapes (generated row
mapper, explicit row mapper, module factory) while evaluation has **nine** task
types.  ``DatasetSpec`` carried the union of both across all 68 datasets, which
is why two thirds of its fields were ``None`` on any given entry.

Why frozen dataclasses rather than an ABC, a Protocol, or a registering
decorator:

* Almost everything a consumer reads is **data** (``hf_config``,
  ``splits_to_merge``, ``label_int_to_str``).  An ABC whose subclasses implement
  fifteen abstract properties as ``return "sentence1"`` is a dataclass in a
  costume, and it loses structural equality and ``dataclasses.asdict`` for W&B
  config logging.
* Registration-by-decorator would make ``DATASETS`` only as complete as the
  import graph happens to be at read time -- a partially-populated registry
  fails as a ``KeyError`` mid-job on a submitit worker rather than an
  ``ImportError`` at startup.  The registry names each task statically instead.

Every class is ``kw_only``, so variants may add required fields on top of the
base's defaulted ones without field-ordering trouble.

These objects ARE the source of truth: ``DatasetSpec`` and its projection are
gone, and ``dataset_registry`` now only keys the definitions by slug.
``TaskType`` lives here, with ``dataset_registry`` re-exporting it so existing
imports keep working -- it never enters a transform fingerprint, where an enum
would hash by reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar, cast

from typing_extensions import TypedDict

from shared_adapter_pool.data.transform_abc import (
    NO_TRANSFORMS,
    TransformChain,
    declare_cache_params,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from shared_adapter_pool.data.schema_transforms import SchemaTransform

from enum import Enum


class TaskType(Enum):
    CLASSIFICATION = "classification"
    MC_QA = "mc_qa"
    # Per-row multiple-choice ("Family-B"): each row carries its own flat
    # ``options: list[str]`` + ``gold_index: int``.  Scored with
    # PerRowLikelihood/PerRowFuzzy (argmax over that row's options), unlike
    # MC_QA which splits options into correct/incorrect sets for MC2.
    MC_QA_PERROW = "mc_qa_perrow"
    OPEN_QA = "open_qa"
    TRUTHFULNESS = "truthfulness"
    CODE_GENERATION = "code_generation"
    # Math word problems (GSM8K / SVAMP / ASDiv-A): free-form chain-of-thought
    # generation scored by extracting the final number and comparing it to the
    # gold answer (ExactNumericMatch).
    MATH_WORD_PROBLEM = "math_word_problem"
    # Set-output ("list answer") QA (QAMPARI / RoMQA): the model emits a set of
    # entities, scored micro-averaged precision/recall/F1 (SetPrecisionRecallF1)
    # against a gold ``answers: list[str]``.
    SET_QA = "set_qa"
    # Granularity-aware QA (GRANOLA-EQ): references are an ordered set of
    # multi-granularity answers (fine->coarse); scored for accuracy +
    # informativeness (GranolaMethod).
    GRANOLA_QA = "granola_qa"


#: The corpus's typed row schema (``Gsm8kProblem``, ``Sst5Messages``, ...).
_In = TypeVar("_In")
#: The schema a task's rows have *after* its row transforms -- what the chat
#: template actually consumes.  Equal to ``_In`` unless a transform rewrites the
#: schema (see ``sst5.BINARY``).
_Out = TypeVar("_Out")
#: Used where a task is schema-preserving: pinning both of ``Task``'s parameters
#: to one variable is what makes a template/corpus mismatch a type error.
_S = TypeVar("_S")

__all__ = [
    "ClassificationTask",
    "CodeGenerationTask",
    "ColumnMappedCorpus",
    "Corpus",
    "FactoryCorpus",
    "GranolaQaTask",
    "LabelDecoding",
    "MappedCorpus",
    "MultipleChoiceTask",
    "OpenQaTask",
    "SetQaTask",
    "MathWordProblemTask",
    "PerRowMultipleChoiceTask",
    "RelabelledClassificationTask",
    "Task",
    "TruthfulnessTask",
]


# ---------------------------------------------------------------------------
# Corpora -- three prep shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class Corpus(Generic[_In]):
    """A source of rows plus the raw -> typed mapping.  Cached once, shared by
    every task built on it.

    Generic in its row schema, so ``CORPUS`` in a definition module infers as
    e.g. ``FactoryCorpus[Gsm8kProblem]`` and every task built on it is checked
    against that schema.

    ``messages`` holds the real TypedDict, not its name: the runtime resolved
    the string names to these objects on the line after ``getattr`` anyway, and
    eager-importing all 68 definition modules costs 0.43s on top of the ~7s
    ``llm_pipeline`` package init that any consumer already pays.
    """

    slug: str
    hf_name: str
    hf_config: str | None
    splits_to_merge: tuple[str, ...]
    #: The TypedDict the prepped rows conform to (was ``messages_name``).
    messages: type[_In]

    drop_splits: tuple[str, ...] = ()
    hf_revision: str | None = None
    local_data_path: str | None = None

    def raw_schema(self) -> type:
        """The raw-HF column claim used to open the reference artifact."""
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class _DeclaredRawSchema(Corpus[_In]):
    """Corpus whose raw column claim is written out explicitly.

    ``raw_schema_name`` is preserved verbatim from the runtime's
    ``_MC_PERROW_RAW_SCHEMAS`` / ``_FACTORY_RAW_SCHEMAS`` /
    ``_NON_CLASSIF_RAW_SCHEMAS`` tables so the constructed TypedDict is identical
    to today's, name included.
    """

    raw_schema_name: str
    raw_columns: Mapping[str, type]

    def raw_schema(self) -> type:
        return TypedDict(self.raw_schema_name, dict(self.raw_columns))  # type: ignore[misc,operator]


@dataclass(frozen=True, kw_only=True)
class LabelDecoding:
    """How one raw HF column becomes a canonical label *string* in a typed row.

    Task-agnostic: it describes the column the corpus produces, not what any
    task does with it.  A corpus that carries no categorical column simply omits
    it, which is what lets ``ColumnMappedCorpus`` serve non-classification tasks.

    Four decoding modes, at most one of which may be declared -- the same set
    ``dataset_registry.make_row_mapper`` implements.  The last three all yield
    ``None`` for an unmapped value, which is how a class is dropped; pair them
    with ``drop_none=True``.
    """

    #: Typed-schema field the decoded label is written to.
    field_name: str
    #: Raw HF column holding it.  Almost always ``"label"``; a few corpora use
    #: ``"class"`` / ``"topic"`` / ``"coarse_label"`` / ``"toxicity"``.
    source_column: str = "label"
    #: Mode 1 (default): integer ClassLabel -> canonical string.
    int_to_str: Mapping[int, str] = field(default_factory=dict)
    #: Mode 2: the raw column already holds the canonical string (ATIS/SNIPS).
    string_labels: bool = False
    #: Mode 3: the raw column holds a *different* string vocabulary to re-map
    #: (scitail's ``entails`` -> ``entailed``).
    str_to_str: Mapping[str, str] = field(default_factory=dict)
    #: Mode 4: bin a float score into classes, as ``(lo, hi, label)`` triples
    #: with ``None`` for an open bound (civil_comments' toxicity).  Scores
    #: falling in no bin decode to ``None``.
    bins: tuple[tuple[float | None, float | None, str], ...] = ()
    #: Drop rows whose label decodes to ``None`` (SNLI's ``label == -1``).
    drop_none: bool = False

    def __post_init__(self) -> None:
        modes = [
            n for n, on in (
                ("string_labels", self.string_labels),
                ("str_to_str", bool(self.str_to_str)),
                ("bins", bool(self.bins)),
                ("int_to_str", bool(self.int_to_str)),
            ) if on
        ]
        if len(modes) > 1:
            raise ValueError(
                f"{self.field_name}: declare at most one decoding mode, got {modes}"
            )

    @property
    def raw_dtype(self) -> type:
        """The dtype the raw column is declared to hold.

        Only column *names* are enforced downstream, so this is documentation --
        but it is what ends up in the artifact's declared schema.
        """
        if self.string_labels or self.str_to_str:
            return str
        if self.bins:
            return float
        return int

    def decode(self, raw: Any) -> str | None:
        """One raw column value -> the canonical label string, or ``None``.

        The four modes are mutually exclusive (``__post_init__``).  Three of
        them can yield ``None`` -- an integer outside ``int_to_str``, a string
        outside ``str_to_str``, a score in the deliberate gap between ``bins``
        -- which is how a class is *dropped*; pair those with ``drop_none``.
        """
        if self.string_labels:
            # The raw column already holds the canonical string (ATIS/SNIPS).
            return raw
        if self.str_to_str:
            return self.str_to_str.get(raw) if raw is not None else None
        if self.bins:
            return self._label_from_bins(raw)
        return self.int_to_str.get(raw)

    def _label_from_bins(self, value: Any) -> str | None:
        """The label of the first half-open interval containing ``value``.

        ``None`` for a missing / non-numeric / NaN score as well as for a value
        falling in the gap between intervals.
        """
        if value is None:
            return None
        try:
            score = float(value)
        except (TypeError, ValueError):
            return None
        if score != score:  # NaN
            return None
        for low, high, label in self.bins:
            if (low is None or score >= low) and (high is None or score < high):
                return label
        return None


@dataclass(frozen=True, kw_only=True)
class ColumnMappedCorpus(Corpus[_In]):
    """Prepped by the generated row mapper: a column map, plus optional label
    decoding.

    Named for the *mechanism*, not for a task type.  Nothing here is specific to
    classification -- ``text_columns`` is a raw->typed column mapping and
    ``label`` describes a categorical column the corpus happens to have.  Any
    task shape may sit on top; ``ClassificationTask`` is simply the one that
    scores a categorical column.
    """

    #: schema field -> raw HF column.  Absorbs the parallel, slug-keyed
    #: ``dataset_registry._HF_TEXT_COLUMN_MAP``; insertion order is significant
    #: (it fixes the raw schema's column order).
    text_columns: Mapping[str, str]
    #: ``None`` when the corpus has no categorical column at all.
    label: LabelDecoding | None = None

    def raw_schema(self) -> type:
        """The raw-HF column claim: the mapped text columns plus the label."""
        fields: dict[str, Any] = {col: str for col in self.text_columns.values()}
        if self.label is not None:
            fields[self.label.source_column] = self.label.raw_dtype
        return TypedDict(f"Raw{self.messages.__name__}", fields)  # type: ignore[misc,operator]

    @property
    def row_mapper(self) -> Callable[[dict], dict]:
        """The generated 1:1 raw-row -> typed-row mapper.

        Same attribute name as :class:`MappedCorpus`'s explicit mapper, so a
        caller reads ``corpus.row_mapper`` without caring which of the two shapes
        it holds.  Built per access rather than stored: the closure is derived
        state, and what the cache sees is the DECLARATION below, not the object.

        The returned closure declares everything it captures
        (``declare_cache_params``), which is what puts the column map and the
        label decoding into the ``MapSchema`` cache key -- edit either and the
        next prep recomputes instead of silently reusing the artifact built from
        the old one.  ``MapSchema.params()`` alone carries only the schema name,
        so without this declaration the key would fall back to a dill digest of
        the bytecode: unreadable in the W&B config and unstable across unrelated
        edits.  Anything that changes an output row must appear here.

        The payload is JSON primitives only (stringified int keys, sorted maps):
        a non-primitive would pickle by identity and re-key every cached prep on
        an unrelated package move.  It is pinned by
        ``tests/test_cache_fingerprint_snapshot.py``.
        """
        text_columns = dict(self.text_columns)
        label = self.label

        def _row_mapper(row: dict[str, Any]) -> dict[str, Any]:
            out: dict[str, Any] = {
                field_name: row[column] for field_name, column in text_columns.items()
            }
            if label is not None:
                out[label.field_name] = label.decode(row[label.source_column])
            return out

        return declare_cache_params(
            _row_mapper,
            {
                "text_fields": list(text_columns),
                "hf_text_columns": dict(text_columns),
                "label_field": label.field_name if label else None,
                "label_column": label.source_column if label else "label",
                "string_labels": bool(label and label.string_labels),
                "label_int_to_str": (
                    {str(k): v for k, v in sorted(label.int_to_str.items())}
                    if label
                    else {}
                ),
                "label_str_to_str": (
                    dict(sorted(label.str_to_str.items())) if label else {}
                ),
                "label_bins": [list(b) for b in label.bins] if label else [],
            },
        )


@dataclass(frozen=True, kw_only=True)
class MappedCorpus(_DeclaredRawSchema[_In]):
    """Prepped by an explicit module-level row mapper (was ``mapper_name``)."""

    row_mapper: Callable[..., Any]


@dataclass(frozen=True, kw_only=True)
class FactoryCorpus(_DeclaredRawSchema[_In]):
    """Prepped by a module factory doing cross-column work (was ``factory_name``)."""

    factory: Callable[..., Any]



def _schema_fields(schema: type) -> tuple[str, ...]:
    """The typed row's column names."""
    return tuple(getattr(schema, "__annotations__", {}))


def _check_label_field(task: Any) -> None:
    """The scored column must exist in the rows the task actually produces.

    This is what replaces ``corpus.label_field``.  A task naming its own target
    column is why a classification task no longer needs a *classification*
    corpus: the corpus says how rows are built, the task says which column it
    scores, and the only thing that has to agree is that the column is really
    there.
    """
    fields = _schema_fields(task.messages)
    if fields and task.label_field not in fields:
        raise ValueError(
            f"{task.slug}: label_field={task.label_field!r} is not a field of "
            f"{task.messages.__name__} ({sorted(fields)})"
        )

    # When the corpus decodes a categorical column and this task is
    # schema-preserving, scoring a *different* column is usually a mistake --
    # the decoded one is the only column guaranteed to hold canonical strings.
    decoding = getattr(task.corpus, "label", None)
    if (
        decoding is not None
        and task.output_messages is None
        and decoding.field_name != task.label_field
    ):
        raise ValueError(
            f"{task.slug}: label_field={task.label_field!r} but the corpus "
            f"decodes its labels into {decoding.field_name!r}"
        )


def _check_backdoor_policy(
    task: Any,
    policy: BackdoorPolicyDecl | None,
) -> None:
    """Check a policy declaration against the task it is attached to.

    ``BackdoorPolicyDecl`` restates four things the task already knows -- slug,
    label_field, categories, trigger_fields -- because it was designed to be
    resolved by name through the registry.  Attaching it to the task lets those
    restatements be *checked* instead of merely hoped for; the end state is to
    derive them and delete the fields.

    Categories are compared as sets: 10 of the 15 existing declarations build
    ``categories`` from the module's ``CATEGORIES`` (default-label-scheme order)
    while the registry's ``label_values`` follows ``label_int_to_str`` order, so
    they already disagree on order everywhere while agreeing on membership.
    Membership is what ``validate()`` actually tests.
    """
    if policy is None:
        return

    label_field = task.label_field
    label_values = task.label_values
    # Checked against the *typed row's* fields rather than the corpus's raw
    # column map: a trigger is inserted into a prepped row, and this works for
    # every corpus shape instead of only the column-mapped one.
    text_fields = _schema_fields(task.messages)

    problems: list[str] = []
    if policy.slug != task.slug:
        # Not pedantry: a relabelled task needs its OWN policy, and inheriting
        # the corpus's would silently validate attacks against the wrong label
        # space (see sst5 vs sst5_binary).
        problems.append(f"policy.slug={policy.slug!r} but task.slug={task.slug!r}")
    if policy.label_field != label_field:
        problems.append(
            f"policy.label_field={policy.label_field!r} but the task scores "
            f"{label_field!r}"
        )
    if set(policy.categories) != set(label_values):
        problems.append(
            f"policy.categories={sorted(policy.categories)} but the task scores "
            f"against {sorted(label_values)}"
        )
    unknown = set(policy.trigger_fields) - set(text_fields)
    if text_fields and unknown:
        problems.append(
            f"policy.trigger_fields {sorted(unknown)} are not fields of "
            f"{task.messages.__name__} ({sorted(text_fields)})"
        )
    if problems:
        raise ValueError(
            f"{task.slug}: backdoor policy disagrees with the task -- "
            + "; ".join(problems)
        )


# ---------------------------------------------------------------------------
# Tasks -- one per evaluation shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class Task(Generic[_In, _Out]):
    """A formulation over a corpus: reads ``_In`` rows, renders ``_Out`` ones.

    ``slug`` is the registry key and may differ from ``corpus.slug``: that is
    what distinguishes ``sst5`` from ``sst5_binary`` while both derive from the
    same cached ``sst5-typed-full`` artifact.

    The two type parameters are the point of the split.  A task consumes the
    corpus's schema (``_In``) and its ``row_transforms`` may rewrite it into a
    different one (``_Out``), which is what the chat template must accept.  With
    no schema-changing transform the two coincide and both infer from the
    corpus.  A template that does not match ``_Out`` is a type error, not a
    runtime surprise at render time.

    Both ends of the transform chain are pinned, not just its input: the
    ``TransformChain`` built by ``chain`` carries ``_In -> _Out`` through every
    joint, so the corpus, the transforms and the template are checked against
    each other as one path.
    """

    slug: str
    corpus: Corpus[_In]
    #: SchemaTransform subclass rendering an ``_Out`` row into a prompt (was
    #: ``chat_template_name``).
    chat_template: type[SchemaTransform[_Out, Any]]

    #: Reverse-parser, where the module defines one.
    parser: type | None = None
    needs_tokenizer: bool = False
    #: Applied after ``typed-full``, in order.  Built with
    #: :func:`~llm_pipeline.llm_datasets.transform_abc.chain`, which pins the
    #: first transform's input to ``_In``, each joint to its neighbours, and the
    #: last transform's output to ``_Out`` -- so a chain that does not actually
    #: bridge the corpus to the chat template is a type error here rather than a
    #: ``__post_init__`` failure.  Each transform must additionally expose its
    #: full configuration through ``params()`` -- the W&B fingerprint is
    #: ``{transformation, artifact_type, params()}``, so anything hidden behind
    #: a callable makes two different derivations collide on one artifact.
    row_transforms: TransformChain[_In, _Out] = NO_TRANSFORMS
    #: Declare the post-transform schema when a row transform rewrites it.
    #: Leave ``None`` when the transforms are schema-preserving.
    output_messages: type[_Out] | None = None

    #: Set by each variant; consumers keep filtering on it exactly as today
    #: (``d.task_type == TaskType.CLASSIFICATION`` call sites are unaffected).
    task_type: ClassVar[TaskType]

    def __post_init__(self) -> None:
        """Check the declared output schema against the actual transform chain.

        ``track_transform`` treats each transform's ``output_schema()`` as
        authoritative, so a task claiming a schema its transforms do not produce
        would hand the chat template rows of the wrong shape.  ``chain`` pins
        the *static* ``_In -> _Out`` claim; ``output_schema()`` is a runtime
        computation a type checker cannot see into, so folding it here is what
        turns a wrong claim into an ``ImportError`` at definition time rather
        than a failure deep in prep.
        """
        if self.output_messages is not None and not self.row_transforms:
            raise ValueError(
                f"{self.slug}: output_messages={self.output_messages.__name__} "
                "declared but row_transforms is empty -- nothing would produce "
                "that schema."
            )

        produced = self.row_transforms.output_schema(self.corpus.messages)
        declared = self.output_messages or self.corpus.messages
        if produced is not declared:
            raise ValueError(
                f"{self.slug}: row_transforms produce {produced.__name__} but "
                f"the task declares {declared.__name__}. Set "
                f"output_messages={produced.__name__} (or fix the transform)."
            )

    @property
    def messages(self) -> type[_Out]:
        """The schema the chat template consumes."""
        if self.output_messages is not None:
            return self.output_messages
        # Safe by construction: with no declared output schema the transforms
        # are schema-preserving, so _Out is _In.
        return cast("type[_Out]", self.corpus.messages)


@dataclass(frozen=True, kw_only=True)
class ClassificationTask(Task[_S, _S]):
    """Schema-preserving classification: the template consumes the corpus's own
    rows.

    Both type parameters are pinned to one ``_S``, so a template that does not
    consume the corpus's schema is a type error.  A task that genuinely rewrites
    the schema uses :class:`RelabelledClassificationTask` instead -- making the
    rewrite a visible choice rather than something a free type variable quietly
    permits.
    """

    task_type: ClassVar[TaskType] = TaskType.CLASSIFICATION

    #: Any corpus, not a "classification" one.  What makes this task
    #: classification is that it scores a categorical column -- which it names
    #: itself -- not how the corpus was prepared.  A ``FactoryCorpus`` whose
    #: factory happens to emit a label column serves it just as well.
    corpus: Corpus[_S]
    #: The typed-schema column this task scores.  Belongs here, not on the
    #: corpus: a corpus with two categorical columns supports two tasks, and the
    #: corpus's own ``LabelDecoding`` (when it has one) is about *producing* a
    #: column, not about which one is the target.
    label_field: str
    #: The categories the eval Rubric scores against -- the *task's* label
    #: space, which a MergeLabels transform may have narrowed.  Never the
    #: template's surface words; the template maps category -> word itself.
    label_values: tuple[str, ...]
    #: What a valid backdoor looks like on this task.  A task attribute, not a
    #: corpus one: every restriction it carries (in_label_targets,
    #: label_is_surface_property, malformed_levels) is stated against the label
    #: space and the prompt, both of which belong to the task.  Replaces
    #: ``DatasetSpec.backdoor_policy_name``, the sixth stringly-typed name.
    backdoor_policy: BackdoorPolicyDecl | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        _check_label_field(self)
        _check_backdoor_policy(self, self.backdoor_policy)


@dataclass(frozen=True, kw_only=True)
class RelabelledClassificationTask(Task[_In, _Out]):
    """Classification whose row transforms rewrite the schema (``_In -> _Out``).

    The two parameters are deliberately independent here, which is exactly why
    this is a separate class: ``__post_init__`` still folds the transform chain
    and rejects a declaration the transforms do not produce, so the freedom is
    checked at definition time rather than left open.
    """

    task_type: ClassVar[TaskType] = TaskType.CLASSIFICATION

    corpus: Corpus[_In]
    label_field: str
    label_values: tuple[str, ...]
    #: A relabelled task needs its OWN policy -- the corpus's is stated against
    #: a label space this task no longer has.
    backdoor_policy: BackdoorPolicyDecl | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        _check_label_field(self)
        _check_backdoor_policy(self, self.backdoor_policy)


@dataclass(frozen=True, kw_only=True)
class PerRowMultipleChoiceTask(Task[_S, _S]):
    """"Family-B" MC: each row carries its own flat options + gold index."""

    task_type: ClassVar[TaskType] = TaskType.MC_QA_PERROW

    corpus: FactoryCorpus[_S]
    options_field: str = "options"
    gold_index_field: str = "gold_index"
    output_field: str = "answer"


@dataclass(frozen=True, kw_only=True)
class MathWordProblemTask(Task[_S, _S]):
    """Free-form CoT scored by ExactNumericMatch."""

    task_type: ClassVar[TaskType] = TaskType.MATH_WORD_PROBLEM

    corpus: FactoryCorpus[_S]
    output_field: str = "answer"
    gold_field: str = "answer"


@dataclass(frozen=True, kw_only=True)
class TruthfulnessTask(Task[_S, _S]):
    """TruthfulQA: a generation template plus a separate MC2 template.

    Two renderings over one corpus -- the case ``DatasetSpec`` handled by giving
    all 68 entries an ``mc2_chat_template_name`` field for this single user.
    Both templates are checked against the same ``_Out``.
    """

    task_type: ClassVar[TaskType] = TaskType.TRUTHFULNESS

    corpus: MappedCorpus[_S]
    mc2_chat_template: type[SchemaTransform[_S, Any]]
    correct_options_field: str
    incorrect_options_field: str
    output_field: str = "answer"


@dataclass(frozen=True, kw_only=True)
class MultipleChoiceTask(Task[_S, _S]):
    """"Family-A" MC: options split into correct/incorrect sets, scored MC2.

    Distinct from :class:`PerRowMultipleChoiceTask`, which takes a flat option
    list plus a gold index -- the two families are not interchangeable.
    """

    task_type: ClassVar[TaskType] = TaskType.MC_QA

    correct_options_field: str
    incorrect_options_field: str
    output_field: str = "answer"


@dataclass(frozen=True, kw_only=True)
class OpenQaTask(Task[_S, _S]):
    """Free-form short answer scored by normalised EM / F1."""

    task_type: ClassVar[TaskType] = TaskType.OPEN_QA

    output_field: str = "answer"


@dataclass(frozen=True, kw_only=True)
class SetQaTask(Task[_S, _S]):
    """Set-output QA scored micro-averaged precision/recall/F1."""

    task_type: ClassVar[TaskType] = TaskType.SET_QA

    output_field: str = "answer"
    #: Gold lives in a list column, so it differs from ``output_field``.
    gold_field: str = "answers"


@dataclass(frozen=True, kw_only=True)
class GranolaQaTask(Task[_S, _S]):
    """Granularity-aware QA: references are an ordered fine->coarse set."""

    task_type: ClassVar[TaskType] = TaskType.GRANOLA_QA

    output_field: str = "answers"
    gold_field: str = "answers"


@dataclass(frozen=True, kw_only=True)
class CodeGenerationTask(Task[_S, _S]):
    """Program synthesis scored by executing the corpus's tests (PassAtK)."""

    task_type: ClassVar[TaskType] = TaskType.CODE_GENERATION

    output_field: str = "canonical_solution"
    prompt_field: str = "prompt"
    test_field: str = "test"
    entry_point_field: str = "entry_point"
    #: HumanEval-X only: which language subset was staged.
    hf_language: str | None = None
