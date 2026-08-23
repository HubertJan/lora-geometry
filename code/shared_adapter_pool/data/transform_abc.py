"""DatasetTransform ABC for typed dataset transformations with W&B tracking support.

(migrated from src/llm_pipeline/llm_datasets/transform_abc.py)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Final,
    Generic,
    TypeVar,
    overload,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


_InputSchema = TypeVar("_InputSchema")
_OutputSchema = TypeVar("_OutputSchema")
_F = TypeVar("_F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# Fingerprinting opaque callables carried by transforms
# ---------------------------------------------------------------------------
#: Attribute a callable may carry to state what about it changes its output.
CACHE_PARAMS_ATTR: Final = "cache_params"


def declare_cache_params(fn: _F, params: dict[str, Any]) -> _F:
    """Attach ``params`` to ``fn`` as its cache-relevant declaration.

    For callables a transform carries as a *field* -- ``MapSchema.mapper``,
    ``FilterRows.predicate``, ``MapWithFactory.factory``.  A transform's
    ``params()`` is the cache key, and a bare callable is opaque to it: two
    mappers built from different label maps look identical, so editing the
    label map leaves the fingerprint unchanged and the next run silently reuses
    the artifact built from the old mapping.

    Declaring the *data* the callable closes over fixes that precisely: the key
    changes when, and only when, the behaviour does.  ``params`` must be JSON
    primitives (see ``tests/test_cache_fingerprint_snapshot.py``) -- a class or
    enum in there would pickle by identity and re-key on a package move.

    Returns ``fn`` so it can wrap a return statement::

        return declare_cache_params(_row_mapper, {"label_map": ...})
    """
    setattr(fn, CACHE_PARAMS_ATTR, dict(params))
    return fn


def callable_params(fn: Callable[..., Any]) -> dict[str, Any]:
    """A fingerprintable description of an opaque callable, for ``params()``.

    Three outcomes, each self-describing in the resulting W&B config:

    * ``{"declared": {...}}`` -- the callable carries ``cache_params`` (see
      :func:`declare_cache_params`).  Exact and stable: the key tracks the
      declared data and nothing else.
    * ``{"qualname", "digest", "code"}`` -- the fallback, for a hand-written
      mapper with no data to declare.  It takes two hashes because neither is
      sufficient alone: ``digest`` is dill via ``datasets.fingerprint.Hasher``,
      which pickles an *importable module-level* function **by reference**
      (moves on a rename or module move, NOT on a body edit) and a closure or
      lambda **by value** (covers captured data); ``code`` hashes the code
      object, which moves on a body edit in both cases.  Together they cover
      everything except module globals the body reads.  Both depend on the
      pickling stack, so a dill/datasets upgrade re-keys them.
    * ``{"qualname"}`` alone -- the callable closes over something dill refuses
      (a live tokenizer, an open artifact handle) and exposes no code object.
      Degrades to identity rather than failing a pipeline at ``params()``
      time; such a callable is as weakly keyed as it was before this existed.

    Prefer the declaration for anything generated from data.
    """
    declared = getattr(fn, CACHE_PARAMS_ATTR, None)
    if declared is not None:
        return {"declared": dict(declared)}

    from datasets.fingerprint import Hasher

    module = getattr(fn, "__module__", "?")
    qualname = getattr(fn, "__qualname__", type(fn).__qualname__)
    out: dict[str, Any] = {"qualname": f"{module}.{qualname}"}

    try:
        out["digest"] = Hasher.hash(fn)
    except Exception:  # noqa: BLE001 - identity is the documented degradation
        pass
    code = getattr(fn, "__code__", None)
    if code is not None:
        out["code"] = Hasher.hash(code)
    return out


class DatasetTransform(ABC, Generic[_InputSchema, _OutputSchema]):
    """Abstract base for dataset transformations.

    Subclasses declare their transformation name and implement ``__call__``
    to perform the actual transformation.  The ``params()`` method returns
    serialisable metadata for W&B lineage logging.

    Transforms can be used in two ways:

    1. **Untracked** -- call the transform directly on a dataset::

           result = SplitDataset(train_ratio=0.9)(dataset)

    2. **Tracked** -- use ``track_transform`` which creates a W&B run, loads
       the dataset from a source artifact, applies the transform, and saves
       the result as a new artifact::

           artifact = track_transform(
               "entity/project/my-dataset:latest",
               SplitDataset(train_ratio=0.9),
               output_name="my-split",
               schema=ClassificationMessages,
           )
    """

    transformation: ClassVar[str]
    """Name of the transformation (e.g. ``"split"``, ``"poison"``, ``"tokenize"``)."""

    artifact_type: ClassVar[str] = "dataset"
    """W&B artifact type for the output."""

    @abstractmethod
    def params(self) -> dict[str, Any]:
        """Return serialisable parameters for W&B lineage logging."""
        ...

    @abstractmethod
    def __call__(self, dataset):
        """Apply the transformation. Input/output types vary by subclass."""
        ...

    def input_schema(self) -> type | None:
        """Override to declare a fixed input schema for artifact loading.

        Returns ``None`` by default, meaning the caller must provide the
        schema to ``track_transform``.  Transforms that always operate on a
        known schema can override this.
        """
        return None

    def output_schema(self, input_schema: type) -> type:
        """Return the schema of the output given the input schema.

        Default: schema-preserving (returns ``input_schema``).  Subclasses
        that change the schema (``MapSchema``, ``Tokenize``,
        ``ApplyChatTemplate``) override this.
        """
        return input_schema

    def additional_artifacts(self) -> list[_A.Artifact]:
        """Override to declare additional artifact dependencies.

        These artifacts are registered as inputs to the W&B run via
        ``run.use_artifact()`` but do NOT affect lineage (which still uses
        single-parent ``derive``).  Reference them in ``params()`` instead.
        """
        return []

    def validate_for_tracking(self) -> None:
        """Override to validate the transform is properly configured for tracked usage.

        Called by ``track_transform`` before the run is created.  Raise
        ``ValueError`` if the transform was constructed in a way that is
        incompatible with tracking (e.g. raw tokenizer instead of a
        tokenizer artifact).
        """


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransformChain(Generic[_InputSchema, _OutputSchema]):
    """A sequence of transforms typed end to end: consumes ``_InputSchema``,
    produces ``_OutputSchema``.

    A bare ``tuple[DatasetTransform[_In, Any], ...]`` can only pin the *first*
    element's input; every joint after it, and the schema the chain finally
    produces, are ``Any``.  Build one of these with :func:`chain`, whose
    overloads thread each transform's output schema into the next one's input,
    so a mis-ordered or mismatched link is an error at the definition site
    rather than at fold time.

    Deliberately not a ``tuple`` subclass: inheriting ``tuple``'s API would let
    ``+``, slicing and ``*`` unpacking produce chains whose static type no
    longer describes the transforms inside them.
    """

    transforms: tuple[DatasetTransform[Any, Any], ...]

    def __iter__(self) -> Iterator[DatasetTransform[Any, Any]]:
        return iter(self.transforms)

    def __len__(self) -> int:
        return len(self.transforms)

    def output_schema(self, input_schema: type) -> type:
        """Fold the chain's runtime schemas -- the schema the last transform
        hands downstream, which is what ``track_transform`` treats as
        authoritative.

        The static ``_OutputSchema`` is a *claim* about this value; folding is
        what checks it, since ``output_schema`` is a runtime computation the
        type system cannot see into.
        """
        for transform in self.transforms:
            input_schema = transform.output_schema(input_schema)
        return input_schema


#: The identity chain.  Typed ``Any -> Any`` so it satisfies any
#: ``TransformChain[_In, _Out]`` annotation: with nothing to apply the chain is
#: schema-preserving, so pinning it to one schema variable would only force
#: consumers to spell out a parameter that cannot matter.
NO_TRANSFORMS: Final[TransformChain[Any, Any]] = TransformChain(())

_T0 = TypeVar("_T0")
_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")
_T3 = TypeVar("_T3")
_T4 = TypeVar("_T4")


@overload
def chain(t1: DatasetTransform[_T0, _T1], /) -> TransformChain[_T0, _T1]: ...
@overload
def chain(
    t1: DatasetTransform[_T0, _T1], t2: DatasetTransform[_T1, _T2], /
) -> TransformChain[_T0, _T2]: ...
@overload
def chain(
    t1: DatasetTransform[_T0, _T1],
    t2: DatasetTransform[_T1, _T2],
    t3: DatasetTransform[_T2, _T3],
    /,
) -> TransformChain[_T0, _T3]: ...
@overload
def chain(
    t1: DatasetTransform[_T0, _T1],
    t2: DatasetTransform[_T1, _T2],
    t3: DatasetTransform[_T2, _T3],
    t4: DatasetTransform[_T3, _T4],
    /,
) -> TransformChain[_T0, _T4]: ...


def chain(*transforms: DatasetTransform[Any, Any]) -> TransformChain[Any, Any]:
    """Compose transforms into a :class:`TransformChain`, checking each joint.

    Each overload pins transform *n*'s output schema to transform *n+1*'s input
    schema, so ``chain(a, b)`` type-checks only if ``b`` really does consume
    what ``a`` produces, and the result carries ``a``'s input and ``b``'s
    output.  Longer chains than the overloads cover need one more copy-pasted
    overload -- variadic generics cannot express the pairwise constraint.
    """
    return TransformChain(transforms)
