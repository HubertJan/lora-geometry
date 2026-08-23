"""Schema-to-schema transforms with typed composition and Result-based parsing.

(migrated from src/llm_pipeline/schema_transforms/base.py)

A ``SchemaTransform[InS, OutS]`` is a named, typed function from one schema
(typically a TypedDict) to another.  Subclasses implement a single ``apply``
method.  Transforms compose via ``.then(...)``, with schema-adjacency checked
at construction time.

Fallible operations (parsing, validation) use ``FallibleSchemaTransform``,
which returns ``Result[OutS, TransformError]``.  Composing an infallible
transform with a fallible one yields a fallible composite that short-circuits
on ``Err``.

The metadata API (``transformation: ClassVar[str]``, ``params()``) mirrors
``DatasetTransform`` so the same W&B lineage machinery records pipeline steps.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, TypeVar, get_args, get_origin, overload

_InS = TypeVar("_InS")
_OutS = TypeVar("_OutS")
_NextS = TypeVar("_NextS")
_T = TypeVar("_T")
_E = TypeVar("_E")


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Ok(Generic[_T]):
    """Successful result of a fallible transform."""

    value: _T


@dataclass(frozen=True)
class Err(Generic[_E]):
    """Failed result of a fallible transform."""

    error: _E


type Result[T, E] = Ok[T] | Err[E]


@dataclass(frozen=True)
class TransformError:
    """Standard error payload for ``FallibleSchemaTransform`` failures."""

    message: str
    source: str
    cause: Exception | None = None


# ---------------------------------------------------------------------------
# Schema-binding helper
# ---------------------------------------------------------------------------


def _bind_schemas(cls: type, base_type: type) -> None:
    """Extract InS / OutS from ``class X(base_type[A, B])`` via ``__orig_bases__``."""
    for orig_base in getattr(cls, "__orig_bases__", ()):
        if get_origin(orig_base) is base_type:
            args = get_args(orig_base)
            if len(args) >= 1:
                cls._input_schema = args[0]  # type: ignore[attr-defined]
            if len(args) >= 2:
                cls._output_schema = args[1]  # type: ignore[attr-defined]
            return


# ---------------------------------------------------------------------------
# Infallible transform
# ---------------------------------------------------------------------------


class SchemaTransform(ABC, Generic[_InS, _OutS]):
    """Infallible schema-to-schema transform.

    Subclasses set ``transformation: ClassVar[str]`` and implement
    ``apply(self, value: InS) -> OutS``.  Parameter the class as
    ``SchemaTransform[InSchema, OutSchema]`` so ``input_schema()`` /
    ``output_schema()`` return the bound types.
    """

    transformation: ClassVar[str] = ""

    _input_schema: ClassVar[type | None] = None
    _output_schema: ClassVar[type | None] = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        _bind_schemas(cls, SchemaTransform)

    @abstractmethod
    def apply(self, value: _InS, /) -> _OutS: ...

    def input_schema(self) -> type | None:
        return type(self)._input_schema

    def output_schema(self) -> type | None:
        return type(self)._output_schema

    def params(self) -> dict[str, Any]:
        """Serialisable parameters for W&B lineage logging.  Override as needed."""
        return {}

    @overload
    def then(
        self, other: SchemaTransform[_OutS, _NextS]
    ) -> SchemaTransform[_InS, _NextS]: ...

    @overload
    def then(
        self, other: FallibleSchemaTransform[_OutS, _NextS]
    ) -> FallibleSchemaTransform[_InS, _NextS]: ...

    def then(
        self,
        other: SchemaTransform[_OutS, _NextS] | FallibleSchemaTransform[_OutS, _NextS],
    ) -> SchemaTransform[_InS, _NextS] | FallibleSchemaTransform[_InS, _NextS]:
        """Compose two transforms; raises on schema mismatch.

        Returns an infallible compose if both sides are infallible; otherwise a
        fallible compose that short-circuits on ``Err``.
        """
        if isinstance(other, FallibleSchemaTransform):
            return _FallibleCompose(
                steps=(*_flatten_any(self), *_flatten_any(other)),
            )
        return _Compose(steps=(*_flatten_infallible(self), *_flatten_infallible(other)))


# ---------------------------------------------------------------------------
# Fallible transform
# ---------------------------------------------------------------------------


class FallibleSchemaTransform(ABC, Generic[_InS, _OutS]):
    """Schema-to-schema transform whose ``apply`` may fail with ``TransformError``."""

    transformation: ClassVar[str] = ""

    _input_schema: ClassVar[type | None] = None
    _output_schema: ClassVar[type | None] = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        _bind_schemas(cls, FallibleSchemaTransform)

    @abstractmethod
    def apply(self, value: _InS, /) -> Result[_OutS, TransformError]: ...

    def input_schema(self) -> type | None:
        return type(self)._input_schema

    def output_schema(self) -> type | None:
        return type(self)._output_schema

    def params(self) -> dict[str, Any]:
        return {}

    def then(
        self,
        other: SchemaTransform[_OutS, _NextS] | FallibleSchemaTransform[_OutS, _NextS],
    ) -> FallibleSchemaTransform[_InS, _NextS]:
        """Compose; result is always fallible because ``self`` is fallible."""
        return _FallibleCompose(
            steps=(*_flatten_any(self), *_flatten_any(other)),
        )


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def _check_adjacency(
    a: SchemaTransform[Any, Any] | FallibleSchemaTransform[Any, Any],
    b: SchemaTransform[Any, Any] | FallibleSchemaTransform[Any, Any],
) -> None:
    out_s = a.output_schema()
    in_s = b.input_schema()
    if out_s is None or in_s is None:
        warnings.warn(
            f"Cannot verify schema adjacency between {type(a).__name__} "
            f"(output={out_s}) and {type(b).__name__} (input={in_s}).",
            stacklevel=3,
        )
        return
    if out_s is not in_s:
        raise TypeError(
            f"Schema mismatch in compose: "
            f"{type(a).__name__}.output_schema={getattr(out_s, '__name__', out_s)} "
            f"!= {type(b).__name__}.input_schema={getattr(in_s, '__name__', in_s)}"
        )


def _flatten_infallible(
    t: SchemaTransform[Any, Any],
) -> tuple[SchemaTransform[Any, Any], ...]:
    if isinstance(t, _Compose):
        return tuple(t.steps)
    return (t,)


def _flatten_any(
    t: SchemaTransform[Any, Any] | FallibleSchemaTransform[Any, Any],
) -> tuple[SchemaTransform[Any, Any] | FallibleSchemaTransform[Any, Any], ...]:
    if isinstance(t, _Compose):
        return tuple(t.steps)
    if isinstance(t, _FallibleCompose):
        return tuple(t.steps)
    return (t,)


def _compose_params(
    steps: tuple[SchemaTransform[Any, Any] | FallibleSchemaTransform[Any, Any], ...],
) -> dict[str, Any]:
    return {
        "steps": [
            {"transformation": s.transformation, "params": s.params()}
            for s in steps
        ]
    }


class _Compose(SchemaTransform[Any, Any]):
    """All-infallible composite."""

    transformation: ClassVar[str] = "compose"

    def __init__(self, steps: tuple[SchemaTransform[Any, Any], ...]) -> None:
        if len(steps) < 2:
            raise ValueError("Compose requires at least 2 steps")
        for a, b in zip(steps[:-1], steps[1:], strict=True):
            _check_adjacency(a, b)
        self.steps = steps

    def apply(self, value: Any, /) -> Any:
        for step in self.steps:
            value = step.apply(value)
        return value

    def input_schema(self) -> type | None:
        return self.steps[0].input_schema()

    def output_schema(self) -> type | None:
        return self.steps[-1].output_schema()

    def params(self) -> dict[str, Any]:
        return _compose_params(self.steps)


class _FallibleCompose(FallibleSchemaTransform[Any, Any]):
    """Mixed or all-fallible composite; short-circuits on ``Err``."""

    transformation: ClassVar[str] = "compose"

    def __init__(
        self,
        steps: tuple[
            SchemaTransform[Any, Any] | FallibleSchemaTransform[Any, Any], ...
        ],
    ) -> None:
        if len(steps) < 2:
            raise ValueError("Compose requires at least 2 steps")
        for a, b in zip(steps[:-1], steps[1:], strict=True):
            _check_adjacency(a, b)
        self.steps = steps

    def apply(self, value: Any, /) -> Result[Any, TransformError]:
        current: Any = value
        for step in self.steps:
            if isinstance(step, FallibleSchemaTransform):
                result = step.apply(current)
                if isinstance(result, Err):
                    return result
                current = result.value
            else:
                current = step.apply(current)
        return Ok(current)

    def input_schema(self) -> type | None:
        return self.steps[0].input_schema()

    def output_schema(self) -> type | None:
        return self.steps[-1].output_schema()

    def params(self) -> dict[str, Any]:
        return _compose_params(self.steps)
