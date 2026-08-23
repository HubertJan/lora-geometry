"""Small adapters bridging FallibleSchemaTransform to legacy callable signatures.

(migrated from src/llm_pipeline/schema_transforms/adapters.py)
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from shared_adapter_pool.data.schema_transforms.base import (
    Err,
    FallibleSchemaTransform,
    Ok,
)


def as_parser_callable(
    transform: FallibleSchemaTransform[str, Any],
) -> Callable[[str], dict[str, Any]]:
    """Adapt a fallible str->dict parser to the legacy ``Callable[[str], dict]`` shape.

    Returns an empty dict on ``Err``.  Used at evaluation call sites that
    still take a plain callable parser (e.g. ``classify.py``); future passes
    can switch those call sites to consume ``Result`` directly.
    """

    def call(text: str) -> dict[str, Any]:
        result = transform.apply(text)
        if isinstance(result, Ok):
            return dict(result.value)
        assert isinstance(result, Err)
        return {}

    return call
