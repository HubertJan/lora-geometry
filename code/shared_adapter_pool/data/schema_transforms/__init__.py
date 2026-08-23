"""Schema-to-schema transform abstraction.

(migrated from src/llm_pipeline/schema_transforms/__init__.py)
"""

from shared_adapter_pool.data.schema_transforms.adapters import as_parser_callable
from shared_adapter_pool.data.schema_transforms.base import (
    Err,
    FallibleSchemaTransform,
    Ok,
    Result,
    SchemaTransform,
    TransformError,
)

__all__ = [
    "Err",
    "FallibleSchemaTransform",
    "Ok",
    "Result",
    "SchemaTransform",
    "TransformError",
    "as_parser_callable",
]
