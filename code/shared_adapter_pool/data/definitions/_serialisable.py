"""No-op ``serialisable`` decorator shim (migrated from SRC: replaces the
cache-serialisation decorator the dataset modules imported from huberts-toolbox).

The upstream decorator registered enums/classes with the W&B cache
serialisation registry so they could be fingerprinted by name. The standalone
pool has no cache/artifact layer, so the decorator degrades to identity: it
accepts the same ``serialisable("Name")`` call form and returns the class
unchanged.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

_T = TypeVar("_T")


def serialisable(name: str | None = None) -> Callable[[_T], _T]:
    """Return an identity class decorator (the registration is dropped)."""

    def _decorate(cls: _T) -> _T:
        return cls

    return _decorate
