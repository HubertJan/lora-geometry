"""Optional experiment logging. No-ops unless wandb is installed and enabled.

Replaces the ``run.log`` / ``set_summary`` calls scattered through the original
trainers. The scientific code never depends on logging succeeding.
"""

from __future__ import annotations

import os
from typing import Any


class _NullRun:
    """A logger that swallows everything (used when wandb is off/unavailable)."""

    def log(self, *args: Any, **kwargs: Any) -> None:  # noqa: D102
        pass

    def summary_update(self, *args: Any, **kwargs: Any) -> None:  # noqa: D102
        pass

    def finish(self) -> None:  # noqa: D102
        pass


def get_run(name: str | None = None, config: dict | None = None) -> Any:
    """Return a logging run. A no-op unless wandb is installed and enabled."""
    if os.environ.get("WANDB_MODE", "disabled") == "disabled":
        return _NullRun()
    try:
        import wandb
    except Exception:
        return _NullRun()
    return wandb.init(
        project=os.environ.get("WANDB_PROJECT", "lora-geometry"),
        entity=os.environ.get("WANDB_ENTITY") or None,
        name=name,
        config=config or {},
    )
