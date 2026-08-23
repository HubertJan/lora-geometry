"""Paths and config, read from the environment (optionally a ``.env`` file).

Replaces ``huberts_toolbox.env``. Everything defaults under ``./_workdir`` so the
project runs with zero configuration; point the vars at real storage for cluster
runs. Keep the env-var *names* stable — scripts and READMEs reference them.
"""

from __future__ import annotations

import os
from pathlib import Path

try:  # optional: load a .env if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is a declared dep, but stay robust
    pass


def _path(var: str, default: str) -> Path:
    p = Path(os.environ.get(var, default)).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


# Root under which trained adapter pools live (one subdir per pool).
POOL_DIR = _path("POOL_DIR", "./_workdir/pools")

# Root under which experiment outputs (tables, figures, npz) are written.
RESULTS_DIR = _path("RESULTS_DIR", "./_workdir/results")

# Scratch cache for prepared datasets and misc intermediates.
CACHE_DIR = _path("CACHE_DIR", "./_workdir/cache")

# The base model everything is fine-tuned from / evaluated against.
BASE_MODEL = os.environ.get("BASE_MODEL", "meta-llama/Llama-3.2-1B")

# Optional HuggingFace token for gated models / private datasets.
HF_TOKEN = os.environ.get("HF_TOKEN") or None


def pool_path(pool_name: str) -> Path:
    """Directory holding one adapter pool (metadata.parquet + adapters/)."""
    p = POOL_DIR / pool_name
    p.mkdir(parents=True, exist_ok=True)
    return p


def results_path(*parts: str) -> Path:
    """A subdir under RESULTS_DIR, created on demand."""
    p = RESULTS_DIR.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p
