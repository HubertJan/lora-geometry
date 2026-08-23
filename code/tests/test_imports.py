"""Import-smoke every module in the project so no migrated file has a broken
import path. Torch-dependent modules are included (torch is a project dep)."""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PACKAGES = ["common", "shared_adapter_pool", "meta_model"]


def _iter_modules():
    for pkg in PACKAGES:
        mod = importlib.import_module(pkg)
        for info in pkgutil.walk_packages(mod.__path__, prefix=f"{pkg}."):
            yield info.name


@pytest.mark.parametrize("modname", list(_iter_modules()))
def test_import_module(modname):
    importlib.import_module(modname)


def _experiment_modules():
    for p in sorted((ROOT / "experiments").glob("*/*.py")):
        if p.name == "__init__.py":
            continue
        yield f"experiments.{p.parent.name}.{p.stem}"


@pytest.mark.parametrize("modname", list(_experiment_modules()))
def test_import_experiment_module(modname):
    # experiments/ is a namespace package (on sys.path via the editable install).
    importlib.import_module(modname)
