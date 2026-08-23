"""Sanity checks for common/ (no torch needed)."""

from __future__ import annotations

from common.runner import chunked, submit_or_run


def test_submit_or_run_local_sequential():
    results = submit_or_run(lambda x: x * x, [1, 2, 3, 4], executor=None)
    assert results == [1, 4, 9, 16]


def test_submit_or_run_return_exceptions():
    def boom(x):
        if x == 2:
            raise ValueError("nope")
        return x

    results = submit_or_run(boom, [1, 2, 3], executor=None, return_exceptions=True)
    assert results[0] == 1 and isinstance(results[1], ValueError) and results[2] == 3


def test_chunked():
    assert chunked(range(5), 2) == [[0, 1], [2, 3], [4]]


def test_env_paths_exist():
    from common import env

    assert env.POOL_DIR.exists() and env.RESULTS_DIR.exists()
    assert env.pool_path("x").name == "x"
