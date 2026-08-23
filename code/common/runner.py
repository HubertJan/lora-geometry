"""Run a batch of tasks either sequentially in-process or on SLURM via submitit.

Replaces ``huberts_toolbox.submitit.*``. The design goal (from the migration
brief): large training jobs assume submitit, but the caller supplies their own
executor object — or passes ``None`` to run the exact same callable locally and
sequentially.

Usage
-----
    from common.runner import submit_or_run

    def train_one(task):          # a plain picklable callable + arg
        ...
        return result

    tasks = [t0, t1, t2, ...]

    # local, sequential (no cluster):
    results = submit_or_run(train_one, tasks, executor=None)

    # SLURM: build and configure your own AutoExecutor, then hand it over:
    import submitit
    ex = submitit.AutoExecutor(folder="_workdir/submitit")
    ex.update_parameters(timeout_min=180, slurm_partition="gpu", gpus_per_node=1)
    results = submit_or_run(train_one, tasks, executor=ex)

``submit_or_run`` always returns the list of results in task order. On SLURM it
submits an array with ``map_array`` and blocks on ``job.result()``; failures are
surfaced (or collected, with ``return_exceptions=True``).
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def submit_or_run(
    fn: Callable[[T], R],
    tasks: Sequence[T],
    *,
    executor: Any | None = None,
    return_exceptions: bool = False,
) -> list[R]:
    """Map ``fn`` over ``tasks``. ``executor=None`` runs locally & sequentially.

    Parameters
    ----------
    fn:
        A picklable callable taking a single task argument. Wrap multi-arg work
        in a small dataclass/tuple task object.
    tasks:
        The work items. Each is passed to ``fn`` as-is.
    executor:
        A configured ``submitit.AutoExecutor`` (or any object exposing
        ``map_array(fn, tasks) -> list[job]`` with ``job.result()``). ``None``
        for local sequential execution.
    return_exceptions:
        If True, a failing task yields the ``Exception`` in the results list
        instead of raising.
    """
    tasks = list(tasks)
    if executor is None:
        return _run_local(fn, tasks, return_exceptions)
    return _run_submitit(fn, tasks, executor, return_exceptions)


def _run_local(fn, tasks, return_exceptions) -> list:
    results: list = []
    n = len(tasks)
    for i, task in enumerate(tasks):
        print(f"[runner:local] {i + 1}/{n}", flush=True)
        try:
            results.append(fn(task))
        except Exception as exc:  # noqa: BLE001
            if not return_exceptions:
                raise
            results.append(exc)
    return results


def _run_submitit(fn, tasks, executor, return_exceptions) -> list:
    jobs = executor.map_array(fn, tasks)
    print(f"[runner:submitit] submitted {len(jobs)} jobs", flush=True)
    results: list = []
    for job in jobs:
        try:
            results.append(job.result())
        except Exception as exc:  # noqa: BLE001
            if not return_exceptions:
                raise
            results.append(exc)
    return results


def chunked(items: Iterable[T], size: int) -> list[list[T]]:
    """Split ``items`` into lists of at most ``size`` (for packing N adapters/job)."""
    out: list[list[T]] = []
    cur: list[T] = []
    for it in items:
        cur.append(it)
        if len(cur) == size:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out
