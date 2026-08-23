"""Build the SST2 in-task adapter pool (train_dataset == eval == sst2).

(migrated from src/discoveries/sst2_perf_prediction/jobs/04_production-sst2.py)

Expands ``DEFAULT_CELLS`` into an HP-jittered spread of rank-16 TRUE_FALSE_V1
adapters, trains + SST2-evaluates each, and writes the pool to
``POOL_DIR/<POOL_NAME>/`` (``metadata.parquet`` + ``adapters/<__key__>/``).

Run locally (sequential, in-process)::

    POOL_DIR=./_workdir/pools uv run python -m shared_adapter_pool.jobs.build_sst2_pool

Run on SLURM: build a ``submitit.AutoExecutor`` (see the commented block in
``main``) and hand it to ``submit_or_run`` as ``EXECUTOR``. Adapters are packed
``ADAPTERS_PER_JOB`` at a time via ``common.runner.chunked`` so a ~200-adapter
pool is ~20 jobs, not ~200.
"""

from __future__ import annotations

from common import env
from common.runner import chunked, submit_or_run
from shared_adapter_pool.pool.production_grid import DEFAULT_CELLS, build_production_cfgs
from shared_adapter_pool.pool.store import write_pool
from shared_adapter_pool.pool.worker import train_and_eval_chunk

POOL_NAME = "sst2_in_task"
REPLICATES = 27             # 21 DEFAULT_CELLS x 27 -> 567 train+test adapters (paper pool)
ADAPTERS_PER_JOB = 10       # pack ~10 adapters per GPU job

#: Set to a configured ``submitit.AutoExecutor`` to run on SLURM; ``None`` runs
#: locally and sequentially in-process.
EXECUTOR = None


def _build_executor():  # pragma: no cover - example wiring for SLURM
    """Example: a SLURM AutoExecutor. Call this and assign to EXECUTOR to submit.

    import submitit

    ex = submitit.AutoExecutor(folder=str(env.RESULTS_DIR / "submitit" / POOL_NAME))
    ex.update_parameters(
        timeout_min=180,
        slurm_partition="gpu",
        gpus_per_node=1,
        tasks_per_node=1,
        cpus_per_task=8,
    )
    return ex
    """
    raise NotImplementedError("uncomment the body to build a SLURM executor")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Build the SST2 in-task adapter pool.")
    ap.add_argument("--pool", default=POOL_NAME, help="pool name under $POOL_DIR")
    ap.add_argument("--limit", type=int, default=None,
                    help="build only the first N adapters (quick local smoke run)")
    args = ap.parse_args()

    pool_dir = env.pool_path(args.pool)

    cfgs = build_production_cfgs(DEFAULT_CELLS, replicates=REPLICATES)
    if args.limit is not None:
        cfgs = cfgs[: args.limit]
    print(f"[build_sst2_pool] {len(cfgs)} adapter configs -> {pool_dir}")

    chunks = chunked(cfgs, ADAPTERS_PER_JOB)
    chunk_results = submit_or_run(
        lambda chunk: train_and_eval_chunk(chunk, pool_dir),
        chunks,
        executor=EXECUTOR,
    )

    rows = [row for chunk in chunk_results for row in chunk]
    ok_rows = [r for r in rows if "error" not in r]
    failed = [r for r in rows if "error" in r]
    if failed:
        print(f"[build_sst2_pool] {len(failed)} adapters FAILED:")
        for r in failed:
            print(f"    {r['__key__']}: {r['error']}")

    out = write_pool(pool_dir, ok_rows)
    print(f"[build_sst2_pool] wrote {len(ok_rows)} rows -> {out}")


if __name__ == "__main__":
    main()
