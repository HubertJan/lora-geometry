"""Build the OOD adapter pool (train on 14 other slugs, all evaluated on SST2).

(migrated from src/discoveries/sst2_perf_prediction/jobs/10_ood-production.py +
22_ood-pool-fam.py)

Trains ``per_dataset`` rank-16 TRUE_FALSE_V1 adapters for each of the OOD training
datasets (``OOD_DATASETS`` + the two extension families ``FAM_DATASETS``), every
one evaluated on the fixed SST2 test target. Writes the pool to
``POOL_DIR/<POOL_NAME>/``; ``train_dataset`` varies across the 14 OOD slugs while
the SST2 benchmark columns are always the eval target.

Run locally::

    POOL_DIR=./_workdir/pools uv run python -m shared_adapter_pool.jobs.build_ood_pool

Run on SLURM: build a ``submitit.AutoExecutor`` (see ``build_sst2_pool.py`` for a
worked example) and assign it to ``EXECUTOR``. Adapters are packed
``ADAPTERS_PER_JOB`` per job via ``common.runner.chunked``.
"""

from __future__ import annotations

from common import env
from common.runner import chunked, submit_or_run
from shared_adapter_pool.pool.ood_grid import (
    FAM_DATASETS,
    OOD_DATASETS,
    build_ood_cfgs,
)
from shared_adapter_pool.pool.store import write_pool
from shared_adapter_pool.pool.worker import train_and_eval_chunk

POOL_NAME = "sst2_ood"
PER_DATASET = 160           # 16 OOD_CELLS x 10 replicates; x14 datasets -> ~2240 (paper pool)
ADAPTERS_PER_JOB = 10

#: The full OOD training-slug set: the 6 original groups + the 8 extension-family
#: datasets. All are evaluated on SST2.
ALL_OOD_DATASETS = [*OOD_DATASETS, *FAM_DATASETS]

#: Set to a configured ``submitit.AutoExecutor`` to run on SLURM; ``None`` = local.
EXECUTOR = None


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Build the OOD (15-dataset) adapter pool.")
    ap.add_argument("--pool", default=POOL_NAME, help="pool name under $POOL_DIR")
    ap.add_argument("--limit", type=int, default=None,
                    help="build only the first N adapters (quick local smoke run)")
    args = ap.parse_args()

    pool_dir = env.pool_path(args.pool)

    cfgs = build_ood_cfgs(datasets=ALL_OOD_DATASETS, per_dataset=PER_DATASET)
    if args.limit is not None:
        cfgs = cfgs[: args.limit]
    print(
        f"[build_ood_pool] {len(cfgs)} adapter configs over "
        f"{len(ALL_OOD_DATASETS)} datasets -> {pool_dir}"
    )

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
        print(f"[build_ood_pool] {len(failed)} adapters FAILED:")
        for r in failed:
            print(f"    {r['__key__']}: {r['error']}")

    out = write_pool(pool_dir, ok_rows)
    print(f"[build_ood_pool] wrote {len(ok_rows)} rows -> {out}")


if __name__ == "__main__":
    main()
