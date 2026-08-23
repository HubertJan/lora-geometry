"""Write / read the on-disk adapter-pool ``metadata.parquet``.

(migrated concept from the W&B-lineage pool bundler; rewritten to the on-disk
contract in ``code/meta_model/CONTRACT.md``.)

Layout produced (adapters are written separately, by ``pool/worker.py``):

    <pool_dir>/
      metadata.parquet
      adapters/<__key__>/adapter_model.safetensors
      adapters/<__key__>/adapter_config.json

``metadata.parquet`` carries exactly the columns the ``meta_model`` loader reads:
the key/train_dataset/split identity columns, the pass-through hyperparameters,
and the six SST2 benchmark target columns
``benchmark.sst2-test.likelihood.{accuracy,f1_macro,auroc,brier,mean_confidence,nll}``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

#: The six SST2 benchmark target columns, in ``meta_model`` head order.
BENCHMARK_METRICS = ("accuracy", "f1_macro", "auroc", "brier", "mean_confidence", "nll")
BENCHMARK_COLUMNS = tuple(
    f"benchmark.sst2-test.likelihood.{m}" for m in BENCHMARK_METRICS
)

#: Identity columns required by the contract.
ID_COLUMNS = ("__key__", "train_dataset", "split")

#: Pass-through hyperparameter columns (used by some baselines; the geometry
#: baseline reads ``shards_total``, ``epochs``, ``label_noise``).
HPARAM_COLUMNS = (
    "learning_rate",
    "weight_decay",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "epochs",
    "shards_total",
    "label_noise",
    "train_seed",
)

#: Full, ordered column set of ``metadata.parquet``.
COLUMNS = (*ID_COLUMNS, *HPARAM_COLUMNS, *BENCHMARK_COLUMNS)

METADATA_FILE = "metadata.parquet"
ADAPTERS_DIR = "adapters"


def write_pool(pool_dir: str | Path, rows: list[dict[str, Any]]) -> Path:
    """Write ``rows`` as ``<pool_dir>/metadata.parquet`` (one row per adapter).

    Each row dict is expected to carry the identity columns, the hyperparameters,
    and the six ``benchmark.sst2-test.likelihood.*`` values. Missing columns are
    filled with ``None`` so the parquet schema is stable across pools.
    """
    import polars as pl

    pool_dir = Path(pool_dir)
    pool_dir.mkdir(parents=True, exist_ok=True)

    normalized = [{col: row.get(col) for col in COLUMNS} for row in rows]
    df = pl.DataFrame(normalized, schema=list(COLUMNS))

    out = pool_dir / METADATA_FILE
    df.write_parquet(out)
    return out


def read_pool(pool_dir: str | Path):
    """Read ``<pool_dir>/metadata.parquet`` into a polars DataFrame."""
    import polars as pl

    return pl.read_parquet(Path(pool_dir) / METADATA_FILE)


def adapter_dir(pool_dir: str | Path, key: str) -> Path:
    """The directory holding one adapter's safetensors + config."""
    return Path(pool_dir) / ADAPTERS_DIR / key
