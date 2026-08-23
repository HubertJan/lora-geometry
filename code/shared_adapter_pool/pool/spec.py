"""AdapterCfg: everything that distinguishes ONE adapter of the pool.

(migrated from src/discoveries/sst2_perf_prediction/flows/train_eval_adapter.py
::AdapterCfg)

Plain dataclass so it cloudpickles cleanly across a submitit boundary.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AdapterCfg:
    """Everything that distinguishes ONE adapter of the pool."""

    adapter_idx: int
    pool: str  # "train" | "test"
    train_slug: str
    eval_slug: str
    label_scheme: str  # e.g. "TRUE_FALSE_V1"

    # data volume: a stratified shard slice of the shared prepped train split
    shards_total: int
    shard_indices: list[int]

    # HPs
    learning_rate: float
    weight_decay: float
    epochs: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: list[str]
    train_seed: int

    # degrader lever: fraction of TRAIN labels randomly flipped (0.0 = clean)
    label_noise: float = 0.0

    # optional per-adapter grad-accum override. None = use the worker-level kwarg.
    gradient_accumulation_steps: int | None = None

    # optional campaign namespace prefixed onto ``label`` (and hence the adapter
    # ``__key__``). Empty = legacy behaviour.
    campaign: str = ""

    @property
    def label(self) -> str:
        prefix = f"{self.campaign}-" if self.campaign else ""
        return f"{prefix}{self.pool}-{self.train_slug}-i{self.adapter_idx:03d}"
