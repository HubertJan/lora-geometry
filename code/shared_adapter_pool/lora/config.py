"""TrainingConfig + LoraConfig dataclasses for SFT / PEFT.

(migrated from src/llm_pipeline/training/config.py)

The dataclasses and their ``to_*_kwargs()`` mappings to ``trl.SFTConfig`` /
``peft.LoraConfig`` are kept intact. The only seam removed is the original
``report_to`` default, which imported the ``llm_pipeline.artifacts`` backend
selector; the standalone trainer does no experiment-tracking callbacks, so
``report_to`` defaults to ``["none"]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainingConfig:
    """Hyperparameters for a SFTTrainer run."""

    learning_rate: float
    weight_decay: float
    epochs: int
    batch_size: int
    gradient_accumulation_steps: int
    seed: int
    group: str
    run_id: int

    do_eval: bool = True
    eval_steps: int = 100
    save_strategy: str = "no"
    save_steps: int = 500
    logging_steps: int = 10
    max_seq_length: int | None = None
    warmup_ratio: float = 0.0
    lr_scheduler_type: str = "linear"
    fp16: bool = False
    bf16: bool = False
    dataloader_num_workers: int = 0
    should_train_embeddings: bool = False
    report_to: list[str] = field(default_factory=lambda: ["none"])
    extra: dict[str, Any] = field(default_factory=dict)

    def to_config_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict for config logging."""
        return {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "seed": self.seed,
            "group": self.group,
            "run_id": self.run_id,
            "eval_steps": self.eval_steps,
            "save_strategy": self.save_strategy,
            "save_steps": self.save_steps,
            "max_seq_length": self.max_seq_length,
            "warmup_ratio": self.warmup_ratio,
            "lr_scheduler_type": self.lr_scheduler_type,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "should_train_embeddings": self.should_train_embeddings,
        }

    def to_training_arguments_kwargs(self) -> dict[str, Any]:
        """Return kwargs suitable for ``transformers.TrainingArguments`` / ``trl.SFTConfig``."""
        kwargs: dict[str, Any] = {
            "num_train_epochs": self.epochs,
            "per_device_train_batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "eval_strategy": "steps" if self.do_eval else "no",
            "eval_steps": self.eval_steps if self.do_eval else None,
            "save_strategy": self.save_strategy,
            # Only meaningful for the "steps" strategy; passing it under "no"
            # is harmless but misleading in the logged TrainingArguments.
            "save_steps": (
                self.save_steps if self.save_strategy == "steps" else None
            ),
            "logging_steps": self.logging_steps,
            "warmup_ratio": self.warmup_ratio,
            "lr_scheduler_type": self.lr_scheduler_type,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "seed": self.seed,
            "data_seed": self.seed,
            "dataloader_num_workers": self.dataloader_num_workers,
            "report_to": self.report_to,
            "dataset_kwargs": {
                "skip_prepare_dataset": True
            },  # we prepare datasets manually in train.py
        }
        kwargs.update(self.extra)
        return kwargs


@dataclass
class LoraConfig:
    """PEFT / LoRA settings bundled separately from TrainingConfig."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "k_proj",
            "q_proj",
            "gate_proj",
            "up_proj",
            "o_proj",
            "v_proj",
            "down_proj",
        ]
    )
    bias: str = "none"
    modules_to_save: list[str] | None = None
    init_seed: int | None = None

    def to_config_dict(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "lora_r": self.r,
            "lora_alpha": self.alpha,
            "lora_dropout": self.dropout,
            "target_modules": self.target_modules,
        }
        if self.modules_to_save is not None:
            config["modules_to_save"] = self.modules_to_save
        if self.init_seed is not None:
            config["init_lora_seed"] = self.init_seed
        return config
