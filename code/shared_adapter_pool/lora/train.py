"""Train one LoRA adapter with TRL's SFTTrainer, then save it to disk.

(migrated from src/llm_pipeline/training/trainer.py::run_sft_training)

The artifact/cache/W&B-callback machinery is stripped. What remains is the
trainer core, byte-faithful to the original: the exact ``TrainingConfig`` ->
``SFTConfig`` argument mapping, the runtime-padding ``DataCollatorForSeq2Seq``
(pad_to_multiple_of=8), the trainable-parameter assertion, ``.train()``, and a
final adapter-only save via ``common.hf.save_adapter``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from shared_adapter_pool.lora.config import LoraConfig, TrainingConfig
from shared_adapter_pool.lora.peft import build_peft_model


def train_lora_adapter(
    base_model: Any,
    tokenizer: Any,
    train_ds: Any,
    training_config: TrainingConfig,
    lora_config: LoraConfig,
    out_dir: str | Path,
) -> Path:
    """Wrap ``base_model`` with a fresh LoRA adapter, SFT-train it, save to ``out_dir``.

    ``train_ds`` is a tokenized HuggingFace ``Dataset`` (``input_ids`` /
    ``attention_mask`` / ``labels``). Returns the directory the adapter weights
    were written to.
    """
    from transformers import DataCollatorForSeq2Seq
    from trl import SFTConfig, SFTTrainer

    from common.hf import save_adapter

    out_dir = Path(out_dir)

    # ---- fresh LoRA wrap (seeded init if requested) -----------------------
    model = build_peft_model(base_model, lora_config, init_seed=lora_config.init_seed)

    # ---- SFTConfig from the training hyperparameters ----------------------
    checkpoint_dir = out_dir / "_checkpoints"
    training_args = SFTConfig(
        output_dir=str(checkpoint_dir),
        **training_config.to_training_arguments_kwargs(),
    )

    # Data collator that pads variable-length sequences at runtime.
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )

    trainable_params, total_params = model.get_nb_trainable_parameters()
    assert trainable_params > 0, (
        "Model has no trainable parameters. Check the base model and that PEFT "
        "adapters are properly applied."
    )
    trainable_ratio = trainable_params / total_params if total_params > 0 else 0.0
    print(
        f"[INFO] Model parameters: trainable={trainable_params:,} / "
        f"total={total_params:,} ({trainable_ratio:.2%})"
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=None,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    # ---- save adapter-only weights (safetensors) --------------------------
    save_adapter(model, out_dir)
    return out_dir
