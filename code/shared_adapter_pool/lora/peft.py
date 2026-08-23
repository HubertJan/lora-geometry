"""Wrap a base causal-LM with a fresh LoRA adapter via PEFT.

(migrated from src/llm_pipeline/peft.py::create_peft_model_artifact)

The artifact-saving / lineage machinery is dropped; this is just the PEFT-wrap
core: build a ``peft.LoraConfig`` from our :class:`LoraConfig`, optionally seed
the ambient torch RNG so the random LoRA-A initialisation is reproducible (B is
always zero), and return the wrapped ``peft.PeftModel``. Target modules default
to all seven Llama projections (k/q/v/o/gate/up/down), rank 16.
"""

from __future__ import annotations

from typing import Any

from shared_adapter_pool.lora.config import LoraConfig


def build_peft_model(
    base_model: Any,
    lora_config: LoraConfig,
    init_seed: int | None = None,
) -> Any:
    """Return ``get_peft_model(base_model, peft.LoraConfig(...))``.

    ``init_seed`` (falling back to ``lora_config.init_seed``) is applied via
    ``torch.manual_seed`` immediately before ``get_peft_model`` so the random
    LoRA-A init is reproducible. ``None`` leaves the init unseeded.
    """
    import torch
    from peft import LoraConfig as PeftLoraConfig
    from peft import TaskType, get_peft_model

    modules_to_save = lora_config.modules_to_save
    peft_kwargs: dict[str, Any] = dict(
        r=lora_config.r,
        lora_alpha=lora_config.alpha,
        lora_dropout=lora_config.dropout,
        bias=lora_config.bias,
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_config.target_modules,
        modules_to_save=modules_to_save,
    )
    if modules_to_save and getattr(base_model.config, "tie_word_embeddings", False):
        # Llama 3.2 ties embed_tokens and lm_head. Keep them tied while saving
        # so the input embedding and output projection stay consistent.
        peft_kwargs["ensure_weight_tying"] = True
    peft_config = PeftLoraConfig(**peft_kwargs)

    seed = init_seed if init_seed is not None else lora_config.init_seed
    if seed is not None:
        torch.manual_seed(seed)

    return get_peft_model(base_model, peft_config)
