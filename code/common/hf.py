"""HuggingFace base-model / tokenizer / LoRA-adapter load & save helpers.

Replaces the weight-bearing parts of ``huberts_toolbox.wandb_logging.hf_*`` and
``llm_pipeline.model_artifact`` with plain ``transformers`` / ``peft`` calls and a
local directory layout. No artifact store, no lineage — an adapter is just a
directory containing ``adapter_model.safetensors`` + ``adapter_config.json``.

Two invariants carried over from the original pipeline because they are
load-bearing for reproducing results:

* **Load dtype.** Llama-3.2-1B/3B must load in ``float16``; bf16 (or a mismatched
  dtype) collapses freshly-applied LoRA adapters to garbage. Larger Llama models
  use ``bfloat16``. See ``recommended_dtype`` below.
* **Pad token.** Training and likelihood eval pad with Llama-3's reserved
  ``<|finetune_right_pad_id|>`` so padding never collides with a real token.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from common import env

PAD_TOKEN = "<|finetune_right_pad_id|>"

# Transcribed from the original ``ModelSpec.recommended_dtype``: fp16 for the two
# smaller Llama-3.2 checkpoints, bf16 for the larger ones, fp16 for anything
# unlisted (the vetted-safe default).
_FP16_MODELS = {
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.2-3B-Instruct",
}
_BF16_MODELS = {
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3-8B",
}


def recommended_dtype(model_name: str, override: str | None = None) -> torch.dtype:
    """Resolve the load dtype for a base model. An explicit ``override`` wins."""
    if override is not None:
        dt = getattr(torch, override, None)
        if not isinstance(dt, torch.dtype):
            raise ValueError(f"torch_dtype={override!r} is not a torch dtype name")
        return dt
    if model_name in _FP16_MODELS:
        return torch.float16
    if model_name in _BF16_MODELS:
        return torch.bfloat16
    return torch.float16


def load_tokenizer(
    model_name: str | None = None, *, pad_token: str | None = PAD_TOKEN
) -> PreTrainedTokenizerBase:
    """Load the base tokenizer and set the reserved pad token used everywhere."""
    model_name = model_name or env.BASE_MODEL
    tok = AutoTokenizer.from_pretrained(model_name, token=env.HF_TOKEN)
    if pad_token is not None:
        # <|finetune_right_pad_id|> is already in Llama-3's vocab; this just
        # points pad_token at it without resizing embeddings.
        tok.pad_token = pad_token
        tok.padding_side = "right"
    return tok


def load_base_model(
    model_name: str | None = None,
    *,
    torch_dtype: str | None = None,
    device_map: str | None = None,
) -> Any:
    """Load the frozen base causal-LM at its recommended dtype."""
    model_name = model_name or env.BASE_MODEL
    dtype = recommended_dtype(model_name, torch_dtype)
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        token=env.HF_TOKEN,
    )


def load_adapter_model(
    adapter_dir: str | Path,
    *,
    base_model: str | None = None,
    torch_dtype: str | None = None,
    device_map: str | None = None,
) -> Any:
    """Load base + a saved LoRA adapter as a ``peft.PeftModel``.

    The base model name is read from the adapter's ``adapter_config.json`` unless
    ``base_model`` overrides it.
    """
    from peft import PeftModel

    adapter_dir = Path(adapter_dir)
    base_name = base_model or _base_from_adapter_config(adapter_dir) or env.BASE_MODEL
    base = load_base_model(base_name, torch_dtype=torch_dtype, device_map=device_map)
    return PeftModel.from_pretrained(base, str(adapter_dir))


def _base_from_adapter_config(adapter_dir: Path) -> str | None:
    import json

    cfg = adapter_dir / "adapter_config.json"
    if not cfg.exists():
        return None
    return json.loads(cfg.read_text()).get("base_model_name_or_path")


def save_adapter(model: Any, adapter_dir: str | Path) -> Path:
    """Save the adapter-only weights (safetensors) + config to ``adapter_dir``."""
    adapter_dir = Path(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir), safe_serialization=True)
    return adapter_dir
