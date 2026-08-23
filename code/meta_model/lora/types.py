"""Migrated from SRC/src/glad/lora/types.py.

LoRA-related type definitions and enumerations.

Migrated from paretune to remove the external dependency for core glad functionality.
"""

from __future__ import annotations

from enum import StrEnum


# ── LoRA matrix types ────────────────────────────────────────────────────────


class LoraType(StrEnum):
    A = "A"
    B = "B"


# ── Transformer module targeting ─────────────────────────────────────────────


class TargetModuleType(StrEnum):
    K_ATTENTION = "k_proj"
    Q_ATTENTION = "q_proj"
    O_ATTENTION = "o_proj"
    V_ATTENTION = "v_proj"
    GATE_MLP = "gate_proj"
    UP_MLP = "up_proj"
    DOWN_MLP = "down_proj"


ALL_ATTENTION_MODULE_TYPES = {
    TargetModuleType.K_ATTENTION,
    TargetModuleType.Q_ATTENTION,
    TargetModuleType.O_ATTENTION,
    TargetModuleType.V_ATTENTION,
}

ALL_MLP_MODULE_TYPES = {
    TargetModuleType.GATE_MLP,
    TargetModuleType.UP_MLP,
    TargetModuleType.DOWN_MLP,
}


# ── Activation functions ─────────────────────────────────────────────────────


class ActivationFunction(StrEnum):
    GL_ACTIVATION = "gl_activation"
    LEAKY_RELU = "leaky_relu"


# ── Llama 3 module paths ────────────────────────────────────────────────────


class Llama3ModuleType(StrEnum):
    QUERY_ATTENTION = "self_attn.q_proj"
    KEY_ATTENTION = "self_attn.k_proj"
    VALUE_ATTENTION = "self_attn.v_proj"
    OUTPUT_ATTENTION = "self_attn.o_proj"
    GATE_MLP = "mlp.gate_proj"
    UP_MLP = "mlp.up_proj"
    DOWN_MLP = "mlp.down_proj"


ALL_LLAMA3_ATTENTION_MODULE_TYPES = {
    Llama3ModuleType.KEY_ATTENTION,
    Llama3ModuleType.QUERY_ATTENTION,
    Llama3ModuleType.VALUE_ATTENTION,
    Llama3ModuleType.OUTPUT_ATTENTION,
}

ALL_LLAMA3_MLP_MODULE_TYPES = {
    Llama3ModuleType.GATE_MLP,
    Llama3ModuleType.UP_MLP,
    Llama3ModuleType.DOWN_MLP,
}


# ── LLM model identifiers ───────────────────────────────────────────────────


class LLMModel(StrEnum):
    QWEN_15_7B = "qwen_15_7b"
    LLAMA_2_7B = "llama_2_7b"
    # ``LLAMA_3_8B`` is a deprecated alias of ``LLAMA_31_8B``.  Its historical
    # size dict was wrong (q/k/v/o=5120, no GQA — matched no real Llama-3
    # model); it is kept only so old serialized configs still unpickle, and now
    # maps to the corrected Llama-3.1-8B shapes.  Prefer ``LLAMA_31_8B``.
    LLAMA_3_8B = "llama_3_8b"
    LLAMA_3_1B = "llama_3_1b"  # == Llama-3.2-1B (16 layers)
    # ── Models added for the multi-base-model effort ─────────────────────────
    LLAMA_32_3B = "llama_32_3b"
    LLAMA_31_8B = "llama_31_8b"
    MISTRAL_7B_V01 = "mistral_7b_v01"
    QWEN_25_7B = "qwen_25_7b"
    QWEN_3_14B = "qwen_3_14b"
