"""Migrated from SRC/src/glad/lora/model_sizes.py.

Model dimension constants for supported LLM architectures.

Migrated from paretune (many_stats.py, llama3_1b_stats.py, custom_model_configs.py).
"""

from __future__ import annotations

from meta_model.lora.types import LLMModel, TargetModuleType

# ── Per-model size dicts ─────────────────────────────────────────────────────

QWEN15_7B: dict[TargetModuleType, tuple[int, int]] = {
    TargetModuleType.K_ATTENTION: (4096, 4096),
    TargetModuleType.Q_ATTENTION: (4096, 4096),
    TargetModuleType.O_ATTENTION: (4096, 4096),
    TargetModuleType.V_ATTENTION: (4096, 4096),
    TargetModuleType.UP_MLP: (4096, 22016),
    TargetModuleType.GATE_MLP: (4096, 22016),
    TargetModuleType.DOWN_MLP: (22016, 4096),
}

LLAMA2_7B: dict[TargetModuleType, tuple[int, int]] = {
    TargetModuleType.K_ATTENTION: (4096, 4096),
    TargetModuleType.Q_ATTENTION: (4096, 4096),
    TargetModuleType.O_ATTENTION: (4096, 4096),
    TargetModuleType.V_ATTENTION: (4096, 4096),
    TargetModuleType.UP_MLP: (4096, 11008),
    TargetModuleType.GATE_MLP: (4096, 11008),
    TargetModuleType.DOWN_MLP: (11008, 4096),
}

# NOTE: shapes below are LoRA target-module ``(in_features, out_features)``.  For
# attention they derive from the HF config as q/o=(hidden, heads*head_dim) and
# k/v=(hidden, kv_heads*head_dim); every model added here uses grouped-query
# attention (GQA), so k/v output < q output — do NOT collapse them.  ``head_dim``
# is read from the config (128 for Llama-3.1/3.2-3B/Qwen3), never hidden//heads.

# Llama-3.1-8B  (hidden 4096, heads 32, kv 8, head_dim 128, inter 14336, 32 layers)
# Verified against meta-llama/Llama-3.1-8B config.json.  Replaces the historical
# (wrong) 5120/no-GQA ``LLAMA3_8B`` entry, which matched no real Llama-3 model.
LLAMA3_8B: dict[TargetModuleType, tuple[int, int]] = {
    TargetModuleType.Q_ATTENTION: (4096, 4096),
    TargetModuleType.K_ATTENTION: (4096, 1024),
    TargetModuleType.V_ATTENTION: (4096, 1024),
    TargetModuleType.O_ATTENTION: (4096, 4096),
    TargetModuleType.GATE_MLP: (4096, 14336),
    TargetModuleType.UP_MLP: (4096, 14336),
    TargetModuleType.DOWN_MLP: (14336, 4096),
}

# Llama-3.1-8B == fixed LLAMA3_8B (kept under an explicit, unambiguous name).
LLAMA31_8B = LLAMA3_8B

# Mistral-7B-v0.1  (hidden 4096, heads 32, kv 8, head_dim 128, inter 14336, 32 layers)
# Verified against mistralai/Mistral-7B-v0.1 config.json — identical LoRA shapes
# to Llama-3.1-8B.
MISTRAL_7B: dict[TargetModuleType, tuple[int, int]] = {
    TargetModuleType.Q_ATTENTION: (4096, 4096),
    TargetModuleType.K_ATTENTION: (4096, 1024),
    TargetModuleType.V_ATTENTION: (4096, 1024),
    TargetModuleType.O_ATTENTION: (4096, 4096),
    TargetModuleType.GATE_MLP: (4096, 14336),
    TargetModuleType.UP_MLP: (4096, 14336),
    TargetModuleType.DOWN_MLP: (14336, 4096),
}

# Llama-3.2-3B  (hidden 3072, heads 24, kv 8, head_dim 128, inter 8192, 28 layers)
# Verified against meta-llama/Llama-3.2-3B config.json.
LLAMA32_3B: dict[TargetModuleType, tuple[int, int]] = {
    TargetModuleType.Q_ATTENTION: (3072, 3072),
    TargetModuleType.K_ATTENTION: (3072, 1024),
    TargetModuleType.V_ATTENTION: (3072, 1024),
    TargetModuleType.O_ATTENTION: (3072, 3072),
    TargetModuleType.GATE_MLP: (3072, 8192),
    TargetModuleType.UP_MLP: (3072, 8192),
    TargetModuleType.DOWN_MLP: (8192, 3072),
}

# Qwen2.5-7B  (hidden 3584, heads 28, kv 4, head_dim 128, inter 18944, 28 layers)
# Verified against Qwen/Qwen2.5-7B config.json.  Attention has q/k/v bias, but
# that does not change the LoRA matrix shapes.
QWEN25_7B: dict[TargetModuleType, tuple[int, int]] = {
    TargetModuleType.Q_ATTENTION: (3584, 3584),
    TargetModuleType.K_ATTENTION: (3584, 512),
    TargetModuleType.V_ATTENTION: (3584, 512),
    TargetModuleType.O_ATTENTION: (3584, 3584),
    TargetModuleType.GATE_MLP: (3584, 18944),
    TargetModuleType.UP_MLP: (3584, 18944),
    TargetModuleType.DOWN_MLP: (18944, 3584),
}

# Qwen3-14B  (hidden 5120, heads 40, kv 8, head_dim 128 explicit, inter 17408, 40 layers)
# Verified against Qwen/Qwen3-14B config.json.
QWEN3_14B: dict[TargetModuleType, tuple[int, int]] = {
    TargetModuleType.Q_ATTENTION: (5120, 5120),
    TargetModuleType.K_ATTENTION: (5120, 1024),
    TargetModuleType.V_ATTENTION: (5120, 1024),
    TargetModuleType.O_ATTENTION: (5120, 5120),
    TargetModuleType.GATE_MLP: (5120, 17408),
    TargetModuleType.UP_MLP: (5120, 17408),
    TargetModuleType.DOWN_MLP: (17408, 5120),
}

LLAMA3_1B: dict[TargetModuleType, tuple[int, int]] = {
    TargetModuleType.Q_ATTENTION: (2048, 2048),
    TargetModuleType.K_ATTENTION: (2048, 512),
    TargetModuleType.V_ATTENTION: (2048, 512),
    TargetModuleType.O_ATTENTION: (2048, 2048),
    TargetModuleType.GATE_MLP: (2048, 8192),
    TargetModuleType.UP_MLP: (2048, 8192),
    TargetModuleType.DOWN_MLP: (8192, 2048),
}

# ── Aggregate mapping ────────────────────────────────────────────────────────

TARGET_MODULE_SIZES_BY_LLM_MODEL: dict[LLMModel, dict[TargetModuleType, tuple[int, int]]] = {
    LLMModel.QWEN_15_7B: QWEN15_7B,
    LLMModel.LLAMA_2_7B: LLAMA2_7B,
    LLMModel.LLAMA_3_8B: LLAMA3_8B,  # deprecated alias → corrected Llama-3.1-8B dims
    LLMModel.LLAMA_31_8B: LLAMA31_8B,
    LLMModel.LLAMA_3_1B: LLAMA3_1B,
    LLMModel.LLAMA_32_3B: LLAMA32_3B,
    LLMModel.MISTRAL_7B_V01: MISTRAL_7B,
    LLMModel.QWEN_25_7B: QWEN25_7B,
    LLMModel.QWEN_3_14B: QWEN3_14B,
}

# ── Llama 3 1B individual constants ──────────────────────────────────────────

V_INPUT_SIZE = 2048
V_OUTPUT_SIZE = 512
Q_INPUT_SIZE = 2048
Q_OUTPUT_SIZE = 2048
K_INPUT_SIZE = 2048
K_OUTPUT_SIZE = 512
O_INPUT_SIZE = 2048
O_OUTPUT_SIZE = 2048
GATE_INPUT_SIZE = 2048
GATE_OUTPUT_SIZE = 8192
UP_INPUT_SIZE = 2048
UP_OUTPUT_SIZE = 8192
DOWN_INPUT_SIZE = 8192
DOWN_OUTPUT_SIZE = 2048
LAYER_COUNT = 16

TARGET_MODULE_TYPE_TO_SIZE: dict[TargetModuleType, tuple[int, int]] = {
    TargetModuleType.K_ATTENTION: (K_INPUT_SIZE, K_OUTPUT_SIZE),
    TargetModuleType.Q_ATTENTION: (Q_INPUT_SIZE, Q_OUTPUT_SIZE),
    TargetModuleType.O_ATTENTION: (O_INPUT_SIZE, O_OUTPUT_SIZE),
    TargetModuleType.V_ATTENTION: (V_INPUT_SIZE, V_OUTPUT_SIZE),
    TargetModuleType.GATE_MLP: (GATE_INPUT_SIZE, GATE_OUTPUT_SIZE),
    TargetModuleType.UP_MLP: (UP_INPUT_SIZE, UP_OUTPUT_SIZE),
    TargetModuleType.DOWN_MLP: (DOWN_INPUT_SIZE, DOWN_OUTPUT_SIZE),
}
