"""Build a synthetic adapter pool on disk for CPU sanity checks.

Writes real-shaped Llama-3.2-1B LoRA adapters (random weights) + a
``metadata.parquet`` matching ``meta_model/CONTRACT.md``, so the meta-model
loader, models, trainer and baselines can be exercised without a GPU, the gated
base model, or any real training.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Llama-3.2-1B LoRA target modules → (in_features, out_features). lora_A is
# (r, in), lora_B is (out, r). Keys follow PEFT's naming so the loader's
# ``.layers.<L>.<submodule>.lora_{A,B}.weight`` regex matches.
MODULES: dict[str, tuple[int, int]] = {
    "self_attn.q_proj": (2048, 2048),
    "self_attn.k_proj": (2048, 512),
    "self_attn.v_proj": (2048, 512),
    "self_attn.o_proj": (2048, 2048),
    "mlp.gate_proj": (2048, 8192),
    "mlp.up_proj": (2048, 8192),
    "mlp.down_proj": (8192, 2048),
}
N_LAYERS = 16
RANK = 16
TARGET_COLS = [
    "benchmark.sst2-test.likelihood.accuracy",
    "benchmark.sst2-test.likelihood.f1_macro",
    "benchmark.sst2-test.likelihood.auroc",
    "benchmark.sst2-test.likelihood.brier",
    "benchmark.sst2-test.likelihood.mean_confidence",
    "benchmark.sst2-test.likelihood.nll",
]


def _write_adapter(adir: Path, rng: np.random.Generator) -> None:
    import torch
    from safetensors.torch import save_file

    tensors: dict[str, "torch.Tensor"] = {}
    for layer in range(N_LAYERS):
        for sub, (fin, fout) in MODULES.items():
            base = f"base_model.model.model.layers.{layer}.{sub}"
            a = rng.standard_normal((RANK, fin)).astype("float32") * 0.02
            b = rng.standard_normal((fout, RANK)).astype("float32") * 0.02
            tensors[f"{base}.lora_A.weight"] = torch.from_numpy(a)
            tensors[f"{base}.lora_B.weight"] = torch.from_numpy(b)
    adir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(adir / "adapter_model.safetensors"))
    (adir / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "meta-llama/Llama-3.2-1B",
                "peft_type": "LORA",
                "r": RANK,
                "lora_alpha": 32,
                "target_modules": [m.split(".")[-1] for m in MODULES],
            }
        )
    )


def make_fake_pool(
    pool_dir: Path, *, n: int = 12, datasets: tuple[str, ...] = ("sst2",), seed: int = 0
) -> Path:
    """Create a pool of ``n`` random adapters (round-robin over ``datasets``)."""
    import polars as pl

    pool_dir = Path(pool_dir)
    (pool_dir / "adapters").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        key = f"adapter_{i:03d}"
        _write_adapter(pool_dir / "adapters" / key, rng)
        # A learnable-ish signal: accuracy correlates with the row index so the
        # tiny model / baselines have something non-degenerate to fit.
        acc = float(np.clip(0.55 + 0.4 * (i / max(1, n - 1)) + rng.normal(0, 0.02), 0, 1))
        row = {
            "__key__": key,
            "train_dataset": datasets[i % len(datasets)],
            "split": "test" if i % 4 == 0 else "train",
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "lora_rank": RANK,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "epochs": 1 + (i % 3),
            "shards_total": 2 ** (i % 5),
            "label_noise": 0.0,
            "train_seed": i,
            TARGET_COLS[0]: acc,
            TARGET_COLS[1]: acc - 0.02,
            TARGET_COLS[2]: float(np.clip(acc + 0.05, 0, 1)),
            TARGET_COLS[3]: float(np.clip(0.5 * (1 - acc), 0, 1)),
            TARGET_COLS[4]: float(np.clip(acc + 0.1, 0, 1)),
            TARGET_COLS[5]: float(max(0.01, 1.5 * (1 - acc))),
        }
        rows.append(row)
    pl.DataFrame(rows).write_parquet(pool_dir / "metadata.parquet")
    return pool_dir
