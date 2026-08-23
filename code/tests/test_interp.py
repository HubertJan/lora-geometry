"""CPU smoke checks for the interpretability experiments: keep-only weight
surgery, and the in_task→LRP checkpoint seam (train → save → LRP loads it)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tests._fake_pool import _write_adapter, make_fake_pool  # noqa: E402


def test_keep_only_ablation(tmp_path):
    from safetensors.torch import load_file

    from experiments.causal_ablation.ablate import ablate_adapter

    src = tmp_path / "src"
    _write_adapter(src, np.random.default_rng(0))

    dst = tmp_path / "dst"
    # keep only mlp.down_proj on layers 0..10, revert everything else to base (zero).
    ablate_adapter(src, dst, modules=["mlp.down_proj"], layers=range(0, 11), keep=True)

    w = load_file(str(dst / "adapter_model.safetensors"))
    # a q_proj tensor (not in the kept set) must be zeroed ...
    q = w["base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight"]
    assert torch.count_nonzero(q) == 0
    # ... while a kept down_proj tensor on a kept layer stays non-zero.
    d = w["base_model.model.model.layers.5.mlp.down_proj.lora_A.weight"]
    assert torch.count_nonzero(d) > 0


def test_in_task_to_lrp_checkpoint_seam(tmp_path):
    """A checkpoint saved by train_meta_model must load into the LRP wrapper."""
    import polars as pl

    from experiments.lrp.lrp_run import (
        LRPFlexibleMetaClassifier,
        group_from_path,
        make_composite_l2,
        run_lrp_single,
    )
    from meta_model.dataset import load_adapter_pool_metadata
    from meta_model.regressor_config import build_regressor_config, sst2_head_specs
    from meta_model.train import train_meta_model

    pool = make_fake_pool(tmp_path / "pool", n=8)
    df = load_adapter_pool_metadata(pool)
    ckpt = tmp_path / "w8_l2.pth"
    train_meta_model(
        build_regressor_config("w8_l2", seed=0),
        df.filter(pl.col("split") == "train"),
        {"test": df.filter(pl.col("split") == "test")},
        sst2_head_specs(),
        pool_dir=pool, epochs=1, batch_size=2, device="cpu", out_path=ckpt,
    )

    # The LRP wrapper loads the {state_dict, hyperparameters} checkpoint strict=True.
    model = LRPFlexibleMetaClassifier.load(str(ckpt), device="cpu").eval()
    adapter = Path(pool) / "adapters" / df["__key__"][0] / "adapter_model.safetensors"
    grouped = group_from_path(adapter, 16)
    res = run_lrp_single(model, grouped, make_composite_l2(), "cpu", "acc", 0)
    assert "attr_dict" in res and len(res["attr_dict"]) > 0
