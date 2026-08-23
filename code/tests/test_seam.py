"""End-to-end seam check (CPU): synthetic pool → meta-model loader → model
forward → 1-epoch train → baseline feature extraction.

Validates that ``shared_adapter_pool``'s on-disk contract is exactly what
``meta_model`` consumes, using random-weight full-shape Llama-3.2-1B adapters —
no GPU, no gated base model, no real training.
"""

from __future__ import annotations

import polars as pl
import pytest

torch = pytest.importorskip("torch")

from tests._fake_pool import make_fake_pool  # noqa: E402


def test_loader_and_model_forward(tmp_path):
    from meta_model.dataset import build_eval_dataloader, load_adapter_pool_metadata
    from meta_model.regressor_config import build_regressor_config, sst2_head_specs

    pool = make_fake_pool(tmp_path / "pool", n=6)
    df = load_adapter_pool_metadata(pool)
    assert "safetensor_path" in df.columns and len(df) == 6

    cfg = build_regressor_config("w8_l2", seed=0)
    model = cfg.create_model("cpu")
    model.eval()

    specs = sst2_head_specs()
    loader = build_eval_dataloader(df, specs, batch_size=2, device="cpu")
    inputs, _targets = next(iter(loader))
    out = model(inputs)
    assert {s.name for s in specs} <= set(out.keys())


def test_train_one_epoch(tmp_path):
    from meta_model.dataset import load_adapter_pool_metadata
    from meta_model.regressor_config import build_regressor_config, sst2_head_specs
    from meta_model.train import train_meta_model

    pool = make_fake_pool(tmp_path / "pool", n=12)
    df = load_adapter_pool_metadata(pool)
    train_df = df.filter(pl.col("split") == "train")
    test_df = df.filter(pl.col("split") == "test")

    cfg = build_regressor_config("w8_l2", seed=0)
    metrics = train_meta_model(
        cfg, train_df, {"test": test_df}, sst2_head_specs(),
        pool_dir=pool, epochs=1, batch_size=2, device="cpu",
    )
    assert "acc" in metrics["test"] and "r2" in metrics["test"]["acc"]


def test_weight_baseline_features(tmp_path):
    from meta_model.baselines.weight_feature_baselines import extract_weight_features
    from meta_model.dataset import load_adapter_pool_metadata

    pool = make_fake_pool(tmp_path / "pool", n=6)
    df = load_adapter_pool_metadata(pool).to_pandas()
    scores = df.rename(
        columns={
            "split": "pool",
            "train_dataset": "source",
            "safetensor_path": "adapter_qname",
            "benchmark.sst2-test.likelihood.accuracy": "accuracy",
            "benchmark.sst2-test.likelihood.f1_macro": "f1_macro",
            "benchmark.sst2-test.likelihood.auroc": "auroc",
            "benchmark.sst2-test.likelihood.brier": "brier",
            "benchmark.sst2-test.likelihood.mean_confidence": "mean_confidence",
            "benchmark.sst2-test.likelihood.nll": "nll",
        }
    )
    scores["adapter_idx"] = range(len(scores))
    feats = extract_weight_features(scores)
    # 112 per-cell Frobenius norms + 112*rank singular values expected.
    assert any(c.startswith("norm.") for c in feats.columns)
    assert any(c.startswith("sv.") for c in feats.columns)
