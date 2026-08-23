"""Meta-model training entry point (migrated from
SRC/src/discoveries/meta_classifier_training/flows/train_meta_classifier.py).

The original ``train()`` was decorated with ``@cached_wandb_run`` and consumed
``LinkedDatasetRef`` inputs, resolving each to a metadata DataFrame and logging
W&B artifacts for lineage. This rewrite strips all of that infrastructure — the
cache decorator, the run group, ``create_file_path_for_artifact``, the trackio
store, ``ref_to_metadata_df`` and the artifact logger — and takes plain polars
DataFrames plus a ``pool_dir`` instead.

The scientific core is kept verbatim: a fixed-epoch loop (no early stopping) with
``Adam(lr, weight_decay)``, a ``CosineAnnealingLR(T_max=epochs, eta_min=5e-6)``
scheduler stepped per optimiser step, per-head masked losses via
:func:`meta_model.heads.compute_multihead_loss`, and per-head R2 / Spearman
evaluation via :func:`meta_model.metrics.evaluate_multihead_performance`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl
import torch
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from common.logging import get_run
from meta_model.dataset import (
    attach_safetensor_paths,
    build_eval_dataloader,
    build_pool_dataloaders,
)
from meta_model.heads import MetaTargetSpec, compute_multihead_loss
from meta_model.metrics import evaluate_multihead_performance

# Training constants carried over verbatim from the original trainer.
_ETA_MIN = 5e-6


def _resolve_device(device: str | None) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_out_file(out_path: str | Path | None) -> Path | None:
    if out_path is None:
        return None
    p = Path(out_path)
    if p.suffix != ".pth":
        p = p / "best-model.pth"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def train_meta_model(
    model_config: Any,
    train_df: pl.DataFrame,
    test_dfs: dict[str, pl.DataFrame],
    target_specs: list[MetaTargetSpec],
    *,
    pool_dir: str | Path,
    epochs: int = 200,
    lr: float = 2e-5,
    weight_decay: float = 1e-5,
    batch_size: int = 8,
    train_val_split: float = 0.85,
    seed: int = 42,
    stratify_by: str | Sequence[str] | None = None,
    materialize: str = "tmp",
    device: str | None = None,
    out_path: str | Path | None = None,
) -> dict[str, dict[str, dict]]:
    """Train a multi-head meta model on a safetensor-backed adapter pool.

    Parameters
    ----------
    model_config:
        Any ``ModelConfig`` exposing ``create_model(device) -> nn.Module``,
        ``name`` and ``to_dict()`` (e.g. :class:`meta_model.regressor_config.
        FlexibleRegressorConfig` or :class:`meta_model.baselines.peftguard.
        PEFTGuardConfig`). The model must return ``{head_name: logits}``.
    train_df:
        Pool metadata (one row per adapter) with a ``__key__`` column and the
        ``target_specs`` benchmark columns. Safetensor paths are resolved as
        ``pool_dir/adapters/<__key__>/adapter_model.safetensors``.
    test_dfs:
        ``{name: metadata_df}`` — one held-out pool per name, evaluated once at
        the end. Each is keyed the same way as ``train_df``.
    target_specs:
        The prediction heads and how each head's target is read from a metadata
        row. Must match ``model_config.head_specs``.
    pool_dir:
        Root of the on-disk pool (contains ``adapters/<__key__>/``).
    out_path:
        If given, the final model ``state_dict`` is saved here (a ``.pth`` file,
        or ``<out_path>/best-model.pth`` for a directory).

    Returns
    -------
    ``{test_df_name: {head_name: metric_dict}}`` where each ``metric_dict``
    carries ``r2`` / ``spearman`` / ``pearson`` / ``mae`` (regression heads) or
    classification metrics, plus ``count`` / ``n_missing``.
    """
    device = _resolve_device(device)
    pool_dir = Path(pool_dir)

    # --- Build train/val loaders (split owned by build_pool_dataloaders) ---
    train_df = attach_safetensor_paths(train_df, pool_dir)
    train_loader, val_loader = build_pool_dataloaders(
        train_df,
        target_specs,
        materialize=materialize,  # type: ignore[arg-type]
        seed=seed,
        val_frac=max(0.0, 1.0 - train_val_split),
        stratify_by=stratify_by,
        batch_size=batch_size,
        device=device,
    )
    n_train = len(train_loader.dataset)  # type: ignore[arg-type]
    n_val = len(val_loader.dataset) if val_loader is not None else 0  # type: ignore[arg-type]
    print(f"Train/val split: {n_train} / {n_val}")

    # --- Held-out test loaders (lazy: iterated once) ---
    test_loaders: dict[str, DataLoader] = {}
    for name, df in test_dfs.items():
        df = attach_safetensor_paths(df, pool_dir)
        test_loaders[name] = build_eval_dataloader(
            df, target_specs, batch_size=batch_size, device=device
        )
        print(f"Test pool {name!r}: {len(df)} adapters")

    # --- Model + optimiser (verbatim schedule) ---
    model = model_config.create_model(device)
    print(f"Created model: {model_config.name}")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=_ETA_MIN)

    run = get_run(
        name=model_config.name,
        config={
            "model_config": model_config.to_dict(),
            "learning_rate": lr,
            "epochs": epochs,
            "weight_decay": weight_decay,
            "eta_min": _ETA_MIN,
            "batch_size": batch_size,
            "target_specs": [spec.to_dict() for spec in target_specs],
            "train_val_split": train_val_split,
            "seed": seed,
            "stratify_by": stratify_by,
            "materialize": materialize,
            "train_size": n_train,
            "val_size": n_val,
            "test_sizes": {name: len(dl.dataset) for name, dl in test_loaders.items()},  # type: ignore[arg-type]
        },
    )

    global_steps = 0
    # Fixed-epoch training — no early stopping; the final model is what is saved.
    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        running_count = 0

        for inputs, targets in train_loader:
            outputs = model(inputs)
            loss, per_head_loss = compute_multihead_loss(outputs, targets, target_specs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            batch_size_current = next(iter(targets.values())).size(0)
            step_loss = loss.item()
            run.log({"train/step_loss": step_loss}, step=global_steps)
            global_steps += 1
            running_train_loss += step_loss
            running_count += batch_size_current

        train_loss = running_train_loss / running_count if running_count else 0.0
        run.log({"train/loss": train_loss}, step=global_steps)

        if val_loader is not None:
            val_metrics = evaluate_multihead_performance(
                {"val": val_loader}, model, target_specs, device
            )
            run.log({"val": val_metrics}, step=global_steps)

        print(f"Epoch {epoch + 1:02d} | train_loss={train_loss:.4f}")

    # --- Save final model ---
    # Prefer the model's own ``save`` (writes {state_dict, hyperparameters}) so the
    # checkpoint can be reloaded by ``FlexibleLoRAMetaClassifier.load`` — this is what
    # the LRP / UV interpretability experiments consume. Fall back to a bare
    # state_dict for models without a ``save`` method (e.g. PEFTGuard).
    out_file = _resolve_out_file(out_path)
    if out_file is not None:
        if hasattr(model, "save"):
            model.save(out_file)
        else:
            torch.save(model.state_dict(), out_file)
        print(f"Saved final model to {out_file}")

    # --- Final held-out evaluation ---
    test_metrics = evaluate_multihead_performance(
        test_loaders, model, target_specs, device
    )
    run.log({"test": test_metrics}, step=global_steps)
    run.summary_update({"test": test_metrics})
    run.finish()

    return test_metrics
