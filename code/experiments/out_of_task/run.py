"""Out-of-Task-Domain (leave-one-task-out) meta-model evaluation.

Report → Results §Out-of-Task-Domain (Tab. loo-perdataset + the per-family
scatter figures fig-loto-*).

For each of the 15 training datasets in the OOD pool, hold out every adapter
trained on that dataset, train the meta model (``base_l2`` by default) on the
adapters from the *other* 14 datasets, and evaluate the held-out adapters. The
prediction target is always the SST2 benchmark; only the adapter's *training
dataset* changes — this is the distribution-shift axis. Emits per-dataset
calibration R² / ranking ρ and the predicted-vs-true scatter records.

Prerequisite: the OOD pool at ``$POOL_DIR/<pool>`` built by
``shared_adapter_pool/jobs/build_ood_pool.py``.

Run:  uv run python experiments/out_of_task/run.py --pool sst2_ood --arch base_l2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from common import env
from common.runner import submit_or_run
from meta_model.dataset import load_adapter_pool_metadata

DEFAULT_EPOCHS = 200


def _pool_dir(pool_name: str) -> Path:
    pool_dir = env.POOL_DIR / pool_name
    if not (pool_dir / "metadata.parquet").exists():
        raise SystemExit(
            f"No pool at {pool_dir}. Build it first:\n"
            f"  uv run python shared_adapter_pool/jobs/build_ood_pool.py"
        )
    return pool_dir


class Fold:
    """One LOTO fold: everything needed to train + score a held-out dataset."""

    def __init__(self, held: str, pool_name: str, arch: str, epochs: int):
        self.held = held
        self.pool_name = pool_name
        self.arch = arch
        self.epochs = epochs

    def __call__(self) -> dict:  # picklable entry for the runner
        return _run_fold(self.held, self.pool_name, self.arch, self.epochs)


def _run_fold(held: str, pool_name: str, arch: str, epochs: int) -> dict:
    """Train on all-but-``held`` adapters, evaluate on the held-out ones."""
    import torch

    from meta_model.dataset import attach_safetensor_paths, build_eval_dataloader
    from meta_model.metrics import predict_per_adapter
    from meta_model.regressor_config import build_regressor_config, sst2_head_specs
    from meta_model.train import train_meta_model

    pool_dir = _pool_dir(pool_name)
    meta = load_adapter_pool_metadata(pool_dir)
    train_df = meta.filter(pl.col("train_dataset") != held)
    held_df = meta.filter(pl.col("train_dataset") == held)
    if len(held_df) == 0:
        return {"held": held, "n": 0}

    specs = sst2_head_specs()
    # Fold-unique checkpoint keeps folds independent. train_val_split=1.0 → no val
    # (mirrors the original LOTO fold which trains on the full rest-of-pool).
    cfg = build_regressor_config(arch, seed=42)
    ckpt = env.results_path("out_of_task", "checkpoints") / f"{arch}_holdout_{held}.pth"
    res = train_meta_model(
        cfg, train_df, {held: held_df}, specs,
        pool_dir=pool_dir, epochs=epochs, seed=42,
        train_val_split=1.0, out_path=ckpt,
    )
    metrics = res[held]

    # Reload for per-adapter scatter records (predict_per_adapter needs the model).
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = cfg.create_model(device)
    ckpt_obj = torch.load(ckpt, map_location=device)
    # train_meta_model saves {state_dict, hyperparameters}; accept a bare state_dict too.
    state = (
        ckpt_obj["state_dict"]
        if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj
        else ckpt_obj
    )
    model.load_state_dict(state)
    loader = build_eval_dataloader(
        attach_safetensor_paths(held_df, pool_dir), specs, batch_size=8, device=device
    )
    preds = predict_per_adapter(loader, model, specs, device)
    keys = held_df["__key__"].to_list()
    scatter = [
        {"held": held, "__key__": keys[i],
         "true_acc": rec["acc.target"], "pred_acc": rec["acc.pred"]}
        for i, rec in enumerate(preds)
    ]
    return {
        "held": held,
        "n": len(held_df),
        "acc_r2": metrics["acc"]["r2"],
        "acc_spearman": metrics["acc"]["spearman"],
        "scatter": scatter,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", default="sst2_ood", help="pool name under $POOL_DIR")
    ap.add_argument("--arch", default="base_l2", help="arch-zoo key for the meta model")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    args = ap.parse_args()

    pool_dir = _pool_dir(args.pool)
    datasets = (
        load_adapter_pool_metadata(pool_dir)["train_dataset"].unique().sort().to_list()
    )
    print(f"LOTO over {len(datasets)} datasets: {datasets}")

    # One fold per held-out dataset. Pass a configured submitit executor here to
    # fan folds out across SLURM; None runs them sequentially in-process.
    folds = [Fold(d, args.pool, args.arch, args.epochs) for d in datasets]
    results = submit_or_run(lambda f: f(), folds, executor=None)

    import pandas as pd

    per_dataset = pd.DataFrame(
        [{"dataset": r["held"], "n": r["n"],
          "acc_r2": r.get("acc_r2"), "acc_spearman": r.get("acc_spearman")}
         for r in results if r]
    )
    scatter = pd.DataFrame(
        [row for r in results if r for row in r.get("scatter", [])]
    )
    out = env.results_path("out_of_task")
    per_dataset.to_csv(out / "loto_per_dataset.csv", index=False)
    scatter.to_csv(out / "loto_scatter.csv", index=False)

    print("\n=== Per-dataset LOTO (acc R² / ρ) ===")
    print(per_dataset.to_string(index=False))
    print(f"\nWrote {out}/loto_per_dataset.csv and loto_scatter.csv")


if __name__ == "__main__":
    main()
