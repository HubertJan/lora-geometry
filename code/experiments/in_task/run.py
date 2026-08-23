"""In-Task meta-model evaluation.

Report → Results §In-Task (Tab. compare) and §Different Metrics (Tab. othermetrics).

Trains every meta-model config and baseline on the SST2 *in-task* pool (adapters
trained on SST2, split into train/test) and reports held-out calibration R² and
Spearman ρ.

* Tab. compare  — accuracy-head R²/ρ for the equivariant regressors
  (``w8_l2``, ``w32_l2``, ``bilin8_l2``, ``base_l2``, ``base``) and the baselines
  (``baserel``/``geom``/``intrinsic``-ridge, ``spectral``/``norms``-ridge, PEFTGuard).
* Tab. othermetrics — R²/ρ across all six SST2 metrics for ``w8_l2`` (4-seed mean)
  and ``w32_l2`` (single seed).

Prerequisite: an SST2 in-task pool at ``$POOL_DIR/<pool>`` built by
``shared_adapter_pool/jobs/build_sst2_pool.py``. Needs a GPU only for PEFTGuard's
size; the equivariant models train fine on CPU (slowly).

Run:  uv run python experiments/in_task/run.py --pool sst2_in_task
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from common import env
from meta_model.dataset import load_adapter_pool_metadata

# Paper name → arch-zoo key, and how many seeds each is averaged over (Tab. compare:
# w8_l2 is a 4-seed mean, the rest single-seed). arch-zoo keys are documented in
# meta_model/arch_zoo.py and meta_model/regressor_config.py::LADDER_ARCHS.
EQUIVARIANT_SEEDS: dict[str, list[int]] = {
    "w8_l2": [42, 43, 44, 45],
    "w32_l2": [42],
    "bilin8_l2": [42],
    "base_l2": [42],
    "base": [42],
}

# The six SST2 targets, short name (baselines) → full metadata column (heads).
TARGET_COLUMNS: dict[str, str] = {
    "accuracy": "benchmark.sst2-test.likelihood.accuracy",
    "f1_macro": "benchmark.sst2-test.likelihood.f1_macro",
    "auroc": "benchmark.sst2-test.likelihood.auroc",
    "brier": "benchmark.sst2-test.likelihood.brier",
    "mean_confidence": "benchmark.sst2-test.likelihood.mean_confidence",
    "nll": "benchmark.sst2-test.likelihood.nll",
}

# Fixed-epoch training, no early stopping (source: gl-regressor complexity ladder).
DEFAULT_EPOCHS = 200


# --------------------------------------------------------------------------- #
# Pool loading
# --------------------------------------------------------------------------- #
def load_intask_pool(pool_name: str) -> tuple[pl.DataFrame, pl.DataFrame, Path]:
    """Return (train_df, test_df, pool_dir) for the SST2 in-task pool."""
    pool_dir = env.POOL_DIR / pool_name
    if not (pool_dir / "metadata.parquet").exists():
        raise SystemExit(
            f"No pool at {pool_dir}. Build it first:\n"
            f"  uv run python shared_adapter_pool/jobs/build_sst2_pool.py"
        )
    meta = load_adapter_pool_metadata(pool_dir)
    # In-task = adapters trained on SST2 (the OOD pool shares the layout).
    if "train_dataset" in meta.columns:
        meta = meta.filter(pl.col("train_dataset") == "sst2")
    train_df = meta.filter(pl.col("split") == "train")
    test_df = meta.filter(pl.col("split") == "test")
    return train_df, test_df, pool_dir


# --------------------------------------------------------------------------- #
# Equivariant regressors + PEFTGuard (trained through the shared trainer)
# --------------------------------------------------------------------------- #
def train_equivariant_models(
    train_df, test_df, pool_dir, *, epochs: int
) -> dict[str, dict[int, dict]]:
    """Train each equivariant config (all seeds). Returns {arch: {seed: metrics}}."""
    from meta_model.regressor_config import build_regressor_config, sst2_head_specs
    from meta_model.train import train_meta_model

    specs = sst2_head_specs()
    out: dict[str, dict[int, dict]] = {}
    for arch, seeds in EQUIVARIANT_SEEDS.items():
        out[arch] = {}
        for seed in seeds:
            cfg = build_regressor_config(arch, seed=seed)
            ckpt = env.results_path("in_task", "checkpoints") / f"{arch}_seed{seed}.pth"
            res = train_meta_model(
                cfg, train_df, {"test": test_df}, specs,
                pool_dir=pool_dir, epochs=epochs, seed=seed, out_path=ckpt,
            )
            out[arch][seed] = res["test"]
    return out


def train_peftguard(train_df, test_df, pool_dir, *, epochs: int) -> dict:
    """Train the PEFTGuard baseline (a ModelConfig) through the shared trainer."""
    from meta_model.baselines.peftguard import build_peftguard_config
    from meta_model.regressor_config import sst2_head_specs
    from meta_model.train import train_meta_model

    cfg = build_peftguard_config()
    res = train_meta_model(
        cfg, train_df, {"test": test_df}, sst2_head_specs(),
        pool_dir=pool_dir, epochs=epochs, seed=42,
    )
    return res["test"]


# --------------------------------------------------------------------------- #
# Ridge / geometry baselines (sklearn, over a scores table)
# --------------------------------------------------------------------------- #
def _scores_df(pool_dir: Path):
    """Build the pandas scores frame the baselines expect from pool metadata."""
    meta = load_adapter_pool_metadata(pool_dir)
    if "train_dataset" in meta.columns:
        meta = meta.filter(pl.col("train_dataset") == "sst2")
    pdf = meta.to_pandas()
    import pandas as pd

    scores = pd.DataFrame(
        {
            "adapter_idx": range(len(pdf)),
            "pool": pdf["split"],                 # baselines split on pool=="train"/"test"
            "source": pdf.get("train_dataset", "sst2"),
            "adapter_qname": pdf["safetensor_path"],
        }
    )
    for short, full in TARGET_COLUMNS.items():
        scores[short] = pdf[full].to_numpy()
    # Geometry baselines also use the per-adapter degrader columns, if present.
    for hp in ("shards_total", "epochs", "label_noise"):
        if hp in pdf.columns:
            scores[hp] = pdf[hp].to_numpy()
    return scores


def run_weight_baselines(pool_dir: Path):
    from meta_model.baselines.weight_feature_baselines import (
        extract_weight_features,
        fit_exact_baselines,
    )

    feats = extract_weight_features(_scores_df(pool_dir))
    leaderboard, _ = fit_exact_baselines(feats)
    return leaderboard  # cols: model, head, r2, spearman, ...


def run_geometry_baselines(pool_dir: Path):
    """Tier-A (intrinsic), Tier-B (baserel), Tier-AB (geom) ridge arms.

    Needs the frozen base model to build its top-k singular subspaces.
    """
    from meta_model.baselines.geometry_baseline import (
        build_base_subspaces,
        extract_geometry_features,
        fit_geometry_leaderboard,
    )

    scores = _scores_df(pool_dir)
    base = build_base_subspaces(env.BASE_MODEL)
    feats = extract_geometry_features(scores, base)
    return fit_geometry_leaderboard(feats)


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def build_tab_compare(equiv, peftguard, weight_lb, geom_lb):
    """Tab. compare: accuracy-head R²/ρ for all models, most-calibrated first."""
    import pandas as pd

    rows = []
    for arch, by_seed in equiv.items():
        r2 = _mean([m["acc"]["r2"] for m in by_seed.values()])
        rho = _mean([m["acc"]["spearman"] for m in by_seed.values()])
        rows.append({"model": arch, "acc_r2": r2, "acc_spearman": rho})
    if peftguard is not None:
        rows.append(
            {"model": "peftguard",
             "acc_r2": peftguard["acc"]["r2"],
             "acc_spearman": peftguard["acc"]["spearman"]}
        )
    for lb, mapping in (
        (weight_lb, {"norms_ridge": "norms-ridge", "spectral_ridge": "spectral-ridge"}),
        (geom_lb, {"tierA_ridge": "intrinsic-ridge", "tierB_ridge": "baserel-ridge",
                   "tierAB_ridge": "geom-ridge"}),
    ):
        if lb is None:
            continue
        acc = lb[lb.head == "accuracy"]
        for model_key, label in mapping.items():
            hit = acc[acc.model == model_key]
            if len(hit):
                rows.append({"model": label,
                             "acc_r2": float(hit.r2.iloc[0]),
                             "acc_spearman": float(hit.spearman.iloc[0])})
    return pd.DataFrame(rows).sort_values("acc_r2", ascending=False)


def build_tab_othermetrics(equiv):
    """Tab. othermetrics: R²/ρ across the 6 SST2 metrics for w8_l2, w32_l2."""
    import pandas as pd

    heads = ["acc", "f1", "auroc", "brier", "meanconf", "nll"]
    rows = []
    for head in heads:
        row = {"metric": head}
        for arch in ("w8_l2", "w32_l2"):
            by_seed = equiv.get(arch, {})
            row[f"{arch}_r2"] = _mean([m[head]["r2"] for m in by_seed.values()])
            row[f"{arch}_spearman"] = _mean(
                [m[head]["spearman"] for m in by_seed.values()]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", default="sst2_in_task", help="pool name under $POOL_DIR")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--no-peftguard", action="store_true")
    ap.add_argument("--no-geometry", action="store_true",
                    help="skip geometry ridges (they load the base model)")
    args = ap.parse_args()

    train_df, test_df, pool_dir = load_intask_pool(args.pool)
    print(f"In-task pool: {len(train_df)} train / {len(test_df)} test adapters")

    equiv = train_equivariant_models(train_df, test_df, pool_dir, epochs=args.epochs)
    peftguard = None if args.no_peftguard else train_peftguard(
        train_df, test_df, pool_dir, epochs=args.epochs
    )
    weight_lb = run_weight_baselines(pool_dir)
    geom_lb = None if args.no_geometry else run_geometry_baselines(pool_dir)

    out = env.results_path("in_task")
    tab_compare = build_tab_compare(equiv, peftguard, weight_lb, geom_lb)
    tab_other = build_tab_othermetrics(equiv)
    tab_compare.to_csv(out / "tab_compare.csv", index=False)
    tab_other.to_csv(out / "tab_othermetrics.csv", index=False)
    if weight_lb is not None:
        weight_lb.to_csv(out / "weight_baselines_leaderboard.csv", index=False)
    if geom_lb is not None:
        geom_lb.to_csv(out / "geometry_leaderboard.csv", index=False)

    print("\n=== Tab. compare (held-out accuracy) ===")
    print(tab_compare.to_string(index=False))
    print("\n=== Tab. othermetrics ===")
    print(tab_other.to_string(index=False))
    print(f"\nWrote tables to {out}")


if __name__ == "__main__":
    main()
