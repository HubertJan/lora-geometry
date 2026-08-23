# experiments/in_task — In-Task performance prediction

**Report:** Results → *In-Task* (Tab. compare) and *Different Metrics* (Tab. othermetrics).

Trains every meta-model config and baseline on the **SST2 in-task pool** (adapters
trained on SST2, held-out train/test split) and reports how well each predicts a
held-out adapter's SST2 performance from its weights alone.

## Prerequisite
An SST2 in-task pool at `$POOL_DIR/sst2_in_task`:
```bash
uv run python shared_adapter_pool/jobs/build_sst2_pool.py
```

## Run
```bash
uv run python experiments/in_task/run.py --pool sst2_in_task
# options: --epochs 200  --no-peftguard  --no-geometry
```

## Outputs (under `$RESULTS_DIR/in_task/`)
- `tab_compare.csv` — accuracy-head R²/ρ for `w8_l2`, `w32_l2`, `bilin8_l2`, `base_l2`,
  `base`, PEFTGuard, and the ridge baselines (`norms`/`spectral`/`intrinsic`/`baserel`/`geom`).
- `tab_othermetrics.csv` — R²/ρ across all six SST2 metrics for `w8_l2` (4-seed mean) and `w32_l2`.
- `weight_baselines_leaderboard.csv`, `geometry_leaderboard.csv` — full baseline leaderboards.
- `checkpoints/<arch>_seed<seed>.pth` — trained meta models (reused by the interpretability experiments).

## Notes
- The equivariant regressors train on CPU (slowly); only PEFTGuard's size really wants a GPU
  (skip it with `--no-peftguard`). The geometry ridges load the frozen base model to build its
  top-k singular subspaces (`--no-geometry` to skip).
- Config → arch-zoo key: `w8-l2`→`w8_l2`, `w32-l2`→`w32_l2`, `base-l2`→`base_l2`, `base`→`base`,
  `bilin8-l2`→`bilin8_l2`. See `meta_model/arch_zoo.py`.
