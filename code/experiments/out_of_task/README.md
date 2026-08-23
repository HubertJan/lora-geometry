# experiments/out_of_task — Out-of-Task-Domain (LOTO)

**Report:** Results → *Out-of-Task-Domain* (Tab. loo-perdataset + the per-family scatter
figures fig-loto-sentiment / -toxicity / -entailment).

Leave-one-task-out robustness: for each of the 15 training datasets, hold out every adapter
trained on it, train the meta model on the adapters from the other 14, and predict the
held-out adapters' **SST2** performance. Only the adapter's *training dataset* changes — the
prediction target is always SST2. This measures whether the weight→capability map survives a
change in what the adapter was trained on.

## Prerequisite
The OOD pool at `$POOL_DIR/sst2_ood` (adapters over all 15 datasets):
```bash
uv run python shared_adapter_pool/jobs/build_ood_pool.py
```

## Run
```bash
uv run python experiments/out_of_task/run.py --pool sst2_ood --arch base_l2
```
Each held-out dataset is one fold. Pass a configured `submitit.AutoExecutor` in `run.py`
(where `submit_or_run(..., executor=None)` is called) to fan the 15 folds across SLURM;
`None` runs them sequentially.

## Outputs (under `$RESULTS_DIR/out_of_task/`)
- `loto_per_dataset.csv` — per-dataset held-out accuracy R² (calibration) and ρ (ranking).
  Reproduces Tab. loo-perdataset.
- `loto_scatter.csv` — per-adapter `(true_acc, pred_acc, held, __key__)` for the scatter figures.
- `checkpoints/base_l2_holdout_<dataset>.pth` — the per-fold meta models.
