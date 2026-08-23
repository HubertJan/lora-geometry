# experiments/ — one folder per Results subsection

Each folder is a runnable experiment that consumes a trained adapter pool (from
`shared_adapter_pool/`) and, for the interpretability experiments, a trained meta-model
checkpoint (from `in_task/`).

| Folder | Report section | Needs |
|---|---|---|
| `in_task/` | Results → In-Task + Different Metrics | SST2 in-task pool |
| `out_of_task/` | Results → Out-of-Task-Domain (LOTO) | OOD pool |
| `lrp/` | Results → LRP Maps | in-task pool + a trained `w8_l2`/`base_l2` checkpoint |
| `causal_ablation/` | Results → Causal Ablation | 3 high-accuracy SST2 adapters + base model |
| `uv_probe/` | Results → UV Explanation | in-task pool + a trained (bilinear) checkpoint |

Typical order: build pools → `in_task` (also trains the checkpoints the interpretability
experiments reuse) → `out_of_task` → `lrp` / `causal_ablation` / `uv_probe`.

Large fan-outs (LOTO folds, per-adapter LRP, ablation variants) go through
`common.runner.submit_or_run`: pass your own `submitit.AutoExecutor` for SLURM, or `None`
to run sequentially in-process.
