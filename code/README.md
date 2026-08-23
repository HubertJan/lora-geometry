# Functional LoRA Geometry — replication code

Training and analysis code to replicate the results in the *LoRA Geometry* seminar
report.

The pipeline is: **train adapter pools → fit GL-equivariant meta models that predict SST2
performance from adapter weights → evaluate under distribution shift → probe what the meta
model reads.** Each folder below corresponds to a part of the report.

## Folder ↔ report map

| Folder | Report section |
|---|---|
| `common/` | Supporting infra (paths, HF model load/save, submitit-or-local runner). Not a report section. |
| `shared_adapter_pool/` | **Experimental Setup** — trains the two adapter pools (SST2 in-task ≈567, OOD ≈2300 over 15 datasets) and scores each adapter on SST2 (6 metrics). Produces the on-disk *pool datasets* every downstream experiment consumes. |
| `meta_model/` | **Approach — Meta Model** — the LoL GL-equivariant classifier (`FlexibleLoRAMetaClassifier`), the 5 configs (`w8-l2`, `w32-l2`, `base-l2`, `base`, `bilin8-l2`), the training loop and R²/Spearman metrics. |
| `meta_model/baselines/` | **Approach — Baselines** — PEFTGuard, `spectral-ridge` / `norms-ridge`, and the geometrical-metric ridges (`intrinsic-ridge` = Tier-A, `baserel-ridge` = Tier-B, `geom-ridge` = Tier-AB). |
| `experiments/in_task/` | **Results — In-Task** (Tab. compare) and **Different Metrics** (Tab. othermetrics). |
| `experiments/out_of_task/` | **Results — Out-of-Task-Domain** — leave-one-task-out (LOTO) per-dataset R²/ρ + scatter data. |
| `experiments/lrp/` | **Results — LRP Maps** — layer-wise relevance propagation on the meta model; PCA / cluster analysis. |
| `experiments/causal_ablation/` | **Results — Causal Ablation** — keep-only module-group sufficiency on high-accuracy SST2 adapters. |
| `experiments/uv_probe/` | **Results — UV Explanation** — bilinear u/v probe directions vs. orthographic/semantic token features. |

## The data contract (how folders connect)

`shared_adapter_pool/` writes each pool as a directory:

```
$POOL_DIR/<pool_name>/
  metadata.parquet          # one row per adapter: __key__, train_dataset, split,
                            # hyperparameters, and benchmark.sst2-test.likelihood.<metric>
  adapters/<__key__>/adapter_model.safetensors   # the LoRA weights (adapter-only)
```

`meta_model/` and every `experiments/` folder read exactly this layout — nothing else is
shared between them.

## Running

```bash
cd code
uv sync                       # CPU-only: comment out the [[tool.uv.index]] block in pyproject
cp .env.example .env          # then edit paths / BASE_MODEL / HF_TOKEN
```

**Large training jobs** (building pools, training many meta models) go through
`common.runner.submit_or_run`. Pass your own `submitit.AutoExecutor` to fan out on SLURM,
or pass `None` to run the same tasks **sequentially in-process** on the local machine:

```python
from common.runner import submit_or_run
import submitit

# local, sequential:
submit_or_run(train_one, tasks, executor=None)

# SLURM via your own executor:
ex = submitit.AutoExecutor(folder="_workdir/submitit")
ex.update_parameters(timeout_min=120, slurm_partition="gpu", gpus_per_node=1)
submit_or_run(train_one, tasks, executor=ex)
```

Each experiment folder has its own `README.md` and a `run.py` entry point.

## Sanity checks

`tests/` contains CPU-only smoke tests (import every module, tiny dummy-model forward /
train / eval / weight-surgery). They do **not** run real training or reproduce the paper
numbers — that needs a GPU and the gated Llama-3.2-1B — but they verify the code paths
execute. Run with `uv run pytest`.
