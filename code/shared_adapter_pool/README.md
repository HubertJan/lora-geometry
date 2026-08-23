# shared_adapter_pool

Trains **LoRA adapter pools** and scores each adapter on **SST2**, producing an
on-disk *pool dataset* (`metadata.parquet` + `adapters/<key>/`) that the
`meta_model` package consumes. Migrated from the research repo `glad`
(`src/llm_pipeline` + `src/discoveries/sst2_perf_prediction`) with the W&B /
trackio / artifact-cache infrastructure stripped out; the numeric/scientific
code (dataset prep, chat templates/verbalizers, SFT training, likelihood
scoring, the six metrics, the HP-sampling grids) is preserved.

## End-to-end flow

For each adapter config (`pool/spec.py::AdapterCfg`), `pool/worker.py`:

1. **Prep** the training corpus — `data/corpus_prep.py::prepare_corpus` does
   raw HF → merge source splits → map to the typed schema (canonical label
   strings via `data/definitions/_definition.py::LabelDecoding.decode`) →
   drop-none; then `task_splits` cuts a shuffled 80/10/10 split (`seed=42`) and
   head-caps it (`max_train=20000`, `max_test=2000`).
2. **Shard** a stratified slice of the train split
   (`data/sharding.py::CategoryStratifiedShardSamplingStrategy`) — this is the
   data-volume lever (`shards_total` → ≈ `20000 / shards_total` samples).
3. Optionally apply **label noise**
   (`data/label_noise.py::LabelNoiseDataset`) — the mid-band degrader lever.
4. Render prompts with the task's **chat template** under the `TRUE_FALSE_V1`
   verbalizer, then **verbalizer preflight**: assert rendered completions ∈
   {`true`, `false`} (guards the label-scheme silent no-op).
5. **Tokenize** (completion-only loss, prompt masked to `-100`).
6. **Train** a fresh rank-16 LoRA adapter — `lora/peft.py::build_peft_model`
   (all 7 Llama projections) + `lora/train.py::train_lora_adapter` (TRL
   `SFTTrainer`/`SFTConfig`), saved to `adapters/<__key__>/`.
7. **Evaluate** on the SST2 test split —
   `eval/run_eval.py::evaluate_on_sst2` runs the likelihood classifier
   (`eval/classify.py`) and computes the six metrics (`eval/metrics.py`).
8. Return a metadata row; `pool/store.py::write_pool` writes `metadata.parquet`.

The HP-sampling grids live in `pool/production_grid.py` (SST2 in-task cells +
per-adapter SeedSequence HP jitter: LR U(5e-5, 3e-4), dropout U(0, 0.1),
alpha ∈ {16, 32}) and `pool/ood_grid.py` (the 14 OOD training slugs, all
evaluated on SST2).

## Running the jobs

Both jobs use `common.runner.submit_or_run`, which runs the identical callable
**locally & sequentially** when `EXECUTOR=None` (the default) or submits a
**SLURM** array when handed a `submitit.AutoExecutor`. Adapters are packed
`ADAPTERS_PER_JOB` at a time via `common.runner.chunked`.

### Local (sequential, in-process)

```bash
POOL_DIR=./_workdir/pools BASE_MODEL=meta-llama/Llama-3.2-1B \
  python -m shared_adapter_pool.jobs.build_sst2_pool     # SST2 in-task pool (567 adapters)

POOL_DIR=./_workdir/pools \
  python -m shared_adapter_pool.jobs.build_ood_pool       # OOD pool (~2240 adapters, 14 slugs)
```

Pool size is controlled by `REPLICATES` (SST2: 21 cells × 27 = 567) and
`PER_DATASET` (OOD: 16 cells × 10 replicates × 14 datasets = 2240).

### SLURM

Uncomment the executor block in `jobs/build_sst2_pool.py` (a worked example is
in `_build_executor`) and assign it to `EXECUTOR`:

```python
import submitit
ex = submitit.AutoExecutor(folder="…/submitit/sst2_in_task")
ex.update_parameters(timeout_min=180, slurm_partition="gpu",
                     gpus_per_node=1, tasks_per_node=1, cpus_per_task=8)
EXECUTOR = ex
```

Config comes from `common.env` (env vars): `POOL_DIR`, `RESULTS_DIR`,
`CACHE_DIR`, `BASE_MODEL`, `HF_TOKEN`.

## Pinned versions

The training/eval stack is version-sensitive; reproduce with:

```
peft==0.19.1  datasets==4.8.4  transformers==5.6.2  trl==1.2.0
```

Rationale: `trl==1.2.0`'s `SFTConfig`/`SFTTrainer` accepts the exact argument
set `TrainingConfig.to_training_arguments_kwargs()` emits (incl.
`dataset_kwargs={"skip_prepare_dataset": True}`, since datasets are tokenized
manually upstream); `transformers==5.6.2` provides the `DataCollatorForSeq2Seq`
runtime-padding behaviour and the Llama-3.2 loader; `peft==0.19.1` fixes the
LoRA-A init RNG semantics (`torch.manual_seed` before `get_peft_model`) and the
`save_pretrained` key layout the `meta_model` loader parses; `datasets==4.8.4`
fixes `train_test_split` shuffling and `Dataset.shard` semantics that the
stratified sharding and 80/10/10 split depend on. The base model loads in
**float16** for Llama-3.2-1B/3B (bf16 collapses freshly-applied LoRA adapters);
see `common/hf.py`.

## Output contract

`pool/store.py` writes `metadata.parquet` to **exactly** match
[`../meta_model/CONTRACT.md`](../meta_model/CONTRACT.md): identity columns
`__key__`, `train_dataset`, `split`; pass-through hyperparameters
(`learning_rate`, `weight_decay`, `lora_rank`, `lora_alpha`, `lora_dropout`,
`epochs`, `shards_total`, `label_noise`, `train_seed`); and the six SST2
benchmark targets

```
benchmark.sst2-test.likelihood.{accuracy, f1_macro, auroc, brier, mean_confidence, nll}
```

Adapters are the standard PEFT `save_pretrained` output under
`adapters/<__key__>/` (keys not re-mapped), which the `meta_model` loader parses
directly. **If the contract changes, update `pool/store.py` to match it.**
