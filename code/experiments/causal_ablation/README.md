# Causal keep-only ablation

Causal test of the LRP reading: take a high-accuracy SST2 adapter, keep only a
selected group of LoRA cells (reverting everything else to the base model) — or
zero / sign-flip a group — and re-measure SST2 likelihood accuracy. If a group is
causally sufficient, keeping just it sustains accuracy above the base floor.

Backs the report **"Causal Ablation"** section (`sections/results.typ`):
`@fig-module-sufficiency` (source experiment `2026-08-25_kv-ablation-high-acc`).
Base floor (no adapter) ≈ **0.558**, a strong original adapter ≈ **0.955**.

## Files

| file | role |
|------|------|
| `ablate.py` | self-contained safetensors weight surgery: `ablate_adapter` with keep/negate/rules, 0-based `layers.{i}` — copied **verbatim** |
| `sst2_eval.py` | **local fallback** SST2 likelihood-accuracy evaluator (torch-only; TRUE_FALSE_V1 prompt, summed continuation log-likelihood, argmax vs gold) — see note below |
| `benchmark.py` | job: `run_benchmark` + the ~35-variant `ABLATIONS` list (keep_mlpdown_all, keep_attn_all, keep_q_all, keep_kv_all, sanity_ablate_all, sanity_keep_all, …) → `variant_accuracy.csv` |

### SST2 eval — fallback used

`shared_adapter_pool.eval.run_eval.evaluate_on_sst2` (and
`shared_adapter_pool.data.definitions.registry.TASKS["sst2"]`) are **not present**
in the migrated `shared_adapter_pool` package, so `benchmark.py` uses the
self-contained `sst2_eval.py`. It reuses the pool's exact label words / prompt
text from `shared_adapter_pool.data.definitions.sst2`, so the un-ablated baseline
still reproduces the stored `benchmark.sst2-test.likelihood.accuracy`.

## Prerequisites

- A pool directory (`POOL_DIR/<name>`) of SST2 adapters; an `acc` column in the
  metadata is used to pick the top-N (a trained meta-model is *not* required here).
- `torch`, `peft`, `transformers`, `datasets` installed (the base model is loaded
  through `common.hf`).

## Run

```bash
# from code/
python -m experiments.causal_ablation.benchmark --pool sst2-perf-v2-test --n 3
```

Outputs land under `RESULTS_DIR/causal_ablation/` (`variant_accuracy.csv` with
per-(adapter, variant) accuracy + delta-vs-original, `results.json`, and the
`ablated/` weight-surgery tree). Replace the local default with a configured
executor to run the whole benchmark as one SLURM job via `common.runner.submit_or_run`.
