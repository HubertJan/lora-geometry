# Adapter-pool data contract (what `meta_model` loads)

The dataset loader (`meta_model/dataset.py`) and the baselines consume an
on-disk adapter pool with this exact layout:

```
<pool_dir>/
  metadata.parquet
  adapters/<__key__>/adapter_model.safetensors   # adapter-only LoRA weights
  adapters/<__key__>/adapter_config.json
```

The pool builder must write exactly what is described below.

## `metadata.parquet` — one row per adapter

Required columns:

| column         | type | meaning                                                        |
|----------------|------|----------------------------------------------------------------|
| `__key__`      | str  | adapter subdirectory name under `adapters/` (path key)         |
| `train_dataset`| str  | training-data slug, e.g. `sst2`, `imdb`                        |
| `split`        | str  | `train` or `test`                                              |

Hyperparameter columns (carried through; used by some baselines):
`learning_rate`, `lora_alpha`, `lora_dropout`, `epochs`, ...
The geometry baseline additionally reads `shards_total`, `epochs`, `label_noise`
(its `HP_COLS`) when present.

### Benchmark target columns

Targets are read **by name** from `MetaTargetSpec.column`. The SST2 regression
heads (`meta_model/regressor_config.py::SST2_HEADS`) expect exactly these six
columns:

```
benchmark.sst2-test.likelihood.accuracy         -> head "acc"      (BOUNDED_REGRESSION)
benchmark.sst2-test.likelihood.f1_macro         -> head "f1"       (BOUNDED_REGRESSION)
benchmark.sst2-test.likelihood.auroc            -> head "auroc"    (BOUNDED_REGRESSION)
benchmark.sst2-test.likelihood.brier            -> head "brier"    (BOUNDED_REGRESSION)
benchmark.sst2-test.likelihood.mean_confidence  -> head "meanconf" (BOUNDED_REGRESSION)
benchmark.sst2-test.likelihood.nll              -> head "nll"      (UNBOUNDED_REGRESSION)
```

Notes for the pool builder:

- Values may be stored as strings; the loader `float()`-parses regression
  targets. A missing / `None` / unparseable value becomes `NaN` and is masked
  out of the loss and metrics per head (a classification target that is not a
  mapping key becomes the `-1` sentinel).
- The loader accepts **any** column name — it reads `spec.column` for each head.
  To use different benchmark columns, pass `target_specs` whose `column` fields
  name them; no code change is needed. The names above are the SST2 default that
  ships with `regressor_config.py`.

## `adapters/<__key__>/adapter_model.safetensors` — LoRA key layout

Adapter-only (PEFT) safetensors. The loader
(`meta_model/lora/weight_utils.py::group_lora_weights_per_submodule`, via
`materialization.default_transform`) parses tensor **keys** with these regexes:

- layer id:  `.layers.<L>.` (`<L>` an integer, 0-based)
- submodule: `.layers.<L>.<submodule>.lora_(A|B).weight`
- lora type: `lora_A` / `lora_B`

So each weight key must look like:

```
base_model.model.model.layers.<L>.<submodule>.lora_A.weight   # shape (r, in_features)
base_model.model.model.layers.<L>.<submodule>.lora_B.weight   # shape (out_features, r)
```

where `<submodule>` is the HF module path used as the grouping key, e.g.:

```
self_attn.q_proj  self_attn.k_proj  self_attn.v_proj  self_attn.o_proj
mlp.gate_proj     mlp.up_proj       mlp.down_proj
```

(the exact prefix before `.layers.` is irrelevant — only the `.layers.<L>.` and
`.lora_[AB].weight` substrings are matched).

### Grouped representation the model consumes

After grouping, each adapter becomes a nested dict keyed by submodule then by
`LoraType`, with layers stacked into the leading dimension:

```
{ "<submodule>": { LoraType.A: Tensor[L, r, in],
                   LoraType.B: Tensor[L, out, r] } }
```

- `L` = number of transformer layers, inferred per adapter from the max layer id
  (`weight_utils.infer_layer_count`), not hardcoded.
- `r` = LoRA rank, inferred from the A/B shapes (`weight_utils.infer_rank`);
  a pool batched at `batch_size > 1` must have a **uniform rank** across all
  adapters (enforced by `dataset.assert_uniform_rank`).

`adapter_config.json` is the standard PEFT config; `common.hf.load_adapter_model`
reads `base_model_name_or_path` from it.
