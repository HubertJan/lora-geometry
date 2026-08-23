# UV bilinear-probe token-feature analysis

Reads the trained `w8_l2` accuracy **regressor** as a per-cell bilinear probe: for
each (module, layer) cell the local Jacobian template `T = Wb^T G_eff Wa` factors
into a read direction `v` and a write direction `u` in the residual/embedding
space. The analysis asks how legible those directions are (alignment with the
sentiment axis `delta = emb("positive") − emb("negative")`, logit-lens, seed
stability) and whether the per-token projections onto them are selective for
interpretable token features.

Backs the report **"UV Explanation"** section (`sections/results.typ`): the
per-feature UV token-cell selectivity table
(`data/uv_token_layers_cell.csv`; source `2026-08-25_gl-regressor-uv-interpretability`).

## Files

| file | role |
|------|------|
| `uv_extract.py` | per-cell Jacobian template `T = Wb^T G_eff Wa`, `module_qr`, `uv_from_core`, `cell_full`, `RESIDUAL_WRITERS = {o_proj, down_proj}` — copied verbatim (imports fixed) |
| `axes.py` | base-model reference axes: `delta`, `cos`, `subspace_capture`, `logit_lens` — embeddings via `common.hf` (hardcoded HF snapshot path removed) |
| `pipeline.py` | `extract_main` (writes `uv_scalars.parquet` + `mean_dirs.npz`) / `analyze_main` (writes the Q1–Q4 CSVs + `findings.json`) |
| `token_features.py` | collect per-token projections onto the 112 read/write dirs + `K_RAND=32` random dirs/layer, then the ~13 token features (orthographic + word-class + semantic) → per-cell specific selectivity `token_layers_cell.csv` + heatmap |

`token_features.collect` sources each cell's top read/write direction from the
pipeline's `mean_dirs.npz` (the original job's `uv_subspaces_full.npz` only ever
used column 0 of each cell's basis, which equals that seed-mean top direction), so
there is no separate frozen-subspace input.

## Prerequisites

- Trained `w8_l2` regressor checkpoints at
  `RESULTS_DIR/uv_probe/checkpoints/w8_l2_seed{42..45}.pth` (from `experiments/in_task`).
- Adapter metadata parquet at `RESULTS_DIR/uv_probe/test_meta.parquet` (columns
  `adapter_id` / `safetensor_path` / `acc`).
- `torch`, `transformers`, `datasets` installed (base model + tokenizer via `common.hf`).

## Run

```bash
# from code/
python -m experiments.uv_probe.pipeline extract     # -> uv_scalars.parquet, mean_dirs.npz
python -m experiments.uv_probe.pipeline analyze      # -> analysis/*.csv, findings.json
python -m experiments.uv_probe.token_features collect
python -m experiments.uv_probe.token_features analyze
```

All I/O is under `RESULTS_DIR/uv_probe/`.

## Not migrated

`jobs/08_probes_abce.py` (the capture-vs-20-trial random rank-8 baseline) is
**omitted**: it consumes `repr_subspaces.npz` + `activations.npz` produced by
non-migrated jobs (06/07), so it would have dangling inputs.
