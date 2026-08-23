# LRP relevance maps

Layer-wise Relevance Propagation (LRP) of a trained equivariant meta-model's
accuracy head onto the LoRA adapters it scores. Attributes the ACC-head logit
back to every LoRA weight, aggregates it to the canonical per-cell
(7 modules × 16 layers) **signed-net** grid, and studies the diversity of those
maps across adapters.

Backs the report **"LRP Maps"** section (`sections/results.typ`):
`@fig-lrp-diversity`, `@fig-lrp-pca`, `@fig-lrp-clusters`
(source experiment `2026-08-25_gl-regressor-lrp-acc`).

## Files

| file | role |
|------|------|
| `rules.py` | zennit LRP rules (`EpsilonNoBias`, `EquivariantRule`, `MatMulEpsilon`/`compute_mat_mul_relevance`, `PassThrough`) — copied verbatim |
| `analysis.py` | `per_rank_relevance` / `per_rank_aggregate` — copied verbatim (import fixed) |
| `aggregate.py` | `cell_signed_net` / `signed_grid` — the canonical 7×16 aggregation, module order [Q,K,V,O,gate,up,down] — copied verbatim |
| `lrp_run.py` | `run_lrp_single`, `per_rank_profiles`, `group_from_path`, `make_composite` + the `LRPFlexibleMetaClassifier` wrapper, `L2NormCell`, `make_composite_l2` (subclasses `meta_model.model_reg.FlexibleLoRAMetaClassifier`) |
| `run_maps.py` | job: per-adapter LRP maps over adapters × checkpoints → `lrp_maps.npz` + conservation check |
| `pc1_clusters.py` | job: PCA scree (PC1 ≈ 92 %), PC1 loading (7×16), pairwise cosine, tercile cohesion → CSVs + summary json + figures |

## Prerequisites

- A trained meta-model checkpoint (`best-model.pth`), config `w8_l2` (paper) or
  `base_l2`, produced by `experiments/in_task`. `run_maps.py` loads it through the
  LRP-hookable subclass, so any stock `FlexibleLoRAMetaClassifier` checkpoint loads.
- A pool directory (`POOL_DIR/<name>` with `metadata.parquet` + `adapters/`) of the
  test adapters to attribute.
- `torch` + `zennit` installed.

## Run

```bash
# from code/  (one checkpoint per seed; here a single one)
python -m experiments.lrp.run_maps \
    --pool sst2-perf-v2-test \
    --checkpoints /path/to/w8_l2/best-model.pth \
    --seeds 42 --head acc --target 0 --n-layers 16

# PCA + cluster analysis of the maps (needs an `acc` column in the meta parquet)
python -m experiments.lrp.pc1_clusters --meta /path/to/test_meta.parquet
```

Outputs land under `RESULTS_DIR/lrp/` (`lrp_maps.npz`, `lrp_scalars.parquet`,
`pc1_loading.csv`, `cluster_cohesion.csv`, `pc1_clusters_summary.json`,
`fig8_pc1_axis.png`, `fig9_accuracy_clusters.png`).
