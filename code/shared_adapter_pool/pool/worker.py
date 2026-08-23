"""Self-contained per-adapter worker: prep -> shard -> train -> SST2 eval.

(migrated from src/discoveries/sst2_perf_prediction/flows/train_eval_adapter.py
::train_and_eval_adapter, rewritten against the standalone stack.)

One job trains ONE rank-16 LoRA adapter on ``cfg.train_slug`` with the true/false
verbalizer, writes it to ``<pool_dir>/adapters/<__key__>/``, then runs a
standalone likelihood evaluation of it on the SST2 test split. Returns a metadata
row dict (identity + hyperparameters + the six ``benchmark.sst2-test.likelihood.*``
values) ready for ``pool/store.py::write_pool``.

The verbalizer preflight is preserved: before training, a handful of real
about-to-be-trained rows are rendered and their completions asserted to be in
{true, false}, which guards the ``label_scheme`` silent-no-op (string labels
bypassing the scheme).
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

from shared_adapter_pool.pool.spec import AdapterCfg
from shared_adapter_pool.pool.store import BENCHMARK_METRICS

# Worker-level defaults, transcribed from the production jobs
# (04_production-sst2 / 10_ood-production).
MAX_TRAIN_PREP = 20000
MAX_TEST_PREP = 2000
MAX_SEQ_LENGTH = 1024
BATCH_SIZE = 8
GRAD_ACCUM = 4
EVAL_BATCH_SIZE = 8
BF16 = True
ADD_EOS = False
LOGGING_STEPS = 50
PREP_SEED = 42


def _benchmark_key(metric: str) -> str:
    return f"benchmark.sst2-test.likelihood.{metric}"


def train_and_eval_adapter(
    cfg: AdapterCfg,
    pool_dir: str | Path,
    *,
    base_model_name: str | None = None,
    max_train_prep: int = MAX_TRAIN_PREP,
    max_test_prep: int | None = MAX_TEST_PREP,
    max_seq_length: int = MAX_SEQ_LENGTH,
    batch_size: int = BATCH_SIZE,
    gradient_accumulation_steps: int = GRAD_ACCUM,
    eval_batch_size: int = EVAL_BATCH_SIZE,
    bf16: bool = BF16,
    add_eos: bool = ADD_EOS,
    logging_steps: int = LOGGING_STEPS,
    prep_seed: int = PREP_SEED,
) -> dict[str, Any]:
    """Train + standalone-eval one adapter; return its ``metadata.parquet`` row."""
    from common import env
    from common.hf import load_adapter_model, load_base_model, load_tokenizer
    from shared_adapter_pool.data.corpus_prep import prepare_corpus, task_splits
    from shared_adapter_pool.data.definitions._variants import resolve_variant_choices
    from shared_adapter_pool.data.definitions.registry import TASKS
    from shared_adapter_pool.data.label_noise import LabelNoiseDataset
    from shared_adapter_pool.data.sharding import (
        CategoryStratifiedShardSamplingStrategy,
        ShardSamplingConfig,
    )
    from shared_adapter_pool.data.transforms import (
        ApplyChatTemplate,
        SampleDataset,
        Tokenize,
        assert_labels_supervised,
    )
    from shared_adapter_pool.eval.run_eval import evaluate_on_sst2
    from shared_adapter_pool.lora.config import LoraConfig, TrainingConfig
    from shared_adapter_pool.lora.train import train_lora_adapter

    base_model_name = base_model_name or env.BASE_MODEL
    pool_dir = Path(pool_dir)
    out_dir = pool_dir / "adapters" / cfg.label

    train_task = TASKS[cfg.train_slug]
    eval_task = TASKS[cfg.eval_slug]
    if train_task.needs_tokenizer:
        raise NotImplementedError(
            f"{cfg.train_slug}: tokenizer-driven chat templates are not supported "
            "by the standalone pool worker."
        )

    train_variant_kwargs = resolve_variant_choices(
        train_task.chat_template, {"label_scheme": cfg.label_scheme}
    )

    tokenizer = load_tokenizer(base_model_name)

    # -- Phase A: raw -> typed -> 80/10/10 -> capped train/test --------------
    typed = prepare_corpus(train_task.corpus)
    splits = task_splits(
        typed, seed=prep_seed, max_train=max_train_prep, max_test=max_test_prep
    )
    train_ref = splits.train

    # -- this adapter's stratified shard slice of the shared train split -----
    shard_ref = SampleDataset(
        strategy=CategoryStratifiedShardSamplingStrategy(
            shard_sampling_config=ShardSamplingConfig(
                number_of_dataset_shards=cfg.shards_total,
                selected_shards=cfg.shard_indices,
            ),
            category_field_name=train_task.label_field,
        )
    )(train_ref)

    # -- degrader lever: random label noise on the TRAIN shard ---------------
    train_source_ref = shard_ref
    if cfg.label_noise > 0.0:
        train_source_ref = LabelNoiseDataset(
            noise_rate=cfg.label_noise,
            label_field=train_task.label_field,
            label_values=tuple(str(v) for v in train_task.label_values),
            seed=cfg.train_seed,
        )(shard_ref)

    # -- chat template with the true/false verbalizer ------------------------
    chat_template = train_task.chat_template(add_eos=add_eos, **train_variant_kwargs)

    # -- VERBALIZER PREFLIGHT: render real rows, assert true/false -----------
    shard_data = shard_ref.data
    surfaces: dict[str, int] = {}
    for i in range(min(16, len(shard_data))):
        row = {c: shard_data[c][i] for c in shard_data.column_names}
        c = chat_template.apply(row)["completion"]
        surfaces[c] = surfaces.get(c, 0) + 1
    print(f"[{cfg.label}] verbalizer completion histogram: {surfaces}")
    bad = {s for s in surfaces if s not in {"true", "false"}}
    if bad:
        raise AssertionError(
            f"[{cfg.label}] verbalizer SILENT NO-OP: completions {sorted(bad)} "
            f"not in {{true,false}} (label_scheme={cfg.label_scheme}). "
            f"train_variant_kwargs={train_variant_kwargs}"
        )

    # -- prepare + tokenize --------------------------------------------------
    prepared = ApplyChatTemplate(transform=chat_template)(train_source_ref)
    tokenized = Tokenize(tokenizer=tokenizer, max_length=max_seq_length)(prepared)
    assert_labels_supervised(
        tokenized.data, context=f"{cfg.label} train split", max_fully_masked_frac=0.0
    )

    # -- fresh LoRA + train --------------------------------------------------
    base_model = load_base_model(base_model_name)
    training_config = TrainingConfig(
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        epochs=cfg.epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=(
            cfg.gradient_accumulation_steps
            if cfg.gradient_accumulation_steps is not None
            else gradient_accumulation_steps
        ),
        seed=cfg.train_seed,
        group=cfg.train_slug,
        run_id=cfg.adapter_idx,
        do_eval=False,
        max_seq_length=max_seq_length,
        bf16=bf16,
        logging_steps=logging_steps,
    )
    lora_config = LoraConfig(
        r=cfg.lora_rank,
        alpha=cfg.lora_alpha,
        dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        modules_to_save=None,
    )
    train_lora_adapter(
        base_model, tokenizer, tokenized.data, training_config, lora_config, out_dir
    )
    print(f"[{cfg.label}] trained adapter -> {out_dir}")

    # Free the training model before loading the eval model.
    del base_model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass

    # -- standalone eval on the SST2 test split (true/false) -----------------
    if cfg.eval_slug == cfg.train_slug:
        sst2_test = splits.test
    else:
        eval_typed = prepare_corpus(eval_task.corpus)
        eval_splits = task_splits(
            eval_typed, seed=prep_seed, max_train=max_train_prep, max_test=max_test_prep
        )
        sst2_test = eval_splits.test

    eval_model = load_adapter_model(out_dir, base_model=base_model_name)
    metrics = evaluate_on_sst2(
        eval_model,
        tokenizer,
        sst2_test,
        label_scheme=cfg.label_scheme,
        add_eos=add_eos,
        batch_size=eval_batch_size,
        max_length=max_seq_length,
    )
    print(f"[{cfg.label}] eval metrics: {metrics}")

    del eval_model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass

    # -- metadata row (matches meta_model/CONTRACT.md) -----------------------
    row: dict[str, Any] = {
        "__key__": cfg.label,
        "train_dataset": cfg.train_slug,
        "split": cfg.pool,
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "lora_rank": cfg.lora_rank,
        "lora_alpha": cfg.lora_alpha,
        "lora_dropout": cfg.lora_dropout,
        "epochs": cfg.epochs,
        "shards_total": cfg.shards_total,
        "label_noise": cfg.label_noise,
        "train_seed": cfg.train_seed,
    }
    for metric in BENCHMARK_METRICS:
        row[_benchmark_key(metric)] = metrics.get(metric)
    return row


def train_and_eval_chunk(
    cfgs: list[AdapterCfg], pool_dir: str | Path, **kwargs: Any
) -> list[dict[str, Any]]:
    """Run several adapters sequentially in ONE job (for ``chunked`` packing).

    A failing adapter is recorded (``error``) and does not abort the chunk.
    """
    results: list[dict[str, Any]] = []
    for cfg in cfgs:
        try:
            results.append(train_and_eval_adapter(cfg, pool_dir, **kwargs))
        except Exception as e:  # noqa: BLE001
            import traceback

            print(f"[chunk] {cfg.label} FAILED: {type(e).__name__}: {e}")
            results.append({
                "__key__": cfg.label,
                "train_dataset": cfg.train_slug,
                "split": cfg.pool,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            })
        gc.collect()
    return results
