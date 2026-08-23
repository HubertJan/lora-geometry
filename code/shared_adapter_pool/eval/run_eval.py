"""Glue: score a trained adapter on the SST2 test split -> the six metrics.

(migrated from the standalone-eval assembly in
src/discoveries/sst2_perf_prediction/flows/train_eval_adapter.py, using the
migrated classify + metrics modules.)

``evaluate_on_sst2`` renders the SST2 test rows with the true/false verbalizer,
runs the likelihood classifier, and returns the six metric values keyed to the
CONTRACT column suffixes (accuracy, f1_macro, auroc, brier, mean_confidence, nll).
"""

from __future__ import annotations

from typing import Any

from shared_adapter_pool.data.definitions._variants import resolve_variant_choices
from shared_adapter_pool.eval.classify import _build_eval_rows, _evaluate_perplexity
from shared_adapter_pool.eval.metrics import compute_metrics


def build_sst2_eval_template(label_scheme: str = "TRUE_FALSE_V1", *, add_eos: bool = False):
    """The SST2 chat template configured with the given label scheme."""
    from shared_adapter_pool.data.definitions.sst2 import DEFINITION

    variant_kwargs = resolve_variant_choices(
        DEFINITION.chat_template, {"label_scheme": label_scheme}
    )
    return DEFINITION.chat_template(add_eos=add_eos, **variant_kwargs)


def evaluate_on_sst2(
    model: Any,
    tokenizer: Any,
    sst2_test_ds: Any,
    *,
    label_scheme: str = "TRUE_FALSE_V1",
    add_eos: bool = False,
    batch_size: int = 8,
    max_length: int = 2048,
) -> dict[str, float]:
    """Score ``model`` on ``sst2_test_ds`` -> {accuracy, f1_macro, auroc, brier,
    mean_confidence, nll}.

    ``sst2_test_ds`` is the SST2 test split (a ``TypedDataset`` or a bare HF
    ``Dataset``) whose ``sentiment`` column holds the canonical label strings
    ("negative"/"positive"). The chat template maps each canonical label to its
    true/false surface word for scoring.
    """
    from shared_adapter_pool.data.definitions.sst2 import DEFINITION

    output_field = DEFINITION.label_field  # "sentiment"
    classes = list(DEFINITION.label_values)  # ("negative", "positive")

    chat_template = build_sst2_eval_template(label_scheme, add_eos=add_eos)

    data = getattr(sst2_test_ds, "data", sst2_test_ds)
    device = next(model.parameters()).device

    eval_rows = _build_eval_rows(data, output_field)
    df = _evaluate_perplexity(
        eval_rows=eval_rows,
        classes=classes,
        model=model,
        tokenizer=tokenizer,
        formatter=chat_template.apply,
        output_field=output_field,
        batch_size=batch_size,
        device=device,
        max_length=max_length,
    )

    metrics = compute_metrics(df, categories=classes)
    # Return exactly the contract's six keys (auroc present for this binary task).
    keys = ["accuracy", "f1_macro", "auroc", "brier", "mean_confidence", "nll"]
    return {k: float(metrics[k]) for k in keys if k in metrics}
