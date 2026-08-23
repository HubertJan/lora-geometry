"""Likelihood (perplexity) classification scorer.

(migrated from src/llm_pipeline/evaluation/classify.py — the ``_evaluate_perplexity``
+ ``_build_eval_rows`` path; the fuzzy-generation path is dropped.)

Self-contained scorer: for each (row, class) it runs a forward pass, sums the
completion log-probs via a fused ``F.cross_entropy(reduction="none")`` kernel,
softmaxes across the classes to produce ``prob_<class>`` columns, and argmaxes
for the prediction. torch / pandas only.
"""

from __future__ import annotations

import sys
from typing import Any, Callable, NamedTuple, TypeVar

import pandas as pd
import torch
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from shared_adapter_pool.data.schemas import (
    PreparedCompletion,
    PreparedText,
    is_prepared_completion,
)
from shared_adapter_pool.data.transforms import tokenize_completion

_S = TypeVar("_S")


class _EvalRow(NamedTuple):
    """Internal representation of a single evaluation row."""

    row: dict
    is_poisoned: bool
    original_reference: str


def _build_eval_rows(
    data: Dataset,
    output_field: str,
    poison_fn: Callable[[Dataset], Dataset] | None = None,
) -> list[_EvalRow]:
    """Build the list of eval rows, including poisoned copies if applicable.

    Clean samples come first, then poisoned copies of eligible samples.
    """
    rows: list[_EvalRow] = []

    # Clean samples
    for idx in range(len(data)):
        row = {col: data[col][idx] for col in data.column_names}
        reference = row.get(output_field)
        if reference is None:
            raise ValueError(
                f"Missing reference label in row {idx} for output_field "
                f"{output_field!r}.  Row data: {row}"
            )
        rows.append(_EvalRow(row=row, is_poisoned=False, original_reference=reference))

    # Poisoned samples (if poison_fn is provided)
    if poison_fn is not None:
        poisoned_data = poison_fn(data)
        for idx in range(len(poisoned_data)):
            row = {col: poisoned_data[col][idx] for col in poisoned_data.column_names}
            original_ref = row.get("original", row.get(output_field))
            if original_ref is None:
                raise ValueError(
                    f"Missing original reference in row {idx} for output_field "
                    f"{output_field!r}.  Row data: {row}"
                )
            rows.append(
                _EvalRow(
                    row=row,
                    is_poisoned=True,
                    original_reference=str(original_ref),
                )
            )

    return rows


def _evaluate_perplexity(
    eval_rows: list[_EvalRow],
    classes: list[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    formatter: Callable[..., PreparedText | PreparedCompletion],
    output_field: str,
    batch_size: int,
    device: torch.device,
    max_length: int = 2048,
    length_normalize: bool = False,
) -> pd.DataFrame:
    """Perplexity-based classification: pick the class with highest log-prob.

    Batched implementation: enumerates ``len(eval_rows) * len(classes)`` items,
    runs them through the model in groups of ``batch_size`` with right-padding
    on ``input_ids``/``attention_mask``/``labels``, and gathers per-token log-
    probabilities in a single vectorised op per batch.  Right-padding is safe
    here because the attention mask zeroes padded positions out of self-
    attention and ``labels == -100`` zeroes their contribution to the gather.

    ``length_normalize=True`` divides each class's summed completion log-prob
    by its completion token count before the cross-class softmax (per-token
    average log-prob).  Default ``False`` keeps the historical raw-sum scoring.
    """
    model.eval()

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError(
            "Tokenizer has no pad_token_id; perplexity batching requires one. "
            "Ensure the tokenizer artifact has a pad token set."
        )

    n_rows = len(eval_rows)
    n_classes = len(classes)
    total_items = n_rows * n_classes
    n_batches = (total_items + batch_size - 1) // batch_size

    print(
        f"[perplexity] {n_rows} rows × {n_classes} classes = {total_items} items, "
        f"batch_size={batch_size}, max_length={max_length}, {n_batches} batches",
        flush=True,
    )

    # Cumulative log-probs per (row_idx, class_idx).  Stored on CPU as float64
    # for numerical stability when the per-batch float32 values are accumulated
    # into the final softmax across classes.
    log_prob_matrix = torch.zeros((n_rows, n_classes), dtype=torch.float64)

    progress = tqdm(
        total=total_items,
        desc="perplexity",
        unit="item",
        file=sys.stdout,
        mininterval=5.0,
        dynamic_ncols=True,
    )
    for batch_start in range(0, total_items, batch_size):
        batch_end = min(batch_start + batch_size, total_items)

        encodings: list[dict[str, list[int]]] = []
        item_coords: list[tuple[int, int]] = []
        for item_idx in range(batch_start, batch_end):
            row_idx, class_idx = divmod(item_idx, n_classes)
            cls = classes[class_idx]
            row_copy = dict(eval_rows[row_idx].row)
            row_copy[output_field] = cls
            formatted = formatter(row_copy)
            if not is_prepared_completion(formatted):
                raise ValueError(
                    "Perplexity evaluation requires a PreparedCompletion "
                    f"output from formatter(), got keys {sorted(formatted)!r}. "
                    "Ensure the output_field is set (not None) so the chat "
                    "template produces a prompt+completion pair."
                )
            encodings.append(
                tokenize_completion(
                    tokenizer=tokenizer,
                    prompt=formatted["prompt"],
                    completion=formatted["completion"],
                    max_length=max_length,
                    add_eos=formatted.get("add_eos", False),
                )
            )
            item_coords.append((row_idx, class_idx))

        max_len = max(len(e["input_ids"]) for e in encodings)
        b = len(encodings)

        input_ids = torch.full((b, max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((b, max_len), dtype=torch.long)
        labels = torch.full((b, max_len), -100, dtype=torch.long)
        for i, enc in enumerate(encodings):
            length = len(enc["input_ids"])
            input_ids[i, :length] = torch.tensor(enc["input_ids"], dtype=torch.long)
            attention_mask[i, :length] = torch.tensor(
                enc["attention_mask"], dtype=torch.long
            )
            labels[i, :length] = torch.tensor(enc["labels"], dtype=torch.long)

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Memory-frugal log-prob accumulation via fused softmax+NLL kernel
        # (``F.cross_entropy`` reduction='none').  This avoids materialising
        # any [B, T, V] intermediate.
        #
        # Shift labels left by one position (label at slot i becomes the
        # target predicted by the logit at slot i), pad the last slot with
        # -100 so cross_entropy ignores it.
        b_dim, t_dim, v_dim = outputs.logits.shape
        shifted_labels = torch.full(
            (b_dim, t_dim), -100, dtype=torch.long, device=labels.device
        )
        shifted_labels[:, :-1] = labels[:, 1:]
        nll = torch.nn.functional.cross_entropy(
            outputs.logits.view(-1, v_dim),
            shifted_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(b_dim, t_dim)                                       # [B, T]
        sums = (-nll).sum(dim=1).double().cpu()                    # [B]
        if length_normalize:
            counts = (shifted_labels != -100).sum(dim=1).cpu().clamp(min=1)
            sums = sums / counts

        for (row_idx, class_idx), val in zip(item_coords, sums.tolist()):
            log_prob_matrix[row_idx, class_idx] = val

        progress.update(batch_end - batch_start)
    progress.close()

    probs_matrix = torch.softmax(log_prob_matrix, dim=1).numpy()  # [N, n_classes]
    pred_indices = probs_matrix.argmax(axis=1)

    results: list[dict[str, Any]] = []
    for row_idx, eval_row in enumerate(eval_rows):
        reference = eval_row.row.get(output_field)
        if reference is None:
            raise ValueError(
                f"Missing reference label in row {row_idx} for output_field "
                f"{output_field!r}.  Row data: {eval_row.row}"
            )
        result: dict[str, Any] = {
            "index": row_idx,
            "reference": reference,
            "original_reference": eval_row.original_reference,
            "is_poisoned": eval_row.is_poisoned,
        }
        for c_idx, cls in enumerate(classes):
            result[f"prob_{cls}"] = float(probs_matrix[row_idx, c_idx])
        prediction = classes[int(pred_indices[row_idx])]
        result["prediction"] = prediction
        result["correct"] = prediction == reference
        results.append(result)

    return pd.DataFrame(results)
