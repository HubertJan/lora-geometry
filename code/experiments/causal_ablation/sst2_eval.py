"""Self-contained SST2 likelihood-accuracy evaluator (torch-only fallback).

Fallback for ``shared_adapter_pool.eval.run_eval.evaluate_on_sst2`` — that exact
API (and ``shared_adapter_pool.data.definitions.registry.TASKS["sst2"]``) is not
present in the migrated ``shared_adapter_pool`` package, so this module reproduces
the likelihood-classification scoring locally:

For each SST2 validation sentence we build the TRUE_FALSE_V1 prompt (the exact
system prompt + ``Sentence: ...\\nResponse: `` layout of the pool's chat template),
tokenize ``prompt + " true"`` and ``prompt + " false"`` continuations, sum the
per-token log-likelihood of the continuation tokens under the model, softmax
across the two classes, argmax and compare to the gold sentiment. Accuracy is the
mean of correct predictions. torch (+ HF datasets for the split) only.

The label words / prompt text are taken verbatim from
``shared_adapter_pool.data.definitions.sst2`` so the un-ablated baseline matches
the pool's stored ``benchmark.sst2-test.likelihood.accuracy``.
"""

from __future__ import annotations

from typing import Any

import torch

# TRUE_FALSE_V1 verbalizer: POSITIVE -> "true", NEGATIVE -> "false".
POS_WORD, NEG_WORD = "true", "false"
CLASSES = [NEG_WORD, POS_WORD]  # index 0 = negative, 1 = positive (gold label int)


def _system_prompt() -> str:
    """The DEFAULT_V1 system prompt rendered with the TRUE_FALSE label words."""
    from shared_adapter_pool.data.definitions.sst2 import (
        Sst2LabelScheme,
        Sst2SystemPrompt,
        render_system_prompt,
    )

    return render_system_prompt(
        Sst2SystemPrompt.DEFAULT_V1, Sst2LabelScheme.TRUE_FALSE_V1
    )


def build_prompt(sentence: str) -> str:
    """Prompt string up to (and including) the ``Response: `` cue."""
    return f"{_system_prompt()}\n\nSentence: {sentence}\nResponse: "


def load_sst2_test(max_test: int | None = None) -> tuple[list[str], list[int]]:
    """SST2 validation split (labeled) as (sentences, gold_label_ints)."""
    from datasets import load_dataset

    ds = load_dataset("stanfordnlp/sst2", split="validation")
    sents = [r["sentence"] for r in ds]
    labels = [int(r["label"]) for r in ds]
    if max_test is not None:
        sents, labels = sents[:max_test], labels[:max_test]
    return sents, labels


@torch.no_grad()
def _completion_logprob(
    model: Any, tokenizer: Any, prompt: str, completion: str, device: str
) -> float:
    """Summed log-prob of ``completion`` tokens given ``prompt`` under the model."""
    prompt_ids = tokenizer(prompt, add_special_tokens=True).input_ids
    full_ids = tokenizer(prompt + completion, add_special_tokens=True).input_ids
    comp_start = len(prompt_ids)
    input_ids = torch.tensor([full_ids], device=device)
    logits = model(input_ids=input_ids).logits[0].float()  # (T, V)
    logprobs = torch.log_softmax(logits, dim=-1)
    total = 0.0
    # token at position t is predicted by the logits at position t-1.
    for t in range(comp_start, len(full_ids)):
        total += float(logprobs[t - 1, full_ids[t]])
    return total


@torch.no_grad()
def evaluate_sst2(
    model: Any,
    tokenizer: Any,
    sentences: list[str],
    labels: list[int],
    *,
    device: str | None = None,
) -> dict[str, float]:
    """Likelihood-classification accuracy of ``model`` on SST2 (TRUE_FALSE_V1).

    Returns ``{"accuracy": ..., "n": ...}``.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    correct = 0
    for sent, gold in zip(sentences, labels):
        prompt = build_prompt(sent)
        lp = [
            _completion_logprob(model, tokenizer, prompt, " " + cls, device)
            for cls in CLASSES
        ]
        pred = int(torch.tensor(lp).argmax().item())  # 0=neg, 1=pos
        correct += int(pred == gold)
    n = len(labels)
    return {"accuracy": correct / n if n else float("nan"), "n": float(n)}
