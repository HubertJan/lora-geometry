"""The six likelihood-classification metrics.

(migrated from src/llm_pipeline/evaluation/methods/likelihood_classification.py
::LikelihoodClassificationMethod._compute_metrics + metric_names)

Given the per-row predictions DataFrame produced by ``eval/classify.py`` (the
``prob_{class}`` / ``prediction`` / ``reference`` / ``correct`` columns) and the
ordered class list, computes accuracy, f1_macro (sklearn macro), nll, brier
(0.5·Σ(p−onehot)², normalized to [0,1]), mean_confidence, and auroc (binary
only). Formulas copied verbatim from the original method.
"""

from __future__ import annotations

import pandas as pd


def metric_names() -> list[str]:
    """The metric keys this module can emit (auroc only for binary tasks)."""
    return ["accuracy", "f1_macro", "nll", "brier", "auroc", "mean_confidence"]


def compute_metrics(df: pd.DataFrame, categories: list) -> dict[str, float]:
    """Compute the six metrics from a predictions DataFrame + ordered categories."""
    import numpy as np
    from sklearn.metrics import f1_score, roc_auc_score

    metrics: dict[str, float] = {}

    clean = df[~df["is_poisoned"]]
    if len(clean) == 0:
        return metrics

    metrics["accuracy"] = float(clean["correct"].mean())
    metrics["f1_macro"] = float(
        f1_score(
            clean["reference"],
            clean["prediction"],
            labels=categories,
            average="macro",
            zero_division=0,
        )
    )

    # Probability-based metrics, derived from the per-row ``prob_{class}``
    # columns aligned to ``categories``. ``reference`` holds a category value;
    # map it to a column index to locate the gold prob.
    categories_str = [str(c) for c in categories]
    prob_cols = [f"prob_{c}" for c in categories_str]
    cat_index = {c: i for i, c in enumerate(categories_str)}
    gold_idx = np.array(
        [cat_index.get(str(r), -1) for r in clean["reference"]]
    )
    valid = gold_idx >= 0
    if valid.any():
        probs = clean[prob_cols].to_numpy(dtype=float)[valid]
        gi = gold_idx[valid]
        n = probs.shape[0]
        p_gold = probs[np.arange(n), gi]

        # Test loss: mean negative log-likelihood of the gold class.
        metrics["nll"] = float(-np.log(np.clip(p_gold, 1e-12, 1.0)).mean())

        # Multiclass Brier score, normalised to [0, 1] (max raw value is 2).
        onehot = np.zeros_like(probs)
        onehot[np.arange(n), gi] = 1.0
        metrics["brier"] = float(0.5 * ((probs - onehot) ** 2).sum(1).mean())

        # Mean top-class confidence.
        metrics["mean_confidence"] = float(probs.max(axis=1).mean())

        # AUROC: threshold-free ranking, binary tasks only. Requires both
        # classes present among the gold labels.
        if len(categories_str) == 2:
            y_true = (gi == 1).astype(int)
            if y_true.min() != y_true.max():
                try:
                    metrics["auroc"] = float(roc_auc_score(y_true, probs[:, 1]))
                except ValueError:
                    pass

    return metrics
