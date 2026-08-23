"""Migrated from SRC/src/glad/meta_classifier/metrics.py."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, TypedDict

import numpy as np
from scipy.stats import pearsonr, spearmanr

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader

    from meta_model.heads import MetaTargetSpec
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


class RegressionMetricMeasurements(TypedDict):
    loss: float
    mse: float
    mae: float
    r2: float
    pearson: float
    spearman: float
    count: int


def calculate_regression_metrics(
    predictions: list[float],
    targets: list[float],
    loss: float,
) -> RegressionMetricMeasurements:
    """Regression metrics over already-NaN-filtered prediction/target pairs.

    ``r2`` / ``pearson`` / ``spearman`` are undefined for fewer than two points
    or zero-variance targets; those degenerate cases return ``0.0`` rather than
    raising.
    """
    preds = np.asarray(predictions, dtype=float)
    tgts = np.asarray(targets, dtype=float)
    count = int(preds.size)

    if count == 0:
        return {
            "loss": loss,
            "mse": 0.0,
            "mae": 0.0,
            "r2": 0.0,
            "pearson": 0.0,
            "spearman": 0.0,
            "count": 0,
        }

    mse = float(mean_squared_error(tgts, preds))
    mae = float(mean_absolute_error(tgts, preds))

    enough = count >= 2
    target_varies = bool(np.std(tgts) > 0)
    pred_varies = bool(np.std(preds) > 0)
    correlatable = enough and target_varies and pred_varies

    r2 = float(r2_score(tgts, preds)) if enough and target_varies else 0.0
    pearson = float(pearsonr(preds, tgts)[0]) if correlatable else 0.0
    spearman = float(spearmanr(preds, tgts).statistic) if correlatable else 0.0

    return {
        "loss": loss,
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "pearson": pearson if not np.isnan(pearson) else 0.0,
        "spearman": spearman if not np.isnan(spearman) else 0.0,
        "count": count,
    }


class StandardMetricMeasurements(TypedDict):
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    negative_precision: float
    negative_recall: float
    negative_f1: float
    auc: float


def calculate_classification_metrics(
    predictions: list[bool] | list[int],
    labels: list[bool] | list[int],
    scores: list[float] | list[list[float]],
    loss: float,
    accuracy: float,
    num_classes: int | None = None,
) -> StandardMetricMeasurements:
    """Calculate classification metrics from predictions and labels.

    Supports both binary (bool lists, scalar scores) and multi-class
    (int lists, per-class probability lists) classification.

    ``num_classes`` makes the binary/multi-class decision explicit; when it is
    ``None`` the number of classes is inferred from the observed labels and
    predictions (preserving the historical behaviour).  Passing it avoids
    mis-classifying a >2-class head as binary when a small batch happens to
    contain only two distinct classes.
    """
    if num_classes is not None:
        is_multiclass = num_classes > 2
    else:
        unique_classes = sorted(set(labels) | set(predictions))
        is_multiclass = len(unique_classes) > 2

    if is_multiclass:
        precision = float(precision_score(labels, predictions, average="macro", zero_division=0))
        recall = float(recall_score(labels, predictions, average="macro", zero_division=0))
        f1 = float(f1_score(labels, predictions, average="macro", zero_division=0))
        scores_array = np.array(scores)
        try:
            auc = float(roc_auc_score(labels, scores_array, multi_class="ovr", average="macro"))
        except ValueError:
            auc = 0.0
        return {
            "loss": loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "negative_precision": 0.0,
            "negative_recall": 0.0,
            "negative_f1": 0.0,
            "auc": auc,
        }

    precision = float(precision_score(labels, predictions, labels=[True, False]))
    recall = float(recall_score(labels, predictions, labels=[True, False]))
    f1 = float(f1_score(labels, predictions, labels=[True, False]))
    try:
        auc = float(roc_auc_score(labels, scores))
    except ValueError:
        auc = 0.0
    negative_precision = float(
        precision_score(labels, predictions, labels=[True, False], pos_label=False),
    )
    negative_recall = float(
        recall_score(labels, predictions, labels=[True, False], pos_label=False),
    )
    negative_f1 = float(
        f1_score(labels, predictions, labels=[True, False], pos_label=False),
    )

    return {
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "negative_precision": negative_precision,
        "negative_recall": negative_recall,
        "negative_f1": negative_f1,
        "auc": auc,
    }


def evaluate_multihead_performance(
    loaders: "Mapping[str, DataLoader]",
    model: "torch.nn.Module",
    specs: "list[MetaTargetSpec]",
    device: str,
) -> dict[str, dict[str, dict]]:
    """Per-loader, per-head metrics for a multi-head meta classifier.

    Each loader must yield ``(batched_input_dict, target_dict)`` pairs (as built
    by ``multihead_collate_fn``).  Missing targets (``NaN`` for regression,
    ``-1`` for classification) are masked out before metrics are computed.

    Returns ``{loader_name: {head_name: metric_dict}}`` where ``metric_dict`` is
    a :class:`StandardMetricMeasurements` for classification heads and a
    :class:`RegressionMetricMeasurements` for regression heads, each extended
    with ``count`` (rows that entered the metrics) and ``n_missing`` (rows
    masked out by the missing sentinel) — so a pool that silently shrank is
    visible in the logged metrics instead of leaving zero trace.
    """
    import torch

    from meta_model.heads import (
        MISSING_CLASS_LABEL,
        compute_multihead_loss,
        head_class_scores,
        head_predictions,
    )

    model.eval()
    results: dict[str, dict[str, dict]] = {}

    with torch.no_grad():
        for loader_name, loader in loaders.items():
            preds_acc: dict[str, list] = {s.name: [] for s in specs}
            tgts_acc: dict[str, list] = {s.name: [] for s in specs}
            scores_acc: dict[str, list] = {
                s.name: [] for s in specs if s.is_classification
            }
            loss_sum: dict[str, float] = {s.name: 0.0 for s in specs}
            valid_count: dict[str, int] = {s.name: 0 for s in specs}
            missing_count: dict[str, int] = {s.name: 0 for s in specs}

            for inputs, targets in loader:
                outputs = model(inputs)
                _, per_head_loss = compute_multihead_loss(outputs, targets, specs)
                preds = head_predictions(outputs, specs)
                cls_scores = head_class_scores(outputs, specs)

                for spec in specs:
                    target = targets[spec.name]
                    if spec.is_classification:
                        mask = target != MISSING_CLASS_LABEL
                    else:
                        target = target.float()
                        mask = ~torch.isnan(target)
                    n_valid = int(mask.sum())
                    missing_count[spec.name] += int(mask.numel()) - n_valid
                    if n_valid == 0:
                        continue

                    mask_cpu = mask.cpu()
                    preds_acc[spec.name].extend(
                        preds[spec.name].cpu()[mask_cpu].tolist()
                    )
                    tgts_acc[spec.name].extend(target.cpu()[mask_cpu].tolist())
                    if spec.is_classification:
                        scores_acc[spec.name].extend(
                            cls_scores[spec.name].cpu()[mask_cpu].tolist()
                        )
                    loss_sum[spec.name] += per_head_loss.get(spec.name, 0.0) * n_valid
                    valid_count[spec.name] += n_valid

            head_results: dict[str, dict] = {}
            for spec in specs:
                count = valid_count[spec.name]
                avg_loss = loss_sum[spec.name] / count if count else 0.0
                if count == 0:
                    head_results[spec.name] = {
                        "loss": 0.0,
                        "count": 0,
                        "n_missing": missing_count[spec.name],
                    }
                    continue

                if spec.is_classification:
                    labels = [int(x) for x in tgts_acc[spec.name]]
                    predictions = [int(x) for x in preds_acc[spec.name]]
                    raw_scores = scores_acc[spec.name]
                    # calculate_classification_metrics expects scalar pos-class
                    # scores for the binary case and per-class rows otherwise.
                    if spec.num_classes == 2:
                        scores: list = [row[1] for row in raw_scores]
                    else:
                        scores = raw_scores
                    accuracy = float(
                        np.mean(
                            [p == t for p, t in zip(predictions, labels, strict=True)]
                        )
                    )
                    head_results[spec.name] = dict(
                        calculate_classification_metrics(
                            predictions,
                            labels,
                            scores,
                            avg_loss,
                            accuracy,
                            num_classes=spec.num_classes,
                        )
                    )
                    head_results[spec.name]["count"] = count
                else:
                    head_results[spec.name] = dict(
                        calculate_regression_metrics(
                            preds_acc[spec.name], tgts_acc[spec.name], avg_loss
                        )
                    )
                head_results[spec.name]["n_missing"] = missing_count[spec.name]

            results[loader_name] = head_results

    return results


def predict_per_adapter(
    loader: "DataLoader",
    model: "torch.nn.Module",
    specs: "list[MetaTargetSpec]",
    device: str,
) -> list[dict]:
    """Per-adapter (per-row) predictions for a multi-head meta classifier.

    Runs *model* over *loader* and returns one record per adapter, in the order
    the loader yields them.  The loader MUST be built with ``shuffle=False`` so
    that the returned list lines up positionally with the source metadata
    DataFrame — callers join on row index to attach adapter identity (qname,
    task, trigger, ...).

    Each record carries, for every head ``<h>``:

    * ``<h>.target`` — ground-truth target (int class / float value), with the
      missing sentinel (``-1`` / ``NaN``) preserved so callers can drop it,
    * ``<h>.pred``   — predicted class index (classification) or regressed value,
    * ``<h>.score``  — positive-class probability for binary heads
      (``num_classes == 2``), the full per-class probability list for multiclass
      heads, absent for regression heads.

    Unlike :func:`evaluate_multihead_performance` (which only returns pooled
    metrics), this keeps every adapter's raw prediction so the full eval table
    can be reported / logged.
    """
    import torch

    from meta_model.heads import head_class_scores, head_predictions

    model.eval()
    records: list[dict] = []
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)
            preds = head_predictions(outputs, specs)
            cls_scores = head_class_scores(outputs, specs)
            batch_n = int(next(iter(targets.values())).size(0))
            for i in range(batch_n):
                rec: dict = {}
                for spec in specs:
                    rec[f"{spec.name}.target"] = targets[spec.name][i].item()
                    rec[f"{spec.name}.pred"] = preds[spec.name][i].item()
                    if spec.is_classification:
                        row_scores = cls_scores[spec.name][i].cpu().tolist()
                        rec[f"{spec.name}.score"] = (
                            row_scores[1] if spec.num_classes == 2 else row_scores
                        )
                records.append(rec)
    return records
