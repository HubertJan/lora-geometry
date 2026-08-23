"""Migrated from SRC/src/glad/meta_classifier/heads.py.

Heterogeneous output heads for the meta classifier.

A single :class:`MetaTargetSpec` is the source of truth for *both* the model
head (its type + output dimension) *and* how the per-adapter target is read out
of a pool metadata row.  Three head families are supported:

* ``BOUNDED_REGRESSION``   — target in ``[0, 1]`` (e.g. accuracy / f1 / pass@k).
  Output dim 1, sigmoid activation, MSE loss on the squashed output.
* ``UNBOUNDED_REGRESSION`` — target in ``[0, +inf)`` (e.g. loss).  Output dim 1,
  softplus activation (keeps predictions non-negative), MSE loss.
* ``CLASSIFICATION``       — ``num_classes`` logits, cross-entropy loss.

Targets are allowed to be *missing* on a per-row basis (different pools carry
different benchmarks): regression targets use ``NaN`` and classification targets
use ``-1`` as the "missing" sentinel.  :func:`compute_multihead_loss` masks
those rows out per head, so a batch that only has some targets still trains the
heads it does have.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch
import torch.nn.functional as F  # noqa: N812

# Sentinel for a missing classification target (regression uses float NaN).
MISSING_CLASS_LABEL = -1


class MetaHeadType(StrEnum):
    """The kind of prediction a head produces."""

    BOUNDED_REGRESSION = "bounded_regression"
    UNBOUNDED_REGRESSION = "unbounded_regression"
    CLASSIFICATION = "classification"


@dataclass(frozen=True)
class MetaTargetSpec:
    """Describes one prediction head and how to extract its target.

    ``mapping`` is a tuple of ``(string, int)`` pairs (rather than a dict) so the
    spec is hashable and its fingerprint is order-canonical — it participated in
    the run-cache fingerprint of the original training flow.
    """

    name: str
    column: str
    type: MetaHeadType
    num_classes: int = 1
    mapping: tuple[tuple[str, int], ...] | None = None
    loss_weight: float = 1.0

    @property
    def is_classification(self) -> bool:
        return self.type is MetaHeadType.CLASSIFICATION

    @property
    def output_dim(self) -> int:
        return self.num_classes if self.is_classification else 1

    @property
    def mapping_dict(self) -> dict[str, int]:
        if self.mapping is None:
            return {}
        return dict(self.mapping)

    def to_dict(self) -> dict:
        """Plain JSON-able representation for persistence / W&B config."""
        return {
            "name": self.name,
            "column": self.column,
            "type": self.type.value,
            "num_classes": self.num_classes,
            "mapping": list(self.mapping) if self.mapping is not None else None,
            "loss_weight": self.loss_weight,
        }

    @classmethod
    def from_dict(cls, data: dict) -> MetaTargetSpec:
        mapping = data.get("mapping")
        return cls(
            name=data["name"],
            column=data["column"],
            type=MetaHeadType(data["type"]),
            num_classes=int(data.get("num_classes", 1)),
            mapping=(
                tuple((str(k), int(v)) for k, v in mapping)
                if mapping is not None
                else None
            ),
            loss_weight=float(data.get("loss_weight", 1.0)),
        )


def _head_specs_from_dicts(dicts: list[dict]) -> list[MetaTargetSpec]:
    return [MetaTargetSpec.from_dict(d) for d in dicts]


def _regression_activation(head_type: MetaHeadType, logits: torch.Tensor) -> torch.Tensor:
    """Squash a (batch, 1) regression logit into the head's value range."""
    flat = logits.squeeze(-1)
    if head_type is MetaHeadType.BOUNDED_REGRESSION:
        return torch.sigmoid(flat)
    if head_type is MetaHeadType.UNBOUNDED_REGRESSION:
        return F.softplus(flat)
    msg = f"{head_type} is not a regression head"
    raise ValueError(msg)


def compute_multihead_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    specs: list[MetaTargetSpec],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Weighted sum of per-head losses, masking missing targets.

    Returns the total loss tensor (for ``.backward()``) and a dict of per-head
    scalar losses (for logging).  Heads whose targets are entirely missing in
    this batch contribute nothing and are absent from the per-head dict.
    """
    device = next(iter(outputs.values())).device
    total = torch.zeros((), device=device)
    per_head: dict[str, float] = {}

    for spec in specs:
        logits = outputs[spec.name]
        target = targets[spec.name].to(device)

        if spec.is_classification:
            mask = target != MISSING_CLASS_LABEL
            if not bool(mask.any()):
                continue
            head_loss = F.cross_entropy(logits[mask], target[mask].long())
        else:
            target = target.float()
            mask = ~torch.isnan(target)
            if not bool(mask.any()):
                continue
            preds = _regression_activation(spec.type, logits)
            head_loss = F.mse_loss(preds[mask], target[mask])

        total = total + spec.loss_weight * head_loss
        per_head[spec.name] = float(head_loss.detach().cpu())

    return total, per_head


def head_predictions(
    outputs: dict[str, torch.Tensor],
    specs: list[MetaTargetSpec],
) -> dict[str, torch.Tensor]:
    """Map raw head logits to predictions in each head's natural space.

    * regression heads → the squashed scalar value (sigmoid / softplus),
    * classification heads → the integer ``argmax`` class index.
    """
    preds: dict[str, torch.Tensor] = {}
    for spec in specs:
        logits = outputs[spec.name]
        if spec.is_classification:
            preds[spec.name] = logits.argmax(dim=-1)
        else:
            preds[spec.name] = _regression_activation(spec.type, logits)
    return preds


def head_class_scores(
    outputs: dict[str, torch.Tensor],
    specs: list[MetaTargetSpec],
) -> dict[str, torch.Tensor]:
    """Softmax probabilities for classification heads (used for AUC etc.)."""
    scores: dict[str, torch.Tensor] = {}
    for spec in specs:
        if spec.is_classification:
            scores[spec.name] = torch.softmax(outputs[spec.name], dim=-1)
    return scores
