"""Migrated from SRC/src/glad/meta_classifier/helpers.py."""

from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812


def collate_fn(batch: list) -> tuple[list, torch.Tensor]:
    """Collate SafetensorDataset items into a list of input dicts and a label tensor."""
    inputs = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    return inputs, labels


def batch_lora_dicts(
    dicts: list[dict[str, dict[Any, torch.Tensor]]],
) -> dict[str, dict[Any, torch.Tensor]]:
    """Batch a list of LoRA weight dicts by stacking tensors along a new batch dimension."""
    batched: dict[str, dict[Any, torch.Tensor]] = {}

    for key in dicts[0].keys():
        batched[key] = {}
        for lora_type in dicts[0][key].keys():
            tensors = [d[key][lora_type] for d in dicts]
            batched[key][lora_type] = torch.stack(tensors, dim=0)

    return batched


def one_hot_labels(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(labels.to(torch.long), num_classes=num_classes).float()


def forward_pass_and_loss(
    model: torch.nn.Module,
    inputs: list | dict,
    labels: torch.Tensor,
    criterion: torch.nn.Module,
    device: str,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Perform forward pass and compute loss."""
    if isinstance(inputs, list):
        inputs = batch_lora_dicts(inputs)
    outputs = model(inputs)
    labels_oh = one_hot_labels(labels, num_classes).to(device)
    return criterion(outputs, labels_oh), outputs


def should_early_stop(
    epoch: int,
    num_epochs: int,
    early_stop_count: int,
    patience: int = 5,
) -> bool:
    """Determine if early stopping should be triggered."""
    if epoch < num_epochs // 2:
        return False
    return (epoch + 1) == num_epochs or early_stop_count >= patience
