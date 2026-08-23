"""Migrated from SRC/src/glad/modules/flattened_input_classifier.py.

Flatten/unflatten utilities for nested LoRA dictionaries and a wrapper classifier.

Migrated from paretune (equivariant_classifiers/general.py).
"""

from __future__ import annotations

import torch
from torch import nn

from meta_model.lora.types import LoraType


def flatten_lora_dict(
    x: dict[str, dict[LoraType, torch.Tensor]],
) -> tuple[torch.Tensor, dict]:
    """Flatten nested LoRA dictionary into a single tensor.

    Args:
        x: Nested dictionary with structure {module_name: {LoraType: tensor}}.
           Tensors can be batched [batch, ...] or unbatched [...].

    Returns:
        Tuple of (flattened_tensor, metadata) where metadata contains information
        needed to reconstruct the original structure.
    """
    tensors = []
    metadata: dict = {"structure": [], "has_batch": True}

    # Check if inputs are batched by examining all tensor first dimensions
    sample_shapes = []
    for module_name in sorted(x.keys()):
        for lora_type in [LoraType.A, LoraType.B]:
            tensor = x[module_name][lora_type]
            sample_shapes.append(tensor.shape)

    # If all first dimensions match, likely batched
    first_dims = [s[0] for s in sample_shapes]
    has_batch = len(set(first_dims)) == 1 and len(sample_shapes) > 0
    metadata["has_batch"] = has_batch

    for module_name in sorted(x.keys()):
        for lora_type in [LoraType.A, LoraType.B]:
            tensor = x[module_name][lora_type]

            if has_batch:
                batch_size = tensor.size(0)
                tensors.append(tensor.reshape(batch_size, -1))
                metadata["structure"].append({
                    "module_name": module_name,
                    "lora_type": lora_type,
                    "shape": list(tensor.shape[1:]),  # Exclude batch dimension
                })
            else:
                tensors.append(tensor.reshape(-1).unsqueeze(0))  # Add batch dim
                metadata["structure"].append({
                    "module_name": module_name,
                    "lora_type": lora_type,
                    "shape": list(tensor.shape),  # Keep all dimensions
                })

    flattened = torch.cat(tensors, dim=1)
    return flattened, metadata


def unflatten_lora_dict(
    flattened: torch.Tensor,
    metadata: dict,
) -> dict[str, dict[LoraType, torch.Tensor]]:
    """Reconstruct nested LoRA dictionary from flattened tensor.

    Args:
        flattened: Flattened tensor from flatten_lora_dict.
        metadata: Metadata dictionary from flatten_lora_dict.

    Returns:
        Nested dictionary with structure {module_name: {LoraType: tensor}}.
    """
    result: dict[str, dict[LoraType, torch.Tensor]] = {}
    batch_size = flattened.size(0)
    current_idx = 0
    has_batch = metadata.get("has_batch", True)

    for item in metadata["structure"]:
        module_name = item["module_name"]
        lora_type = item["lora_type"]
        shape = item["shape"]

        # Calculate number of elements for this tensor
        num_elements = 1
        for dim in shape:
            num_elements *= dim

        # Extract and reshape tensor
        tensor_flat = flattened[:, current_idx : current_idx + num_elements]

        if has_batch:
            tensor = tensor_flat.reshape(batch_size, *shape)
        else:
            # Remove the batch dimension we added
            tensor = tensor_flat.reshape(*shape)

        # Add to result dict
        if module_name not in result:
            result[module_name] = {}
        result[module_name][lora_type] = tensor

        current_idx += num_elements

    return result


class FlattenedInputClassifier(nn.Module):
    """Wrapper that converts flat tensor input to nested dict for a classifier.

    This wrapper allows using a classifier with flat tensor inputs, which is useful
    for LRP analysis and other optimization tasks.
    """

    def __init__(
        self,
        classifier: nn.Module,
        metadata: dict,
    ) -> None:
        super().__init__()
        self.classifier = classifier
        self.metadata = metadata

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        unflattened = unflatten_lora_dict(x, self.metadata)
        return self.classifier(unflattened)

    def __getattr__(self, name: str):
        """Delegate attribute access to wrapped classifier when appropriate."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.classifier, name)
