"""Migrated from SRC/src/glad/meta_classifier/config.py.

Configuration for EquivariantLoRAMetaClassifier.

Migrated from experiments/meta_classifier_training/equivariant_lora_meta_classifier_config.py
with the paretune ModelConfig ABC dropped in favour of a standalone class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from meta_model.lora.model_sizes import TARGET_MODULE_SIZES_BY_LLM_MODEL
from meta_model.lora.types import ActivationFunction, LLMModel, TargetModuleType
from meta_model.equivariant_lora_meta_classifier import (
    EquivariantLoRAMetaClassifier,
)
from meta_model.heads import MetaHeadType, MetaTargetSpec

if TYPE_CHECKING:
    import torch


class ModelConfig(ABC):
    """Abstract base class for model configurations (replaces paretune.ModelConfig)."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    def to_dict(self) -> dict:
        return {"name": self.name}

    @abstractmethod
    def create_model(self, device: str) -> torch.nn.Module: ...


class EquivariantLoRAMetaClassifierConfig(ModelConfig):
    """Configuration for EquivariantLoRAMetaClassifier with multi-head support."""

    equivariant_layer_sizes: list[int]
    head_layer_sizes: list[int]
    target_modules: set[TargetModuleType]
    llm_layer_count: int
    llm_model: LLMModel
    head_specs: list[MetaTargetSpec]
    equivariant_layers_activation: ActivationFunction
    seed: int | None

    def __init__(
        self,
        equivariant_layer_sizes: list[int],
        head_layer_sizes: list[int],
        equivariant_layers_activation: ActivationFunction,
        target_modules: set[TargetModuleType],
        llm_layer_count: int,
        llm_model: LLMModel,
        head_specs: list[MetaTargetSpec] | None = None,
        num_classes: int | None = None,
        seed: int | None = None,
    ) -> None:
        head_specs = self._resolve_head_specs(head_specs, num_classes)
        self.equivariant_layer_sizes = equivariant_layer_sizes
        self.head_layer_sizes = head_layer_sizes
        self.equivariant_layers_activation = equivariant_layers_activation
        self.target_modules = target_modules
        self.llm_layer_count = llm_layer_count
        self.llm_model = llm_model
        self.head_specs = head_specs
        self.seed = seed

    @staticmethod
    def _resolve_head_specs(
        head_specs: list[MetaTargetSpec] | None,
        num_classes: int | None,
    ) -> list[MetaTargetSpec]:
        """Resolve heads from explicit specs or the legacy ``num_classes`` arg.

        Passing ``num_classes`` (and no ``head_specs``) builds a single
        classification head named ``"label"`` — this keeps pre-multi-head
        callers (e.g. ``standard_sizes(..., num_classes=2)``) working.
        """
        if head_specs:
            return head_specs
        if num_classes is not None:
            return [
                MetaTargetSpec(
                    name="label",
                    column="label",
                    type=MetaHeadType.CLASSIFICATION,
                    num_classes=num_classes,
                )
            ]
        msg = "Either head_specs or num_classes must be provided"
        raise ValueError(msg)

    @property
    def name(self) -> str:
        return "equivariant_lora_meta_classifier"

    @property
    def num_classes(self) -> int:
        """Class count of the sole classification head (legacy single-head API).

        Raises if the config is genuinely multi-head, so callers that assume a
        single classification output fail loudly rather than silently.
        """
        classification_heads = [
            s for s in self.head_specs if s.type is MetaHeadType.CLASSIFICATION
        ]
        if len(self.head_specs) == 1 and len(classification_heads) == 1:
            return classification_heads[0].num_classes
        msg = (
            "num_classes is only defined for a single-classification-head config; "
            f"this config has heads {[s.name for s in self.head_specs]}"
        )
        raise ValueError(msg)

    def to_dict(self) -> dict:
        return {
            "equivariant_layer_sizes": [str(s) for s in self.equivariant_layer_sizes],
            "head_layer_sizes": [str(s) for s in self.head_layer_sizes],
            "equivariant_layers_activation": str(self.equivariant_layers_activation),
            "target_modules": sorted([str(t) for t in self.target_modules]),
            "llm_layer_count": self.llm_layer_count,
            "llm_model": str(self.llm_model),
            "head_specs": [spec.to_dict() for spec in self.head_specs],
            "seed": str(self.seed) if self.seed is not None else None,
            **super().to_dict(),
        }

    def create_model(self, device: str) -> torch.nn.Module:
        return EquivariantLoRAMetaClassifier(
            equivariant_layer_sizes=self.equivariant_layer_sizes,
            head_layer_sizes=self.head_layer_sizes,
            targeted_modules=self.target_modules,
            target_module_sizes=TARGET_MODULE_SIZES_BY_LLM_MODEL[self.llm_model],
            head_specs=self.head_specs,
            equivariant_layers_activation=self.equivariant_layers_activation,
            llm_layers=self.llm_layer_count,
            device=device,
            seed=self.seed,
        )

    @classmethod
    def standard_sizes(
        cls,
        target_modules: set[TargetModuleType],
        llm_layer_count: int,
        equivariant_layers_activation: ActivationFunction,
        llm_model: LLMModel,
        head_specs: list[MetaTargetSpec] | None = None,
        num_classes: int | None = None,
        seed: int | None = None,
    ) -> EquivariantLoRAMetaClassifierConfig:
        return cls(
            equivariant_layer_sizes=[128],
            head_layer_sizes=[64],
            equivariant_layers_activation=equivariant_layers_activation,
            target_modules=target_modules,
            llm_layer_count=llm_layer_count,
            llm_model=llm_model,
            head_specs=head_specs,
            num_classes=num_classes,
            seed=seed,
        )

    @classmethod
    def single_classification(
        cls,
        target_modules: set[TargetModuleType],
        llm_layer_count: int,
        equivariant_layers_activation: ActivationFunction,
        llm_model: LLMModel,
        num_classes: int = 2,
        *,
        name: str = "label",
        column: str = "label",
        seed: int | None = None,
    ) -> EquivariantLoRAMetaClassifierConfig:
        """Convenience builder reproducing the old single-head classifier."""
        return cls.standard_sizes(
            target_modules=target_modules,
            llm_layer_count=llm_layer_count,
            equivariant_layers_activation=equivariant_layers_activation,
            llm_model=llm_model,
            head_specs=[
                MetaTargetSpec(
                    name=name,
                    column=column,
                    type=MetaHeadType.CLASSIFICATION,
                    num_classes=num_classes,
                )
            ],
            seed=seed,
        )
