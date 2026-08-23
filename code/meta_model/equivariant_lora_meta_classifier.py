"""Migrated from SRC/src/glad/meta_classifier/equivariant_lora_meta_classifier.py.

Equivariant LoRA meta classifier for backdoor detection.

Migrated from experiments/meta_classifier_training/equivariant_lora_meta_classifier.py
with all paretune imports replaced by glad equivalents.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Self

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from meta_model.lora.types import ActivationFunction, LoraType, TargetModuleType
from meta_model.heads import MetaHeadType, MetaTargetSpec
from meta_model.modules.activations import gl_activation
from meta_model.modules.equivariant_linear_layer import EquivariantLinearLayer
from meta_model.modules.mat_mul import MatMul

# Expected tensor dimensions: (batch_size, layer_count, width, length)
EXPECTED_TENSOR_DIMS = 4

# Named tensor dimension labels for readability
BATCH = "batch"
LAYER = "layer"
WIDTH = "width"
LENGTH = "length"
FEATURES = "features"


class EquivariantLoRAMetaClassifier(nn.Module):
    def __init__(
        self,
        equivariant_layer_sizes: list[int],
        head_layer_sizes: list[int],
        targeted_modules: set[TargetModuleType],
        target_module_sizes: dict[TargetModuleType, tuple[int, int]],
        head_specs: list[MetaTargetSpec],
        equivariant_layers_activation: ActivationFunction = (
            ActivationFunction.LEAKY_RELU
        ),
        llm_layers: int = 1,
        device: torch.device | str = "cuda",
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if not head_specs:
            msg = "head_specs must contain at least one head"
            raise ValueError(msg)
        self.equivariant_layers_activation = equivariant_layers_activation
        self.device = torch.device(device) if isinstance(device, str) else device
        self.targeted_modules = sorted(targeted_modules)
        self.llm_layers = llm_layers
        self.head_specs = head_specs
        self.seed = seed

        # Set seed for reproducible weight initialization
        if seed is not None:
            torch.manual_seed(seed)

        # Store hyperparameters for saving
        self.equivariant_layer_sizes = equivariant_layer_sizes
        self.head_layer_sizes = head_layer_sizes
        self.target_module_sizes = target_module_sizes

        self.module_a_layers = nn.ModuleDict()
        self.module_b_layers = nn.ModuleDict()

        for module_type in self.targeted_modules:
            input_size, output_size = target_module_sizes[module_type]

            a_layers = nn.ModuleList()
            b_layers = nn.ModuleList()

            for in_size, out_size in itertools.pairwise(
                [input_size, *equivariant_layer_sizes],
            ):
                a_layers.append(
                    EquivariantLinearLayer(in_size, out_size, device=self.device),
                )

            for in_size, out_size in itertools.pairwise(
                [output_size, *equivariant_layer_sizes],
            ):
                b_layers.append(
                    EquivariantLinearLayer(in_size, out_size, device=self.device),
                )

            self.module_a_layers[module_type.value] = a_layers
            self.module_b_layers[module_type.value] = b_layers

        last_equivariant_size = equivariant_layer_sizes[-1]
        module_matrix_size = last_equivariant_size * last_equivariant_size
        # Account for multiple layers: each layer contributes to the concatenated input
        input_dim = module_matrix_size * len(self.targeted_modules) * self.llm_layers

        # Shared head body: the dense stack [input_dim, *head_layer_sizes] with a
        # LeakyReLU after every layer (including the last) so the per-head output
        # linears see a non-linear representation.  With a single classification
        # head and head_layer_sizes=[H] this reproduces the historical
        # [Linear(input, H), LeakyReLU, Linear(H, num_classes)] architecture.
        body_sizes = [input_dim, *head_layer_sizes]
        body_layers: list[nn.Module] = []
        for in_size, out_size in itertools.pairwise(body_sizes):
            body_layers.append(nn.Linear(in_size, out_size, device=self.device))
            body_layers.append(nn.LeakyReLU())
        self.head_body = nn.Sequential(*body_layers)

        last_hidden = body_sizes[-1]
        self.output_heads = nn.ModuleDict({
            spec.name: nn.Linear(last_hidden, spec.output_dim, device=self.device)
            for spec in self.head_specs
        })

        self.to(self.device)

        # Each (module, layer) needs its own MatMul instance so that
        # zennit LRP hooks store per-call activations (shared instance
        # would overwrite stored tensors on every forward call).
        self.matrix_linear_products = nn.ModuleDict({
            f"{module_type.value}_{layer_idx}": MatMul()
            for module_type in self.targeted_modules
            for layer_idx in range(self.llm_layers)
        })

        # Fast path: batch the per-(module, layer) work over the layer dim in a
        # single set of ops instead of a Python loop, cutting kernel launches
        # ~llm_layers-fold (measured 9160 -> 774 / step, 12x, on a 7-module x
        # 16-layer pool). It is numerically identical (every op acts on the
        # trailing matrix dims and broadcasts over (batch, layer)); verified
        # bit-identical to the loop (max|Δ| ~1e-9 fp32). It bypasses the
        # per-(module, layer) MatMul instances, so it is auto-disabled whenever
        # LRP hooks are live on them (see :meth:`_lrp_hooks_active`); set this
        # flag False to force the loop path for any other reason.
        self.vectorized_forward = True

    @property
    def num_classes(self) -> int:
        """Class count of the sole classification head (legacy single-head API).

        Raises if the model is genuinely multi-head; callers that assume one
        classification output then fail loudly rather than silently.
        """
        classification_heads = [
            s for s in self.head_specs if s.is_classification
        ]
        if len(self.head_specs) == 1 and len(classification_heads) == 1:
            return classification_heads[0].num_classes
        msg = (
            "num_classes is only defined for a single-classification-head model; "
            f"this model has heads {[s.name for s in self.head_specs]}"
        )
        raise ValueError(msg)

    def _get_module_key(self, module_type: TargetModuleType) -> str:
        """Map module type to the actual key in the input dict."""
        mapping = {
            TargetModuleType.Q_ATTENTION: "self_attn.q_proj",
            TargetModuleType.K_ATTENTION: "self_attn.k_proj",
            TargetModuleType.V_ATTENTION: "self_attn.v_proj",
            TargetModuleType.O_ATTENTION: "self_attn.o_proj",
            TargetModuleType.GATE_MLP: "mlp.gate_proj",
            TargetModuleType.UP_MLP: "mlp.up_proj",
            TargetModuleType.DOWN_MLP: "mlp.down_proj",
        }
        if module_type not in mapping:
            msg = f"Unknown module type: {module_type}"
            raise ValueError(msg)
        return mapping[module_type]

    def _validate_tensor_shapes(self, a: torch.Tensor, b: torch.Tensor) -> None:
        """Validate that tensors have the expected 4D shape (batch, layer, width, length)."""
        if a.dim() != EXPECTED_TENSOR_DIMS or b.dim() != EXPECTED_TENSOR_DIMS:
            msg = (
                f"Expected {EXPECTED_TENSOR_DIMS}D tensors ({BATCH}, {LAYER}, {WIDTH}, {LENGTH}), "
                f"got A: {a.shape}, B: {b.shape}"
            )
            raise ValueError(msg)

        if a.size(1) != self.llm_layers or b.size(1) != self.llm_layers:
            msg = (
                f"Expected {self.llm_layers} layers, got A: {a.size(1)}, B: {b.size(1)}"
            )
            raise ValueError(msg)

    def _process_layer_matrices(
        self,
        a_layer: torch.Tensor,
        b_layer: torch.Tensor,
        module_type: TargetModuleType,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """Process A and B matrices for a single layer through equivariant layers.

        Args:
            a_layer: LoRA A matrix (batch, width, length).
            b_layer: LoRA B matrix (batch, width, length).
            module_type: Target module type for layer lookup.
            layer_idx: Layer index (used to select the correct MatMul instance).

        Returns:
            Flattened weight matrix (batch, features).
        """
        a_layers = self.module_a_layers[module_type.value]
        b_layers = self.module_b_layers[module_type.value]

        # Ensure we have ModuleLists
        if not isinstance(a_layers, nn.ModuleList):
            msg = f"Expected ModuleList for a_layers, got {type(a_layers)}"
            raise TypeError(msg)
        if not isinstance(b_layers, nn.ModuleList):
            msg = f"Expected ModuleList for b_layers, got {type(b_layers)}"
            raise TypeError(msg)

        # Apply equivariant layers (a_layer, b_layer: batch x width x length)
        a_processed = a_layer
        b_processed = b_layer

        for i, (a_eq_layer, b_eq_layer) in enumerate(
            zip(a_layers, b_layers, strict=True),
        ):
            a_processed = a_eq_layer(a_processed)
            b_processed = b_eq_layer(b_processed)
            if i != len(a_layers) - 1:
                match self.equivariant_layers_activation:
                    case ActivationFunction.GL_ACTIVATION:
                        b_processed, a_processed = gl_activation(
                            b_processed,
                            a_processed,
                        )
                    case ActivationFunction.LEAKY_RELU:
                        a_processed = F.leaky_relu(a_processed)
                        b_processed = F.leaky_relu(b_processed)

        # Multiply B @ A^T for this layer -> (batch, width, width)
        mat_mul_key = f"{module_type.value}_{layer_idx}"
        w_layer = self.matrix_linear_products[mat_mul_key](
            b_processed,
            a_processed.transpose(-1, -2),
        )
        # Flatten to (batch, features) for concatenation
        return w_layer.view(w_layer.size(0), -1)

    def _lrp_hooks_active(self) -> bool:
        """True if any zennit/LRP hook is live on the per-cell MatMul instances.

        The vectorised path bypasses ``matrix_linear_products`` (it uses one
        batched matmul per module), so per-cell attribution rules could not fire.
        LRP composites register forward/backward hooks on those modules; when any
        are present we fall back to the loop path automatically, so attribution
        flows need no change and never silently read the wrong path.
        """
        hook_attrs = (
            "_forward_hooks",
            "_forward_pre_hooks",
            "_backward_hooks",
            "_full_backward_hooks",
            "_full_backward_pre_hooks",
        )
        for m in self.matrix_linear_products.values():
            if any(getattr(m, a, None) for a in hook_attrs):
                return True
        return False

    def forward(
        self, x: dict[str, dict[LoraType, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        if self.vectorized_forward and not self._lrp_hooks_active():
            return self._forward_vectorised(x)
        return self._forward_looped(x)

    def _forward_looped(
        self, x: dict[str, dict[LoraType, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        """Reference path: one Python iteration per (module, layer).

        Retained for LRP (needs the per-cell :class:`MatMul` instances) and as
        the exact numerical reference for :meth:`_forward_vectorised`.
        """
        module_matrices = []

        for module_type in self.targeted_modules:
            key = self._get_module_key(module_type)

            # Extract A and B matrices for this module
            a = x[key][LoraType.A].to(self.device).transpose(-1, -2)
            b = x[key][LoraType.B].to(self.device)

            # Handle 3D tensors (single layer case) by adding layer dimension
            if a.dim() == 3:
                a = a.unsqueeze(1)  # (batch, width, length) -> (batch, 1, width, length)
            if b.dim() == 3:
                b = b.unsqueeze(1)  # (batch, width, length) -> (batch, 1, width, length)

            # Validate the expected shape
            self._validate_tensor_shapes(a, b)

            # Add named dimensions for readability: (batch, layer, width, length)
            a = a.refine_names(BATCH, LAYER, WIDTH, LENGTH)
            b = b.refine_names(BATCH, LAYER, WIDTH, LENGTH)

            layer_matrices = []

            # Process each layer separately
            for layer_idx in range(self.llm_layers):
                # Extract matrices for this layer (batch, width, length)
                a_layer = a.select(LAYER, layer_idx).rename(None)
                b_layer = b.select(LAYER, layer_idx).rename(None)

                # Process through this module's equivariant layers
                w_layer = self._process_layer_matrices(a_layer, b_layer, module_type, layer_idx)
                layer_matrices.append(w_layer)

            # Concatenate all layers for this module along features
            module_matrix = torch.cat(layer_matrices, dim=1)
            module_matrices.append(module_matrix)

        # Concatenate all module matrices along features
        concatenated = torch.cat(module_matrices, dim=1)

        shared = self.head_body(concatenated)
        return {name: head(shared) for name, head in self.output_heads.items()}

    def _forward_vectorised(
        self, x: dict[str, dict[LoraType, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        """Layer-batched equivalent of :meth:`_forward_looped`.

        Keeps the ``layer`` axis through the equivariant stack and the ``B @ Aᵀ``
        product instead of looping — ``EquivariantLinearLayer`` (``weights @ x``),
        ``gl_activation`` / ``leaky_relu`` and the final matmul all act on the
        trailing matrix dims and broadcast over ``(batch, layer)``, and the
        layer-major ``reshape`` reproduces the loop's ``cat`` order exactly. No
        named tensors, one matmul per module rather than ``llm_layers``.
        """
        module_matrices = []

        for module_type in self.targeted_modules:
            key = self._get_module_key(module_type)

            a = x[key][LoraType.A].to(self.device).transpose(-1, -2)
            b = x[key][LoraType.B].to(self.device)
            if a.dim() == 3:
                a = a.unsqueeze(1)
            if b.dim() == 3:
                b = b.unsqueeze(1)
            self._validate_tensor_shapes(a, b)

            a_layers = self.module_a_layers[module_type.value]
            b_layers = self.module_b_layers[module_type.value]
            a_proc, b_proc = a, b
            last = len(a_layers) - 1
            for i, (a_eq, b_eq) in enumerate(zip(a_layers, b_layers, strict=True)):
                a_proc = a_eq(a_proc)
                b_proc = b_eq(b_proc)
                if i != last:
                    match self.equivariant_layers_activation:
                        case ActivationFunction.GL_ACTIVATION:
                            b_proc, a_proc = gl_activation(b_proc, a_proc)
                        case ActivationFunction.LEAKY_RELU:
                            a_proc = F.leaky_relu(a_proc)
                            b_proc = F.leaky_relu(b_proc)

            # (batch, layer, d, d) -> flatten layer-major to match the loop's cat.
            w = b_proc @ a_proc.transpose(-1, -2)
            module_matrices.append(w.reshape(w.size(0), -1))

        concatenated = torch.cat(module_matrices, dim=1)
        shared = self.head_body(concatenated)
        return {name: head(shared) for name, head in self.output_heads.items()}

    def save(self, path: str | Path) -> None:
        """Save model weights and hyperparameters to a file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        hyperparams = {
            "equivariant_layer_sizes": self.equivariant_layer_sizes,
            "head_layer_sizes": self.head_layer_sizes,
            "targeted_modules": [m.value for m in self.targeted_modules],
            "target_module_sizes": {
                k.value: v for k, v in self.target_module_sizes.items()
            },
            "head_specs": [spec.to_dict() for spec in self.head_specs],
            "equivariant_layers_activation": self.equivariant_layers_activation.value,
            "llm_layers": self.llm_layers,
            "device": str(self.device),
            "seed": self.seed,
        }

        torch.save(
            {
                "state_dict": self.state_dict(),
                "hyperparameters": hyperparams,
            },
            path,
        )

    @staticmethod
    def _head_specs_from_hyperparams(hyperparams: dict) -> list[MetaTargetSpec]:
        """Rebuild head specs, falling back to a single classification head.

        Checkpoints saved before the multi-head refactor only store
        ``num_classes``; synthesise a single ``CLASSIFICATION`` head named
        ``"label"`` so those artifacts still load.
        """
        raw_specs = hyperparams.get("head_specs")
        if raw_specs is not None:
            return [MetaTargetSpec.from_dict(d) for d in raw_specs]
        num_classes = int(hyperparams.get("num_classes", 2))
        return [
            MetaTargetSpec(
                name="label",
                column="label",
                type=MetaHeadType.CLASSIFICATION,
                num_classes=num_classes,
            )
        ]

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: torch.device | str | None = None,
    ) -> Self:
        """Load model weights and hyperparameters from a file."""
        path = Path(path)
        checkpoint = torch.load(path, map_location="cpu")

        hyperparams = checkpoint["hyperparameters"]

        targeted_modules = {
            TargetModuleType(m) for m in hyperparams["targeted_modules"]
        }
        target_module_sizes = {
            TargetModuleType(k): tuple(v)
            for k, v in hyperparams["target_module_sizes"].items()
        }
        equivariant_layers_activation = ActivationFunction(
            hyperparams["equivariant_layers_activation"],
        )
        head_specs = cls._head_specs_from_hyperparams(hyperparams)

        model_device = device if device is not None else hyperparams["device"]

        model = cls(
            equivariant_layer_sizes=hyperparams["equivariant_layer_sizes"],
            head_layer_sizes=hyperparams["head_layer_sizes"],
            targeted_modules=targeted_modules,
            target_module_sizes=target_module_sizes,
            head_specs=head_specs,
            equivariant_layers_activation=equivariant_layers_activation,
            llm_layers=hyperparams["llm_layers"],
            device=model_device,
            seed=hyperparams.get("seed"),
        )

        model.load_state_dict(checkpoint["state_dict"])
        model.to(model_device)

        return model
