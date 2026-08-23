"""Migrated from SRC/src/discoveries/sst2_perf_prediction/flows/peftguard.py.

A faithful PEFTGuard detector, re-cast as a 6-head SST2 performance regressor.

Baseline for the in-task SST2 study: reproduce the architecture of PEFTGuard
(Sun et al., IEEE S&P 2025, arXiv:2411.17453 — see ``model/PEFTGuard_*.py`` in
https://github.com/Vincent-HKUSTGZ/PEFTGuard) and score it on the SAME 504-train /
63-test v2 pool used by the arch-benchmark, so its numbers sit directly next to
``base`` / ``base_l2`` from ``flexible_meta_config``.

What PEFTGuard does (from the released source, e.g. ``PEFTGuard_llama3_8b.py``)
------------------------------------------------------------------------------
* Input = the **merged** low-rank update ``dW = B @ A`` per (layer, component),
  the full ``d_out x d_in`` dense matrix — NOT the raw A/B factors, and NO
  equivariant preprocessing.
* Transformer **layers become conv channels**; each targeted **component gets its
  own conv tower** (the released Llama code has ``conv1_q`` / ``conv1_v`` because
  GQA gives q and v different widths).
* Per tower: one big-stride ``Conv2d`` downsamples the weight matrix, then a
  single very large ``Linear`` flattens it to 512. Towers are concatenated →
  ``Linear(->128)`` → output head. ``LeakyReLU`` throughout.
* The published detector is a **binary** classifier (backdoored vs benign) and
  targets **q,v only** (its Table-5 default; more components did not help
  detection).

Faithful here, three deliberate task-adaptations
------------------------------------------------
1. **q,v-only**, exactly the released default (this file is the q,v baseline; the
   all-7 "hard mode" is a separate arm).
2. Dimensions scaled to **Llama-3.2-1B** (hidden 2048, 16 layers, GQA kv=512):
   q ``dW`` is 2048x2048, v ``dW`` is 512x2048. With ``k=8, s=8`` the conv output
   is 16x256x256 (q) and 16x64x256 (v) — same recipe as the 7B code, smaller grid.
3. The 2-class softmax head is replaced by the study's **6 regression heads**
   (acc/f1/auroc/brier/mean-conf + unbounded nll). Heads emit **raw** logits;
   ``compute_multihead_loss`` applies sigmoid/softplus + MSE, identical to every
   other model in this discovery — so the trainer, loss and metrics are untouched.

Everything else — no feature normalisation, the giant flatten-FC, the conv recipe
— is kept as published. On 504 adapters the flatten-FC alone is ~0.67 B params,
so this arm is expected to be data-hungry / prone to overfitting and to inherit
the raw-magnitude confound that ``base_l2`` removes; that contrast is the point of
having it as a baseline.

Input contract (shared with the equivariant models)
---------------------------------------------------
``forward(x)`` takes ``x[module_key][LoraType.A] : (batch, L, r, in)`` and
``x[module_key][LoraType.B] : (batch, L, out, r)`` (module_key e.g.
``"self_attn.q_proj"``) and returns ``{head_name: logits}``. ``dW = B @ A`` is
formed inside forward.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Self

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from meta_model.lora.types import LLMModel, LoraType, TargetModuleType
from meta_model.meta_classifier_config import ModelConfig
from meta_model.heads import MetaTargetSpec

if TYPE_CHECKING:
    pass

# Verbatim from the library / architectures.py — TargetModuleType -> safetensor key.
_MODULE_KEY = {
    TargetModuleType.Q_ATTENTION: "self_attn.q_proj",
    TargetModuleType.K_ATTENTION: "self_attn.k_proj",
    TargetModuleType.V_ATTENTION: "self_attn.v_proj",
    TargetModuleType.O_ATTENTION: "self_attn.o_proj",
    TargetModuleType.GATE_MLP: "mlp.gate_proj",
    TargetModuleType.UP_MLP: "mlp.up_proj",
    TargetModuleType.DOWN_MLP: "mlp.down_proj",
}

#: PEFTGuard's released default target set (q,v only).
DEFAULT_QV: set[TargetModuleType] = {
    TargetModuleType.Q_ATTENTION,
    TargetModuleType.V_ATTENTION,
}


def _conv_out(size: int, kernel: int, stride: int) -> int:
    """Spatial extent after ``Conv2d(kernel, stride, padding=0)``."""
    return (size - kernel) // stride + 1


class PEFTGuardMetaClassifier(nn.Module):
    """PEFTGuard's per-component conv-tower detector, multi-head regression variant.

    One conv tower per targeted component (layers = input channels), each ending in
    a large flatten-``Linear`` to ``fc1_dim``; towers concatenate into ``fc2_dim``;
    then one ``Linear(fc2_dim, output_dim)`` per head. Matches ``PEFTGuard_*`` up to
    the head (6 regression heads instead of a 2-class softmax).
    """

    def __init__(
        self,
        *,
        targeted_modules: set[TargetModuleType],
        target_module_sizes: dict[TargetModuleType, tuple[int, int]],
        head_specs: list[MetaTargetSpec],
        llm_layers: int = 16,
        conv_out_channels: int = 16,
        kernel_size: int = 8,
        stride: int = 8,
        fc1_dim: int = 512,
        fc2_dim: int = 128,
        device: torch.device | str = "cuda",
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if not head_specs:
            msg = "head_specs must contain at least one head"
            raise ValueError(msg)
        if seed is not None:
            torch.manual_seed(seed)

        self.device = torch.device(device) if isinstance(device, str) else device
        self.targeted_modules = sorted(targeted_modules)
        self.target_module_sizes = target_module_sizes
        self.head_specs = head_specs
        self.llm_layers = llm_layers
        self.conv_out_channels = conv_out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.seed = seed

        # One conv tower + flatten-FC per component. dW has shape (out, in); the
        # conv sees layers as channels, (out, in) as the H x W grid.
        self.conv = nn.ModuleDict()
        self.fc1 = nn.ModuleDict()
        for module_type in self.targeted_modules:
            in_size, out_size = target_module_sizes[module_type]  # (in, out)
            self.conv[module_type.value] = nn.Conv2d(
                llm_layers, conv_out_channels, kernel_size=kernel_size, stride=stride,
                padding=0,
            )
            h = _conv_out(out_size, kernel_size, stride)  # dW rows = out_features
            w = _conv_out(in_size, kernel_size, stride)   # dW cols = in_features
            self.fc1[module_type.value] = nn.Linear(conv_out_channels * h * w, fc1_dim)

        self.fc2 = nn.Linear(fc1_dim * len(self.targeted_modules), fc2_dim)
        self.output_heads = nn.ModuleDict({
            s.name: nn.Linear(fc2_dim, s.output_dim) for s in head_specs
        })

        self.to(self.device)

    # ── introspection ────────────────────────────────────────────────────────

    def param_counts(self) -> dict[str, int]:
        conv = sum(p.numel() for p in self.conv.parameters())
        fc1 = sum(p.numel() for p in self.fc1.parameters())
        rest = sum(p.numel() for p in self.fc2.parameters()) + sum(
            p.numel() for p in self.output_heads.parameters()
        )
        return {"conv": conv, "fc1_flatten": fc1, "fc2_heads": rest,
                "total": conv + fc1 + rest}

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(
        self, x: dict[str, dict[LoraType, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        tower_outputs: list[torch.Tensor] = []
        for module_type in self.targeted_modules:
            key = _MODULE_KEY[module_type]
            a = x[key][LoraType.A].to(self.device, dtype=torch.float32)  # (b, L, r, in)
            b = x[key][LoraType.B].to(self.device, dtype=torch.float32)  # (b, L, out, r)
            if a.dim() == 3:  # single-layer inputs -> add the layer axis
                a = a.unsqueeze(1)
            if b.dim() == 3:
                b = b.unsqueeze(1)
            if a.size(1) != self.llm_layers or b.size(1) != self.llm_layers:
                msg = (f"expected {self.llm_layers} layers, got "
                       f"A: {a.size(1)}, B: {b.size(1)}")
                raise ValueError(msg)

            dw = torch.matmul(b, a)          # (b, L, out, in) — the merged update
            feat = self.conv[module_type.value](dw)          # (b, C, h, w)
            feat = feat.reshape(feat.size(0), -1)
            feat = F.leaky_relu(self.fc1[module_type.value](feat))  # (b, fc1_dim)
            tower_outputs.append(feat)

        shared = F.leaky_relu(self.fc2(torch.cat(tower_outputs, dim=1)))
        return {name: head(shared) for name, head in self.output_heads.items()}

    # ── persistence (trainer calls model.save(...)) ───────────────────────────

    def hyperparameters(self) -> dict:
        return {
            "targeted_modules": [m.value for m in self.targeted_modules],
            "target_module_sizes": {
                k.value: list(v) for k, v in self.target_module_sizes.items()
            },
            "head_specs": [s.to_dict() for s in self.head_specs],
            "llm_layers": self.llm_layers,
            "conv_out_channels": self.conv_out_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "fc1_dim": self.fc1_dim,
            "fc2_dim": self.fc2_dim,
            "device": str(self.device),
            "seed": self.seed,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self.state_dict(), "hyperparameters": self.hyperparameters()},
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: torch.device | str | None = None) -> Self:
        ckpt = torch.load(Path(path), map_location="cpu")
        hp = ckpt["hyperparameters"]
        model = cls(
            targeted_modules={TargetModuleType(m) for m in hp["targeted_modules"]},
            target_module_sizes={
                TargetModuleType(k): tuple(v)
                for k, v in hp["target_module_sizes"].items()
            },
            head_specs=[MetaTargetSpec.from_dict(d) for d in hp["head_specs"]],
            llm_layers=hp["llm_layers"],
            conv_out_channels=hp["conv_out_channels"],
            kernel_size=hp["kernel_size"],
            stride=hp["stride"],
            fc1_dim=hp["fc1_dim"],
            fc2_dim=hp["fc2_dim"],
            device=device if device is not None else hp["device"],
            seed=hp.get("seed"),
        )
        model.load_state_dict(ckpt["state_dict"])
        return model


class PEFTGuardConfig(ModelConfig):
    """``ModelConfig`` that builds a :class:`PEFTGuardMetaClassifier`.

    Plain-data fields only (enums, ints, frozen specs) so it cloudpickles across
    a cluster executor; ``to_dict`` carries every knob + ``seed`` into the cache
    fingerprint, keeping each (config, seed) a distinct cached entry — no
    collision with the equivariant arms.
    """

    def __init__(
        self,
        *,
        target_modules: set[TargetModuleType],
        head_specs: list[MetaTargetSpec],
        llm_layer_count: int = 16,
        llm_model: LLMModel = LLMModel.LLAMA_3_1B,
        conv_out_channels: int = 16,
        kernel_size: int = 8,
        stride: int = 8,
        fc1_dim: int = 512,
        fc2_dim: int = 128,
        seed: int | None = None,
    ) -> None:
        self.target_modules = target_modules
        self.head_specs = head_specs
        self.llm_layer_count = llm_layer_count
        self.llm_model = llm_model
        self.conv_out_channels = conv_out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.seed = seed

    @property
    def name(self) -> str:
        tag = "".join(sorted(m.value[0] for m in self.target_modules))
        return f"peftguard_{tag}"

    def to_dict(self) -> dict:
        return {
            "target_modules": sorted(str(t) for t in self.target_modules),
            "llm_layer_count": self.llm_layer_count,
            "llm_model": str(self.llm_model),
            "conv_out_channels": self.conv_out_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "fc1_dim": self.fc1_dim,
            "fc2_dim": self.fc2_dim,
            "head_specs": [spec.to_dict() for spec in self.head_specs],
            "seed": str(self.seed) if self.seed is not None else None,
            **super().to_dict(),
        }

    def create_model(self, device: str) -> nn.Module:
        from meta_model.lora.model_sizes import TARGET_MODULE_SIZES_BY_LLM_MODEL

        return PEFTGuardMetaClassifier(
            targeted_modules=self.target_modules,
            target_module_sizes=TARGET_MODULE_SIZES_BY_LLM_MODEL[self.llm_model],
            head_specs=self.head_specs,
            llm_layers=self.llm_layer_count,
            conv_out_channels=self.conv_out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            fc1_dim=self.fc1_dim,
            fc2_dim=self.fc2_dim,
            device=device,
            seed=self.seed,
        )


def build_peftguard_config(
    *,
    head_specs: list[MetaTargetSpec] | None = None,
    target_modules: set[TargetModuleType] | None = None,
    llm_layer_count: int = 16,
    llm_model: LLMModel = LLMModel.LLAMA_3_1B,
    seed: int | None = 42,
) -> PEFTGuardConfig:
    """Build the faithful q,v-only PEFTGuard config for the 6 SST2 heads."""
    from meta_model.regressor_config import (
        sst2_head_specs,
    )

    return PEFTGuardConfig(
        target_modules=target_modules or set(DEFAULT_QV),
        head_specs=head_specs or sst2_head_specs(),
        llm_layer_count=llm_layer_count,
        llm_model=llm_model,
        seed=seed,
    )


__all__ = [
    "DEFAULT_QV",
    "PEFTGuardConfig",
    "PEFTGuardMetaClassifier",
    "build_peftguard_config",
]
