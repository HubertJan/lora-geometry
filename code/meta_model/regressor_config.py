"""Migrated from SRC/src/discoveries/sst2_perf_prediction/flows/flexible_meta_config.py.

A ``ModelConfig`` wrapper so the cached library trainer can drive ANY
architecture from the ``meta_classifier_architectures`` zoo (mlp / deepsets /
attn heads, GL depth, feature-norm) as a 6-head SST2 performance REGRESSOR.

Why this exists
---------------
``discoveries.meta_classifier_training.flows.train_meta_classifier.train`` is
model-agnostic: its loop, ``compute_multihead_loss`` and
``evaluate_multihead_performance`` (per-head R2/Spearman/Pearson/MAE) work for any
``nn.Module`` returning ``{head_name: logits}``. The ONLY coupling to a concrete
model is ``model = model_config.create_model(device)`` (+ ``.name`` / ``.to_dict``).
So to benchmark architectures through the SAME cached, lineage-tracked trainer we
only need a ``ModelConfig`` whose ``create_model`` returns a
:class:`FlexibleLoRAMetaClassifier` instead of the fixed
``EquivariantLoRAMetaClassifier``.

The alternative (the arch-zoo's own ``train_arch.py``) forks the loop and is
single-head CLASSIFICATION, so it is not reusable for the 6-head regression.

Cache correctness
-----------------
The original cached trainer fingerprinted ``model_config`` via ``to_dict()``.
This wrapper's ``to_dict`` includes the full ``arch`` dict and ``seed``, so every
(arch, seed) is a DISTINCT entry — no accidental collision across arms.

Picklability
------------
Plain-data fields only (arch dict, frozen ``MetaTargetSpec`` list, enums), so the
config cloudpickles across a cluster executor to the GPU node, where
``create_model`` builds the torch module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from meta_model.lora.types import LLMModel, TargetModuleType
from meta_model.meta_classifier_config import ModelConfig
from meta_model.heads import MetaHeadType, MetaTargetSpec

if TYPE_CHECKING:
    import torch

#: The 6 SST2 likelihood metrics -> (head name, benchmark column, head type).
#: Mirrors ``jobs/05`` HEADS: 5 bounded [0,1] metrics + unbounded nll.
SST2_HEADS: list[tuple[str, str, str]] = [
    ("acc", "benchmark.sst2-test.likelihood.accuracy", "BOUNDED_REGRESSION"),
    ("f1", "benchmark.sst2-test.likelihood.f1_macro", "BOUNDED_REGRESSION"),
    ("auroc", "benchmark.sst2-test.likelihood.auroc", "BOUNDED_REGRESSION"),
    ("brier", "benchmark.sst2-test.likelihood.brier", "BOUNDED_REGRESSION"),
    ("meanconf", "benchmark.sst2-test.likelihood.mean_confidence", "BOUNDED_REGRESSION"),
    ("nll", "benchmark.sst2-test.likelihood.nll", "UNBOUNDED_REGRESSION"),
]

#: The curated architecture set for the Stage-B benchmark. Keys index the arch zoo
#: (``arch_zoo.ARCHS``). ``base`` = the production repro (single equivariant layer,
#: no-op activation, 117 M dense MLP head, no feature-norm) -> the baseline that must
#: reproduce the parent's numbers. Every multi-layer / attention arm carries
#: ``feature_norm="l2"`` (MANDATORY: un-normalised deep GL trunks collapse features
#: ~40x/layer and the head sits at chance -- arch_zoo.py:54-92).
BENCH_ARCHS: list[str] = [
    "base",          # production repro (mlp [128], activation none, no norm)
    "base_l2",       # + L2 feature-norm (cheap magnitude-removal fix)
    "mlp32_l2",      # [128,32] glsum trunk, mlp head [64]
    "deepsets32_l2",  # shared per-cell proj + mean-pool (sharing without attention)
    "attn128_l2",    # attention head, [128] trunk
    "attn32_l2",     # attention head, [128,32] glsum trunk
    "base_d2_l2",    # real GL depth [128,128] glsum, mlp head
]

#: The complexity-ladder sweep (jobs/22): how small can the equivariant regressor get
#: before held-out calibration R2 falls off. All single-layer, no-op-activation,
#: L2-normalised trunks (only the reducible dim moves) — ``base`` / ``base_l2`` are the
#: in-sweep faithfulness gate + ceiling anchor; ``w{d}`` shrink the equivariant width;
#: ``bilin{d}`` empty the head to a pure linear-in-dW (bilinear) readout; ``bilin128``
#: is the no-norm exact-linear control. See ``arch_zoo.py`` round-3 block.
LADDER_ARCHS: list[str] = [
    "base",          # no-norm faithfulness gate
    "base_l2",       # d=128 dense-head CEILING anchor
    "w64_l2", "w32_l2", "w16_l2", "w8_l2", "w4_l2",   # width ladder
    "bilin128_l2", "bilin32_l2", "bilin8_l2",          # bilinear (empty head), L2-normed
    "bilin128",      # exact linear-in-dW, no feature-norm
]


def sst2_head_specs() -> list[MetaTargetSpec]:
    """The 6 SST2 regression heads (shared by jobs 14/15/16)."""
    return [
        MetaTargetSpec(name=name, column=col, type=MetaHeadType[htype],
                       num_classes=1, mapping=None, loss_weight=1.0)
        for name, col, htype in SST2_HEADS
    ]


class FlexibleRegressorConfig(ModelConfig):
    """``ModelConfig`` that builds a :class:`FlexibleLoRAMetaClassifier`.

    ``arch`` is one entry of ``arch_zoo.ARCHS`` (``equivariant_layer_sizes``,
    ``activation``, ``head``, ``feature_norm`` and, for attention heads, ``d_model``
    / ``n_heads`` / ``n_attn_layers`` / ``dropout``); ``arch_name`` is its zoo key,
    carried into the run name and the cache fingerprint.
    """

    def __init__(
        self,
        *,
        arch: dict,
        arch_name: str,
        head_specs: list[MetaTargetSpec],
        target_modules: set[TargetModuleType],
        llm_layer_count: int,
        llm_model: LLMModel,
        seed: int | None = None,
    ) -> None:
        self.arch = dict(arch)
        self.arch_name = arch_name
        self.head_specs = head_specs
        self.target_modules = target_modules
        self.llm_layer_count = llm_layer_count
        self.llm_model = llm_model
        self.seed = seed

    @property
    def name(self) -> str:
        return f"flexible_lora_meta_classifier_{self.arch_name}"

    def to_dict(self) -> dict:
        def _jsonable(v):
            if isinstance(v, (list, tuple)):
                return [_jsonable(x) for x in v]
            return v

        return {
            "arch_name": self.arch_name,
            "arch": {k: _jsonable(v) for k, v in sorted(self.arch.items())},
            "target_modules": sorted(str(t) for t in self.target_modules),
            "llm_layer_count": self.llm_layer_count,
            "llm_model": str(self.llm_model),
            "head_specs": [spec.to_dict() for spec in self.head_specs],
            "seed": str(self.seed) if self.seed is not None else None,
            **super().to_dict(),
        }

    def create_model(self, device: str) -> "torch.nn.Module":
        from meta_model.model import (
            FlexibleLoRAMetaClassifier,
        )
        from meta_model.lora.model_sizes import TARGET_MODULE_SIZES_BY_LLM_MODEL

        return FlexibleLoRAMetaClassifier(
            targeted_modules=self.target_modules,
            target_module_sizes=TARGET_MODULE_SIZES_BY_LLM_MODEL[self.llm_model],
            head_specs=self.head_specs,
            llm_layers=self.llm_layer_count,
            device=device,
            seed=self.seed,
            **self.arch,
        )


def build_regressor_config(
    arch_name: str,
    *,
    head_specs: list[MetaTargetSpec] | None = None,
    llm_layer_count: int = 16,
    llm_model: LLMModel = LLMModel.LLAMA_3_1B,
    seed: int | None = 42,
) -> FlexibleRegressorConfig:
    """Build a :class:`FlexibleRegressorConfig` for one arch-zoo key (all 7 modules)."""
    from meta_model.arch_zoo import ARCHS

    if arch_name not in ARCHS:
        msg = f"unknown arch {arch_name!r}; expected one of {sorted(ARCHS)}"
        raise ValueError(msg)
    return FlexibleRegressorConfig(
        arch=ARCHS[arch_name],
        arch_name=arch_name,
        head_specs=head_specs or sst2_head_specs(),
        target_modules=set(TargetModuleType),
        llm_layer_count=llm_layer_count,
        llm_model=llm_model,
        seed=seed,
    )

# NOTE: the Campaign-3 magnitude-reintroduction arms (FlexibleRegressorConfigMag /
# build_mag_regressor_config / MAG_ARCHS) were dropped in migration: they depend on
# a separate ``architectures_mag`` module that is not part of this package.
