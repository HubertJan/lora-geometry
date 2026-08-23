"""Migrated from SRC/src/discoveries/meta_classifier_architectures/flows/arch_zoo.py.

The architecture grid, in an importable module.

Kept out of ``jobs/02_train-archs_*.py`` so the SLURM worker can import the arch
dict by name instead of relying on the job module being picklable by value.

``base`` is the production detector. Its published ``activation="gl_activation"``
is a NO-OP at depth 1 (``jobs/01``), so ``"none"`` reproduces it exactly; naming
it honestly here keeps the table from implying a non-linearity that never ran.

The grid is designed so each pair isolates one factor:

  base    -> base_l2       : per-cell L2 normalisation, nothing else
  base_l2 -> base_l2g      : magnitude removal alone vs. magnitude + cell balancing
  base    -> attn128       : the HEAD (117 M dense -> 2.5 M attention), same trunk
  attn32  -> deepsets32    : attention vs. mean-pooling at equal weight sharing
  attn32  -> attn32_none   : whether the GL activation earns its place
  attn32  -> attn32_glpinv : the row-sum GL gate vs. the library's pinv gate
  attn32  -> mlp32         : the head again, at the narrow bottleneck
  attn32  -> attn32_d3     : a third equivariant layer (two GL activations)
"""

from __future__ import annotations

_ATTN = {"head": "attn", "d_model": 128, "n_heads": 4, "n_attn_layers": 2, "dropout": 0.1}

ARCHS: dict[str, dict] = {
    "base": {"equivariant_layer_sizes": [128], "activation": "none",
             "head": "mlp", "head_layer_sizes": [64]},
    "base_l2": {"equivariant_layer_sizes": [128], "activation": "none",
                "head": "mlp", "head_layer_sizes": [64], "feature_norm": "l2"},
    # Splits what `base_l2` confounds: `l2_global` removes per-adapter MAGNITUDE only,
    # `base_l2` also equalises the 112 cells. The delta between them is the cell-balance
    # term; the delta from `base` is the magnitude term.
    "base_l2g": {"equivariant_layer_sizes": [128], "activation": "none",
                 "head": "mlp", "head_layer_sizes": [64],
                 "feature_norm": "l2_global"},
    "attn128": {"equivariant_layer_sizes": [128], "activation": "none", **_ATTN},
    "attn128_l2": {"equivariant_layer_sizes": [128], "activation": "none",
                   "feature_norm": "l2", **_ATTN},
    "attn32": {"equivariant_layer_sizes": [128, 32], "activation": "glsum", **_ATTN},
    "attn32_l2": {"equivariant_layer_sizes": [128, 32], "activation": "glsum",
                  "feature_norm": "l2", **_ATTN},
    "attn32_none": {"equivariant_layer_sizes": [128, 32], "activation": "none", **_ATTN},
    "attn32_glpinv": {"equivariant_layer_sizes": [128, 32], "activation": "gl_pinv",
                      **_ATTN},
    "attn32_d3": {"equivariant_layer_sizes": [128, 64, 32], "activation": "glsum",
                  **_ATTN},
    "deepsets32": {"equivariant_layer_sizes": [128, 32], "activation": "glsum",
                   "head": "deepsets", "d_model": 128, "dropout": 0.1},
    "mlp32": {"equivariant_layer_sizes": [128, 32], "activation": "glsum",
              "head": "mlp", "head_layer_sizes": [64]},
}

# ── round 2 ──────────────────────────────────────────────────────────────────
#
# Stage 1 showed EVERY un-normalised multi-layer arm sitting at chance (0.500), for a
# reason that has nothing to do with GL depth: stacking Kaiming-initialised
# ``EquivariantLinearLayer``s shrinks the cell-feature norm by ~40x per layer
# (2.7e-03 at [128], 7e-05 at [128,32], 8.9e-06 at [128,64,32]). A dense head
# initialised for unit-scale inputs then emits ~zero logits and never moves. So the
# depth and activation contrasts as first run measured an initialisation artefact, not
# the thing they were meant to measure.
#
# Round 2 re-runs every one of them WITH ``feature_norm="l2"``, which restores unit
# scale, so "does a GL layer help" is finally a fair question. The three ``base_d*``
# arms are the most direct form of it: identical trunk width and identical dense head
# to ``base_l2``, one and two extra equivariant layers with a GL activation between.

_L2 = {"feature_norm": "l2"}

ARCHS.update({
    # production trunk + REAL GL depth, head held fixed at base_l2's
    "base_d2_l2": {"equivariant_layer_sizes": [128, 128], "activation": "glsum",
                   "head": "mlp", "head_layer_sizes": [64], **_L2},
    # the control for the above: the extra layer WITHOUT the activation, so the two
    # separate "more linear capacity" from "a GL non-linearity".
    "base_d2none_l2": {"equivariant_layer_sizes": [128, 128], "activation": "none",
                       "head": "mlp", "head_layer_sizes": [64], **_L2},
    "base_d3_l2": {"equivariant_layer_sizes": [128, 128, 128], "activation": "glsum",
                   "head": "mlp", "head_layer_sizes": [64], **_L2},
    # the stage-1 arms that died, now numerically viable
    "attn32_none_l2": {"equivariant_layer_sizes": [128, 32], "activation": "none",
                       **_L2, **_ATTN},
    "attn32_glpinv_l2": {"equivariant_layer_sizes": [128, 32], "activation": "gl_pinv",
                         **_L2, **_ATTN},
    "attn32_d3_l2": {"equivariant_layer_sizes": [128, 64, 32], "activation": "glsum",
                     **_L2, **_ATTN},
    "deepsets32_l2": {"equivariant_layer_sizes": [128, 32], "activation": "glsum",
                      "head": "deepsets", "d_model": 128, "dropout": 0.1, **_L2},
    "mlp32_l2": {"equivariant_layer_sizes": [128, 32], "activation": "glsum",
                 "head": "mlp", "head_layer_sizes": [64], **_L2},
})

# ── round 3: the complexity ladder ───────────────────────────────────────────
#
# The parent zoo already established that ADDING capacity (attention, deepsets, GL
# depth) does not beat the simplest arm ``base_l2`` (one equivariant layer d=128 +
# L2 feature-norm + 117 M dense head). The open question is the other direction: how
# far can we SHRINK before held-out calibration R2 falls off — and does it collapse
# all the way to a pure BILINEAR readout (empty head -> one Linear over the flattened
# projected dW = the exact ``<T, dW_proj> + b`` template, the regression analogue of
# the single-cell bilinear meta-classifiers).
#
# Two axes, both on the single-layer, no-op-activation, L2-normalised trunk (so the
# ONLY thing that moves is the reducible dimension):
#   width ladder  w{d}_l2    : shrink the equivariant width d, keep base_l2's [64] head
#   bilinear      bilin{d}_l2: empty head (head_layer_sizes=[]) -> linear-in-dW readout
# ``bilin128`` (no L2-norm) is the one deliberate control: the EXACT linear-in-dW form
# with no normalisation nonlinearity — the truest bilinear-template analogue.
#
# feature_norm="l2" is mandatory on every normalised rung (un-normalised trunks sit at
# chance, see round-2 note above).

_MLP64 = {"activation": "none", "head": "mlp", "head_layer_sizes": [64]}
_BILIN = {"activation": "none", "head": "mlp", "head_layer_sizes": []}

ARCHS.update({
    # width ladder: shrink d only, base_l2's dense [64] head held fixed
    "w64_l2": {"equivariant_layer_sizes": [64], **_MLP64, **_L2},
    "w32_l2": {"equivariant_layer_sizes": [32], **_MLP64, **_L2},
    "w16_l2": {"equivariant_layer_sizes": [16], **_MLP64, **_L2},
    "w8_l2": {"equivariant_layer_sizes": [8], **_MLP64, **_L2},
    "w4_l2": {"equivariant_layer_sizes": [4], **_MLP64, **_L2},
    # bilinear rungs: empty head -> exact <T, dW_proj> + b, at three widths
    "bilin128_l2": {"equivariant_layer_sizes": [128], **_BILIN, **_L2},
    "bilin32_l2": {"equivariant_layer_sizes": [32], **_BILIN, **_L2},
    "bilin8_l2": {"equivariant_layer_sizes": [8], **_BILIN, **_L2},
    # control: exact linear-in-dW closed form, NO feature-norm (no nonlinearity at all)
    "bilin128": {"equivariant_layer_sizes": [128], **_BILIN},
})

__all__ = ["ARCHS"]
