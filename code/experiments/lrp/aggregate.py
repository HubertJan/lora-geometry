"""Aggregate per-rank LRP relevance to one **signed-net** score per (layer × component).

(migrated from SRC/src/discoveries/sentiment_crosstone_lrp/flows/aggregate.py)


The canonical recipe (``glad.lrp.analysis`` docstring; jobs 28e / emergent 06): for each
``(module, layer)`` cell combine A and B **signed** per rank first, then sum over rank —

    v_k          = A_signed[k] + B_signed[k]      # one signed value per rank
    signed(cell) = Σ_k v_k                        # PRIMARY: the signed net
    magnitude    = Σ_k |v_k|                       # abs at RANK level (for concentration)

The documented mistake is per-weight ``Σ|weight|`` (``A_mag + B_mag`` / ``total_mag``);
those fields are deliberately NOT used here.
"""

from __future__ import annotations

import numpy as np

MODULE_ORDER = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]
MODULE_SHORT = {
    "self_attn.q_proj": "Attn Q", "self_attn.k_proj": "Attn K",
    "self_attn.v_proj": "Attn V", "self_attn.o_proj": "Attn O",
    "mlp.gate_proj": "MLP Gate", "mlp.up_proj": "MLP Up", "mlp.down_proj": "MLP Down",
}


def cell_signed_net(prof: dict) -> list[dict]:
    """Per-(component, layer) signed net + rank-level magnitude from a per_rank profile.

    ``prof`` is :func:`glad.lrp.analysis.per_rank_relevance` output
    (``{module_key: {layer: {A_signed, B_signed, ...}}}``). Returns a flat list of
    ``{component, module_key, layer, signed, magnitude}``.
    """
    out: list[dict] = []
    for mk in prof:
        for layer, cell in prof[mk].items():
            v = np.asarray(cell["A_signed"], float) + np.asarray(cell["B_signed"], float)
            out.append({
                "component": MODULE_SHORT.get(mk, mk),
                "module_key": mk,
                "layer": int(layer),
                "signed": float(v.sum()),          # PRIMARY signed net
                "magnitude": float(np.abs(v).sum()),  # abs at rank level
            })
    return out


def signed_grid(cell_rows: list[dict], n_layers: int) -> np.ndarray:
    """Stack ``cell_signed_net`` rows into a ``(7 components × n_layers)`` signed matrix."""
    grid = np.full((len(MODULE_ORDER), n_layers), np.nan)
    idx = {mk: i for i, mk in enumerate(MODULE_ORDER)}
    for r in cell_rows:
        if r["module_key"] in idx and 0 <= r["layer"] < n_layers:
            grid[idx[r["module_key"]], r["layer"]] = r["signed"]
    return grid
