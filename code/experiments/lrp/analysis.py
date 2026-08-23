"""Aggregations of LRP attributions over the grouped LoRA layout.

(migrated from SRC/src/glad/lrp/analysis.py)


Operates on ``attr_dict`` as produced by unflattening a zennit attribution back
into the grouped shape (``{module_key: {LoraType.A: (1, n_layers, rank, d_in),
LoraType.B: (1, n_layers, d_out, rank)}}``, leading batch axis of 1) — see
``glad.modules.flattened_input_classifier`` / the discoveries' ``run_lrp_single``.

Two conventions are load-bearing (see ``feedback_lrp_signed_component_level_abs``):

* relevance is **signed**; the magnitude variant sums ``|relevance|`` *first*,
  at the component level, so a rank's internal cancellation cannot hide it —
  never per-weight ``Σ|weight|`` after a net sum;
* everything is normalised by ``norm`` = the target-class logit total, so
  profiles are comparable across adapters.

Promoted from ``per_rank_profiles`` in
``task_meta_classifier_lrp/flows/lrp_svd.py`` (the head-agnostic live copy of
``lrp_on_meta_classifiers/jobs/28_lrp-rank-relevance-svd_2026-06-29.py``, which
keeps its inline copy as a dated record) plus the aggregate from job 28d.
When the input ``grouped`` dict was SVD-balanced first
(:func:`glad.lora.svd.svd_balance_grouped`), rank index ``k`` is the ``k``-th
singular direction and these profiles answer "how much relevance do the top
singular directions carry".
"""

from __future__ import annotations

import numpy as np

from meta_model.lora.types import LoraType

__all__ = ["per_rank_aggregate", "per_rank_relevance"]


def per_rank_relevance(attr_dict: dict, norm: float | None) -> dict:
    """Per-(module, layer) per-rank signed & magnitude relevance for A and B.

    A is ``(rank, d_in)``  → sum over ``d_in``  (axis 1) gives one value per rank.
    B is ``(d_out, rank)`` → sum over ``d_out`` (axis 0) gives one value per rank.
    Everything normalised by the target-logit total ``norm`` (``0``/``None``
    falls back to 1 so a zero-logit decision degrades to unnormalised values
    instead of dividing by zero).

    Returns ``{module_key: {layer: {"A_signed", "A_mag", "B_signed", "B_mag",
    "total_mag"}}}`` with per-rank ndarrays and a scalar ``total_mag``.
    """
    norm = norm if norm not in (0.0, None) else 1.0
    out: dict = {}
    for mk in attr_dict:
        A = attr_dict[mk][LoraType.A][0]  # (n_layers, rank, d_in)
        B = attr_dict[mk][LoraType.B][0]  # (n_layers, d_out, rank)
        n_l = A.shape[0]
        layers = {}
        for layer in range(n_l):
            ra = (A[layer] / norm).numpy()  # (rank, d_in)
            rb = (B[layer] / norm).numpy()  # (d_out, rank)
            layers[layer] = {
                "A_signed": ra.sum(axis=1),
                "A_mag": np.abs(ra).sum(axis=1),
                "B_signed": rb.sum(axis=0),
                "B_mag": np.abs(rb).sum(axis=0),
                "total_mag": float(np.abs(ra).sum() + np.abs(rb).sum()),
            }
        out[mk] = layers
    return out


def per_rank_aggregate(prof: dict) -> tuple[np.ndarray, np.ndarray]:
    """Collapse a :func:`per_rank_relevance` profile over all (module, layer) cells.

    Returns ``(S, M)`` — per-rank signed relevance ``S[k] = Σ (A_signed +
    B_signed)`` and magnitude ``M[k] = Σ (A_mag + B_mag)``.  Meaningful only
    when every cell shares one rank (i.e. a uniform-rank adapter, which is also
    what makes rank index ``k`` comparable across cells after SVD balancing).
    """
    first_module = next(iter(prof.values()))
    first_cell = first_module[next(iter(first_module))]
    R = len(first_cell["A_signed"])
    S = np.zeros(R)
    M = np.zeros(R)
    for mk in prof:
        for layer in prof[mk]:
            c = prof[mk][layer]
            S += c["A_signed"] + c["B_signed"]
            M += c["A_mag"] + c["B_mag"]
    return S, M
