"""Base-model reference axes for interpreting the regressor's (u,v) directions.

(migrated from SRC/src/discoveries/sst2_perf_regressor_uv/flows/axes.py)

The functional decision axis for SST2 sentiment is delta = emb(positive) - emb(negative)
in the residual/embedding space (2048-d for Llama-3.2-1B).  Every cell's residual-side
singular vector (u for down/o writers, v for the readers) lives in that same 2048-d space,
so it can be (a) cosine-compared with delta and (b) logit-lensed against the tied
embeddings E.  This mirrors the reference uv-synthesis machinery.

The tied embeddings + tokenizer are obtained through ``common.hf`` (``load_base_model`` /
``load_tokenizer``) rather than a hardcoded HF snapshot path.
"""

from __future__ import annotations

import numpy as np
import torch


def load_axes():
    """Return dict with tied embeddings + sentiment axes (all float64 numpy / torch).

    Keys: E (vocab,2048 torch f32), E_norm (unit rows, torch f32), tok (tokenizer),
    delta (2048,), delta_sp (2048,) numpy float64; plus pos/neg token ids.
    """
    from common.hf import load_base_model, load_tokenizer

    tok = load_tokenizer()
    pos_id = tok("positive", add_special_tokens=False).input_ids[0]
    neg_id = tok("negative", add_special_tokens=False).input_ids[0]
    pos_sp = tok(" positive", add_special_tokens=False).input_ids[0]
    neg_sp = tok(" negative", add_special_tokens=False).input_ids[0]
    model = load_base_model()
    E = model.get_input_embeddings().weight.detach().float()  # (vocab, 2048)
    E_norm = E / (E.norm(dim=-1, keepdim=True) + 1e-12)
    delta = (E[pos_id] - E[neg_id]).double().numpy()
    delta_sp = (E[pos_sp] - E[neg_sp]).double().numpy()
    return {
        "E": E, "E_norm": E_norm, "tok": tok,
        "delta": delta, "delta_sp": delta_sp,
        "pos_id": int(pos_id), "neg_id": int(neg_id),
        "pos_sp": int(pos_sp), "neg_sp": int(neg_sp),
    }


def cos(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def subspace_capture(basis: np.ndarray, vec: np.ndarray) -> float:
    """||P_span(basis) vec|| / ||vec||, basis = (k, dim) rows (need not be orthonormal).

    Fraction of ``vec`` captured by the row-span of ``basis`` — a gauge-free measure of
    'does this subspace contain the direction vec'.  1 = fully inside, 0 = orthogonal.
    """
    B = np.asarray(basis, float)
    v = np.asarray(vec, float)
    Q, _ = np.linalg.qr(B.T)             # (dim, k) orthonormal columns
    proj = Q @ (Q.T @ v)
    return float(np.linalg.norm(proj) / (np.linalg.norm(v) + 1e-30))


def logit_lens(vec: np.ndarray, tok, E_norm: torch.Tensor, top_k: int = 15):
    """Top-k / bottom-k tokens by cosine of vec (2048,) with tied embeddings."""
    if vec.shape[0] != E_norm.shape[1]:
        return None
    unit = torch.from_numpy(np.asarray(vec) / (np.linalg.norm(vec) + 1e-12)).float()
    c = (E_norm @ unit).numpy()
    top = np.argsort(-c)[:top_k]; bot = np.argsort(c)[:top_k]
    return {
        "top_tokens": [tok.decode([int(i)]) for i in top],
        "top_cos": c[top].tolist(),
        "bot_tokens": [tok.decode([int(i)]) for i in bot],
        "bot_cos": c[bot].tolist(),
    }
