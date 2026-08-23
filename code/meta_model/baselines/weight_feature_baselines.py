"""Migrated from SRC/src/discoveries/sst2_perf_prediction/flows/weight_feature_baselines.py.

Simple weight-feature baselines for the in-task SST2 performance regressor.

Campaign 3 (`RESEARCH_LOG.md`): is the deep equivariant meta-classifier
(`base_l2`, held-out acc R2 0.83) more complex than the task needs? This module
builds a ladder of deliberately simple models on the SAME per-adapter input the
deep model reads (`meta_model.materialization.default_transform` — the
grouped LoRA weights ``{module: {LoraType.A: [L,r,in], B: [L,out,r]}}``) and the
SAME 504-train / 63-test split, so the numbers drop straight into the arch
leaderboard's frame.

Two families of features, both derived from the effective per-cell update
``dW = B @ A`` (invariant to the LoRA gauge ``A -> G A, B -> B G^-1``, so two
adapters that induce the same update look the same):

* **exact summaries** — per-cell Frobenius norm ``||dW||_F`` and the singular-value
  spectrum ``sigma(dW)`` (obtained cheaply from the ``r x r`` core of a per-factor
  QR, never forming the full ``out x in`` matrix). Fed to ``RidgeCV`` / cosine kNN.
* **full directional dW** — the exact Frobenius Gram of ``dW`` is infeasible for the
  8192-wide MLP cells, so each cell is measured by a two-sided Gaussian sketch
  ``S (B A) R = (S B)(A R)`` and the linear kernel ``K = sum_cells <dW_a, dW_b>`` is
  accumulated. Fed to kernel ridge / cosine kNN. The per-cell sketch is noisy but K
  sums 112 cells; report the off-diagonal rank-agreement between two independent
  sketch seeds as a stability check (sketching only *loses* signal, so a sketched
  linear probe is a conservative LOWER bound on the exact linear-on-dW ceiling).

Findings frozen into the results store
(`2026-08-23_simple-baselines-vs-deep-regressor`): on accuracy the exact linear
probes cap at R2 0.19, the sketched dW linear probe reaches 0.33, cosine kNN 0.53,
vs the deep model's 0.83 — but for *ranking* the simple models already hit Spearman
0.82 (ridge) / 0.90 (kNN), above the parent's un-normalised regressor (0.705).

The feature-extraction functions read adapter weights via the library
(``meta_model.materialization.default_transform``); the fitting functions are
pure sklearn/numpy so they run from a frozen feature table alone.
"""

from __future__ import annotations

import glob
import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

#: The six per-adapter SST2 targets (columns of the pool scores CSV).
SST2_TARGETS: list[str] = [
    "accuracy", "f1_macro", "auroc", "brier", "mean_confidence", "nll",
]
LORA_RANK = 16


# ── feature extraction (imports glad) ────────────────────────────────────────

def _cell_singular_values(b_mat, a_mat, rank: int = LORA_RANK):
    """Sorted-desc singular values (len ``rank``, zero-padded) of ``dW = B @ A``.

    ``a_mat``: ``(r, in)``; ``b_mat``: ``(out, r)``. The nonzero singular values of
    ``B @ A`` equal those of ``R_b @ R_a^T`` (an ``r x r`` matrix, from the per-factor
    QR), so the full ``out x in`` product is never formed.
    """
    import torch

    a_mat = a_mat.float()
    b_mat = b_mat.float()
    _, r_a = torch.linalg.qr(a_mat.T, mode="reduced")
    _, r_b = torch.linalg.qr(b_mat, mode="reduced")
    s = torch.linalg.svdvals(r_b @ r_a.T)
    out = torch.zeros(rank)
    out[: s.numel()] = s
    return out


def _resolve_safetensor(adapter_ref: str) -> str:
    """Resolve an adapter reference to its local ``adapter_model.safetensors``.

    Migration note: the original resolved an artifact-store qname via
    ``llm_pipeline`` (fetch + download). The standalone pool is a plain on-disk
    layout (``<pool_dir>/adapters/<__key__>/adapter_model.safetensors``), so this
    now accepts either the safetensors file path directly, or any directory /
    adapter key under which exactly one ``adapter_model.safetensors`` lives.
    """
    p = Path(adapter_ref)
    if p.is_file():
        return str(p)
    if p.is_dir():
        hits = glob.glob(
            os.path.join(str(p), "**", "adapter_model.safetensors"), recursive=True
        )
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"could not resolve adapter_model.safetensors from {adapter_ref!r}; pass a "
        "path to the file, or to a directory containing it "
        "(e.g. <pool_dir>/adapters/<__key__>)."
    )


def extract_weight_features(
    scores_df: pd.DataFrame,
    *,
    rank: int = LORA_RANK,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Build the gauge-invariant per-cell feature table for every adapter.

    ``scores_df`` must carry ``adapter_idx``, ``pool``, ``source``, ``adapter_qname``
    and the six :data:`SST2_TARGETS` columns (the parent's ``production_scores_v2.csv``).
    Returns one row per adapter with: the targets, ``gate_global_dw_norm`` (the single
    global ``||dW||`` gate feature), 112 ``norm.<cell>`` per-cell Frobenius norms, and
    112*rank ``sv.<cell>.<k>`` singular values. Each cell is ``<module>.l<layer>``.
    """
    import torch

    from meta_model.lora.types import LoraType
    from meta_model.materialization import default_transform

    records: list[dict[str, Any]] = []
    module_order: list[str] | None = None
    for i, row in enumerate(scores_df.reset_index(drop=True).to_dict("records")):
        grouped = default_transform(_resolve_safetensor(row["adapter_qname"]))
        if module_order is None:
            module_order = sorted(grouped.keys())
        rec: dict[str, Any] = {
            "adapter_idx": int(row["adapter_idx"]),
            "pool": row["pool"],
            "source": row["source"],
            **{t: float(row[t]) for t in SST2_TARGETS},
        }
        total_sq = 0.0
        for m in module_order:
            a_w = grouped[m][LoraType.A]  # (L, r, in)
            b_w = grouped[m][LoraType.B]  # (L, out, r)
            for lyr in range(a_w.shape[0]):
                sv = _cell_singular_values(b_w[lyr], a_w[lyr], rank)
                fro = float((sv ** 2).sum().sqrt())
                total_sq += fro * fro
                cell = f"{m}.l{lyr}"
                rec[f"norm.{cell}"] = fro
                for k in range(rank):
                    rec[f"sv.{cell}.{k}"] = float(sv[k])
        rec["gate_global_dw_norm"] = float(np.sqrt(total_sq))
        records.append(rec)
        if progress is not None and (i + 1) % 50 == 0:
            progress(f"{i + 1}/{len(scores_df)} adapters featurised")
    return pd.DataFrame.from_records(records)


# ── metrics ──────────────────────────────────────────────────────────────────

def score_predictions(y_true, y_pred) -> dict[str, float]:
    """Held-out r2 / spearman / pearson / mae; constant predictors report rho=0."""
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import mean_absolute_error, r2_score

    if np.std(y_pred) == 0:  # the mean floor
        sp = pe = 0.0
    else:
        sp = float(spearmanr(y_true, y_pred).statistic)
        pe = float(pearsonr(y_true, y_pred)[0])
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": sp,
        "pearson": pe,
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


# ── exact-feature baselines (pure sklearn) ───────────────────────────────────

def fit_exact_baselines(
    features_df: pd.DataFrame,
    *,
    heads: Iterable[str] = SST2_TARGETS,
    alphas=np.logspace(-3, 5, 25),
    knn_grid=(3, 5, 7, 10, 15, 20),
    cv_seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """mean / gate_lin / norms_ridge / spectral_ridge / spectral_knn, per head.

    Model selection (ridge alpha, kNN k) is 5-fold CV on the ``pool=="train"`` rows;
    the ``pool=="test"`` rows are scored once. Returns (leaderboard, chosen-k-by-head).
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import Normalizer, StandardScaler

    tr = features_df[features_df.pool == "train"].reset_index(drop=True)
    te = features_df[features_df.pool == "test"].reset_index(drop=True)
    cv = KFold(n_splits=5, shuffle=True, random_state=cv_seed)
    norm_cols = [c for c in features_df.columns if c.startswith("norm.")]
    sv_cols = [c for c in features_df.columns if c.startswith("sv.")]
    feature_sets = {
        "gate_lin": ["gate_global_dw_norm"],
        "norms_ridge": norm_cols,
        "spectral_ridge": sv_cols,
        "spectral_knn": sv_cols,
    }

    def build(model_name: str):
        if model_name == "spectral_knn":
            pipe = make_pipeline(
                Normalizer(norm="l2"), KNeighborsRegressor(weights="distance"))
            return GridSearchCV(
                pipe, {"kneighborsregressor__n_neighbors": list(knn_grid)},
                cv=cv, scoring="neg_mean_squared_error")
        return make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))

    rows: list[dict[str, Any]] = []
    chosen: dict[str, int] = {}
    for model_name, cols in feature_sets.items():
        x_tr, x_te = tr[cols].to_numpy(), te[cols].to_numpy()
        for head in heads:
            est = build(model_name)
            est.fit(x_tr, tr[head].to_numpy())
            pred = est.predict(x_te)
            rows.append({"model": model_name, "head": head,
                         "n_features": len(cols),
                         **score_predictions(te[head].to_numpy(), pred)})
            if model_name == "spectral_knn":
                chosen[head] = int(
                    est.best_params_["kneighborsregressor__n_neighbors"])
    for head in heads:  # mean floor
        pred = np.full(len(te), tr[head].to_numpy().mean())
        rows.append({"model": "mean", "head": head, "n_features": 0,
                     **score_predictions(te[head].to_numpy(), pred)})
    return pd.DataFrame(rows), chosen


# ── full-directional dW sketch kernel (imports glad) ─────────────────────────

def build_dw_sketch_kernel(
    scores_df: pd.DataFrame,
    *,
    sketch_dim: int = 128,
    seed: int = 42,
    progress: Callable[[str], None] | None = None,
) -> np.ndarray:
    """Accumulate the linear kernel ``K = sum_cells <phi_a, phi_b>`` over adapters.

    ``phi_a^c = (S_c B_a^c)(A_a^c R_c)`` is a two-sided Gaussian sketch of ``dW`` per
    cell (``sketch_dim`` per side); K is streamed so ``sketch_dim`` can be large
    cheaply. Row order matches ``scores_df`` (used to slice train/test).
    """
    import torch

    from meta_model.lora.types import LoraType
    from meta_model.materialization import default_transform

    gen = torch.Generator().manual_seed(seed)
    s_cache: dict[tuple[int, int], "torch.Tensor"] = {}

    def smat(rows: int, cols: int):
        key = (rows, cols)
        if key not in s_cache:
            s_cache[key] = torch.randn(rows, cols, generator=gen) / np.sqrt(rows)
        return s_cache[key]

    n = len(scores_df)
    phi_mat: np.ndarray | None = None
    for i, row in enumerate(scores_df.reset_index(drop=True).to_dict("records")):
        grouped = default_transform(_resolve_safetensor(row["adapter_qname"]))
        feats = []
        for m in sorted(grouped.keys()):
            a_w = grouped[m][LoraType.A].float()
            b_w = grouped[m][LoraType.B].float()
            s = smat(sketch_dim, b_w.shape[1])
            r = smat(a_w.shape[2], sketch_dim)
            for lyr in range(a_w.shape[0]):
                feats.append(((s @ b_w[lyr]) @ (a_w[lyr] @ r)).reshape(-1))
        phi = torch.cat(feats).numpy()
        if phi_mat is None:
            phi_mat = np.empty((n, phi.shape[0]), dtype=np.float32)
        phi_mat[i] = phi
        if progress is not None and (i + 1) % 100 == 0:
            progress(f"{i + 1}/{n} adapters sketched (seed {seed})")
    return (phi_mat @ phi_mat.T).astype(np.float64)


def _kernel_ridge_predict(k_train, k_test, y, alpha):
    a = np.linalg.solve(k_train + alpha * np.eye(k_train.shape[0]), y)
    return k_test @ a


def fit_dw_kernel_baselines(
    kernel: np.ndarray,
    scores_df: pd.DataFrame,
    *,
    heads: Iterable[str] = SST2_TARGETS,
    alphas=np.logspace(-6, 3, 28),
    knn_grid=(3, 5, 7, 10, 15, 20),
    cv_seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """dw_ridge (kernel ridge) + dw_knn_cos (cosine kNN) on the dW kernel, per head.

    ``kernel`` rows/cols align with ``scores_df`` order. Model selection is 5-fold CV
    on the train rows; test rows scored once.
    """
    from sklearn.model_selection import KFold

    df = scores_df.reset_index(drop=True)
    tr_idx = df.index[df.pool == "train"].to_numpy()
    te_idx = df.index[df.pool == "test"].to_numpy()
    k_tr = kernel[np.ix_(tr_idx, tr_idx)]
    k_te = kernel[np.ix_(te_idx, tr_idx)]
    diag = np.sqrt(np.diag(kernel))
    kcos = kernel / np.outer(diag, diag)
    cv = KFold(n_splits=5, shuffle=True, random_state=cv_seed)

    rows: list[dict[str, Any]] = []
    chosen: dict[str, dict[str, float]] = {}
    for head in heads:
        y_tr = df.loc[tr_idx, head].to_numpy()
        y_te = df.loc[te_idx, head].to_numpy()
        mu = y_tr.mean()

        best_a, best_cv = alphas[0], np.inf
        for alpha in alphas:
            errs = []
            for trn, val in cv.split(tr_idx):
                m = y_tr[trn].mean()
                pred = _kernel_ridge_predict(
                    k_tr[np.ix_(trn, trn)], k_tr[np.ix_(val, trn)],
                    y_tr[trn] - m, alpha) + m
                errs.append(np.mean((y_tr[val] - pred) ** 2))
            if np.mean(errs) < best_cv:
                best_cv, best_a = np.mean(errs), alpha
        pred = _kernel_ridge_predict(k_tr, k_te, y_tr - mu, best_a) + mu
        rows.append({"model": "dw_ridge", "head": head,
                     "param": float(best_a),
                     **score_predictions(y_te, pred)})

        ctr = kcos[np.ix_(tr_idx, tr_idx)]

        def knn_predict(sims, targets, k):
            preds = []
            for r_ in range(sims.shape[0]):
                nn = np.argsort(-sims[r_])[:k]
                w = np.clip(sims[r_][nn], 1e-6, None)
                preds.append(np.average(targets[nn], weights=w))
            return np.array(preds)

        best_k, best_cv = knn_grid[0], np.inf
        for k in knn_grid:
            errs = []
            for trn, val in cv.split(tr_idx):
                pred = knn_predict(ctr[np.ix_(val, trn)], y_tr[trn], k)
                errs.append(np.mean((y_tr[val] - pred) ** 2))
            if np.mean(errs) < best_cv:
                best_cv, best_k = np.mean(errs), k
        pred = knn_predict(kcos[np.ix_(te_idx, tr_idx)], y_tr, best_k)
        rows.append({"model": "dw_knn_cos", "head": head,
                     "param": float(best_k),
                     **score_predictions(y_te, pred)})
        chosen[head] = {"ridge_alpha": float(best_a), "knn_k": int(best_k)}
    return pd.DataFrame(rows), chosen


def kernel_stability(scores_df: pd.DataFrame, *, sketch_dim: int = 128,
                     seeds: tuple[int, int] = (42, 1234),
                     progress: Callable[[str], None] | None = None) -> float:
    """Off-diagonal Spearman between the dW kernels from two independent sketch seeds.

    A high value means the *summed* (112-cell) kernel is a stable estimate of the true
    dW geometry despite each per-cell sketch being noisy.
    """
    from scipy.stats import spearmanr

    k1 = build_dw_sketch_kernel(scores_df, sketch_dim=sketch_dim, seed=seeds[0],
                                progress=progress)
    k2 = build_dw_sketch_kernel(scores_df, sketch_dim=sketch_dim, seed=seeds[1],
                                progress=progress)
    iu = np.triu_indices(len(scores_df), k=1)
    return float(spearmanr(k1[iu], k2[iu]).statistic)


def load_scores(scores_csv: str | Path) -> pd.DataFrame:
    """Load the pool scores CSV (parent's ``production_scores_v2.csv``)."""
    return pd.read_csv(scores_csv)
