"""Migrated from SRC/src/discoveries/sst2_perf_prediction/flows/geometry_baseline.py.

Tier-A + Tier-B weight-space geometry baselines for SST2 perf prediction.

Campaign question (`RESEARCH_LOG.md`): the deep meta-classifier reaches held-out
acc R2 0.83; the existing *simple* baselines (`weight_feature_baselines.py`) show raw
per-cell Frobenius / full-spectrum ridge caps at R2 ~0.19 (kNN ~0.53). Do the *richer*
geometry descriptors — Tier-A derived spectral scalars and Tier-B base-relative metrics
— predict adapter performance, and (the honest test) do they add anything beyond a
**hyperparameter-only** regressor on the training recipe `{shards_total, epochs,
label_noise}` that the pool was designed around?

Everything here is CPU-only. Two feature families, both built from the effective per-cell
update ``dW = (alpha/r)*B@A`` (gauge-invariant to ``A->GA, B->BG^-1``):

* **Tier A** (`extract_geometry_features`, prefix ``ta.``) — scalars of the dW singular
  spectrum sigma = sigma(dW): spectral norm, nuclear norm, stable rank, effective/
  participation rank, spectral entropy, log-log decay slope (crude — only rank=16 points),
  spectrum kurtosis/skew, and the base-relative norm ratio ``||dW||_F / ||W0||_F``.
* **Tier B** (prefix ``tb.``) — dW *relative to the base weight* W0's leading singular
  subspace (top-k SVD, k=32, precomputed once via :func:`build_base_subspaces`):
  amplification (energy of dW inside col/row span of W0's top-k), novel-direction fraction,
  principal angles between col(B) and W0's leading left subspace, and spectral displacement
  of W0's top singular values under dW. Ratio metrics are scale-invariant; ``sdisp*`` and
  the norm ratio use the true alpha/r-scaled update.

The feature-extraction functions read adapter/base weights via the library and
torch; the fitting functions
(:func:`fit_geometry_leaderboard`, :func:`incremental_over_hp`) are pure sklearn/numpy and
reuse :func:`weight_feature_baselines.score_predictions` and the same 504-train/63-test
split on the ``pool`` column, so the numbers drop straight into the campaign leaderboard.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import pandas as pd

from meta_model.baselines.weight_feature_baselines import (
    LORA_RANK,
    SST2_TARGETS,
    _cell_singular_values,
    _resolve_safetensor,
    score_predictions,
)

#: PEFT effective LoRA scale for this pool (adapter_config.json: alpha=32, r=16).
ALPHA_OVER_R = 2.0
#: Base model every adapter in the pool was trained on.
BASE_MODEL = "meta-llama/Llama-3.2-1B"
#: Rank of the base-weight leading subspace used for the Tier-B projections.
BASE_SUBSPACE_K = 32
#: Hyperparameter columns that generated the performance spread (the honest baseline).
HP_COLS = ["shards_total", "epochs", "label_noise"]


# ── base-weight leading subspaces (Tier-B precompute, one-time) ───────────────

def _hf_cache_roots() -> list[str]:
    """Candidate HF hub-cache roots, resolved from the environment at *call* time.

    ``huggingface_hub.constants.HF_HUB_CACHE`` is frozen at import; when ``HF_HOME`` is set
    by ``load_dotenv`` after that import it can be stale, so read the env directly here.
    """
    import os

    roots: list[str] = []
    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(var):
            roots.append(os.environ[var])
    if os.environ.get("HF_HOME"):
        roots.append(os.path.join(os.environ["HF_HOME"], "hub"))
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        roots.append(HF_HUB_CACHE)
    except Exception:
        pass
    roots.append(os.path.expanduser("~/.cache/huggingface/hub"))
    # dedupe, keep order
    seen: set[str] = set()
    return [r for r in roots if not (r in seen or seen.add(r))]


def _resolve_local_model_dir(model_name: str) -> str | None:
    """Local HF-cache snapshot dir for ``model_name``, or ``None`` if not cached.

    Reads the cache layout directly (``<root>/models--org--name/snapshots/<commit>``,
    commit from ``refs/main``) so the base weights load with no network / hub calls.
    """
    import glob
    import os

    repo = "models--" + model_name.replace("/", "--")
    for root in _hf_cache_roots():
        base = os.path.join(root, repo)
        snap = None
        ref = os.path.join(base, "refs", "main")
        if os.path.exists(ref):
            with open(ref) as fh:
                snap = os.path.join(base, "snapshots", fh.read().strip())
        if snap is None or not os.path.isdir(snap):
            cands = sorted(glob.glob(os.path.join(base, "snapshots", "*")))
            snap = cands[-1] if cands else None
        # require a COMPLETE snapshot: config + a weights file (a partial cache may hold
        # only config.json — e.g. a stray ~/.cache alongside the real /data cache).
        if (snap and os.path.exists(os.path.join(snap, "config.json"))
                and (os.path.exists(os.path.join(snap, "model.safetensors"))
                     or os.path.exists(os.path.join(snap, "pytorch_model.bin"))
                     or glob.glob(os.path.join(snap, "*.safetensors")))):
            return snap
    return None


def build_base_subspaces(
    k: int = BASE_SUBSPACE_K,
    *,
    model_name: str = BASE_MODEL,
    progress: Callable[[str], None] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Top-``k`` SVD of every ``(module, layer)`` base weight ``W0`` of ``model_name``.

    Returns ``{cell: {"U": (out,k), "s": (k,), "V": (in,k), "fro": scalar}}`` where the
    cell key is ``"<module>.l<layer>"`` matching the adapter grouped-dict module keys
    (e.g. ``self_attn.q_proj.l3``). ``W0`` is HF ``nn.Linear.weight`` of shape ``(out,in)``,
    the same orientation as ``dW = B@A``. Loaded on CPU in float32.
    """
    import time

    import torch

    from common.hf import load_base_model

    # W0 is fully cached under HF_HOME. Resolve the local snapshot DIRECTORY from the cache
    # and hand that path to the loader: given an existing local dir, transformers loads
    # directly with zero hub calls — avoids the gated repo entirely. The cache lives on a
    # shared (networked) FS whose reads occasionally blip, so retry resolve+load a few times.
    # Base weights load in float32 (geometry needs full precision), overriding the
    # library's fp16 default via common.hf.load_base_model's torch_dtype arg.
    model = None
    for attempt in range(6):
        try:
            src = _resolve_local_model_dir(model_name) or model_name
            model = load_base_model(src, torch_dtype="float32")
            break
        except OSError as exc:  # transient cache/network read failure
            if attempt == 5:
                raise
            if progress is not None:
                progress(f"base model load retry {attempt + 1}/6 ({type(exc).__name__})")
            time.sleep(2.0 * (attempt + 1))
    assert model is not None
    modules = [
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    ]
    out: dict[str, dict[str, np.ndarray]] = {}
    layers = model.model.layers
    for li, layer in enumerate(layers):
        for m in modules:
            parent, leaf = m.split(".")
            w = getattr(getattr(layer, parent), leaf).weight.detach().float()  # (out,in)
            kk = min(k, w.shape[0], w.shape[1])
            # top-k SVD without forming the full Gram; svd_lowrank oversamples internally.
            U, s, V = torch.svd_lowrank(w, q=min(kk + 8, min(w.shape)), niter=4)
            U, s, V = U[:, :kk], s[:kk], V[:, :kk]  # W0 ~ U diag(s) V^T
            out[f"{m}.l{li}"] = {
                "U": U.contiguous().numpy(),
                "s": s.numpy(),
                "V": V.contiguous().numpy(),
                "fro": float(w.norm()),
            }
        if progress is not None:
            progress(f"base subspaces: layer {li + 1}/{len(layers)}")
    del model
    return out


def save_base_subspaces(base: dict[str, dict[str, np.ndarray]], path: str) -> None:
    """Flatten the nested base-subspace dict into a single ``.npz``."""
    flat: dict[str, np.ndarray] = {}
    for cell, d in base.items():
        for key, arr in d.items():
            flat[f"{cell}|{key}"] = np.asarray(arr)
    np.savez(path, **flat)


def load_base_subspaces(path: str) -> dict[str, dict[str, np.ndarray]]:
    """Inverse of :func:`save_base_subspaces`."""
    z = np.load(path)
    out: dict[str, dict[str, np.ndarray]] = {}
    for flat_key in z.files:
        cell, key = flat_key.split("|")
        out.setdefault(cell, {})[key] = z[flat_key]
    return out


# ── per-cell metric kernels ──────────────────────────────────────────────────

def _tierA_cell(sv: np.ndarray) -> dict[str, float]:
    """Tier-A spectral scalars from the (already alpha/r-scaled) dW spectrum ``sv``.

    ``sv`` is sorted-desc, length rank (zero-padded). Shape statistics (stable rank,
    effective rank, entropy, decay, kurtosis, skew) are scale-invariant; ``specnorm``
    and ``nuclear`` carry the alpha/r scale.
    """
    s = np.sort(sv[sv > 0])[::-1]  # positive, desc
    fro2 = float((s ** 2).sum())
    if s.size == 0 or fro2 == 0.0:
        return {kk: 0.0 for kk in
                ("specnorm", "nuclear", "stable_rank", "eff_rank", "spec_entropy",
                 "decay", "kurtosis", "skew")}
    p = (s ** 2) / fro2                      # spectral energy distribution
    entropy = float(-(p * np.log(p)).sum())
    # log-log decay slope (crude: <=rank points); intercept-free least squares on slope.
    if s.size >= 2:
        x = np.log(np.arange(1, s.size + 1))
        y = np.log(s)
        decay = float(np.polyfit(x, y, 1)[0])
    else:
        decay = 0.0
    mu, sd = float(s.mean()), float(s.std())
    if sd > 0:
        z = (s - mu) / sd
        skew = float((z ** 3).mean())
        kurt = float((z ** 4).mean() - 3.0)
    else:
        skew = kurt = 0.0
    return {
        "specnorm": float(s[0]),
        "nuclear": float(s.sum()),
        "stable_rank": fro2 / float(s[0] ** 2),
        "eff_rank": float(np.exp(entropy)),
        "spec_entropy": entropy,
        "decay": decay,
        "kurtosis": kurt,
        "skew": skew,
    }


def _tierB_cell(a: np.ndarray, b: np.ndarray, base: dict[str, np.ndarray],
                fro_dw: float) -> dict[str, float]:
    """Tier-B base-relative scalars for one cell.

    ``a``: (r,in), ``b``: (out,r) — the *raw* LoRA factors. ``base``: ``{U,s,V,fro}`` top-k
    SVD of W0 for this cell. ``fro_dw = ||dW||_F`` with the alpha/r scale already applied.
    All energy ratios cancel the alpha/r scale; ``sdisp*`` reintroduces it via ``c``.
    """
    import numpy.linalg as la

    c = ALPHA_OVER_R
    U, s0, V = base["U"], base["s"], base["V"]     # (out,k),(k,),(in,k)
    # projections built from small matrices only — dW = c * b @ a is never formed.
    UtB = U.T @ b                                   # (k,r)
    AV = a @ V                                       # (r,k)
    fro_ba = fro_dw / c if c else fro_dw             # ||B@A||_F (unscaled)
    if fro_ba <= 0:
        return {"amp": 0.0, "novel": 0.0, "pangle_cos2": 0.0, "pangle_min": 0.0,
                **{f"sdisp{i}": 0.0 for i in range(1, 5)}}
    # amplification: energy of dW inside W0's top-k left AND right subspace.
    core = UtB @ AV                                  # (k,k) = U^T (B A) V
    amp = float(la.norm(core) / fro_ba)
    # novel-direction fraction: 1 - fraction of dW energy in W0's top-k left subspace.
    left_proj = UtB @ a                              # (k,in) = U^T (B A)
    novel = float(1.0 - la.norm(left_proj) / fro_ba)
    # principal angles between col(B) and W0's leading left subspace U.
    QB, _ = la.qr(b)                                 # (out,r) orthonormal basis of col(B)
    r = b.shape[1]
    cos_sv = la.svd(QB.T @ U, compute_uv=False)      # cos of principal angles, len<=r
    pangle_cos2 = float((cos_sv ** 2).sum() / r)
    pangle_min = float(np.arccos(np.clip(cos_sv.max(), -1.0, 1.0)))  # smallest angle
    # spectral displacement of W0's top singular values under dW, in W0's top-k basis:
    # U^T (W0 + dW) V = diag(s0) + c * core  (U^T W0 V ~ diag(s0)).
    M = np.diag(s0) + c * core
    sM = la.svd(M, compute_uv=False)                 # perturbed top singular values
    sdisp = {f"sdisp{i}": float(sM[i - 1] - s0[i - 1]) for i in range(1, 5)}
    return {"amp": amp, "novel": novel, "pangle_cos2": pangle_cos2,
            "pangle_min": pangle_min, **sdisp}


# ── full feature table ───────────────────────────────────────────────────────

def extract_geometry_features(
    scores_df: pd.DataFrame,
    base: dict[str, dict[str, np.ndarray]],
    *,
    rank: int = LORA_RANK,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """One-pass Tier-A + Tier-B feature table for every adapter in ``scores_df``.

    ``scores_df`` must carry ``adapter_idx, pool, source, adapter_qname``, the six
    :data:`SST2_TARGETS`, and the :data:`HP_COLS`. ``base`` is the output of
    :func:`build_base_subspaces` (same base model, cells keyed ``<module>.l<layer>``).

    Emits per adapter: the targets + HP columns carried through, then per-cell columns
    ``ta.<metric>.<cell>`` (8 Tier-A scalars), ``ta.relnorm.<cell>``,
    ``tb.<metric>.<cell>`` (8 Tier-B scalars), plus a compact per-module mean of every
    metric (``ta.<metric>.mean.<module>`` / ``tb.<metric>.mean.<module>``). Also keeps
    the raw ``sv.<cell>.<k>``, ``norm.<cell>`` and ``gate_global_dw_norm`` columns so the
    reference spectral baselines can be reproduced from the same table.
    """
    import torch

    from meta_model.lora.types import LoraType
    from meta_model.materialization import default_transform

    carry = ["adapter_idx", "pool", "source", *SST2_TARGETS, *HP_COLS]
    records: list[dict[str, Any]] = []
    module_order: list[str] | None = None
    rows = scores_df.reset_index(drop=True).to_dict("records")
    for i, row in enumerate(rows):
        grouped = default_transform(_resolve_safetensor(row["adapter_qname"]))
        if module_order is None:
            module_order = sorted(grouped.keys())
        rec: dict[str, Any] = {c: row[c] for c in carry if c in row}
        rec["adapter_idx"] = int(row["adapter_idx"])
        # per-module accumulators for the compact mean variant
        acc: dict[str, dict[str, list[float]]] = {}
        total_sq = 0.0
        for m in module_order:
            a_w = grouped[m][LoraType.A]  # (L,r,in)
            b_w = grouped[m][LoraType.B]  # (L,out,r)
            for lyr in range(a_w.shape[0]):
                cell = f"{m}.l{lyr}"
                a = a_w[lyr].float().numpy()
                b = b_w[lyr].float().numpy()
                sv_raw = _cell_singular_values(
                    torch.from_numpy(b), torch.from_numpy(a), rank).numpy()
                fro_ba = float(np.sqrt((sv_raw ** 2).sum()))
                fro_dw = ALPHA_OVER_R * fro_ba
                total_sq += fro_dw * fro_dw
                # raw columns (reproduce existing spectral/norm baselines)
                rec[f"norm.{cell}"] = fro_dw
                for kk in range(rank):
                    rec[f"sv.{cell}.{kk}"] = float(sv_raw[kk])
                # Tier A
                ta = _tierA_cell(ALPHA_OVER_R * sv_raw)
                base_cell = base.get(cell)
                ta["relnorm"] = (fro_dw / base_cell["fro"]) if base_cell else 0.0
                for name, val in ta.items():
                    rec[f"ta.{name}.{cell}"] = val
                    acc.setdefault(f"ta.{name}", {}).setdefault(m, []).append(val)
                # Tier B
                if base_cell is not None:
                    tb = _tierB_cell(a, b, base_cell, fro_dw)
                    for name, val in tb.items():
                        rec[f"tb.{name}.{cell}"] = val
                        acc.setdefault(f"tb.{name}", {}).setdefault(m, []).append(val)
        rec["gate_global_dw_norm"] = float(np.sqrt(total_sq))
        # compact per-module means
        for metric, per_mod in acc.items():
            for m, vals in per_mod.items():
                rec[f"{metric}.mean.{m}"] = float(np.mean(vals))
        records.append(rec)
        if progress is not None and (i + 1) % 25 == 0:
            progress(f"{i + 1}/{len(rows)} adapters featurised")
    return pd.DataFrame.from_records(records)


# ── fitting (pure sklearn) ───────────────────────────────────────────────────

def _cols(df: pd.DataFrame, *prefixes: str, percell: bool = True) -> list[str]:
    """Feature columns with any of the given ``ta.``/``tb.`` prefixes.

    ``percell=False`` keeps only the compact ``.mean.<module>`` columns.
    """
    out = []
    for c in df.columns:
        if not any(c.startswith(p) for p in prefixes):
            continue
        is_mean = ".mean." in c
        if percell and is_mean:
            continue
        if not percell and not is_mean:
            continue
        out.append(c)
    return out


def _fit_arm(model_name: str, cols: list[str], tr: pd.DataFrame, te: pd.DataFrame,
             head: str, alphas, knn_grid, cv):
    """Fit one (model, feature-set, head) and score once on the test rows."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GridSearchCV
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import Normalizer, StandardScaler

    x_tr, x_te = tr[cols].to_numpy(), te[cols].to_numpy()
    y_tr, y_te = tr[head].to_numpy(), te[head].to_numpy()
    if model_name == "knn" or model_name.endswith("_knn"):
        pipe = make_pipeline(Normalizer(norm="l2"), KNeighborsRegressor(weights="distance"))
        est = GridSearchCV(pipe, {"kneighborsregressor__n_neighbors": list(knn_grid)},
                           cv=cv, scoring="neg_mean_squared_error")
    else:
        est = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
    est.fit(x_tr, y_tr)
    pred = est.predict(x_te)
    return score_predictions(y_te, pred), pred


def fit_geometry_leaderboard(
    features_df: pd.DataFrame,
    *,
    heads: Iterable[str] = SST2_TARGETS,
    alphas=np.logspace(-3, 5, 25),
    knn_grid=(3, 5, 7, 10, 15, 20),
    cv_seed: int = 42,
) -> pd.DataFrame:
    """Full arm x head leaderboard on the fixed 504-train/63-test split (``pool`` col).

    Arms: ``mean`` floor; ``hp_only`` (ridge & knn on :data:`HP_COLS` — the honest
    control); reference ``gate_lin``/``norms_ridge``/``spectral_ridge``/``spectral_knn``;
    new ``tierA``/``tierB``/``tierAB`` (ridge & knn, per-cell); their compact
    ``*_mean`` variants; and the incremental ``hp_plus_*`` arms (HP concatenated with
    each geometry family). Columns: ``model, head, n_features, r2, spearman, pearson, mae``.
    """
    from sklearn.model_selection import KFold

    heads = list(heads)
    tr = features_df[features_df.pool == "train"].reset_index(drop=True)
    te = features_df[features_df.pool == "test"].reset_index(drop=True)
    cv = KFold(n_splits=5, shuffle=True, random_state=cv_seed)

    ta = _cols(features_df, "ta.")
    tb = _cols(features_df, "tb.")
    ta_m = _cols(features_df, "ta.", percell=False)
    tb_m = _cols(features_df, "tb.", percell=False)
    norm_cols = [c for c in features_df.columns if c.startswith("norm.")]
    sv_cols = [c for c in features_df.columns if c.startswith("sv.")]
    hp = HP_COLS

    # (arm_name, model_kind, feature-columns)
    arms: list[tuple[str, str, list[str]]] = [
        ("hp_only_ridge", "ridge", hp),
        ("hp_only_knn", "knn", hp),
        ("gate_lin", "ridge", ["gate_global_dw_norm"]),
        ("norms_ridge", "ridge", norm_cols),
        ("spectral_ridge", "ridge", sv_cols),
        ("spectral_knn", "knn", sv_cols),
        ("tierA_ridge", "ridge", ta),
        ("tierA_knn", "knn", ta),
        ("tierB_ridge", "ridge", tb),
        ("tierB_knn", "knn", tb),
        ("tierAB_ridge", "ridge", ta + tb),
        ("tierAB_knn", "knn", ta + tb),
        ("tierA_mean_ridge", "ridge", ta_m),
        ("tierB_mean_ridge", "ridge", tb_m),
        ("hp_plus_tierA_ridge", "ridge", hp + ta),
        ("hp_plus_tierB_ridge", "ridge", hp + tb),
        ("hp_plus_tierAB_ridge", "ridge", hp + ta + tb),
        ("hp_plus_tierA_knn", "knn", hp + ta),
        ("hp_plus_tierB_knn", "knn", hp + tb),
        ("hp_plus_tierAB_knn", "knn", hp + ta + tb),
    ]

    rows: list[dict[str, Any]] = []
    for head in heads:  # mean floor
        pred = np.full(len(te), tr[head].to_numpy().mean())
        rows.append({"model": "mean", "head": head, "n_features": 0,
                     **score_predictions(te[head].to_numpy(), pred)})
    for arm, kind, cols in arms:
        for head in heads:
            metrics, _ = _fit_arm(kind, cols, tr, te, head, alphas, knn_grid, cv)
            rows.append({"model": arm, "head": head, "n_features": len(cols), **metrics})
    return pd.DataFrame(rows)


def incremental_over_hp(leaderboard: pd.DataFrame) -> pd.DataFrame:
    """ΔR² / ΔSpearman of each ``hp_plus_*`` arm over the matching ``hp_only`` baseline.

    Answers the campaign's honest test: does geometry add signal *beyond the training
    recipe*? Positive Δ means the geometry family carries information HP alone does not.
    """
    lb = leaderboard.set_index(["model", "head"])
    out: list[dict[str, Any]] = []
    for model in leaderboard["model"].unique():
        if not model.startswith("hp_plus_"):
            continue
        kind = "knn" if model.endswith("_knn") else "ridge"
        base_arm = f"hp_only_{kind}"
        for head in leaderboard["head"].unique():
            try:
                combo = lb.loc[(model, head)]
                base = lb.loc[(base_arm, head)]
            except KeyError:
                continue
            out.append({
                "model": model, "head": head,
                "r2": float(combo.r2), "hp_only_r2": float(base.r2),
                "delta_r2": float(combo.r2 - base.r2),
                "spearman": float(combo.spearman), "hp_only_spearman": float(base.spearman),
                "delta_spearman": float(combo.spearman - base.spearman),
            })
    return pd.DataFrame(out)
