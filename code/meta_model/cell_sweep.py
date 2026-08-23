"""Migrated from SRC/src/glad/meta_classifier/cell_sweep.py.

Per-cell detector training on RAM-preloaded adapter rows.

The harness almost every interpretability discovery trains its detectors with:
preload a pool of adapters into RAM once (``{path, label}`` rows → grouped LoRA
dicts), slice each adapter down to one :class:`~meta_model.lora.cells.Cell` (or keep
the full read-set), and train a small :class:`EquivariantLoRAMetaClassifier` on
the slices.  Preloading is what makes the per-cell sweeps tractable: reading the
pool from NFS every epoch was ~11 s/epoch; from RAM it is ~0.18 s/epoch (see
``imdb_detector_locality/RESEARCH_LOG.md``).

Two training protocols coexisted in the copies, and both are kept:

* :func:`train_detector_on_rows` — **fixed-epoch, no validation** (the
  ``train_from_rows`` lineage: ``imdb_backdoor_locality``,
  ``imdb_detector_locality``, ``target_polarity_meta_classifier``,
  ``emergent_misalignment_meta_classifier``).  Assumes a balanced pool and a
  budget chosen by the caller.
* :func:`train_cell` / :func:`train_cell_multi` / :func:`train_full_multi` —
  **per-epoch validation with best-by-val selection** (the ``sweep_train``
  lineage: ``component_sufficiency_sweep``, ``per_cell_xtask_stability``,
  ``meta_classifier_robustness``), including that study's seeding split
  (``init_seed`` vs ``shuffle_seed``), the dual ACC/AUC selection rules, probe
  epochs, and the ``select="last"`` fast path.

What was fork-specific in the copies is now a parameter:

* the classification **head** (``is_imdb`` / ``is_backdoored`` /
  ``target_polarity`` / ``is_phenomenon`` were the *only* difference between the
  four binary ``train_from_rows`` copies),
* the **module-size table** (``LLAMA3_1B`` vs ``LLAMA31_8B``),
* the **layer count** in :func:`preload` — default ``None`` infers it from each
  adapter (the copies hardcoded 16 or 32; fewer real layers than the hardcode
  silently zero-pads, see :func:`meta_model.lora.cells.load_grouped`).

Rows everywhere are ``{"path": str, "label": int}`` dicts; preloaded data is
``list[(grouped_dict, int)]`` **in input row order** (callers build subsets by
zipping rows against the preloaded list — do not reorder).
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from meta_model.lora.cells import KEY_TO_MODULE_TYPE, Cell, cell_dict, cell_rank_dict, load_grouped
from meta_model.lora.model_sizes import LLAMA3_1B
from meta_model.lora.types import ActivationFunction, TargetModuleType
from meta_model.equivariant_lora_meta_classifier import (
    EquivariantLoRAMetaClassifier,
)
from meta_model.heads import (
    MetaHeadType,
    MetaTargetSpec,
    compute_multihead_loss,
)
from meta_model.helpers import batch_lora_dicts

Slicer = Callable[[dict], dict]
PreloadedRow = tuple[dict, int]

# Reference hyperparameters, identical across every copy of the harness.
SEED = 42
LR = 2e-5
WD = 1e-5
ETA_MIN = 5e-6
BATCH = 8
EQUIV_SIZES = (128,)
HEAD_SIZES = (64,)

# The most common head across the copies (backdoor detection).  Discoveries with
# a different label pass their own spec.
DEFAULT_HEAD = MetaTargetSpec(
    name="is_backdoored",
    column="is_backdoored",
    type=MetaHeadType.CLASSIFICATION,
    num_classes=2,
    mapping=(("False", 0), ("True", 1)),
    loss_weight=1.0,
)

__all__ = [
    "BATCH",
    "DEFAULT_HEAD",
    "EQUIV_SIZES",
    "ETA_MIN",
    "HEAD_SIZES",
    "LR",
    "SEED",
    "WD",
    "PreloadedRow",
    "Slicer",
    "evaluate_on_rows",
    "identity_slicer",
    "iter_batches",
    "make_cell_rank_slicer",
    "make_cell_slicer",
    "metrics_from_scores",
    "predict_classes_on_rows",
    "predict_on_rows",
    "preload",
    "train_cell",
    "train_cell_multi",
    "train_detector_on_rows",
    "train_full_multi",
]


# ── Preload + slicing ────────────────────────────────────────────────────────


def preload(
    rows: Iterable[dict],
    desc: str | None = None,
    *,
    progress_desc: str | None = None,
    layer_count: int | None = None,
) -> list[PreloadedRow]:
    """Load each row's adapter into RAM once, as ``(grouped_dict, label)``.

    Output order matches input row order — callers rely on ``zip(rows, data)``
    to build subsets.  Tensors keep their as-loaded dtype (usually bf16) to
    save RAM; :func:`iter_batches` casts to float at batch time.

    ``desc`` / ``progress_desc`` are the same tqdm label (the two lineages named
    the keyword differently); ``None`` disables the progress bar.
    ``layer_count=None`` infers each adapter's layer span from its own file.
    Pin it only to reproduce a historical run that assumed a fixed count.

    RAM cost is the full pool (a 380-adapter rank-16 1B pool is ~10 GB in bf16);
    preloading many pools at once has OOM'd drivers before — prefer loading one
    pool at a time.  For DataLoader-based training use
    ``create_safetensor_dataset(..., materialize="ram")`` instead.
    """
    if desc is None:
        desc = progress_desc
    it: Iterable[dict] = rows
    if desc is not None:
        from tqdm import tqdm

        it = tqdm(rows, desc=desc)
    return [
        (load_grouped(r["path"], layer_count=layer_count), int(r["label"])) for r in it
    ]


def identity_slicer(g: dict) -> dict:
    """Full read-set: the grouped dict, untouched."""
    return g


def _as_cell(cell: Cell | str, layer: int | None) -> Cell:
    if isinstance(cell, Cell):
        if layer is not None:
            raise TypeError("pass either a Cell or (module_key, layer), not both")
        return cell
    if layer is None:
        raise TypeError(f"layer is required when addressing by module key {cell!r}")
    return Cell(KEY_TO_MODULE_TYPE[cell], layer)


def make_cell_slicer(cell: Cell | str, layer: int | None = None) -> Slicer:
    """Slicer restricting the input to one cell (native rank, single layer)."""
    c = _as_cell(cell, layer)
    return lambda g: cell_dict(g, c)


def make_cell_rank_slicer(
    cell: Cell | str, layer: int | None = None, k: int = 1
) -> Slicer:
    """Slicer restricting the input to singular direction ``k`` of one cell."""
    c = _as_cell(cell, layer)
    return lambda g: cell_rank_dict(g, c, k)


def iter_batches(
    data: Sequence[PreloadedRow],
    slicer: Slicer,
    *,
    batch_size: int = BATCH,
    shuffle: bool = False,
    rng: random.Random | None = None,
) -> Iterator[tuple[dict, torch.Tensor]]:
    """Yield ``(batched_lora_dict, labels)`` batches, slicing + float-casting lazily."""
    idx = list(range(len(data)))
    if shuffle:
        (rng if rng is not None else random).shuffle(idx)
    for i in range(0, len(idx), batch_size):
        chunk = idx[i : i + batch_size]
        dicts = []
        for j in chunk:
            sl = slicer(data[j][0])
            dicts.append(
                {k: {lt: t.float() for lt, t in v.items()} for k, v in sl.items()}
            )
        labels = torch.tensor([data[j][1] for j in chunk], dtype=torch.long)
        yield batch_lora_dicts(dicts), labels


def _to_device(inp: dict, device: str) -> dict:
    return {k: {lt: t.to(device) for lt, t in v.items()} for k, v in inp.items()}


# ── Prediction + metrics ─────────────────────────────────────────────────────


def _positive_index(spec: MetaTargetSpec, positive_class: int | str | None) -> int:
    if isinstance(positive_class, int):
        return positive_class
    if isinstance(positive_class, str):
        idx = spec.mapping_dict.get(positive_class)
        if idx is None:
            raise KeyError(
                f"positive_class {positive_class!r} not in head {spec.name!r} mapping "
                f"{spec.mapping_dict}"
            )
        return idx
    if spec.num_classes == 2:
        return 1
    raise TypeError(
        f"head {spec.name!r} has {spec.num_classes} classes; pass positive_class "
        "(an index or a mapped label) to score one of them"
    )


@torch.no_grad()
def predict_on_rows(
    model: EquivariantLoRAMetaClassifier,
    data: Sequence[PreloadedRow],
    slicer: Slicer,
    device: str = "cpu",
    *,
    batch_size: int = BATCH,
    positive_class: int | str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Softmax score of the positive class for every row: ``(labels, scores)``.

    ``positive_class`` defaults to index 1 for binary heads; for an N-way head
    pass the class index or its mapped label (restores the multiclass scoring
    the ``imdb_detector_locality`` original had and later copies dropped).
    """
    model.eval()
    spec = model.head_specs[0]
    col = _positive_index(spec, positive_class)
    labs: list[int] = []
    scores: list[float] = []
    for inp, lab in iter_batches(data, slicer, batch_size=batch_size):
        out = model(_to_device(inp, device))[spec.name]
        scores += torch.softmax(out, dim=1)[:, col].cpu().tolist()
        labs += lab.tolist()
    return np.array(labs), np.array(scores)


@torch.no_grad()
def predict_classes_on_rows(
    model: EquivariantLoRAMetaClassifier,
    data: Sequence[PreloadedRow],
    slicer: Slicer,
    device: str = "cpu",
    *,
    batch_size: int = BATCH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Argmax prediction for every row: ``(labels, predictions, confidences)``."""
    model.eval()
    name = model.head_specs[0].name
    labs: list[int] = []
    preds: list[int] = []
    confs: list[float] = []
    for inp, lab in iter_batches(data, slicer, batch_size=batch_size):
        probs = torch.softmax(model(_to_device(inp, device))[name], dim=1)
        conf, pred = probs.max(dim=1)
        preds += pred.cpu().tolist()
        confs += conf.cpu().tolist()
        labs += lab.tolist()
    return np.array(labs), np.array(preds), np.array(confs)


def metrics_from_scores(labs: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Binary metrics at threshold 0.5 (+ threshold-free ROC/PR AUC).

    Single-class inputs get ``nan`` AUCs instead of a sklearn crash.
    """
    preds = (scores >= 0.5).astype(int)
    two = len(set(labs.tolist())) == 2
    return {
        "n": int(labs.size),
        "n_pos": int(labs.sum()),
        "roc_auc": float(roc_auc_score(labs, scores)) if two else float("nan"),
        "pr_auc": float(average_precision_score(labs, scores)) if two else float("nan"),
        "accuracy": float((preds == labs).mean()),
        "precision": float(precision_score(labs, preds, zero_division=0)),
        "recall": float(recall_score(labs, preds, zero_division=0)),
        "f1": float(f1_score(labs, preds, zero_division=0)),
    }


@torch.no_grad()
def evaluate_on_rows(
    model: EquivariantLoRAMetaClassifier,
    data: Sequence[PreloadedRow],
    slicer: Slicer,
    device: str,
    *,
    batch_size: int = BATCH,
) -> dict[str, float]:
    """Argmax-based binary metrics (the ``sweep_train`` lineage's dict).

    Key names follow that lineage: ``auc`` (not ``roc_auc``), and
    ``backdoored_recall`` / ``clean_recall`` mean label-1 / label-0 recall
    regardless of what the head is called.
    """
    model.eval()
    name = model.head_specs[0].name
    preds_l: list[int] = []
    labs_l: list[int] = []
    scores: list[float] = []
    for inp, lab in iter_batches(data, slicer, batch_size=batch_size):
        out = model(_to_device(inp, device))[name]
        preds_l += out.argmax(dim=1).cpu().tolist()
        scores += torch.softmax(out, dim=1)[:, 1].cpu().tolist()
        labs_l += lab.tolist()
    preds, labs = np.array(preds_l), np.array(labs_l)
    try:
        auc = float(roc_auc_score(labs, scores))
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float((preds == labs).mean()),
        "auc": auc,
        "precision": float(precision_score(labs, preds, zero_division=0)),
        "recall": float(recall_score(labs, preds, zero_division=0)),
        "f1": float(f1_score(labs, preds, zero_division=0)),
        "backdoored_recall": float((preds[labs == 1] == 1).mean())
        if (labs == 1).any()
        else float("nan"),
        "clean_recall": float((preds[labs == 0] == 0).mean())
        if (labs == 0).any()
        else float("nan"),
    }


# ── Model construction ───────────────────────────────────────────────────────


def _build_model(
    *,
    target_modules: set[TargetModuleType],
    model_sizes: Mapping[TargetModuleType, tuple[int, int]],
    head: MetaTargetSpec,
    llm_layers: int,
    equiv_sizes: Sequence[int],
    head_sizes: Sequence[int],
    seed: int,
    device: str,
    activation: ActivationFunction,
) -> EquivariantLoRAMetaClassifier:
    torch.manual_seed(seed)
    return EquivariantLoRAMetaClassifier(
        equivariant_layer_sizes=list(equiv_sizes),
        head_layer_sizes=list(head_sizes),
        targeted_modules=set(target_modules),
        target_module_sizes={m: model_sizes[m] for m in target_modules},
        head_specs=[head],
        equivariant_layers_activation=activation,
        llm_layers=llm_layers,
        device=device,
        seed=seed,
    )


def _resolve_device(device: str) -> str:
    return "cpu" if device == "cuda" and not torch.cuda.is_available() else device


# ── Protocol 1: fixed-epoch training, no validation ──────────────────────────


def train_detector_on_rows(
    train_d: Sequence[PreloadedRow],
    test_d: Sequence[PreloadedRow],
    *,
    llm_layers: int,
    head: MetaTargetSpec = DEFAULT_HEAD,
    model_sizes: Mapping[TargetModuleType, tuple[int, int]] = LLAMA3_1B,
    target_modules: set[TargetModuleType] | None = None,
    slicer: Slicer = identity_slicer,
    equiv_sizes: Sequence[int] = EQUIV_SIZES,
    head_layer_sizes: Sequence[int] | None = None,
    epochs: int = 100,
    seed: int = SEED,
    device: str = "cuda",
    batch_size: int = BATCH,
    lr: float = LR,
    weight_decay: float = WD,
    eta_min: float = ETA_MIN,
    activation: ActivationFunction = ActivationFunction.GL_ACTIVATION,
    positive_class: int | str | None = None,
) -> tuple[EquivariantLoRAMetaClassifier, dict[str, float]]:
    """Train for a fixed epoch budget and score on ``test_d``.

    No validation split, no early stopping, no checkpoint selection: this
    protocol assumes a balanced pool and treats the epoch budget as part of the
    experiment definition.  ``head_layer_sizes=[]`` makes the detector an exact
    bilinear ``s = ⟨T, ΔW⟩ + β`` (what the template-extraction analyses rely
    on); ``None`` defaults to ``[64]``.

    ``llm_layers`` must match what ``slicer`` emits: 1 for a single-cell slicer,
    the model's layer count for the full read-set.
    """
    if head_layer_sizes is None:
        head_layer_sizes = [64]
    if target_modules is None:
        target_modules = set(KEY_TO_MODULE_TYPE.values())
    device = _resolve_device(device)

    model = _build_model(
        target_modules=target_modules,
        model_sizes=model_sizes,
        head=head,
        llm_layers=llm_layers,
        equiv_sizes=equiv_sizes,
        head_sizes=head_layer_sizes,
        seed=seed,
        device=device,
        activation=activation,
    )
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=eta_min)
    rng = random.Random(seed)

    for ep in range(epochs):
        model.train()
        running, count = 0.0, 0
        for inp, lab in iter_batches(
            train_d, slicer, batch_size=batch_size, shuffle=True, rng=rng
        ):
            out = model(_to_device(inp, device))
            loss, _ = compute_multihead_loss(out, {head.name: lab.to(device)}, [head])
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.detach()) * lab.size(0)
            count += lab.size(0)
        sched.step()
        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"  epoch {ep + 1:3d} | train_loss={running / max(count, 1):.4f}")

    labs, scores = predict_on_rows(
        model, test_d, slicer, device, batch_size=batch_size, positive_class=positive_class
    )
    return model, metrics_from_scores(labs, scores)


# ── Protocol 2: per-epoch validation with best-by-val selection ──────────────

Select = Literal["best_val", "last"]


def _fit_with_selection(
    model: EquivariantLoRAMetaClassifier,
    slicer: Slicer,
    train_d: Sequence[PreloadedRow],
    val_d: Sequence[PreloadedRow],
    *,
    head: MetaTargetSpec,
    epochs: int,
    select: Select,
    shuffle_seed: int,
    device: str,
    batch_size: int,
    lr: float,
    weight_decay: float,
    eta_min: float,
    probe_epochs: Sequence[int] = (),
    probe_test: Sequence[PreloadedRow] | None = None,
) -> dict[str, Any]:
    """Run the epoch loop; track ACC- and AUC-selected checkpoints and probes.

    The two selection rules score the same per-epoch candidate set and differ
    only in the leading val metric: ``val_acc + 1e-3·val_auc`` (the production
    rule) vs ``val_auc + 1e-3·val_acc``.  ``select="last"`` skips per-epoch
    validation entirely (much faster) and keeps the final-epoch weights.
    """
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=eta_min)
    rng = random.Random(shuffle_seed)

    probe_set = {int(e) for e in probe_epochs}
    probe: dict[int, dict] = {}
    best_acc_score, best_acc_state, best_acc_ep = -1.0, None, -1
    best_auc_score, best_auc_state, best_auc_ep = -1.0, None, -1

    for ep in range(epochs):
        model.train()
        for inp, lab in iter_batches(
            train_d, slicer, batch_size=batch_size, shuffle=True, rng=rng
        ):
            out = model(_to_device(inp, device))
            loss, _ = compute_multihead_loss(out, {head.name: lab.to(device)}, [head])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

        if select == "best_val":
            vm = evaluate_on_rows(model, val_d, slicer, device, batch_size=batch_size)
            vauc = vm["auc"] if vm["auc"] == vm["auc"] else 0.0
            sel_acc = vm["accuracy"] + 1e-3 * vauc
            sel_auc = vauc + 1e-3 * vm["accuracy"]
            if sel_acc > best_acc_score:
                best_acc_score, best_acc_ep = sel_acc, ep
                best_acc_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            if sel_auc > best_auc_score:
                best_auc_score, best_auc_ep = sel_auc, ep
                best_auc_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            if (ep + 1) in probe_set and probe_test is not None:
                probe[ep + 1] = {
                    "test": evaluate_on_rows(
                        model, probe_test, slicer, device, batch_size=batch_size
                    ),
                    "val": vm,
                }

    return {
        "best_acc_state": best_acc_state,
        "best_acc_ep": best_acc_ep,
        "best_acc_score": best_acc_score,
        "best_auc_state": best_auc_state,
        "best_auc_ep": best_auc_ep,
        "probe": probe,
    }


def train_cell(
    train_d: Sequence[PreloadedRow],
    val_d: Sequence[PreloadedRow],
    test_d: Sequence[PreloadedRow],
    cell: Cell | str,
    layer: int | None = None,
    *,
    rank: int | None = None,
    head: MetaTargetSpec = DEFAULT_HEAD,
    model_sizes: Mapping[TargetModuleType, tuple[int, int]] = LLAMA3_1B,
    epochs: int,
    device: str = "cuda",
    init_seed: int = SEED,
    shuffle_seed: int = SEED,
    equiv_sizes: Sequence[int] = EQUIV_SIZES,
    head_sizes: Sequence[int] = HEAD_SIZES,
    probe_epochs: Sequence[int] = (),
    save_path: str | Path | None = None,
    batch_size: int = BATCH,
    lr: float = LR,
    weight_decay: float = WD,
    eta_min: float = ETA_MIN,
) -> dict[str, Any]:
    """Train a single-cell classifier with best-by-val selection.

    ``init_seed`` seeds weight init (``torch.manual_seed`` + the model RNG);
    ``shuffle_seed`` seeds batch shuffling — the robustness study's
    decomposition, with the original single-seed behaviour as the default.
    ``rank`` slices the cell to one SVD-balanced singular direction.
    ``probe_epochs`` are 1-indexed epochs at which test/val metrics are
    *measured* (never used for selection).  ``save_path`` persists the
    val-accuracy-selected weights via ``model.save`` (default: discarded).

    Returns ``test`` / ``val`` / ``best_epoch`` / ``best_val_score`` for the
    val-accuracy-selected checkpoint, plus ``test_aucsel`` /
    ``best_epoch_aucsel`` for the val-AUC-selected one and the ``probe`` dict.
    """
    c = _as_cell(cell, layer)
    slicer = make_cell_slicer(c) if rank is None else make_cell_rank_slicer(c, k=rank)
    device = _resolve_device(device)
    model = _build_model(
        target_modules={c.module},
        model_sizes=model_sizes,
        head=head,
        llm_layers=1,
        equiv_sizes=equiv_sizes,
        head_sizes=head_sizes,
        seed=init_seed,
        device=device,
        activation=ActivationFunction.GL_ACTIVATION,
    )
    fit = _fit_with_selection(
        model,
        slicer,
        train_d,
        val_d,
        head=head,
        epochs=epochs,
        select="best_val",
        shuffle_seed=shuffle_seed,
        device=device,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        eta_min=eta_min,
        probe_epochs=probe_epochs,
        probe_test=test_d,
    )

    if fit["best_acc_state"] is not None:
        model.load_state_dict(fit["best_acc_state"])
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(save_path))
    test_m = evaluate_on_rows(model, test_d, slicer, device, batch_size=batch_size)
    val_m = evaluate_on_rows(model, val_d, slicer, device, batch_size=batch_size)
    if fit["best_auc_state"] is not None:
        model.load_state_dict(fit["best_auc_state"])
    test_aucsel = evaluate_on_rows(model, test_d, slicer, device, batch_size=batch_size)
    return {
        "test": test_m,
        "val": val_m,
        "best_epoch": fit["best_acc_ep"],
        "best_val_score": fit["best_acc_score"],
        "test_aucsel": test_aucsel,
        "best_epoch_aucsel": fit["best_auc_ep"],
        "probe": fit["probe"],
    }


def _train_multi(
    model: EquivariantLoRAMetaClassifier,
    slicer: Slicer,
    train_d: Sequence[PreloadedRow],
    val_d: Sequence[PreloadedRow],
    test_pools: Mapping[str, Sequence[PreloadedRow]],
    *,
    head: MetaTargetSpec,
    epochs: int,
    select: Select,
    shuffle_seed: int,
    device: str,
    batch_size: int,
    lr: float,
    weight_decay: float,
    eta_min: float,
) -> dict[str, Any]:
    fit = _fit_with_selection(
        model,
        slicer,
        train_d,
        val_d,
        head=head,
        epochs=epochs,
        select=select,
        shuffle_seed=shuffle_seed,
        device=device,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        eta_min=eta_min,
    )
    if select == "best_val" and fit["best_acc_state"] is not None:
        model.load_state_dict(fit["best_acc_state"])
        used_ep = fit["best_acc_ep"]
    else:
        used_ep = epochs - 1
    per_test = {
        name: evaluate_on_rows(model, td, slicer, device, batch_size=batch_size)
        for name, td in test_pools.items()
    }
    val_m = evaluate_on_rows(model, val_d, slicer, device, batch_size=batch_size)
    return {"per_test": per_test, "val": val_m, "best_epoch": used_ep}


def train_cell_multi(
    train_d: Sequence[PreloadedRow],
    val_d: Sequence[PreloadedRow],
    test_pools: Mapping[str, Sequence[PreloadedRow]],
    cell: Cell | str,
    layer: int | None = None,
    *,
    head: MetaTargetSpec = DEFAULT_HEAD,
    model_sizes: Mapping[TargetModuleType, tuple[int, int]] = LLAMA3_1B,
    epochs: int,
    select: Select = "last",
    device: str = "cuda",
    seed: int = SEED,
    equiv_sizes: Sequence[int] = EQUIV_SIZES,
    head_sizes: Sequence[int] = HEAD_SIZES,
    batch_size: int = BATCH,
    lr: float = LR,
    weight_decay: float = WD,
    eta_min: float = ETA_MIN,
) -> dict[str, Any]:
    """Train one single-cell classifier and evaluate it on every test pool.

    One row of a per-cell N×N cross-eval matrix: ``{"per_test": {pool:
    metrics}, "val": metrics, "best_epoch": int}``.
    """
    c = _as_cell(cell, layer)
    device = _resolve_device(device)
    model = _build_model(
        target_modules={c.module},
        model_sizes=model_sizes,
        head=head,
        llm_layers=1,
        equiv_sizes=equiv_sizes,
        head_sizes=head_sizes,
        seed=seed,
        device=device,
        activation=ActivationFunction.GL_ACTIVATION,
    )
    return _train_multi(
        model,
        make_cell_slicer(c),
        train_d,
        val_d,
        test_pools,
        head=head,
        epochs=epochs,
        select=select,
        shuffle_seed=seed,
        device=device,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        eta_min=eta_min,
    )


def train_full_multi(
    train_d: Sequence[PreloadedRow],
    val_d: Sequence[PreloadedRow],
    test_pools: Mapping[str, Sequence[PreloadedRow]],
    *,
    llm_layers: int,
    head: MetaTargetSpec = DEFAULT_HEAD,
    model_sizes: Mapping[TargetModuleType, tuple[int, int]] = LLAMA3_1B,
    target_modules: set[TargetModuleType] | None = None,
    epochs: int,
    select: Select = "last",
    device: str = "cuda",
    seed: int = SEED,
    equiv_sizes: Sequence[int] = EQUIV_SIZES,
    head_sizes: Sequence[int] = HEAD_SIZES,
    batch_size: int = BATCH,
    lr: float = LR,
    weight_decay: float = WD,
    eta_min: float = ETA_MIN,
) -> dict[str, Any]:
    """Train the full-read-set baseline and evaluate it on every test pool.

    The slicer restricts each grouped dict to the targeted module keys, so
    adapters carrying extra submodules don't crash the stacked batch.
    """
    if target_modules is None:
        target_modules = set(KEY_TO_MODULE_TYPE.values())
    keys = [k for k, mt in KEY_TO_MODULE_TYPE.items() if mt in target_modules]
    device = _resolve_device(device)
    model = _build_model(
        target_modules=target_modules,
        model_sizes=model_sizes,
        head=head,
        llm_layers=llm_layers,
        equiv_sizes=equiv_sizes,
        head_sizes=head_sizes,
        seed=seed,
        device=device,
        activation=ActivationFunction.GL_ACTIVATION,
    )
    return _train_multi(
        model,
        lambda g: {k: g[k] for k in keys if k in g},
        train_d,
        val_d,
        test_pools,
        head=head,
        epochs=epochs,
        select=select,
        shuffle_seed=seed,
        device=device,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        eta_min=eta_min,
    )
