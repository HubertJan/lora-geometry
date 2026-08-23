"""Benchmark SST2 adapters, original vs a battery of keep-only / ablation surgeries.

(migrated from SRC/src/discoveries/kv_ablation_high_acc/jobs/01_benchmark_kv_ablation.py)

Causal test of the LRP / UV readings: zero (or negate, or keep-only) selected
LoRA cells of a high-accuracy adapter and re-measure SST2 likelihood accuracy. If
a group is causally sufficient, keeping only it sustains accuracy above the base
floor; if a group is causally irrelevant, ablating it leaves accuracy intact.

Baselines (Llama-3.2-1B, TRUE_FALSE_V1): base floor (no adapter) ~0.558, a strong
original adapter ~0.955.

The SST2 eval is the local likelihood-accuracy scorer (``sst2_eval.py``) — the
``shared_adapter_pool.eval.run_eval.evaluate_on_sst2`` API is not importable, so
the fallback scorer is used (see ``sst2_eval`` docstring). The ``--submit``
JobSpec/RestrictedExecutor/BerGpu path is replaced by ``common.runner.submit_or_run``
with a local (in-process) default.

Output (under ``results_path("causal_ablation")``):
- ``variant_accuracy.csv`` : one row per (adapter, variant) with accuracy + delta
  vs original + stored accuracy; plus the BASE floor row.
- ``results.json``         : the same rows as JSON.

Example
-------
    python -m experiments.causal_ablation.benchmark --pool sst2-perf-v2-test --n 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Ablation variants scored per adapter (besides the un-ablated "original").
# ``modules`` are LoRA-key substrings; ``layers`` are 0-based model layer indices
# (Llama-3.2-1B: 0..15), ``None`` = all layers. NB query_1_11 (idx 0..10) and
# query_12_16 (idx 11..15) partition all 16 layers.
_MLP_MODULES = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
_ATTN_MODULES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")
_ALL_MODULES = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    *_MLP_MODULES,
)
ABLATIONS = [
    {"name": "kv_all",       "modules": ("self_attn.k_proj", "self_attn.v_proj"), "layers": None},
    {"name": "layer1_all",   "modules": _ALL_MODULES,           "layers": [0]},                 # layer 1
    {"name": "mlpdown_2_11", "modules": ("mlp.down_proj",),     "layers": list(range(1, 11))},  # layers 2..11
    {"name": "query_all",    "modules": ("self_attn.q_proj",),  "layers": None},                # all layers
    {"name": "query_12_16",  "modules": ("self_attn.q_proj",),  "layers": list(range(11, 16))}, # layers 12..16
    {"name": "query_1_11",   "modules": ("self_attn.q_proj",),  "layers": list(range(0, 11))},  # layers 1..11
    {"name": "mlp_all",      "modules": _MLP_MODULES,           "layers": None},                # all MLP, all layers
    {"name": "mlp_2_11",     "modules": _MLP_MODULES,           "layers": list(range(1, 11))},  # all MLP, layers 2..11
    # keep-only variants: retain JUST this group (all layers), zero every other
    # module -> how much accuracy a single group alone sustains above the base.
    {"name": "keep_mlpdown_all", "modules": ("mlp.down_proj",), "layers": None, "keep": True},
    {"name": "keep_kv_all",  "modules": ("self_attn.k_proj", "self_attn.v_proj"), "layers": None, "keep": True},
    # keep-only localisation: where does the MLP sufficiency live (by layer)?
    {"name": "keep_mlpdown_1_11",  "modules": ("mlp.down_proj",), "layers": list(range(0, 11)), "keep": True},  # down, layers 1..11
    {"name": "keep_mlpdown_12_16", "modules": ("mlp.down_proj",), "layers": list(range(11, 16)), "keep": True}, # down, layers 12..16
    {"name": "keep_mlp_1_11_16",   "modules": _MLP_MODULES, "layers": list(range(0, 11)) + [15], "keep": True}, # all MLP, layers 1..11 & 16
    {"name": "keep_mlp_16",        "modules": _MLP_MODULES, "layers": [15], "keep": True},                      # all MLP, layer 16
    # keep-only gate / up, split into the same layer halves as down.
    {"name": "keep_gate_1_11",  "modules": ("mlp.gate_proj",), "layers": list(range(0, 11)),  "keep": True},   # gate, layers 1..11
    {"name": "keep_gate_12_16", "modules": ("mlp.gate_proj",), "layers": list(range(11, 16)), "keep": True},   # gate, layers 12..16
    {"name": "keep_up_1_11",    "modules": ("mlp.up_proj",),   "layers": list(range(0, 11)),  "keep": True},   # up,   layers 1..11
    {"name": "keep_up_12_16",   "modules": ("mlp.up_proj",),   "layers": list(range(11, 16)), "keep": True},   # up,   layers 12..16
    # keep-only the whole attention block (q,k,v,o, all layers) — no MLP at all.
    {"name": "keep_attn_all",   "modules": _ATTN_MODULES,      "layers": None,                "keep": True},
    # keep-only attention sub-combinations (all layers) to localise the attn signal.
    {"name": "keep_qo_all",     "modules": ("self_attn.q_proj", "self_attn.o_proj"), "layers": None, "keep": True},
    {"name": "keep_qkv_all",    "modules": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"), "layers": None, "keep": True},
    {"name": "keep_q_all",      "modules": ("self_attn.q_proj",), "layers": None, "keep": True},
    # keep-only Q from layer 8 onward; and Q from layer 8 + K/V/O from layer 11.
    {"name": "keep_q_from8",    "modules": ("self_attn.q_proj",), "layers": list(range(7, 16)), "keep": True},  # Q, layers 8..16
    {"name": "keep_q8_attn11",  "rules": [
        (("self_attn.q_proj",), list(range(7, 16))),                                            # Q, layers 8..16
        (("self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"), list(range(10, 16))),    # K/V/O, layers 11..16
    ], "keep": True},
    # sign-INVERT (×-1) the MLP delta in layers 1..11, full adapter otherwise.
    {"name": "neg_mlp_1_11",    "modules": _MLP_MODULES,       "layers": list(range(0, 11)),  "negate": True},
    # --- more combinations ---
    {"name": "keep_o_all",      "modules": ("self_attn.o_proj",), "layers": None, "keep": True}, # O alone
    {"name": "keep_mlp_all",    "modules": _MLP_MODULES,       "layers": None,   "keep": True},  # all MLP alone (ref)
    {"name": "remove_attn_all", "modules": _ATTN_MODULES,      "layers": None},                  # remove all attn (=MLP-only)
    {"name": "neg_attn_all",    "modules": _ATTN_MODULES,      "layers": None,   "negate": True},# invert all attn
    # --- sanity checks (validate the ablation machinery) ---
    {"name": "sanity_ablate_all", "modules": _ALL_MODULES,     "layers": None},                  # zero ALL -> must == base
    {"name": "sanity_keep_all",   "modules": _ALL_MODULES,     "layers": None,   "keep": True},  # zero nothing -> must == original
]


def resolve_adapters(pool: str | None, meta_path: str | None, n: int) -> list[dict]:
    """Top-``n`` adapter dirs (by stored ``acc`` if present) from a pool / parquet.

    Each spec has ``id`` (adapter key), ``orig`` (directory holding
    ``adapter_model.safetensors`` + ``adapter_config.json``) and, when available,
    ``stored_acc``.
    """
    import polars as pl

    if meta_path is not None:
        meta = pl.read_parquet(meta_path)
    else:
        from common.env import pool_path
        from meta_model.dataset import load_adapter_pool_metadata

        meta = load_adapter_pool_metadata(pool_path(pool))
    if "acc" in meta.columns:
        meta = meta.sort("acc", descending=True)
    id_col = "adapter_id" if "adapter_id" in meta.columns else "__key__"
    specs = []
    for r in meta.head(n).iter_rows(named=True):
        adir = Path(r["safetensor_path"]).parent
        specs.append({
            "id": str(r[id_col])[:8],
            "orig": str(adir),
            "stored_acc": float(r["acc"]) if "acc" in meta.columns else None,
        })
    return specs


def run_benchmark(adapter_specs: list[dict], *, max_test: int | None = None) -> list[dict]:
    """Build every ablated variant (CPU weight surgery) and score base + variants.

    Returns one row per (adapter, variant), plus a BASE (no-adapter) floor row.
    """
    import torch
    from peft import PeftModel

    from common.env import results_path
    from common.hf import load_base_model, load_tokenizer
    from experiments.causal_ablation import sst2_eval
    from experiments.causal_ablation.ablate import ablate_adapter

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ablated_root = results_path("causal_ablation") / "ablated"

    # -- build every ablated variant per adapter (CPU weight surgery) --
    for a in adapter_specs:
        aid, orig = a["id"], a["orig"]
        assert (Path(orig) / "adapter_config.json").is_file(), f"no adapter at {orig}"
        variants = {}
        for ab in ABLATIONS:
            adst = ablated_root / aid / ab["name"]
            man = ablate_adapter(orig, adst,
                                 ab.get("modules", ()), ab.get("layers"),
                                 keep=ab.get("keep", False),
                                 negate=ab.get("negate", False),
                                 rules=ab.get("rules"))
            variants[ab["name"]] = str(adst)
            tag = ("negated" if ab.get("negate") else
                   "kept-only, zeroed" if ab.get("keep") else "zeroed")
            print(f"[ablate] {aid} {ab['name']:17s} {tag} {man['n_zeroed']:3d} tensors "
                  f"norm-sum {man['prezero_norm_sum']:.1f}", flush=True)
        a["variants"] = variants

    tokenizer = load_tokenizer()
    sents, labels = sst2_eval.load_sst2_test(max_test=max_test)
    print(f"[prep] SST2 test rows: {len(sents)}", flush=True)

    def score(adapter_dir: str | None) -> dict:
        base = load_base_model()
        if adapter_dir is None:                         # base model floor: no adapter
            model = base.to(device).eval()
        else:
            model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False).to(device).eval()
        res = sst2_eval.evaluate_sst2(model, tokenizer, sents, labels, device=device)
        del model, base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return res

    rows = []
    mb = score(None)
    print(f"[eval] {'BASE':8s} {'none':13s} acc={mb['accuracy']:.4f}", flush=True)
    rows.append({"adapter": "BASE", "variant": "none", "stored_acc": None, **mb})
    for s in adapter_specs:
        variant_dirs = [("original", s["orig"])]
        variant_dirs += [(ab["name"], s["variants"][ab["name"]]) for ab in ABLATIONS]
        for variant, adir in variant_dirs:
            m = score(adir)
            print(f"[eval] {s['id']:8s} {variant:13s} acc={m['accuracy']:.4f}", flush=True)
            rows.append({"adapter": s["id"], "variant": variant,
                         "stored_acc": s.get("stored_acc"), **m})
    return rows


def main() -> None:
    from functools import partial

    import polars as pl

    from common.env import results_path
    from common.runner import submit_or_run

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", default=None, help="adapter pool name (under POOL_DIR)")
    ap.add_argument("--meta", default=None, help="explicit metadata parquet")
    ap.add_argument("--n", type=int, default=3, help="top-N adapters to benchmark")
    ap.add_argument("--max-test", type=int, default=None, help="cap SST2 test rows")
    args = ap.parse_args()
    if args.pool is None and args.meta is None:
        ap.error("pass --pool or --meta")

    specs = resolve_adapters(args.pool, args.meta, args.n)
    print("[resolve] adapters:", [(a["id"], a.get("stored_acc")) for a in specs], flush=True)

    # local (in-process) default; hand a configured executor to submit_or_run to
    # run the whole benchmark as one SLURM job instead.
    rows = submit_or_run(
        partial(run_benchmark, max_test=args.max_test), [specs], executor=None
    )[0]

    out = results_path("causal_ablation")
    (out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")

    # per-(adapter, variant) accuracy table with delta-vs-original.
    orig_by_adapter = {r["adapter"]: r["accuracy"]
                       for r in rows if r["variant"] == "original"}
    table = []
    for r in rows:
        o = orig_by_adapter.get(r["adapter"])
        table.append({
            "adapter": r["adapter"], "variant": r["variant"],
            "accuracy": r["accuracy"], "stored_acc": r.get("stored_acc"),
            "delta_vs_original": (r["accuracy"] - o) if o is not None else None,
        })
    pl.DataFrame(table).write_csv(out / "variant_accuracy.csv")

    base = next((r["accuracy"] for r in rows if r["adapter"] == "BASE"), None)
    if base is not None:
        print(f"\nBASE (no adapter) acc = {base:.4f}")
    print(f"wrote {out / 'variant_accuracy.csv'} and results.json")


if __name__ == "__main__":
    main()
