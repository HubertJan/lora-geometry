"""Compute per-adapter LRP relevance maps for the ACC head of the w8_l2 regressor.

(migrated from SRC/src/discoveries/sst2_perf_regressor_lrp/jobs/02_run_lrp.py)

For every (checkpoint/seed, adapter) pair, run one LRP pass and aggregate the
per-weight relevance to the canonical per-cell (7 modules × N layers) SIGNED-net
grid and a rank-level MAGNITUDE grid.

Outputs (under ``results_path("lrp")``):
- ``lrp_maps.npz``    : signed (S, A, 7, L), magnitude (S, A, 7, L), seeds (S,),
  adapter_ids (A,), module_order, n_layers.
- ``lrp_scalars.parquet`` : one row per (seed, adapter) with predicted accuracy +
  the conservation relative error.

CPU only.  w8_l2 is tiny, so this runs in-process in a couple of minutes; pass a
configured executor to fan the per-seed passes onto SLURM.

Example
-------
    python -m experiments.lrp.run_maps \
        --pool sst2-perf-v2-test \
        --checkpoints /path/best-model.pth \
        --head acc --target 0 --n-layers 16
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

MODULE_ORDER = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]


def _grids(cell_rows, n_layers):
    """Return (signed 7×L, magnitude 7×L) from cell_signed_net rows."""
    import numpy as np
    idx = {mk: i for i, mk in enumerate(MODULE_ORDER)}
    sg = np.full((len(MODULE_ORDER), n_layers), np.nan)
    mg = np.full((len(MODULE_ORDER), n_layers), np.nan)
    for r in cell_rows:
        if r["module_key"] in idx and 0 <= r["layer"] < n_layers:
            sg[idx[r["module_key"]], r["layer"]] = r["signed"]
            mg[idx[r["module_key"]], r["layer"]] = r["magnitude"]
    return sg, mg


def _load_meta(pool: str, meta_path: str | None):
    """Return (adapter_ids, safetensor_paths) from a pool name or an explicit parquet."""
    import polars as pl

    if meta_path is not None:
        meta = pl.read_parquet(meta_path)
    else:
        from common.env import pool_path
        from meta_model.dataset import load_adapter_pool_metadata

        meta = load_adapter_pool_metadata(pool_path(pool))
    id_col = "adapter_id" if "adapter_id" in meta.columns else "__key__"
    ids = meta[id_col].to_list()
    paths = meta["safetensor_path"].to_list()
    return ids, paths


class _SeedTask:
    """One LRP-map pass over all adapters for a single checkpoint. Picklable."""

    def __init__(self, seed, ckpt, paths, head, target, n_layers):
        self.seed, self.ckpt, self.paths = seed, ckpt, paths
        self.head, self.target, self.n_layers = head, target, n_layers

    def __call__(self):
        import numpy as np
        import torch

        from experiments.lrp.aggregate import cell_signed_net
        from experiments.lrp.lrp_run import (
            LRPFlexibleMetaClassifier, group_from_path, make_composite_l2,
            per_rank_profiles, run_lrp_single,
        )

        composite = make_composite_l2()
        model = LRPFlexibleMetaClassifier.load(str(self.ckpt), device="cpu").eval()
        grouped = [group_from_path(p, self.n_layers) for p in self.paths]

        A = len(self.paths)
        signed = np.full((A, 7, self.n_layers), np.nan)
        magnit = np.full((A, 7, self.n_layers), np.nan)
        scalar_rows = []
        t0 = time.time()
        for ai in range(A):
            res = run_lrp_single(model, grouped[ai], composite, "cpu", self.head, self.target)
            prof = per_rank_profiles(res["attr_dict"], res["norm"])
            sg, mg = _grids(cell_signed_net(prof), self.n_layers)
            signed[ai], magnit[ai] = sg, mg
            logit = float(res["logits"][0])
            attr_sum = float(sum(
                res["attr_dict"][mk][lt].sum().item()
                for mk in res["attr_dict"] for lt in res["attr_dict"][mk]))
            rel_err = abs(attr_sum - res["norm"]) / (abs(res["norm"]) + 1e-12)
            scalar_rows.append({
                "seed": self.seed,
                "pred_acc_logit": logit,
                "pred_acc": float(torch.sigmoid(torch.tensor(logit)).item()),
                "lrp_norm": float(res["norm"]),
                "conservation_rel_err": float(rel_err),
                "signed_sum": float(np.nansum(sg)),
            })
        print(f"[lrp] seed {self.seed}: {A} adapters in {time.time() - t0:.1f}s", flush=True)
        return signed, magnit, scalar_rows


def _run_task(task: "_SeedTask"):
    """Module-level (picklable) entry point for ``submit_or_run``."""
    return task()


def main() -> None:
    import numpy as np
    import polars as pl

    from common.env import results_path
    from common.runner import submit_or_run

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", default=None, help="adapter pool name (under POOL_DIR)")
    ap.add_argument("--meta", default=None,
                    help="explicit metadata parquet (adapter_id/__key__ + safetensor_path)")
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="one LRPFlexible checkpoint per seed (w8_l2 / base_l2 best-model.pth)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="seed label per checkpoint (default: 0..N-1)")
    ap.add_argument("--head", default="acc")
    ap.add_argument("--target", type=int, default=0)
    ap.add_argument("--n-layers", type=int, default=16)
    args = ap.parse_args()
    if args.pool is None and args.meta is None:
        ap.error("pass --pool or --meta")

    ids, paths = _load_meta(args.pool, args.meta)
    ckpts = [Path(c) for c in args.checkpoints]
    seeds = args.seeds if args.seeds is not None else list(range(len(ckpts)))
    assert len(seeds) == len(ckpts), "one --seeds label per checkpoint"
    A = len(ids)
    print(f"[lrp] {A} adapters × {len(ckpts)} checkpoints", flush=True)

    tasks = [_SeedTask(seed, ckpt, paths, args.head, args.target, args.n_layers)
             for seed, ckpt in zip(seeds, ckpts)]
    results = submit_or_run(_run_task, tasks, executor=None)

    signed = np.stack([r[0] for r in results])   # (S, A, 7, L)
    magnit = np.stack([r[1] for r in results])
    scalar_rows = []
    for (_, _, rows), aid_list in zip(results, [ids] * len(results)):
        for row, aid in zip(rows, aid_list):
            scalar_rows.append({**row, "adapter_id": aid})

    out_dir = results_path("lrp")
    out_npz = out_dir / "lrp_maps.npz"
    out_scalars = out_dir / "lrp_scalars.parquet"
    np.savez_compressed(
        out_npz,
        signed=signed, magnitude=magnit,
        seeds=np.array(seeds), adapter_ids=np.array(ids, dtype=object),
        module_order=np.array(MODULE_ORDER, dtype=object), n_layers=args.n_layers,
    )
    pl.DataFrame(scalar_rows).write_parquet(out_scalars)
    rels = [r["conservation_rel_err"] for r in scalar_rows]
    print(f"[lrp] wrote {out_npz.name} signed{signed.shape} and {out_scalars.name}")
    print(f"[lrp] conservation rel_err median={np.median(rels):.3e} max={np.max(rels):.3e}")


if __name__ == "__main__":
    main()
