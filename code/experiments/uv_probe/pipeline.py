"""Importable pipeline for the UV bilinear-probe token-feature analysis.

(migrated from SRC/src/discoveries/sst2_perf_regressor_uv/flows/pipeline.py)

Holds the extraction (``extract_main``) and analysis (``analyze_main``) entry points.

I/O lives under ``results_path("uv_probe")`` (the ``INST`` dir): place the trained
regressor checkpoints at ``INST/checkpoints/w8_l2_seed{seed}.pth`` and the adapter
metadata parquet (columns ``adapter_id`` / ``safetensor_path`` / ``acc``) at
``INST/test_meta.parquet`` (or ``test_meta_shared.parquet``). ``extract_main`` writes
``uv_scalars.parquet`` + ``mean_dirs.npz``; ``analyze_main`` writes the analysis CSVs +
``findings.json`` under ``INST/analysis/``.
"""
from __future__ import annotations

import json
import time

from common.env import results_path

INST = results_path("uv_probe")
CKPT_DIR = INST / "checkpoints"
_shared = INST / "test_meta_shared.parquet"
META = _shared if _shared.exists() else INST / "test_meta.parquet"
OUT_PARQUET = INST / "uv_scalars.parquet"
OUT_NPZ = INST / "mean_dirs.npz"
DATA = INST / "analysis"
DATA.mkdir(exist_ok=True, parents=True)
SEEDS = [42, 43, 44, 45]
NL = 16
RESIDUAL_WRITERS = frozenset({"self_attn.o_proj", "mlp.down_proj"})
MODULE_SHORT = {
    "self_attn.q_proj": "Q", "self_attn.k_proj": "K", "self_attn.v_proj": "V",
    "self_attn.o_proj": "O", "mlp.gate_proj": "gate", "mlp.up_proj": "up",
    "mlp.down_proj": "down",
}


def _cos(a, b):
    import numpy as np
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def extract_main():
    import numpy as np, polars as pl

    from experiments.uv_probe import axes as ax
    from experiments.uv_probe import uv_extract as uv
    from meta_model.lora.types import LoraType

    meta = pl.read_parquet(META)
    ids = meta["adapter_id"].to_list()
    paths = meta["safetensor_path"].to_list()
    real_acc = np.array(meta["acc"].to_list(), float)
    A = len(ids)
    # accuracy terciles (global) for high/low binning
    q33, q67 = np.nanpercentile(real_acc, [33.333, 66.667])
    tercile = np.where(real_acc <= q33, "low", np.where(real_acc >= q67, "high", "mid"))
    print(f"[extract] {A} adapters x {len(SEEDS)} seeds; acc terciles low<= {q33:.3f} high>= {q67:.3f}")

    axd = ax.load_axes()
    delta, delta_sp = axd["delta"], axd["delta_sp"]
    print(f"[axes] delta dim={delta.shape[0]}  cos(delta,delta_sp)={ax.cos(delta,delta_sp):.3f}")

    # preload adapters once
    grouped_all = [uv.group_from_path(p, NL) for p in paths]

    rows = []
    # mean dirs: per (seed, cell) accumulate sign-aligned sums for all/high/low
    n_cells = 7 * NL
    mean_all = np.zeros((len(SEEDS), n_cells, 2048))
    mean_high = np.zeros((len(SEEDS), n_cells, 2048))
    mean_low = np.zeros((len(SEEDS), n_cells, 2048))
    cnt_all = np.zeros((len(SEEDS), n_cells)); cnt_hi = np.zeros_like(cnt_all); cnt_lo = np.zeros_like(cnt_all)
    ref_dir = {}  # (si, ci) -> reference r0 for sign alignment

    for si, seed in enumerate(SEEDS):
        model = uv.load_model(str(CKPT_DIR / f"w8_l2_seed{seed}.pth"), "cpu")
        mw = uv.module_weights(model)
        qr = uv.module_qr(mw)
        order = [k for k, _ in uv.cell_order(model)]
        t0 = time.time()
        for ai in range(A):
            grouped = grouped_all[ai]
            G, acc_logit = uv.raw_feature_grads(model, mw, grouped, order, NL, "acc")
            ci = 0
            for key in order:
                side = "u" if key in RESIDUAL_WRITERS else "v"
                sub = grouped[key]
                for layer in range(NL):
                    f = uv.cell_full(G[key][layer], qr[key], side)
                    S = f["S"]; r0 = f["res_top"]; basis = f["res_basis"]
                    dW = (sub[LoraType.B][layer].double() @ sub[LoraType.A][layer].double())
                    own = float(f["u0"] @ (dW.numpy() @ f["v0"]))
                    own_load = float(S[0]) * own / (float(np.linalg.norm(dW.numpy())) + 1e-30)
                    # sign align to per-(seed,cell) reference
                    key_ci = (si, ci)
                    if key_ci not in ref_dir:
                        ref_dir[key_ci] = r0.copy()
                    ref = ref_dir[key_ci]
                    sgn = 1.0 if float(r0 @ ref) >= 0 else -1.0
                    cos_to_ref = abs(float(r0 @ ref) / (np.linalg.norm(r0) * np.linalg.norm(ref) + 1e-30))
                    r0a = sgn * r0
                    mean_all[si, ci] += r0a; cnt_all[si, ci] += 1
                    if tercile[ai] == "high":
                        mean_high[si, ci] += r0a; cnt_hi[si, ci] += 1
                    elif tercile[ai] == "low":
                        mean_low[si, ci] += r0a; cnt_lo[si, ci] += 1
                    rows.append({
                        "seed": seed, "adapter_id": ids[ai], "real_acc": float(real_acc[ai]),
                        "tercile": tercile[ai], "acc_logit": acc_logit,
                        "module_key": key, "layer": layer,
                        "residual_side": side,
                        **{f"sigma{j}": float(S[j]) if j < len(S) else 0.0 for j in range(8)},
                        "sigma0_share": float(S[0] / S.sum()),
                        "sigma0_over1": float(S[0] / S[1]),
                        "pr_sing": float((S.sum() ** 2) / (np.square(S).sum())),
                        "abscos_r0_delta": abs(ax.cos(r0, delta)),
                        "abscos_r0_deltasp": abs(ax.cos(r0, delta_sp)),
                        "delta_capture": ax.subspace_capture(basis, delta),
                        "deltasp_capture": ax.subspace_capture(basis, delta_sp),
                        "own_top_load": own_load,
                        "cos_to_seedref": cos_to_ref,
                    })
                    ci += 1
        print(f"[extract] seed {seed}: {A} adapters in {time.time()-t0:.1f}s", flush=True)

    df = pl.DataFrame(rows)
    df.write_parquet(OUT_PARQUET)
    print(f"[extract] wrote {OUT_PARQUET}  ({df.shape})")

    for m, c in ((mean_all, cnt_all), (mean_high, cnt_hi), (mean_low, cnt_lo)):
        c3 = np.where(c[..., None] > 0, c[..., None], 1.0)
        m /= c3
    np.savez_compressed(
        OUT_NPZ,
        mean_all=mean_all.astype(np.float32), mean_high=mean_high.astype(np.float32),
        mean_low=mean_low.astype(np.float32),
        cnt_all=cnt_all, cnt_hi=cnt_hi, cnt_lo=cnt_lo,
        seeds=np.array(SEEDS),
        order=np.array([k for k, _ in uv.cell_order(uv.load_model(str(CKPT_DIR/f'w8_l2_seed{SEEDS[0]}.pth'),'cpu'))], dtype=object),
        delta=delta.astype(np.float32), delta_sp=delta_sp.astype(np.float32),
        n_layers=NL,
    )
    print(f"[extract] wrote {OUT_NPZ}")


def analyze_main():
    import numpy as np, polars as pl

    from experiments.uv_probe import axes as ax

    df = pl.read_parquet(OUT_PARQUET)
    npz = np.load(OUT_NPZ, allow_pickle=True)
    order = list(npz["order"]); NL = int(npz["n_layers"])
    delta = npz["delta"].astype(float); delta_sp = npz["delta_sp"].astype(float)
    mean_all = npz["mean_all"]; mean_high = npz["mean_high"]; mean_low = npz["mean_low"]
    seeds = list(npz["seeds"]); S = len(seeds)
    cell_key = [(order[c // NL], c % NL) for c in range(len(order) * NL)]

    findings = {}

    # ---------- Q1 fidelity ----------
    sh = df["sigma0_share"].to_numpy()
    findings["Q1_fidelity"] = {
        "sigma0_share_median": float(np.median(sh)),
        "sigma0_share_p05": float(np.percentile(sh, 5)),
        "sigma0_share_p95": float(np.percentile(sh, 95)),
        "sigma0_over1_median": float(np.median(df["sigma0_over1"].to_numpy())),
        "pr_sing_median": float(np.median(df["pr_sing"].to_numpy())),
        "note": "rank-1 iff sigma0_share~1; classifiers were 0.86-0.999.",
    }
    # per module/layer grid (mean over seeds+adapters)
    g = (df.group_by(["module_key", "layer"])
           .agg(pl.col("sigma0_share").mean().alias("sigma0_share"),
                pl.col("sigma0_over1").mean().alias("sigma0_over1"),
                pl.col("pr_sing").mean().alias("pr_sing"),
                pl.col("abscos_r0_delta").mean().alias("abscos_r0_delta"),
                pl.col("delta_capture").mean().alias("delta_capture"),
                pl.col("deltasp_capture").mean().alias("deltasp_capture"),
                pl.col("cos_to_seedref").mean().alias("within_seed_cos"),
                pl.col("residual_side").first().alias("residual_side"))
           .with_columns(pl.col("module_key").replace(MODULE_SHORT).alias("module")))
    g = g.sort(["module_key", "layer"])
    g.write_csv(DATA / "cell_grid.csv")

    # per module summary
    gm = (df.group_by("module_key")
            .agg(pl.col("sigma0_share").mean(), pl.col("delta_capture").mean(),
                 pl.col("abscos_r0_delta").mean(), pl.col("deltasp_capture").mean())
            .with_columns(pl.col("module_key").replace(MODULE_SHORT).alias("module"))
            .sort("module_key"))
    gm.write_csv(DATA / "module_summary.csv")

    # ---------- Q2 delta legibility ----------
    findings["Q2_delta"] = {
        "abscos_r0_delta_median": float(np.median(df["abscos_r0_delta"].to_numpy())),
        "abscos_r0_delta_max": float(np.max(df["abscos_r0_delta"].to_numpy())),
        "delta_capture_median": float(np.median(df["delta_capture"].to_numpy())),
        "delta_capture_max": float(np.max(df["delta_capture"].to_numpy())),
        "deltasp_capture_median": float(np.median(df["deltasp_capture"].to_numpy())),
        "deltasp_capture_max": float(np.max(df["deltasp_capture"].to_numpy())),
        "random_subspace_capture_expected": 8.0 / 2048.0,  # rank-8 in 2048-d ~ 0.0625 rms
        "random_capture_rms": float(np.sqrt(8.0 / 2048.0)),
    }
    # top cells by delta capture
    topcap = g.sort("delta_capture", descending=True).head(12)
    topcap.write_csv(DATA / "top_delta_capture_cells.csv")

    # ---------- Q3 stability + accuracy rotation ----------
    rows = []
    for c in range(len(order) * NL):
        key, layer = cell_key[c]
        # seed stability: mean pairwise |cos| of the per-seed mean_all dirs
        sc = []
        for i in range(S):
            for j in range(i + 1, S):
                sc.append(abs(_cos(mean_all[i, c], mean_all[j, c])))
        seed_stab = float(np.mean(sc)) if sc else float("nan")
        # accuracy rotation: |cos| between high-acc mean and low-acc mean (avg over seeds)
        rot = [abs(_cos(mean_high[s, c], mean_low[s, c])) for s in range(S)]
        rot = float(np.mean(rot))
        # alignment of the seed-averaged mean_all dir with delta
        mean_dir = mean_all.mean(axis=0)[c]
        rows.append({"module_key": key, "layer": layer, "module": MODULE_SHORT[key],
                     "seed_stability_abscos": seed_stab, "acc_rotation_abscos": rot,
                     "meandir_abscos_delta": abs(_cos(mean_dir, delta))})
    stab = pl.DataFrame(rows).sort(["module_key", "layer"])
    stab.write_csv(DATA / "stability.csv")
    findings["Q3_stability"] = {
        "seed_stability_median": float(np.median(stab["seed_stability_abscos"].to_numpy())),
        "acc_rotation_median": float(np.median(stab["acc_rotation_abscos"].to_numpy())),
        "within_seed_cos_median": float(np.median(df["cos_to_seedref"].to_numpy())),
        "note": "seed_stability/acc_rotation are |cos| of mean directions; 1=identical, 0=orthogonal",
    }

    # ---------- Q4 accuracy signal ----------
    # acc_logit vs real_acc (per seed, then pooled consensus)
    cons = (df.group_by("adapter_id")
              .agg(pl.col("acc_logit").mean().alias("acc_logit"),
                   pl.col("real_acc").first().alias("real_acc")))
    al = cons["acc_logit"].to_numpy(); ra = cons["real_acc"].to_numpy()
    findings["Q4_accuracy"] = {
        "acc_logit_vs_real_pearson": float(np.corrcoef(al, ra)[0, 1]),
        "n_adapters": int(cons.shape[0]),
    }
    # own_top_load vs real_acc, correlation per cell (does the top direction's own load track acc?)
    corr_rows = []
    for c in range(len(order) * NL):
        key, layer = cell_key[c]
        sub = df.filter((pl.col("module_key") == key) & (pl.col("layer") == layer))
        # consensus per adapter
        cc = (sub.group_by("adapter_id")
                 .agg(pl.col("own_top_load").mean().alias("load"),
                      pl.col("real_acc").first().alias("acc")))
        x = cc["load"].to_numpy(); y = cc["acc"].to_numpy()
        r = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 else float("nan")
        corr_rows.append({"module_key": key, "layer": layer, "module": MODULE_SHORT[key],
                          "own_load_vs_acc_pearson": r})
    cr = pl.DataFrame(corr_rows).sort("own_load_vs_acc_pearson")
    cr.write_csv(DATA / "own_load_corr.csv")
    findings["Q4_accuracy"]["own_load_vs_acc_absmax"] = float(
        np.nanmax(np.abs(cr["own_load_vs_acc_pearson"].to_numpy())))

    # ---------- logit lens of the mean_all direction for key residual writers ----------
    axd = ax.load_axes()
    lens = {}
    for key in ("mlp.down_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj"):
        for layer in (15, 14, 0):
            c = order.index(key) * NL + layer
            mean_dir = mean_all.mean(axis=0)[c]
            L = ax.logit_lens(mean_dir, axd["tok"], axd["E_norm"], top_k=12)
            if L is not None:
                lens[f"{MODULE_SHORT[key]}-L{layer}"] = {
                    "top_tokens": L["top_tokens"], "top_cos": [round(x, 3) for x in L["top_cos"]],
                    "abscos_delta": abs(_cos(mean_dir, delta)),
                }
    with open(DATA / "logit_lens.json", "w") as f:
        json.dump(lens, f, indent=2)

    with open(DATA / "findings.json", "w") as f:
        json.dump(findings, f, indent=2)
    print(json.dumps(findings, indent=2))
    print(f"\n[analyze] wrote CSVs + findings to {DATA}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["extract", "analyze"])
    args = ap.parse_args()
    if args.stage == "extract":
        extract_main()
    else:
        analyze_main()
