"""PC1 (the accuracy axis) in cell space + high/low-accuracy cluster cohesion.

(migrated from SRC/src/discoveries/sst2_perf_regressor_lrp/jobs/06_pc1_and_clusters.py)

Reads ``lrp_maps.npz`` (from ``run_maps.py``) plus a metadata parquet carrying
the measured ``acc`` per adapter, and computes:

- the variance participation ratio + PC1 variance share (~92%);
- the PC1 loading over the 7×L cell grid, oriented to +accuracy;
- pairwise map cosine similarity + tercile cohesion (within-high / within-low /
  low-vs-high, corr(accuracy, mean-cosine-to-others)).

Writes ``pc1_loading.csv``, ``cluster_cohesion.csv``, example-grid CSVs,
``pc1_clusters_summary.json`` and the two figures under ``results_path("lrp")``.
numpy / matplotlib only.

Example
-------
    python -m experiments.lrp.pc1_clusters --meta /path/test_meta.parquet
"""

from __future__ import annotations

import argparse
import json

import numpy as np

MODS = ["Q", "K", "V", "O", "gate", "up", "down"]


def rn(Y):
    n = np.linalg.norm(Y, axis=-1, keepdims=True); n[n == 0] = 1.0
    return Y / n


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import polars as pl

    from common.env import results_path

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", required=True,
                    help="parquet with adapter_id + acc (the measured SST2 accuracy)")
    ap.add_argument("--maps", default=None, help="lrp_maps.npz (default: results_path('lrp'))")
    args = ap.parse_args()

    out = results_path("lrp")
    z = np.load(args.maps or (out / "lrp_maps.npz"), allow_pickle=True)
    NL = int(z["n_layers"])
    A = z["signed"].shape[1]
    X = z["signed"].reshape(z["signed"].shape[0], A, -1)
    map_ids = [str(x) for x in z["adapter_ids"]]

    meta = pl.read_parquet(args.meta)
    id_col = "adapter_id" if "adapter_id" in meta.columns else "__key__"
    acc_by_id = dict(zip(meta[id_col].to_list(), meta["acc"].to_list()))
    real = np.array([acc_by_id[i] for i in map_ids], float)
    ids = map_ids

    M = X.mean(0)                     # (A, 7*NL)
    grid = M.reshape(A, 7, NL)

    Mc = M - M.mean(0)
    U, sv, Vt = np.linalg.svd(Mc, full_matrices=False)
    lam = sv ** 2
    pr_var = float((lam.sum() ** 2) / (lam ** 2).sum())
    pc1 = U[:, 0] * sv[0]; load = Vt[0].reshape(7, NL)
    if np.corrcoef(pc1, real)[0, 1] < 0:
        pc1, load = -pc1, -load

    Sim = rn(M) @ rn(M).T
    S_off = Sim.copy(); np.fill_diagonal(S_off, np.nan)
    mto = np.nanmean(S_off, axis=1)                    # mean cosine to all others

    qlo, qhi = np.quantile(real, [1 / 3, 2 / 3])
    hi = np.where(real >= qhi)[0]; lo = np.where(real < qlo)[0]

    def within(idx):
        return float(np.mean([Sim[i, j] for a, i in enumerate(idx) for j in idx[a + 1:]]))
    cohesion = {"within_high_acc": within(hi), "within_low_acc": within(lo),
                "low_vs_high": float(np.mean([Sim[i, j] for i in lo for j in hi])),
                "corr_acc_vs_meancos": float(np.corrcoef(real, mto)[0, 1]),
                "meancos_high": float(mto[hi].mean()), "meancos_low": float(mto[lo].mean())}

    typ = hi[np.argsort(mto[hi])[::-1][:3]]            # most-typical high-acc
    dist = lo[np.argsort(mto[lo])[:3]]                 # most-distinctive low-acc
    targets = np.linspace(pc1.min(), pc1.max(), 4)
    pc1_examples = [int(np.argmin(np.abs(pc1 - t))) for t in targets]

    summary = {
        "variance_participation_ratio": pr_var,
        "pc1_var_ratio": float(lam[0] / lam.sum()),
        "pc1_corr_accuracy": float(np.corrcoef(pc1, real)[0, 1]),
        "pc1_loading_per_module": {MODS[i]: float(load[i].sum()) for i in range(7)},
        "cluster_cohesion": cohesion,
        "typical_high_acc": [{"adapter_id": ids[a], "acc": float(real[a]), "mean_cos": float(mto[a])} for a in typ],
        "distinctive_low_acc": [{"adapter_id": ids[a], "acc": float(real[a]), "mean_cos": float(mto[a])} for a in dist],
        "pc1_examples": [{"adapter_id": ids[a], "pc1": float(pc1[a]), "acc": float(real[a])} for a in pc1_examples],
    }
    (out / "pc1_clusters_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

    # ── CSV tables ──
    pl.DataFrame([{"module": MODS[i], "layer": l, "loading": float(load[i, l])}
                  for i in range(7) for l in range(NL)]).write_csv(out / "pc1_loading.csv")
    pl.DataFrame([{"group": k, "value": v} for k, v in cohesion.items()]).write_csv(out / "cluster_cohesion.csv")

    def grid_rows(indices, role_tag, extra):
        rows = []
        for a in indices:
            for mi in range(7):
                for li in range(NL):
                    rows.append({"adapter_id": ids[a][:8], "role": role_tag, "acc": float(real[a]),
                                 **{k: extra(a)[k] for k in extra(a)},
                                 "module": MODS[mi], "layer": li, "signed": float(grid[a, mi, li])})
        return rows
    pl.DataFrame(grid_rows(pc1_examples, "pc1", lambda a: {"pc1": float(pc1[a])})).write_csv(
        out / "pc1_example_grids.csv")
    clus_rows = (grid_rows(list(typ), "typical_high", lambda a: {"mean_cos": float(mto[a])}) +
                 grid_rows(list(dist), "distinctive_low", lambda a: {"mean_cos": float(mto[a])}))
    pl.DataFrame(clus_rows).write_csv(out / "cluster_example_grids.csv")

    # ── figures ──
    plt.rcParams.update({"figure.dpi": 130, "font.size": 8})

    fig = plt.figure(figsize=(12.5, 3.6))
    gs = fig.add_gridspec(4, 3, width_ratios=[1.25, 1.1, 1.5], wspace=0.55, hspace=0.35)
    axL = fig.add_subplot(gs[:, 0]); axS = fig.add_subplot(gs[:, 1])
    vmax = float(np.nanmax(np.abs(load)))
    im = axL.imshow(load, cmap="PuOr_r", vmin=-vmax, vmax=vmax, aspect="auto")
    axL.set_yticks(range(7)); axL.set_yticklabels(MODS); axL.set_xlabel("LLM layer")
    axL.set_title("PC1 loading (accuracy axis)\n+ = high-accuracy direction")
    fig.colorbar(im, ax=axL, fraction=0.046, pad=0.04)
    axS.scatter(real, pc1, s=16, c=real, cmap="viridis")
    axS.set_xlabel("real accuracy"); axS.set_ylabel("PC1 score")
    axS.set_title(f"PC1 score vs accuracy\nr={np.corrcoef(pc1, real)[0, 1]:+.2f}, PC1={lam[0] / lam.sum() * 100:.0f}% var")
    for s in ("top", "right"):
        axS.spines[s].set_visible(False)
    for k, a in enumerate(pc1_examples[::-1]):
        axx = fig.add_subplot(gs[k, 2])
        vm = float(np.nanmax(np.abs(grid[a])))
        axx.imshow(grid[a], cmap="RdBu_r", vmin=-vm, vmax=vm, aspect="auto")
        axx.set_yticks([]); axx.set_xticks([])
        axx.set_ylabel(f"pc1 {pc1[a]:+.2f}\nacc {real[a]:.2f}", fontsize=7, rotation=0,
                       labelpad=20, va="center", ha="right")
        if k == 0:
            axx.set_title("maps spanning PC1  (down_proj → attention)", fontsize=8)
    fig.suptitle("PC1 — the accuracy axis is the attention→MLP-down contrast", y=1.04)
    fig.savefig(out / "fig8_pc1_axis.png", bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(12, 5.6))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.15, 1.0], hspace=0.5, wspace=0.5)
    axA = fig.add_subplot(gs[0, 0:3])
    axA.scatter(real, mto, s=18, c=real, cmap="viridis")
    axA.set_xlabel("real accuracy"); axA.set_ylabel("mean map-cosine to all others")
    axA.set_title(f"Low-accuracy adapters are the distinctive ones\nr={cohesion['corr_acc_vs_meancos']:+.2f}")
    for s in ("top", "right"):
        axA.spines[s].set_visible(False)
    axB = fig.add_subplot(gs[0, 3:6])
    axB.bar([0, 1, 2], [cohesion["within_high_acc"], cohesion["within_low_acc"], cohesion["low_vs_high"]],
            color=["#41ae76", "#d7301f", "#999999"])
    axB.set_xticks([0, 1, 2]); axB.set_xticklabels(["within\nhigh-acc", "within\nlow-acc", "low vs\nhigh"])
    axB.set_ylabel("mean pairwise cosine"); axB.set_ylim(0, 1)
    axB.set_title("High-acc cluster is tight; low-acc sit far from it")
    for s in ("top", "right"):
        axB.spines[s].set_visible(False)
    for col, a in enumerate(list(typ) + list(dist)):
        axx = fig.add_subplot(gs[1, col])
        vm = float(np.nanmax(np.abs(grid[a])))
        axx.imshow(grid[a], cmap="RdBu_r", vmin=-vm, vmax=vm, aspect="auto")
        axx.set_yticks(range(7)); axx.set_yticklabels(MODS, fontsize=5); axx.set_xticks([])
        col_typ = col < 3
        axx.set_title(f"{'typical hi' if col_typ else 'distinct lo'}\nacc {real[a]:.2f}",
                      fontsize=7, color="#1b7837" if col_typ else "#b30000")
    fig.suptitle("Accuracy clusters — high-acc maps are near-identical; low-acc maps are distinctive", y=0.98)
    fig.savefig(out / "fig9_accuracy_clusters.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[pc1] wrote CSVs + summary + figures to {out}")


if __name__ == "__main__":
    main()
