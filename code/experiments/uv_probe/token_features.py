"""Per-token projections onto the regressor read/write directions, + token-feature selectivity.

(migrated from SRC/src/discoveries/sst2_perf_regressor_uv/jobs/16_token_layers.py [collect]
 and .../jobs/17_token_layer_analysis.py [analyse])

collect (``collect_main``)
    For every token of N SST2 sentences, record the norm-controlled projection of the
    residual stream onto each of the 112 regressor read/write directions (read v0 for
    readers, write u0 for writers — the per-cell TOP residual direction, read from the
    pipeline's ``mean_dirs.npz``) and onto ``K_RAND=32`` random unit directions PER LAYER
    (the rigorous 'any-direction' control). Writes ``token_layers.npz``.

    (The upstream ``uv_subspaces_full.npz`` of the original job only ever used column 0
    of each cell's U/V basis, which is exactly the seed-mean top direction stored in
    ``mean_dirs.npz`` — so the collect step is sourced from the migrated pipeline output
    rather than a separately-frozen subspace file.)

analyse (``analyze_main``)
    A battery of ~13 token features (orthographic + word-class + semantic). For each of
    the 112 directions, |Pearson r| with each feature, and the SPECIFIC selectivity =
    |r_cell| − p95(|r| over the 32 random directions at that cell's layer). Writes
    ``token_layers_cell.csv``, a summary ``token_layers_findings.json`` and a
    layer×component heatmap ``fig9_token_layers.png``.

Base-model residual collection uses ``common.hf``; the SST2 prompt uses
``shared_adapter_pool.data.definitions.sst2``. All I/O under ``results_path("uv_probe")``.
"""
from __future__ import annotations

import json
import re

from common.env import results_path

INST = results_path("uv_probe")
MEAN_DIRS = INST / "mean_dirs.npz"
OUT_NPZ = INST / "token_layers.npz"
DATA = INST / "analysis"
NL = 16
N_SENT = 500
K_RAND = 32
RESIDUAL_WRITERS = frozenset({"self_attn.o_proj", "mlp.down_proj"})
SHORT = {"self_attn.q_proj": "Q", "self_attn.k_proj": "K", "self_attn.v_proj": "V",
         "mlp.gate_proj": "gate", "mlp.up_proj": "up", "self_attn.o_proj": "O",
         "mlp.down_proj": "down"}
COMPS = ["Q", "K", "V", "O", "gate", "up", "down"]

STOP = set("the a an of to and or but in on at for with as is are was were be been being this that "
           "these those it its i you he she they we me him her them my your his our their s".split())
NEG = {"not", "n't", "no", "never", "nor", "none", "cannot", "without", "nothing", "neither"}
QUOTES = set("\"'`“”‘’")


def build_prompt(s):
    from shared_adapter_pool.data.definitions.sst2 import (
        Sst2LabelScheme, Sst2SystemPrompt, render_system_prompt,
    )
    t = render_system_prompt(Sst2SystemPrompt.DEFAULT_V1, Sst2LabelScheme.POSITIVE_NEGATIVE_V1)
    return f"{t}\n\nSentence: {s}\nResponse: "


def collect_main():
    import numpy as np, torch
    from datasets import load_dataset

    from common.hf import load_base_model, load_tokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_tokenizer()
    model = load_base_model().to(device).eval()

    # per-cell TOP residual direction (seed-mean) from the pipeline output.
    md = np.load(MEAN_DIRS, allow_pickle=True)
    order = [str(k) for k in md["order"]]
    mean_all = md["mean_all"]                 # (S, 112, 2048)
    D_np = mean_all.mean(0)                    # (112, 2048) seed-mean direction per cell
    D_np = D_np / (np.linalg.norm(D_np, axis=-1, keepdims=True) + 1e-12)
    cell_list = [(order[c // NL], c % NL) for c in range(len(order) * NL)]
    sides = {(k, L): ("write" if k in RESIDUAL_WRITERS else "read") for k, L in cell_list}

    D = torch.tensor(D_np.astype(np.float32), device=device)                    # (112,2048)
    layers_of = torch.tensor([L for (_, L) in cell_list], device=device)

    # random directions per layer boundary (17), norm 1
    g = torch.Generator(device="cpu").manual_seed(0)
    R = torch.randn(NL + 1, K_RAND, 2048, generator=g)
    R = (R / R.norm(dim=-1, keepdim=True)).to(device)   # (17,K,2048)

    ds = load_dataset("stanfordnlp/sst2", split="validation")
    sents = [r["sentence"] for r in ds][:N_SENT]; labels = [int(r["label"]) for r in ds][:N_SENT]
    prefix = build_prompt("").split("Sentence: ")[0] + "Sentence: "
    plen = len(tok(prefix, add_special_tokens=True).input_ids)

    toks_s, sent_l, posn = [], [], []
    proj_cell = []   # (Ntok,112)
    proj_rand = []   # (Ntok,17,K)
    bs = 16
    with torch.no_grad():
        for i in range(0, len(sents), bs):
            batch = [build_prompt(s) for s in sents[i:i+bs]]
            enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=True).to(device)
            hs = model(**enc, output_hidden_states=True).hidden_states  # 17 x (B,T,2048)
            H = torch.stack(hs, dim=2).float()      # (B,T,17,2048)
            Hn = H / (H.norm(dim=-1, keepdim=True) + 1e-6)
            lastpos = enc["attention_mask"].sum(1) - 1
            for b in range(len(batch)):
                lp = int(lastpos[b])
                for pos in range(plen, lp + 1):
                    hn = Hn[b, pos]                  # (17,2048)
                    # regressor cells: pick each cell's own layer
                    pc = (D * hn[layers_of]).sum(-1)          # (112,)
                    pr = torch.einsum("lkd,ld->lk", R, hn)    # (17,K)
                    proj_cell.append(pc.cpu().numpy()); proj_rand.append(pr.cpu().numpy())
                    toks_s.append(tok.decode([int(enc["input_ids"][b, pos])]))
                    sent_l.append(labels[i + b]); posn.append((pos - plen) / max(lp - plen, 1))
            if i % (bs * 5) == 0:
                print(f"[H2] {i}/{len(sents)} sents, {len(toks_s)} tokens", flush=True)

    np.savez_compressed(
        OUT_NPZ,
        cells=np.array([f"{SHORT[k]}-L{l}" for k, l in cell_list], dtype=object),
        cell_comp=np.array([SHORT[k] for k, l in cell_list], dtype=object),
        cell_layer=np.array([l for k, l in cell_list]),
        cell_side=np.array([sides[c] for c in cell_list], dtype=object),
        token=np.array(toks_s, dtype=object), sent_label=np.array(sent_l),
        position=np.array(posn, np.float32),
        proj_cell=np.array(proj_cell, np.float32),
        proj_rand=np.array(proj_rand, np.float32),
    )
    print(f"[H2] wrote {OUT_NPZ}: {len(toks_s)} tokens x {len(cell_list)} cells + {K_RAND} rand/layer")


def features(toks, cnt, pos, sent):
    import numpy as np
    def wl(t): return t.strip().lower()
    F = {}
    F["word_initial"] = np.array([t.startswith(" ") or t.startswith("▁") for t in toks], float)
    F["punctuation"] = np.array([bool(re.fullmatch(r"\W+", t.strip())) and t.strip() != "" for t in toks], float)
    F["capitalized"] = np.array([any(c.isupper() for c in t) for t in toks], float)
    F["is_digit"] = np.array([t.strip().isdigit() for t in toks], float)
    F["char_len"] = np.array([len(t.strip()) for t in toks], float)
    F["log_freq"] = np.array([np.log(cnt.get(wl(t), [0, 1])[1] + 1) for t in toks])
    F["position"] = pos.astype(float)
    F["stopword"] = np.array([wl(t) in STOP for t in toks], float)
    F["negation"] = np.array([wl(t) in NEG for t in toks], float)
    F["content_word"] = np.array([(t.strip().isalpha() and (t.startswith(" ")) and wl(t) not in STOP and len(t.strip()) >= 3) for t in toks], float)
    F["is_alpha"] = np.array([t.strip().isalpha() for t in toks], float)
    F["quote"] = np.array([t.strip() in QUOTES for t in toks], float)
    ts = np.array([(cnt[wl(t)][0] / cnt[wl(t)][1]) if (wl(t) in cnt and cnt[wl(t)][1] >= 5) else np.nan for t in toks])
    F["token_sentiment"] = ts
    return F


def corr(a, b):
    import numpy as np
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 30 or np.std(a[m]) == 0 or np.std(b[m]) == 0: return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def analyze_main():
    import numpy as np, polars as pl
    from datasets import load_dataset

    z = np.load(OUT_NPZ, allow_pickle=True)
    toks = [t for t in z["token"]]; pos = z["position"]; sent = z["sent_label"]
    pc = z["proj_cell"]; pr = z["proj_rand"]  # (N,112),(N,17,K)
    cells = list(z["cells"]); comp = list(z["cell_comp"]); clay = z["cell_layer"]; side = list(z["cell_side"])
    ds = load_dataset("stanfordnlp/sst2", split="train")
    from collections import defaultdict
    cnt = defaultdict(lambda: [0, 0])
    for r in ds:
        lab = int(r["label"])
        for w in set(re.findall(r"[a-z']+", r["sentence"].lower())): cnt[w][0] += lab; cnt[w][1] += 1
    F = features(toks, cnt, pos, sent)
    feats = list(F)
    print(f"[H2] {pc.shape[0]} tokens, {len(cells)} cells, {len(feats)} features, K_rand={pr.shape[2]}")

    # random baseline per layer per feature: p95 of |r| over the K random dirs
    randp95 = {}   # (layer,feat)->p95
    for L in range(NL + 1):
        for f in feats:
            rs = [abs(corr(pr[:, L, k], F[f])) for k in range(pr.shape[2])]
            rs = [x for x in rs if np.isfinite(x)]
            randp95[(L, f)] = float(np.percentile(rs, 95)) if rs else np.nan

    rows = []
    for ci, cell in enumerate(cells):
        L = int(clay[ci])
        rec = {"cell": cell, "component": comp[ci], "layer": L, "side": side[ci]}
        for f in feats:
            r = abs(corr(pc[:, ci], F[f]))
            rec[f] = r
            rec[f + "_spec"] = r - randp95[(L, f)]   # specific selectivity vs random at this layer
        rows.append(rec)
    df = pl.DataFrame(rows)
    DATA.mkdir(exist_ok=True, parents=True)
    df.write_csv(DATA / "token_layers_cell.csv")

    # summary: best specific selectivity per feature + where
    summary = {}
    for f in feats:
        sp = df[f + "_spec"].to_numpy(); raw = df[f].to_numpy()
        bi = int(np.nanargmax(sp))
        summary[f] = {"best_cell": cells[bi], "best_raw_absr": float(raw[bi]),
                      "best_specific": float(sp[bi]),
                      "n_cells_specific_gt_0.05": int(np.nansum(sp > 0.05)),
                      "median_raw_absr": float(np.nanmedian(raw))}
    order = sorted(feats, key=lambda f: -summary[f]["best_specific"])
    findings = {"features_ranked_by_specific_selectivity":
                {f: summary[f] for f in order},
                "note": "specific = |r| minus p95 of |r| over 32 random dirs at same layer; >0 beats an arbitrary direction"}
    json.dump(findings, open(DATA / "token_layers_findings.json", "w"), indent=2)
    print(json.dumps({f: summary[f] for f in order[:8]}, indent=2))

    # heatmaps layer x component for the top-6 features (specific selectivity)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    def grid(colspec):
        G = np.full((7, NL), np.nan)
        for ci in range(len(cells)):
            mi = COMPS.index(comp[ci]); G[mi, int(clay[ci])] = df[colspec][ci]
        return G
    top6 = order[:6]
    fig, axes = plt.subplots(2, 3, figsize=(14, 6))
    for ax, f in zip(axes.flat, top6):
        G = grid(f + "_spec")
        vmax = max(0.1, np.nanmax(np.abs(G)))
        im = ax.imshow(G, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_yticks(range(7)); ax.set_yticklabels(COMPS, fontsize=7)
        ax.set_xlabel("layer", fontsize=7)
        ax.set_title(f"{f}  (best spec {summary[f]['best_specific']:+.2f} @ {summary[f]['best_cell']})", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Token-function SPECIFIC selectivity (regressor − random control) by layer × component", fontsize=11)
    fig.tight_layout(); fig.savefig(DATA.parent / "fig9_token_layers.png"); plt.close(fig)
    print("[H2] wrote token_layers_cell.csv + fig9_token_layers.png")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["collect", "analyze"])
    args = ap.parse_args()
    if args.stage == "collect":
        collect_main()
    else:
        analyze_main()
