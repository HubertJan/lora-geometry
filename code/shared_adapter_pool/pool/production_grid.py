"""Expand production "recipe cells" into a spread, HP-jittered pool of AdapterCfgs.

(migrated from src/discoveries/sst2_perf_prediction/flows/production_grid.py)

Goal: ~200 sst2 rank-16 true/false adapters whose SST2 accuracy densely spans
~0.55–0.96, with within-band diversity and nuisance HPs (lr / dropout / alpha /
seed / data subset) DECORRELATED from performance, so the regressor cannot cheat
by reading an HP. Performance is driven by the cell (samples × epochs × label
noise); everything else is jittered per replicate.

A "cell" is one (shards_total, epochs, label_noise) recipe. Each cell is
replicated ``replicates`` times; replicate ``r`` gets:
  * a distinct shard slice ``[r % shards_total]`` (a different data subset — the
    user's "different subsets of SST2"),
  * jittered lr / dropout / alpha-mult / seed from a per-adapter RNG,
  * a pool assignment (~1/``test_every`` replicates → the held-out TEST pool),
so both pools span every band.
"""

from __future__ import annotations

from dataclasses import dataclass

from shared_adapter_pool.pool.spec import AdapterCfg

ALL7 = ["k_proj", "q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass(frozen=True)
class Cell:
    shards_total: int   # shard slice -> ~ MAX_TRAIN_PREP / shards_total samples
    epochs: int
    label_noise: float
    kind: str           # "clean" | "noise" | "collapse" (label only)


def build_production_cfgs(
    cells: list[Cell],
    *,
    replicates: int,
    train_slug: str = "sst2",
    eval_slug: str = "sst2",
    label_scheme: str = "TRUE_FALSE_V1",
    lr_min: float = 5e-5,
    lr_max: float = 3e-4,
    dropout_min: float = 0.0,
    dropout_max: float = 0.1,
    alpha_mult_choices: tuple[int, ...] = (1, 2),
    lora_rank: int = 16,
    master_seed: int = 20260820,
    test_every: int = 4,
    campaign: str = "",
) -> list[AdapterCfg]:
    import numpy as np

    cfgs: list[AdapterCfg] = []
    idx = 0
    for cell in cells:
        for r in range(replicates):
            ss = np.random.SeedSequence([master_seed, idx])
            hp_rng = np.random.default_rng(ss)
            lr = float(hp_rng.uniform(lr_min, lr_max))
            dropout = float(hp_rng.uniform(dropout_min, dropout_max))
            alpha_mult = int(hp_rng.choice(alpha_mult_choices))
            train_seed = int(ss.generate_state(1)[0])
            pool = "test" if (r % test_every == 0) else "train"
            cfgs.append(
                AdapterCfg(
                    adapter_idx=idx,
                    pool=pool,
                    train_slug=train_slug,
                    eval_slug=eval_slug,
                    label_scheme=label_scheme,
                    shards_total=cell.shards_total,
                    shard_indices=[r % cell.shards_total],
                    learning_rate=lr,
                    weight_decay=1e-5,
                    epochs=cell.epochs,
                    lora_rank=lora_rank,
                    lora_alpha=lora_rank * alpha_mult,
                    lora_dropout=dropout,
                    lora_target_modules=list(ALL7),
                    train_seed=train_seed,
                    label_noise=cell.label_noise,
                    campaign=campaign,
                )
            )
            idx += 1
    return cfgs


#: Default cell set, tuned from the clean + noise calibrations. MAX_TRAIN_PREP=
#: 20000, so shards_total -> approx samples: 1->20000, 2->10000, 4->5000, 8->2500,
#: 16->1250, 32->625, 64->312, 128->156, 256->78.
#:
#: Findings driving the mix (both bias toward the UNDER-represented low/mid band,
#: since the high band is trivially reachable):
#:   * Clean cliff: <=156 samples collapse to ~0.56 (1-2 ep) but escape to ~0.81
#:     at 3 ep; >=625 samples reach ~0.89-0.96.
#:   * Label noise is only potent at LOW sample counts (2500/0.4 still ~0.90;
#:     625/0.4 -> 0.68). So mid/low fill uses ~625 & ~312 samples with high noise.
DEFAULT_CELLS: list[Cell] = [
    # -- low band ~0.55-0.68 --
    Cell(256, 2, 0.0, "collapse"),    # ~78 samples, collapse ~0.56
    Cell(128, 2, 0.0, "collapse"),    # ~156, collapse ~0.56
    Cell(32, 2, 0.48, "noise"),       # ~625 + 48% noise -> ~0.56
    Cell(32, 2, 0.45, "noise"),       # ~625 + 45% -> ~0.60
    Cell(64, 2, 0.40, "noise"),       # ~312 + 40% -> ~0.62
    Cell(32, 2, 0.40, "noise"),       # ~625 + 40% -> 0.68
    # -- mid band ~0.70-0.88 --
    Cell(128, 3, 0.0, "clean"),       # ~156, 3 ep -> ~0.81 (cliff escape)
    Cell(64, 3, 0.0, "clean"),        # ~312, 3 ep -> ~0.85
    Cell(64, 2, 0.30, "noise"),       # ~312 + 30% -> ~0.78
    Cell(32, 2, 0.35, "noise"),       # ~625 + 35% -> ~0.83
    Cell(32, 3, 0.30, "noise"),       # ~625 + 30% -> ~0.86
    Cell(32, 1, 0.0, "clean"),        # ~625, 1 ep -> 0.888
    # -- high band ~0.89-0.96 --
    Cell(32, 2, 0.0, "clean"),        # ~625, 2 ep -> 0.916
    Cell(16, 2, 0.0, "clean"),        # ~1250 -> ~0.92
    Cell(8, 2, 0.40, "noise"),        # ~2500 + 40% -> ~0.90
    Cell(8, 2, 0.0, "clean"),         # ~2500 -> 0.9365
    Cell(8, 3, 0.0, "clean"),         # ~2500, 3 ep -> ~0.94
    Cell(4, 2, 0.0, "clean"),         # ~5000 -> ~0.95
    Cell(2, 2, 0.0, "clean"),         # ~10000 -> ~0.95
    Cell(1, 2, 0.0, "clean"),         # ~20000 -> 0.955
    Cell(1, 3, 0.0, "clean"),         # ~20000, 3 ep -> 0.95
]
