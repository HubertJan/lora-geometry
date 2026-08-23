"""OOD arm: adapter pools trained on OTHER binary true/false datasets, all

(migrated from src/discoveries/sst2_perf_prediction/flows/ood_grid.py)
evaluated on the SST2 test split (the fixed target).

The in-dataset grid (``production_grid.py``) calibrated cells to hit specific
SST2-accuracy bands *for SST2-trained adapters*. The OOD arm does NOT need
per-dataset band calibration — the LOTO study only needs **within-dataset spread**
in SST2-transfer performance, which a diverse recipe set (under-training × label
noise × data volume) produces naturally on any dataset. So we reuse one
dataset-agnostic ``OOD_CELLS`` set across every training dataset and let the
transfer distribution fall where it may (that distribution IS the object of study:
between-group = which dataset, within-group = training quality).

``build_ood_cfgs`` emits ~``per_dataset`` rank-16 TRUE_FALSE_V1 adapters for each
of ``OOD_DATASETS``, ``train_slug=<dataset>`` / ``eval_slug="sst2"``. Every
dataset uses its own ``master_seed`` so HP jitter is decorrelated across groups,
and ``AdapterCfg.label`` already embeds ``train_slug`` so labels/qnames never
collide across datasets.
"""

from __future__ import annotations

from shared_adapter_pool.pool.production_grid import (
    Cell,
    build_production_cfgs,
)
from shared_adapter_pool.pool.spec import AdapterCfg

#: 6 OOD training datasets (+ SST2 itself = 7 LOTO groups). 4 sentiment sets that
#: should transfer well to SST2, 2 non-sentiment binary sets that should transfer
#: poorly — the pair that stresses the provenance-vs-quality confound.
OOD_DATASETS: list[str] = [
    "imdb",
    "rotten_tomatoes",
    "yelp_polarity",
    "amazon_polarity",
    "boolq",
    "subj",
]

#: Two NEW non-sentiment binary FAMILIES (job 22 extension pool), each 4 datasets
#: deep to mirror the 4-deep sentiment family. All are genuinely binary and already
#: carry the shared ``TRUE_FALSE_V1`` verbalizer (proven in ``btf_yesno_families``),
#: so no dataset-module code is needed. ``toxicity`` = single-text affect/harm
#: ("near" off-family); ``entailment`` = sentence/passage-pair relational reasoning
#: ("far" off-family). All evaluated on SST2, exactly like the existing groups.
TOXICITY: list[str] = [
    "wiki_toxic",
    "toxigen",
    "civil_comments",
    "wildguard_prompt_harm",
]
ENTAILMENT: list[str] = [
    "qnli",
    "scitail",
    "doc_nli",
    "vitaminc",
]

#: The NEW datasets (job 22 pool), in the order their per-group seeds are derived
#: (``base_seed + 1009*position``). Built as a SEPARATE pool from the existing 7
#: groups so their positions never shift and re-seed the validated pool.
FAM_DATASETS: list[str] = [*TOXICITY, *ENTAILMENT]

#: train_slug -> family, for family-level (between/within) decomposition and the
#: leave-one-family-out (LOFO) folds. ``sst2`` is the in-task reference / eval
#: target; ``boolq``/``subj`` are pre-existing non-sentiment singletons.
FAMILY_OF: dict[str, str] = {
    "imdb": "sentiment",
    "rotten_tomatoes": "sentiment",
    "yelp_polarity": "sentiment",
    "amazon_polarity": "sentiment",
    "wiki_toxic": "toxicity",
    "toxigen": "toxicity",
    "civil_comments": "toxicity",
    "wildguard_prompt_harm": "toxicity",
    "qnli": "entailment",
    "scitail": "entailment",
    "doc_nli": "entailment",
    "vitaminc": "entailment",
    "boolq": "qa",
    "subj": "subjectivity",
    "sst2": "sst2",
}

#: Dataset-agnostic diversity: 16 recipes spanning heavy under-training / high
#: label noise (low transfer) → full-data clean (high transfer). Sample counts
#: assume MAX_TRAIN_PREP=20000 (shards_total -> ~samples: 256->78, 128->156,
#: 64->312, 32->625, 16->1250, 8->2500, 4->5000, 2->10000, 1->20000).
OOD_CELLS: list[Cell] = [
    # -- expect-low transfer: collapse / heavy noise --
    Cell(256, 2, 0.0, "collapse"),   # ~78 samples
    Cell(128, 2, 0.0, "collapse"),   # ~156
    Cell(64, 2, 0.40, "noise"),      # ~312 + 40% noise
    Cell(32, 2, 0.40, "noise"),      # ~625 + 40%
    # -- expect-mid transfer --
    Cell(128, 3, 0.0, "clean"),      # ~156, 3 ep
    Cell(64, 3, 0.0, "clean"),       # ~312, 3 ep
    Cell(32, 2, 0.30, "noise"),      # ~625 + 30%
    Cell(32, 1, 0.0, "clean"),       # ~625, 1 ep
    # -- expect-high transfer: clean, growing data --
    Cell(32, 2, 0.0, "clean"),       # ~625
    Cell(16, 2, 0.0, "clean"),       # ~1250
    Cell(8, 2, 0.40, "noise"),       # ~2500 + 40%
    Cell(8, 2, 0.0, "clean"),        # ~2500
    Cell(8, 3, 0.0, "clean"),        # ~2500, 3 ep
    Cell(4, 2, 0.0, "clean"),        # ~5000
    Cell(2, 2, 0.0, "clean"),        # ~10000
    Cell(1, 2, 0.0, "clean"),        # ~20000
]


def build_ood_cfgs(
    *,
    datasets: list[str] | None = None,
    per_dataset: int = 80,
    eval_slug: str = "sst2",
    base_seed: int = 20260821,
    campaign: str = "",
) -> list[AdapterCfg]:
    """~``per_dataset`` adapters for each OOD dataset, all evaluated on SST2.

    ``per_dataset`` is rounded to a whole number of replicates over ``OOD_CELLS``
    (16 cells → 80 = 5 replicates). Each dataset gets a distinct master seed so HP
    jitter / seeds are independent across groups.

    ``campaign`` namespaces every adapter's ``label`` (and hence its tracked run_id
    / artifact output_names). Set it for a NEW pool so its qnames stay disjoint from
    an existing pool that reuses the same (pool, train_slug, adapter_idx) triple —
    e.g. this denser 160/dataset pool vs the parent's 80/dataset one.
    """
    datasets = datasets or OOD_DATASETS
    replicates = max(1, round(per_dataset / len(OOD_CELLS)))
    cfgs: list[AdapterCfg] = []
    for d_i, ds in enumerate(datasets):
        cfgs.extend(
            build_production_cfgs(
                OOD_CELLS,
                replicates=replicates,
                train_slug=ds,
                eval_slug=eval_slug,
                master_seed=base_seed + 1009 * d_i,
                test_every=4,
                campaign=campaign,
            )
        )
    return cfgs
