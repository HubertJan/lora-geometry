"""Pure-Python sanity checks for shared_adapter_pool (no torch needed)."""

from __future__ import annotations


def test_registry_has_15_tasks():
    from shared_adapter_pool.data.definitions.registry import TASKS

    expected = {
        "sst2", "imdb", "rotten_tomatoes", "yelp_polarity", "amazon_polarity",
        "wiki_toxic", "toxigen", "civil_comments", "wildguard_prompt_harm",
        "qnli", "scitail", "doc_nli", "vitaminc", "boolq", "subj",
    }
    assert expected <= set(TASKS), sorted(expected - set(TASKS))


def test_hp_grids_build():
    from shared_adapter_pool.pool.production_grid import (
        DEFAULT_CELLS,
        build_production_cfgs,
    )
    from shared_adapter_pool.pool.ood_grid import build_ood_cfgs

    sst2 = build_production_cfgs(DEFAULT_CELLS, replicates=27)
    assert len(sst2) == len(DEFAULT_CELLS) * 27
    assert all(c.eval_slug == "sst2" for c in sst2)

    ood = build_ood_cfgs(
        datasets=["imdb", "qnli", "wiki_toxic"], per_dataset=160
    )
    assert len(ood) > 100 and all(c.eval_slug == "sst2" for c in ood)


def test_store_roundtrip(tmp_path):
    from shared_adapter_pool.pool.store import read_pool, write_pool

    rows = [
        {"__key__": "a0", "train_dataset": "sst2", "split": "train",
         "benchmark.sst2-test.likelihood.accuracy": 0.9},
        {"__key__": "a1", "train_dataset": "sst2", "split": "test",
         "benchmark.sst2-test.likelihood.accuracy": 0.6},
    ]
    write_pool(tmp_path, rows)
    df = read_pool(tmp_path)
    assert set(df["__key__"]) == {"a0", "a1"}


def test_six_metrics():
    import pandas as pd

    from shared_adapter_pool.eval.metrics import compute_metrics

    # 4 rows, binary true/false. prob_true/prob_false are the per-class probs;
    # prediction = argmax class; correct = (prediction == reference).
    df = pd.DataFrame(
        {
            "is_poisoned": [False, False, False, False],
            "reference": ["true", "false", "true", "false"],
            "prediction": ["true", "false", "true", "true"],
            "prob_true": [0.8, 0.3, 0.9, 0.6],
            "prob_false": [0.2, 0.7, 0.1, 0.4],
            "correct": [True, True, True, False],
        }
    )
    m = compute_metrics(df, categories=["true", "false"])
    for k in ("accuracy", "f1_macro", "brier", "mean_confidence", "nll", "auroc"):
        assert k in m
