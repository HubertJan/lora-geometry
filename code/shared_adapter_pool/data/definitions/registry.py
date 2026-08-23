"""Task registry: slug -> DEFINITION for exactly the 15 pool datasets.

(migrated concept from src/llm_pipeline/dataset_registry.py, restricted to the
15 task modules the paper's SST2 in-task + OOD pools use. The other ~75 dataset
modules are deliberately NOT imported.)

The 15: sst2 (the eval target and in-task train set) plus the OOD training sets
imdb, rotten_tomatoes, yelp_polarity, amazon_polarity (sentiment),
wiki_toxic, toxigen, civil_comments, wildguard_prompt_harm (toxicity),
qnli, scitail, doc_nli, vitaminc (entailment), and boolq, subj.
"""

from __future__ import annotations

from shared_adapter_pool.data.definitions.amazon_polarity import (
    DEFINITION as amazon_polarity,
)
from shared_adapter_pool.data.definitions.boolq import DEFINITION as boolq
from shared_adapter_pool.data.definitions.civil_comments import (
    DEFINITION as civil_comments,
)
from shared_adapter_pool.data.definitions.doc_nli import DEFINITION as doc_nli
from shared_adapter_pool.data.definitions.imdb import DEFINITION as imdb
from shared_adapter_pool.data.definitions.qnli import DEFINITION as qnli
from shared_adapter_pool.data.definitions.rotten_tomatoes import (
    DEFINITION as rotten_tomatoes,
)
from shared_adapter_pool.data.definitions.scitail import DEFINITION as scitail
from shared_adapter_pool.data.definitions.sst2 import DEFINITION as sst2
from shared_adapter_pool.data.definitions.subj import DEFINITION as subj
from shared_adapter_pool.data.definitions.toxigen import DEFINITION as toxigen
from shared_adapter_pool.data.definitions.vitaminc import DEFINITION as vitaminc
from shared_adapter_pool.data.definitions.wiki_toxic import DEFINITION as wiki_toxic
from shared_adapter_pool.data.definitions.wildguard_prompt_harm import (
    DEFINITION as wildguard_prompt_harm,
)
from shared_adapter_pool.data.definitions.yelp_polarity import (
    DEFINITION as yelp_polarity,
)

TASKS = {
    "sst2": sst2,
    "imdb": imdb,
    "rotten_tomatoes": rotten_tomatoes,
    "yelp_polarity": yelp_polarity,
    "amazon_polarity": amazon_polarity,
    "wiki_toxic": wiki_toxic,
    "toxigen": toxigen,
    "civil_comments": civil_comments,
    "wildguard_prompt_harm": wildguard_prompt_harm,
    "qnli": qnli,
    "scitail": scitail,
    "doc_nli": doc_nli,
    "vitaminc": vitaminc,
    "boolq": boolq,
    "subj": subj,
}

__all__ = ["TASKS"]
