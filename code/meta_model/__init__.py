"""meta_model: the paper's LoRA meta-model, datasets, trainer and baselines.

Migrated from the glad research repo. Submodules are imported lazily (import the
specific module you need, e.g. ``from meta_model.model import
FlexibleLoRAMetaClassifier``) to avoid pulling torch in at package import time.
"""
