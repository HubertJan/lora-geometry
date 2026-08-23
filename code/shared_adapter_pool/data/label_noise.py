"""Seeded, schema-preserving random label noise -- a degrader lever.

(migrated from src/discoveries/sst2_perf_prediction/flows/label_noise.py)

SST2 (and binary sentiment generally) has a sharp train-size cliff: below ~500
samples the adapter collapses to one-class chance, above it jumps to ~0.9. To
make the performance target REGRESSABLE we need adapters densely spanning the
mid-band (~0.6-0.9), which sample count alone cannot reach. Flipping a fraction
of TRAIN labels degrades clean-test accuracy smoothly (roughly acc ~ 1 - noise
for an easy task the model otherwise fits), filling the gap.

Deterministic (per-row ``SeedSequence([seed, idx])``) so the noise pattern is
reproducible independent of batching. It operates on the CANONICAL label strings
that registry prep produces (e.g. sst2 "negative"/"positive"), swapping between
the two provided ``label_values``; rows whose label is neither are left
untouched. The original ``DatasetTransform`` base coupling is dropped -- this is
now a plain dataclass with a ``__call__``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from shared_adapter_pool.data.typed_dataset import TypedDataset


@dataclass
class LabelNoiseDataset:
    """Flip ``label_field`` for a seeded ``noise_rate`` fraction of rows.

    Binary only: ``label_values`` must be the two canonical class strings; a
    flipped row's label becomes the other one.
    """

    transformation: ClassVar[str] = "label_noise"

    noise_rate: float = 0.0
    label_field: str = "label"
    label_values: tuple[str, ...] = ()
    seed: int = 42

    def params(self) -> dict[str, Any]:
        return {
            "noise_rate": self.noise_rate,
            "label_field": self.label_field,
            "label_values": list(self.label_values),
            "seed": self.seed,
        }

    def _flip_map(self) -> dict[str, str]:
        if len(self.label_values) != 2:
            raise ValueError(
                f"LabelNoiseDataset is binary-only; got label_values="
                f"{self.label_values!r}"
            )
        a, b = self.label_values
        return {a: b, b: a}

    def __call__(self, dataset):
        import numpy as np

        if self.noise_rate <= 0.0:
            return dataset

        flip = self._flip_map()
        field_name = self.label_field
        rate = float(self.noise_rate)
        seed = int(self.seed)

        def _noise(row: dict, idx: int) -> dict:
            cur = row.get(field_name)
            # Only flip canonical string labels we recognise; leave anything
            # else (None, an unexpected type) untouched.
            if cur in flip:
                rng = np.random.default_rng(np.random.SeedSequence([seed, idx]))
                if rng.random() < rate:
                    return {field_name: flip[cur]}
            return {}

        if isinstance(dataset, TypedDataset):
            new_data = dataset.data.map(_noise, with_indices=True)
            return TypedDataset(data=new_data, schema=dataset.schema)
        return dataset.map(_noise, with_indices=True)
