"""Create an ablated copy of a LoRA adapter by zeroing selected LoRA tensors.

(migrated from SRC/src/discoveries/kv_ablation_high_acc/flows/ablate.py — VERBATIM)


Zeroing a module's ``lora_A``/``lora_B`` pair makes its delta ``B @ A = 0`` — the
projection reverts to the base model, removing the adapter's contribution *there*
while leaving every other cell untouched. The adapter config (which still lists
the module as a target) and tokenizer are copied verbatim; PEFT loads the zeroed
weights as a no-op on those cells.

:func:`ablate_adapter` is the general surgeon: pass the module substrings to zero
and, optionally, the set of layer indices to restrict to (``None`` = all layers).
:func:`ablate_kv` is the original K/V-all-layers special case kept for callers.

Layer indices are the model's own 0-based ``layers.{i}`` numbering (Llama-3.2-1B:
0..15). Callers that think in 1-based "layer 1..16" convert on the way in.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Iterable
from pathlib import Path

# All LoRA-targeted modules in this pool (7 per layer), for "whole layer" ablations.
ALL_MODULES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def _is_lora_key(key: str) -> bool:
    return ".lora_A." in key or ".lora_B." in key


def _layer_of(key: str) -> int | None:
    m = _LAYER_RE.search(key)
    return int(m.group(1)) if m else None


def ablate_adapter(
    src_dir: str | Path,
    dst_dir: str | Path,
    modules: Iterable[str] = (),
    layers: Iterable[int] | None = None,
    keep: bool = False,
    negate: bool = False,
    rules: "list[tuple[Iterable[str], Iterable[int] | None]] | None" = None,
) -> dict:
    """Copy ``src_dir`` -> ``dst_dir``, mutating the selected LoRA tensors.

    Parameters
    ----------
    modules:
        Module substrings to match against each LoRA key (e.g.
        ``("self_attn.q_proj",)`` or :data:`ALL_MODULES`).
    layers:
        0-based model layer indices to restrict the (module, layer) selector to.
        ``None`` = every layer.
    keep:
        ``False`` (default) targets the LoRA tensors that MATCH the selector — an
        *ablation* (remove this group). ``True`` inverts the selection: target
        every LoRA tensor that does NOT match, i.e. **keep only** the selected
        group. Non-LoRA tensors are never touched.
    negate:
        ``False`` (default) **zeros** the targeted tensors. ``True`` instead
        **sign-flips** the targeted modules' delta (``B @ A → -(B @ A)``) by
        negating each targeted module's ``lora_B`` once — leaving every other
        component at its trained value. Use with ``keep=False`` to invert one
        group inside the otherwise-intact adapter.
    rules:
        Optional list of ``(modules, layers)`` selectors, each with its OWN layer
        set — a tensor matches if it satisfies ANY rule. Use for compound
        selections where different modules span different layers (e.g. Q on
        layers 8–16 AND K/V/O on layers 11–16). Overrides ``modules``/``layers``.

    Returns a small manifest: which tensors changed and their pre-op norm.
    """
    import torch
    from safetensors.torch import load_file, save_file

    if rules is None:
        norm_rules = [(tuple(modules), None if layers is None else set(layers))]
    else:
        norm_rules = [
            (tuple(mods), None if lyrs is None else set(lyrs)) for mods, lyrs in rules
        ]

    src, dst = Path(src_dir), Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    # copy everything except the weights (config, tokenizer, README, training_args)
    for f in src.iterdir():
        if f.name != "adapter_model.safetensors" and f.is_file():
            shutil.copy2(f, dst / f.name)

    sd = load_file(str(src / "adapter_model.safetensors"))
    changed, pre_norm, changed_layers = [], 0.0, set()
    for k in list(sd.keys()):
        if not _is_lora_key(k):
            continue
        li = _layer_of(k)
        matches = any(
            any(m in k for m in mods)
            and (ls is None or (li is not None and li in ls))
            for mods, ls in norm_rules
        )
        # remove-mode targets the matches; keep-mode targets the complement.
        if matches == keep:
            continue
        if negate:
            # flip the delta sign exactly once per module: negate lora_B only
            # (negating both A and B would cancel back to +1).
            if ".lora_B." not in k:
                continue
            pre_norm += float(sd[k].float().norm().item())
            sd[k] = -sd[k]
        else:
            pre_norm += float(sd[k].float().norm().item())
            sd[k] = torch.zeros_like(sd[k])
        changed.append(k)
        if li is not None:
            changed_layers.add(li)
    save_file(sd, str(dst / "adapter_model.safetensors"), metadata={"format": "pt"})
    return {
        "n_tensors_total": len(sd),
        "n_zeroed": len(changed),
        "prezero_norm_sum": pre_norm,
        "layers_touched": sorted(changed_layers),
        "rules": [(list(mods), None if ls is None else sorted(ls)) for mods, ls in norm_rules],
        "keep": keep,
        "op": "negate" if negate else "zero",
        "dst": str(dst),
    }


def ablate_kv(src_dir: str | Path, dst_dir: str | Path) -> dict:
    """K/V-in-all-layers ablation (the original special case)."""
    man = ablate_adapter(
        src_dir, dst_dir, ("self_attn.k_proj", "self_attn.v_proj"), layers=None
    )
    # keep the historical key name for older callers/log lines
    man["kv_prezero_norm_sum"] = man["prezero_norm_sum"]
    return man


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src_dir")
    ap.add_argument("dst_dir")
    ap.add_argument("--modules", nargs="+", default=["self_attn.k_proj", "self_attn.v_proj"],
                    help="module substrings to zero")
    ap.add_argument("--layers", type=int, nargs="*", default=None,
                    help="0-based layer indices to restrict to (default: all)")
    args = ap.parse_args()
    print(json.dumps(ablate_adapter(args.src_dir, args.dst_dir, args.modules, args.layers),
                     indent=2))
