"""Shared `VariantEnum` base for chat-template configuration enums.

(migrated from src/llm_pipeline/dataset_definitions/_variants.py)

Variant enums tag each member with a `(family, version)` tuple value so
that related variants (e.g. several `concise` rewrites of one system
prompt) can be grouped without overloading the member name.

Two directions of resolution live here alongside the base class, because
both are needed to make a recorded variant choice *actionable* rather than
merely documented:

* `VariantEnum.resolve(family, version)` — from the ``*_family`` /
  ``*_version`` pair a template's ``params()`` writes into W&B lineage back
  to the member, so a downstream eval can rebuild the exact template that
  produced an adapter (see `template_from_params`).
* `resolve_variant_choices(template_cls, {...})` — from member *names* (the
  JSON-safe, explicitly versioned form a git-tracked pool plan stores) to
  template kwargs.

`render_prompt` lives here for the same reason: a system prompt that names its
label words has to be rendered *from the chosen label scheme*, or selecting a
scheme silently produces a prompt that contradicts the completions.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from enum import Enum
from typing import Any, get_args, get_type_hints


class VariantEnum(Enum):
    """Base for `(family, version)`-tagged variant enums.

    Subclasses declare members with `("family_name", version_int)` tuple
    values. This class supplies `.family` / `.version` properties and the
    `.in_family()` / `.families()` classmethods.

    `VariantEnum` itself has no members, so subclasses are free to define
    them (Python forbids extending an enum that already has members).
    """

    @property
    def family(self) -> str:
        return self.value[0]

    @property
    def version(self) -> int:
        return self.value[1]

    @classmethod
    def in_family(cls, family: str) -> list["VariantEnum"]:
        """All members of the given family, ordered by version ascending."""
        return sorted(
            (m for m in cls if m.family == family),
            key=lambda m: m.version,
        )

    @classmethod
    def families(cls) -> list[str]:
        """Distinct family names, in declaration order."""
        seen: dict[str, None] = {}
        for m in cls:
            seen.setdefault(m.family, None)
        return list(seen)

    @classmethod
    def resolve(cls, family: str, version: int | None = None) -> "VariantEnum":
        """The member for ``(family, version)``; latest in family if unversioned.

        This is the inverse of the ``*_family`` / ``*_version`` pair that
        `SchemaTransform.params()` records, so a recorded lineage entry can be
        turned back into a live template.

        Passing ``version=None`` resolves to the highest version in the family,
        which is convenient interactively but **not** reproducible — adding a
        ``concise_v3`` would silently change what an unversioned reference
        means. Callers replaying recorded provenance always have the version
        and should pass it.
        """
        members = cls.in_family(family)
        if not members:
            raise ValueError(
                f"{cls.__name__} has no family {family!r}; "
                f"known families: {cls.families()}"
            )
        if version is None:
            return members[-1]
        for member in members:
            if member.version == version:
                return member
        raise ValueError(
            f"{cls.__name__} family {family!r} has no version {version!r}; "
            f"known versions: {[m.version for m in members]}"
        )


def label_format_kwargs(label_texts: Mapping[Enum, str]) -> dict[str, str]:
    """``str.format`` kwargs for one label scheme's ``{member: surface word}`` map.

    **The key is the enum member's name, lower-cased — never the surface word and
    never the registry's canonical spelling.** That rule is what keeps
    non-identifier labels out of the keys: ``QnliLabel.NOT_ENTAILMENT`` yields
    ``{not_entailment}`` and ``WikiToxicToxicity.NON_TOXIC`` yields
    ``{non_toxic}``, whereas the canonical spellings ``"not entailment"`` and
    ``"non-toxic"`` are not valid ``str.format`` field names at all.

    It also means the key names the *class*, not the scheme: ``BoolQLabel.TRUE``
    is ``{true}`` even under ``YES_NO_V1``, where it renders as ``yes``.  That
    reads oddly and is correct -- a prompt keyed on the surface word could not be
    shared across schemes, which is the whole point.
    """
    return {member.name.lower(): text for member, text in label_texts.items()}


def render_prompt(text: str, label_texts: Mapping[Enum, str]) -> str:
    """Substitute *label_texts*' words into a system-prompt format template.

    Raises ``KeyError`` naming the valid keys, so a prompt written against the
    surface word (``{yes}``) or the canonical spelling fails loudly at import or
    first render rather than producing a prompt that contradicts its own
    completions.
    """
    kwargs = label_format_kwargs(label_texts)
    try:
        return text.format(**kwargs)
    except KeyError as exc:
        raise KeyError(
            f"system prompt references {exc.args[0]!r}, which is not a label "
            f"member name; valid keys are {sorted(kwargs)} (enum member names "
            f"lower-cased, not surface words)"
        ) from None


def variant_enum_of(annotation: Any) -> type[VariantEnum] | None:
    """The `VariantEnum` subclass referenced by an annotation, if any.

    Looks through unions so an optional dimension (``ImdbLabelScheme | None``,
    used for "mirror the other field unless overridden" knobs) is recognised
    just like a required one.
    """
    for candidate in (annotation, *get_args(annotation)):
        if isinstance(candidate, type) and issubclass(candidate, VariantEnum):
            return candidate
    return None


def variant_fields(template_cls: type) -> dict[str, type[VariantEnum]]:
    """Map each variant-selecting dataclass field of *template_cls* to its enum.

    Detected from the field's *annotation* rather than its default, so fields
    defaulting to ``None`` are included.
    """
    try:
        hints = get_type_hints(template_cls)
    except Exception:  # noqa: BLE001 - unresolvable forward refs must not be fatal
        hints = {}
    fields: dict[str, type[VariantEnum]] = {}
    for field in dataclasses.fields(template_cls):
        enum_cls = variant_enum_of(hints.get(field.name))
        if enum_cls is None and isinstance(field.default, VariantEnum):
            enum_cls = type(field.default)
        if enum_cls is not None:
            fields[field.name] = enum_cls
    return fields


def resolve_variant_choices(
    template_cls: type, choices: Mapping[str, str | None]
) -> dict[str, VariantEnum]:
    """Turn ``{"label_scheme": "YES_NO_V1"}`` into template constructor kwargs.

    Choices are enum **member names** — the form a stored pool plan uses,
    because it is JSON-safe, explicitly versioned (``CONCISE_V2`` names its own
    version) and self-documenting in the plan file. ``None`` values are dropped,
    so a plan that omits a dimension gets the template's default.

    Raises ``ValueError`` naming the valid options for an unknown field or an
    unknown member, so a typo in a plan fails at validation time rather than
    silently training a pool with the default template.
    """
    available = variant_fields(template_cls)
    kwargs: dict[str, VariantEnum] = {}
    for name, member_name in choices.items():
        if member_name is None:
            continue
        enum_cls = available.get(name)
        if enum_cls is None:
            raise ValueError(
                f"{template_cls.__name__} has no variant dimension {name!r}; "
                f"available: {sorted(available)}"
            )
        try:
            kwargs[name] = enum_cls[member_name]
        except KeyError:
            raise ValueError(
                f"{enum_cls.__name__} has no member {member_name!r}; "
                f"valid: {[m.name for m in enum_cls]}"
            ) from None
    return kwargs


def template_from_params(template_cls: type, params: Mapping[str, Any]):
    """Rebuild a chat template from the ``params()`` dict it recorded.

    Reads each variant dimension's ``<field>_family`` / ``<field>_version``
    pair, plus ``add_eos`` when the template takes it. Dimensions absent from
    *params* keep their default — which is what makes this tolerant of
    provenance written before a dimension existed.

    Only usable for templates whose constructor needs no external objects; a
    tokenizer-driven template must be rebuilt by its caller.
    """
    kwargs: dict[str, Any] = {}
    for name, enum_cls in variant_fields(template_cls).items():
        family = params.get(f"{name}_family")
        if family is None:
            continue
        kwargs[name] = enum_cls.resolve(family, params.get(f"{name}_version"))
    field_names = {f.name for f in dataclasses.fields(template_cls)}
    if "add_eos" in field_names and "add_eos" in params:
        kwargs["add_eos"] = bool(params["add_eos"])
    return template_cls(**kwargs)
