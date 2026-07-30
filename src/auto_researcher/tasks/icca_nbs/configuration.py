"""Strict, canonical DIRECT configuration at the v2 adapter boundary."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from auto_researcher.tasks.icca_nbs.bindings import ICCABindings


def resolve_enum_alias(enum_type: type, value: str) -> Any:
    requested = value.casefold()
    for member in enum_type:
        aliases = {
            str(getattr(member, "name", "")),
            str(getattr(member, "doc_name", "")),
            str(getattr(member, "code_name", "")),
            str(getattr(member, "slug", "")),
        }
        raw_value = getattr(member, "value", None)
        if isinstance(raw_value, str):
            aliases.add(raw_value)
        if requested in {alias.casefold() for alias in aliases if alias}:
            return member
    valid = ", ".join(str(getattr(member, "doc_name", member.name)) for member in enum_type)
    raise ValueError(f"unknown {enum_type.__name__} alias {value!r}; valid values: {valid}")


class ICCADirectConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    network: str = Field(min_length=1)
    alignment: str = Field(min_length=1)
    alpha: float
    K: int
    r: int = Field(default=100, gt=0)

    @classmethod
    def normalise(
        cls,
        configuration: dict,
        bindings: ICCABindings,
    ) -> "ICCADirectConfiguration":
        parsed = cls.model_validate(configuration)
        alpha_low, alpha_high = bindings.alpha_bounds
        if not alpha_low <= parsed.alpha <= alpha_high:
            raise ValueError(
                f"alpha {parsed.alpha} is outside v2 bounds [{alpha_low}, {alpha_high}]"
            )
        k_low, k_high = bindings.k_bounds
        if not k_low <= parsed.K <= k_high:
            raise ValueError(f"K {parsed.K} is outside v2 bounds [{k_low}, {k_high}]")
        network = resolve_enum_alias(bindings.network_type, parsed.network)
        alignment = resolve_enum_alias(bindings.alignment_type, parsed.alignment)
        return parsed.model_copy(
            update={
                "network": str(getattr(network, "doc_name", network.name)),
                "alignment": str(getattr(alignment, "doc_name", alignment.name)),
            }
        )
