"""Strict, canonical DIRECT configuration at the v2 adapter boundary."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from auto_researcher.tasks.icca_nbs.bindings import ICCABindings


# Ten resamples is the executable floor used by lightweight compatibility tests. It
# is deliberately not presented as statistically final: normal scientific runs use
# the reference evaluator's resolved production default of 100 resamples.
ICCA_MINIMUM_RESAMPLING_ITERATIONS = 10
ICCA_DEFAULT_RESAMPLING_ITERATIONS = 100


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
    valid = ", ".join(
        str(getattr(member, "doc_name", member.name)) for member in enum_type
    )
    raise ValueError(
        f"unknown {enum_type.__name__} alias {value!r}; valid values: {valid}"
    )


class ICCADirectConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    network: str = Field(min_length=1)
    alignment: str = Field(min_length=1)
    alpha: float
    K: int
    r: int = Field(default=ICCA_DEFAULT_RESAMPLING_ITERATIONS)

    @field_validator("r")
    @classmethod
    def validate_consensus_resampling_iterations(cls, value: int) -> int:
        if value < ICCA_MINIMUM_RESAMPLING_ITERATIONS:
            raise ValueError(
                "r must be at least "
                f"{ICCA_MINIMUM_RESAMPLING_ITERATIONS} consensus resampling iterations"
            )
        return value

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
