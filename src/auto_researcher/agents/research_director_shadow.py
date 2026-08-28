"""Offline evaluation of Research Director decisions against a locked envelope."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from auto_researcher.agents.models import ResearchDirective
from auto_researcher.contracts.enums import SearchType
from auto_researcher.runtime.identity import payload_hash


class ShadowModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ResearchDirectorShadowPolicy(ShadowModel):
    policy_id: str = Field(min_length=1)
    allowed_operators: frozenset[SearchType] = Field(min_length=1)
    allowed_dimensions: frozenset[str] = Field(min_length=1)
    maximum_allocation_by_operator: dict[SearchType, int]
    maximum_total_allocation: int = Field(ge=1)


class ResearchDirectorShadowReport(ShadowModel):
    policy_id: str
    directive_id: str
    passed: bool
    violations: tuple[str, ...]
    total_allocation: int
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def evaluate_shadow_directive(
    directive: ResearchDirective,
    policy: ResearchDirectorShadowPolicy,
) -> ResearchDirectorShadowReport:
    """Evaluate one recorded decision without dispatching models or experiments."""

    violations: list[str] = []
    selected = set(directive.selected_operators)
    if not selected.issubset(policy.allowed_operators):
        violations.append("operator_outside_locked_envelope")
    if not set(directive.targeted_dimensions).issubset(policy.allowed_dimensions):
        violations.append("dimension_outside_locked_envelope")
    allocation: dict[SearchType, int] = {}
    for raw_operator, raw_count in directive.experiment_allocation.items():
        try:
            operator = SearchType(raw_operator)
        except ValueError:
            violations.append("allocation_operator_invalid")
            continue
        if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count < 0:
            violations.append("allocation_value_invalid")
            continue
        allocation[operator] = raw_count
        if raw_count > policy.maximum_allocation_by_operator.get(operator, 0):
            violations.append("operator_allocation_exceeds_locked_envelope")
    if selected != {operator for operator, count in allocation.items() if count > 0}:
        violations.append("selected_operator_allocation_mismatch")
    total = sum(allocation.values())
    if total > policy.maximum_total_allocation:
        violations.append("total_allocation_exceeds_locked_envelope")
    violations = sorted(set(violations))
    base = {
        "policy_id": policy.policy_id,
        "directive_id": directive.directive_id,
        "passed": not violations,
        "violations": violations,
        "total_allocation": total,
    }
    return ResearchDirectorShadowReport(
        **base,
        report_sha256=payload_hash(base),
    )


__all__ = [
    "ResearchDirectorShadowPolicy",
    "ResearchDirectorShadowReport",
    "evaluate_shadow_directive",
]
