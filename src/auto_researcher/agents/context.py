"""Deterministic assembly of compact, task-safe model contexts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from auto_researcher.agents.models import (
    ContractAgentSummary,
    HypothesisAgentContext,
    PlannerAgentContext,
    PriorResearchSummary,
    TaskAgentContext,
)
from auto_researcher.contracts.enums import (
    EvidenceStatus,
    EventType,
    GroundingStatus,
    SearchType,
)
from auto_researcher.provenance.protocols import ProvenanceStore
from auto_researcher.search.protocols import SearchCapability

if TYPE_CHECKING:
    from auto_researcher.graph.state import ResearchState


@dataclass(frozen=True)
class AgentContextLimits:
    maximum_prior_hypotheses: int = 5
    maximum_prior_results: int = 5
    maximum_context_characters: int = 24_000
    maximum_artefact_references: int = 8


def stable_context_hash(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


class AgentContextAssembler:
    def __init__(
        self,
        provenance_store: ProvenanceStore,
        *,
        limits: AgentContextLimits | None = None,
    ) -> None:
        self._provenance_store = provenance_store
        self.limits = limits or AgentContextLimits()

    def _contract_summary(self, state: ResearchState) -> ContractAgentSummary:
        contract = state["contract"]
        return ContractAgentSummary(
            contract_id=contract.contract_id,
            task_id=contract.task_id,
            task_version=contract.task_version,
            objective_version=contract.objective_version,
            question=contract.question,
            objective=contract.objective,
            primary_metric=contract.primary_metric,
            constraints=dict(contract.constraints),
            allowed_search_types=tuple(
                sorted(contract.allowed_search_types, key=lambda item: item.value)
            ),
            maximum_experiments=contract.maximum_experiments,
            requires_approval_for=tuple(
                sorted(contract.requires_approval_for, key=lambda item: item.value)
            ),
        )

    def _prior(self, run_id: str, current_cycle: int) -> tuple[
        tuple[str, ...], tuple[PriorResearchSummary, ...]
    ]:
        events = tuple(
            event
            for event in self._provenance_store.list_events(run_id)
            if event.cycle < current_cycle
        )
        hypotheses = sorted(
            {
                event.output_references[0]
                for event in events
                if event.event_type == EventType.HYPOTHESIS_PROPOSED
                and event.output_references
            }
        )[-self.limits.maximum_prior_hypotheses :]
        results: list[PriorResearchSummary] = []
        for event in events:
            if event.event_type != EventType.EVIDENCE_VERIFIED:
                continue
            values = {
                key: value
                for reference in event.output_references
                if ":" in reference
                for key, value in [reference.split(":", 1)]
            }
            if values.get("verified") != "true":
                continue
            try:
                status = EvidenceStatus(values["evidence"])
                search_type = SearchType(values["search_type"])
            except (KeyError, ValueError):
                continue
            artefacts = tuple(
                reference
                for reference in event.output_references
                if reference.startswith("artefact:")
            )[: self.limits.maximum_artefact_references]
            score = values.get("score")
            results.append(
                PriorResearchSummary(
                    hypothesis_reference=values.get("hypothesis", "unknown"),
                    experiment_reference=(
                        event.input_references[0]
                        if event.input_references
                        else "unknown"
                    ),
                    search_type=search_type,
                    primary_score=None if score in {None, "None"} else float(score),
                    evidence_status=status,
                    constraint_compliant=values.get("constraints") == "true",
                    concise_verified_finding=event.rationale[:400],
                    safe_artefact_references=artefacts,
                )
            )
        results.sort(
            key=lambda item: (
                item.hypothesis_reference,
                item.experiment_reference,
            )
        )
        return tuple(hypotheses), tuple(results[-self.limits.maximum_prior_results :])

    def _ensure_size(self, model) -> None:
        if len(model.model_dump_json()) > self.limits.maximum_context_characters:
            raise ValueError("agent_context_too_large")

    def hypothesis_context(
        self,
        state: ResearchState,
        task_context: TaskAgentContext,
    ) -> HypothesisAgentContext:
        previous, prior = self._prior(state["run_id"], state["cycle"])
        references = tuple(
            sorted(
                {state["contract"].contract_id}
                | {
                    item.hypothesis_reference
                    for item in prior
                }
                | {item.experiment_reference for item in prior}
                | {
                    reference
                    for item in prior
                    for reference in item.safe_artefact_references
                }
            )
        )
        availability = [GroundingStatus.CONTRACT_GROUNDED]
        if prior:
            availability.append(GroundingStatus.PRIOR_RESULTS_GROUNDED)
        payload = {
            "run_id": state["run_id"],
            "contract": self._contract_summary(state),
            "task": task_context,
            "cycle": state["cycle"],
            "remaining_experiment_budget": max(
                0,
                state["budget"].maximum_experiments
                - state["budget"].experiments_used,
            ),
            "remaining_cost_budget": max(
                0.0,
                state["budget"].maximum_cost - state["budget"].cost_used,
            ),
            "model_calls_used": state["budget"].model_calls_used,
            "previous_hypotheses": previous,
            "prior_verified_findings": prior,
            "permitted_evidence_reference_ids": references,
            "grounding_availability": tuple(availability),
        }
        serialisable = {
            key: (
                value.model_dump(mode="json")
                if hasattr(value, "model_dump")
                else [
                    item.model_dump(mode="json")
                    if hasattr(item, "model_dump")
                    else getattr(item, "value", item)
                    for item in value
                ]
                if isinstance(value, tuple)
                else value
            )
            for key, value in payload.items()
        }
        context = HypothesisAgentContext(
            **payload,
            context_hash=stable_context_hash(serialisable),
        )
        self._ensure_size(context)
        return context

    def planner_context(
        self,
        state: ResearchState,
        task_context: TaskAgentContext,
        capabilities: dict[SearchType, SearchCapability],
    ) -> PlannerAgentContext:
        hypothesis = state["active_hypothesis"]
        assert hypothesis is not None
        _, prior = self._prior(state["run_id"], state["cycle"])
        installed = tuple(
            sorted(
                (
                    search_type
                    for search_type, capability in capabilities.items()
                    if capability.available
                    and search_type in state["contract"].allowed_search_types
                    and search_type in task_context.available_search_types
                    and search_type != SearchType.OPENEVOLVE
                ),
                key=lambda item: item.value,
            )
        )
        payload = {
            "run_id": state["run_id"],
            "contract": self._contract_summary(state),
            "task": task_context,
            "hypothesis": hypothesis,
            "cycle": state["cycle"],
            "installed_search_capabilities": installed,
            "remaining_experiment_budget": max(
                0,
                state["budget"].maximum_experiments
                - state["budget"].experiments_used,
            ),
            "remaining_cost_budget": max(
                0.0,
                state["budget"].maximum_cost - state["budget"].cost_used,
            ),
            "model_calls_used": state["budget"].model_calls_used,
            "approval_requirements": tuple(
                sorted(
                    state["contract"].requires_approval_for,
                    key=lambda item: item.value,
                )
            ),
            "prior_verified_findings": prior,
            "permitted_direct_configuration_schema": (
                task_context.direct_configuration_schema
            ),
            "permitted_optuna_maximum_space": task_context.optuna_space_summary,
            "optuna_narrowing_rules": (
                "Parameters may be fixed or narrowed within the registered space.",
                "Unknown parameters, widening, and fixed-context changes are forbidden.",
            ),
        }
        serialisable = {
            key: (
                value.model_dump(mode="json")
                if hasattr(value, "model_dump")
                else [
                    item.model_dump(mode="json")
                    if hasattr(item, "model_dump")
                    else getattr(item, "value", item)
                    for item in value
                ]
                if isinstance(value, tuple)
                else value
            )
            for key, value in payload.items()
        }
        context = PlannerAgentContext(
            **payload,
            context_hash=stable_context_hash(serialisable),
        )
        self._ensure_size(context)
        return context
