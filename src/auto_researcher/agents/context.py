"""Deterministic assembly of compact, task-safe model contexts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from auto_researcher.agents.models import (
    ContractAgentSummary,
    HypothesisAgentContext,
    PlannerAgentContext,
    PriorResearchSummary,
    ResearchDirectorContext,
    ResearchLandscapeEvidence,
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
from auto_researcher.knowledge.models import (
    KnowledgeContextReference,
    KnowledgeRetrievalStatus,
)
from auto_researcher.knowledge.store import KnowledgeRetrievalStore

if TYPE_CHECKING:
    from auto_researcher.graph.state import ResearchState


@dataclass(frozen=True)
class AgentContextLimits:
    maximum_prior_hypotheses: int = 5
    maximum_prior_results: int = 5
    maximum_context_characters: int = 24_000
    maximum_artefact_references: int = 8
    maximum_knowledge_references: int = 20


class AgentContextAssemblyError(ValueError):
    """Safe, closed failure raised before any model request is dispatched."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


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
        knowledge_retrieval_store: KnowledgeRetrievalStore | None = None,
        clock: Callable[[], datetime] | None = None,
        research_landscape: tuple[ResearchLandscapeEvidence, ...] = (),
    ) -> None:
        self._provenance_store = provenance_store
        self._knowledge_store = knowledge_retrieval_store
        self._clock = clock or (lambda: datetime.now(UTC))
        self._research_landscape = research_landscape
        self.limits = limits or AgentContextLimits()

    @staticmethod
    def _compact_aggregate_metrics(metrics: dict) -> dict:
        """Keep useful learning-curve evidence without copying full epoch history."""

        compact = {
            key: metrics[key]
            for key in (
                "primary_score",
                "per_tissue_dice",
                "reconstruction_gap",
                "best_epoch",
                "training_duration_seconds",
            )
            if key in metrics
        }
        raw_history = metrics.get("validation_history")
        if not isinstance(raw_history, (list, tuple)):
            return compact
        entries = [item for item in raw_history if isinstance(item, dict)]
        selected: list[dict] = []
        interesting = [item for item in entries if item.get("milestone") is True]
        if entries:
            interesting.extend((entries[0], entries[-1]))
        raw_best_epoch = metrics.get("best_epoch")
        interesting.extend(
            item for item in entries if item.get("epoch") == raw_best_epoch
        )
        seen: set[tuple] = set()
        for item in sorted(interesting, key=lambda value: value.get("epoch", -1)):
            summary = {
                key: item[key]
                for key in (
                    "epoch",
                    "validation_score",
                    "best_epoch",
                    "best_validation_score",
                    "milestone",
                )
                if key in item
            }
            identity = tuple(sorted(summary.items()))
            if identity not in seen:
                seen.add(identity)
                selected.append(summary)
        compact["validation_history_summary"] = {
            "observation_count": len(entries),
            "selected_entries": selected[-6:],
        }
        return compact

    def _bounded_prior(
        self, prior: tuple[PriorResearchSummary, ...]
    ) -> tuple[PriorResearchSummary, ...]:
        """Deterministically retain newest findings within half the context budget."""

        maximum = max(1_000, self.limits.maximum_context_characters // 2)
        selected: list[PriorResearchSummary] = []
        used = 2
        for item in reversed(prior):
            size = len(item.model_dump_json()) + 1
            if selected and used + size > maximum:
                break
            selected.append(item)
            used += size
        return tuple(reversed(selected))

    def _knowledge(
        self,
        state: ResearchState,
    ) -> tuple[
        tuple[KnowledgeContextReference, ...],
        str | None,
        str | None,
    ]:
        compact = state.get("knowledge_bundle_reference")
        if isinstance(compact, dict):
            from auto_researcher.knowledge.models import KnowledgeBundleReference

            compact = KnowledgeBundleReference.model_validate(compact)
        if (
            compact is None
            or compact.status != KnowledgeRetrievalStatus.COMPLETED
            or compact.retrieval_id is None
            or self._knowledge_store is None
        ):
            return (), None, None
        records = self._knowledge_store.records_for_retrieval(compact.retrieval_id)
        completed = [item for item in records if item.bundle is not None]
        if not completed:
            return (), None, None
        bundle = completed[-1].bundle
        assert bundle is not None
        if (
            not bundle.validation_result.passed
            or bundle.bundle_hash != compact.bundle_hash
        ):
            return (), None, None
        permitted = set(compact.reference_ids)
        references = tuple(
            KnowledgeContextReference(
                reference_id=item.reference_id,
                concise_claim=item.concise_claim,
                citation_label=item.citation_label,
                trust_tier=item.trust_tier,
                confidence=item.confidence,
                entity_curies=(item.subject_curie, item.object_curie),
                source_ids=item.source_references,
                bundle_id=item.bundle_id,
                relevant_parameters=item.relevant_parameters,
                prior_weight_cap=item.prior_weight_cap,
            )
            for item in sorted(bundle.references, key=lambda value: value.reference_id)
            if item.reference_id in permitted
        )[: self.limits.maximum_knowledge_references]
        return references, bundle.bundle_id, bundle.bundle_hash

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

    def _prior(
        self, run_id: str, current_cycle: int
    ) -> tuple[tuple[str, ...], tuple[PriorResearchSummary, ...]]:
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
                    concise_verified_finding=event.rationale[:240],
                    safe_artefact_references=artefacts,
                    safe_configuration=dict(event.safe_payload.get("configuration", {}))
                    if isinstance(event.safe_payload.get("configuration"), dict)
                    else {},
                    aggregate_metrics=self._compact_aggregate_metrics(
                        dict(event.safe_payload.get("aggregate_metrics", {}))
                    )
                    if isinstance(event.safe_payload.get("aggregate_metrics"), dict)
                    else {},
                )
            )
        results.sort(
            key=lambda item: (
                item.hypothesis_reference,
                item.experiment_reference,
            )
        )
        prior = tuple(results[-self.limits.maximum_prior_results :])
        return tuple(hypotheses), self._bounded_prior(prior)

    def _recent_failure_codes(self, run_id: str) -> tuple[str, ...]:
        codes: list[str] = []
        for event in self._provenance_store.list_events(run_id):
            if event.event_type != EventType.RUN_STOPPED:
                continue
            for reference in event.output_references:
                if reference.startswith("error_code:"):
                    codes.append(reference.split(":", 1)[1])
        return tuple(codes[-8:])

    def _ensure_size(self, model) -> None:
        if len(model.model_dump_json()) > self.limits.maximum_context_characters:
            raise AgentContextAssemblyError("agent_context_too_large")

    def hypothesis_context(
        self,
        state: ResearchState,
        task_context: TaskAgentContext,
    ) -> HypothesisAgentContext:
        previous, prior = self._prior(state["run_id"], state["cycle"])
        knowledge, bundle_id, bundle_hash = self._knowledge(state)
        references = tuple(
            sorted(
                {state["contract"].contract_id}
                | {item.hypothesis_reference for item in prior}
                | {item.experiment_reference for item in prior}
                | {
                    reference
                    for item in prior
                    for reference in item.safe_artefact_references
                }
                | {item.reference_id for item in knowledge}
            )
        )
        availability = [GroundingStatus.CONTRACT_GROUNDED]
        if prior:
            availability.append(GroundingStatus.PRIOR_RESULTS_GROUNDED)
        if knowledge:
            availability.append(GroundingStatus.KNOWLEDGE_GROUNDED)
        payload = {
            "run_id": state["run_id"],
            "contract": self._contract_summary(state),
            "task": task_context,
            "cycle": state["cycle"],
            "remaining_experiment_budget": max(
                0,
                state["budget"].maximum_experiments - state["budget"].experiments_used,
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
            "knowledge_references": knowledge,
            "knowledge_bundle_id": bundle_id,
            "knowledge_bundle_hash": bundle_hash,
            "research_directive": state.get("active_research_directive"),
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
        knowledge, bundle_id, bundle_hash = self._knowledge(state)
        prior_references = (
            {item.hypothesis_reference for item in prior}
            | {item.experiment_reference for item in prior}
            | {
                reference
                for item in prior
                for reference in item.safe_artefact_references
            }
        )
        permitted_references = tuple(
            sorted(
                {state["contract"].contract_id}
                | prior_references
                | {item.reference_id for item in knowledge}
            )
        )
        installed = tuple(
            sorted(
                (
                    search_type
                    for search_type, capability in capabilities.items()
                    if capability.available
                    and search_type in state["contract"].allowed_search_types
                    and search_type in task_context.available_search_types
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
                state["budget"].maximum_experiments - state["budget"].experiments_used,
            ),
            "remaining_cost_budget": max(
                0.0,
                state["budget"].maximum_cost - state["budget"].cost_used,
            ),
            "remaining_time_seconds": state["budget"].remaining_seconds(self._clock()),
            "campaign_deadline_at": (
                state["budget"].deadline_at.isoformat()
                if state["budget"].deadline_at is not None
                else None
            ),
            "model_calls_used": state["budget"].model_calls_used,
            "approval_requirements": tuple(
                sorted(
                    state["contract"].requires_approval_for,
                    key=lambda item: item.value,
                )
            ),
            "prior_verified_findings": prior,
            "permitted_evidence_reference_ids": permitted_references,
            "knowledge_references": knowledge,
            "knowledge_bundle_id": bundle_id,
            "knowledge_bundle_hash": bundle_hash,
            "permitted_direct_configuration_schema": (
                task_context.direct_configuration_schema
            ),
            "permitted_optuna_maximum_space": task_context.optuna_space_summary,
            "optuna_narrowing_rules": (
                "Parameters may be fixed or narrowed within the registered space.",
                "Unknown parameters, widening, and fixed-context changes are forbidden.",
            ),
            "research_directive": state.get("active_research_directive"),
            "recovered_error_codes": tuple(state.get("recovered_error_codes", ())),
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

    def research_director_context(
        self,
        state: ResearchState,
        task_context: TaskAgentContext,
        capabilities: dict[SearchType, SearchCapability],
        *,
        trigger: str,
        finalisation_reserve_seconds: float,
    ) -> ResearchDirectorContext:
        _, prior = self._prior(state["run_id"], state["cycle"])
        references = tuple(
            sorted(
                {state["contract"].contract_id}
                | {item.hypothesis_reference for item in prior}
                | {item.experiment_reference for item in prior}
                | {
                    reference
                    for item in prior
                    for reference in item.safe_artefact_references
                }
                | {
                    item.source_reference for item in self._research_landscape
                }
                | {
                    reference
                    for item in self._research_landscape
                    for reference in item.reference_ids
                }
            )
        )
        installed = tuple(
            sorted(
                (
                    search_type
                    for search_type, capability in capabilities.items()
                    if capability.available
                    and search_type in state["contract"].allowed_search_types
                    and search_type in task_context.available_search_types
                ),
                key=lambda item: item.value,
            )
        )
        dimensions = tuple(
            sorted(
                set(task_context.direct_configuration_schema)
                | set(task_context.optuna_space_summary)
                | set(task_context.openevolve_space_summary)
            )
        )
        payload = {
            "run_id": state["run_id"],
            "contract": self._contract_summary(state),
            "task": task_context,
            "cycle": state["cycle"],
            "trigger": trigger,
            "installed_search_capabilities": installed,
            "remaining_experiment_budget": max(
                0,
                state["budget"].maximum_experiments - state["budget"].experiments_used,
            ),
            "remaining_cost_budget": max(
                0.0,
                state["budget"].maximum_cost - state["budget"].cost_used,
            ),
            "remaining_time_seconds": state["budget"].remaining_seconds(self._clock()),
            "model_calls_used": state["budget"].model_calls_used,
            "prior_verified_findings": prior,
            "research_landscape": self._research_landscape,
            "recent_failure_codes": self._recent_failure_codes(state["run_id"]),
            "permitted_evidence_reference_ids": references,
            "permitted_target_dimensions": dimensions,
            "finalisation_reserve_seconds": finalisation_reserve_seconds,
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
        context = ResearchDirectorContext(
            **payload,
            context_hash=stable_context_hash(serialisable),
        )
        self._ensure_size(context)
        return context
