"""Deterministic validation of untrusted model proposals."""

from __future__ import annotations

import hashlib

from auto_researcher.agents.models import (
    HypothesisAgentContext,
    HypothesisProposal,
    PlannerAgentContext,
    PlannerProposal,
    PriorResearchSummary,
)
from auto_researcher.contracts.enums import (
    GroundingStatus,
    HypothesisStatus,
    ProposalSource,
    ProvenanceKind,
    SearchType,
)
from auto_researcher.contracts.models import Hypothesis, ResearchContract, SearchRequest
from auto_researcher.knowledge.models import KnowledgeContextReference
from auto_researcher.tasks.protocols import (
    OpenEvolveCapableTask,
    OptunaCapableTask,
    ResearchTask,
)


class ReconciliationError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:20]
    return f"{prefix}-{digest}"


def _prior_reference_ids(
    findings: tuple[PriorResearchSummary, ...],
) -> set[str]:
    return (
        {item.hypothesis_reference for item in findings}
        | {item.experiment_reference for item in findings}
        | {
            reference
            for item in findings
            for reference in item.safe_artefact_references
        }
    )


def _require_relevant_knowledge(
    cited_knowledge: list[KnowledgeContextReference],
    proposal_parameters: set[str],
) -> None:
    if any(
        not item.relevant_parameters
        or proposal_parameters.isdisjoint(item.relevant_parameters)
        for item in cited_knowledge
    ):
        raise ReconciliationError("knowledge_reference_not_relevant")


def _normalise_predicted_subspace(
    raw: dict,
    permitted_parameters: set[str],
) -> dict:
    """Keep registered task parameters, including those under model-added wrappers."""

    compatible: dict = {}

    def collect(value: dict) -> None:
        for key in sorted(value):
            item = value[key]
            if key in permitted_parameters:
                if key in compatible and compatible[key] != item:
                    raise ReconciliationError("predicted_subspace_ambiguous")
                compatible[key] = item
            elif isinstance(item, dict):
                collect(item)

    collect(raw)
    return compatible


class HypothesisReconciler:
    def reconcile(
        self,
        proposal: HypothesisProposal,
        context: HypothesisAgentContext,
        *,
        call_id: str,
        prompt_version: str,
    ) -> Hypothesis:
        permitted = set(context.permitted_evidence_reference_ids)
        unknown = set(proposal.evidence_references) - permitted
        if unknown:
            raise ReconciliationError("unknown_evidence_reference")
        lowered_statement = proposal.statement.casefold()
        if any(
            claim in lowered_statement
            for claim in ("is supported", "has been proven", "is confirmed")
        ):
            raise ReconciliationError("proposal_claims_existing_support")
        metric = context.contract.primary_metric.casefold().replace("_", " ")
        observation = proposal.expected_observation.casefold().replace("_", " ")
        if metric not in observation:
            raise ReconciliationError(
                "expected_observation_not_measurable_by_primary_metric"
            )
        if (
            proposal.expected_observation.casefold().strip()
            == proposal.falsification_condition.casefold().strip()
        ):
            raise ReconciliationError("falsification_condition_not_distinct")
        permitted_parameters = set(context.task.direct_configuration_schema) | set(
            context.task.optuna_space_summary
        )
        if not proposal.predicted_subspace:
            raise ReconciliationError("predicted_subspace_is_empty")
        predicted_subspace = _normalise_predicted_subspace(
            dict(proposal.predicted_subspace),
            permitted_parameters,
        )
        if not predicted_subspace:
            raise ReconciliationError("predicted_subspace_not_task_compatible")
        if len(predicted_subspace) > 32:
            raise ReconciliationError("predicted_subspace_too_large")
        prior_refs = _prior_reference_ids(context.prior_verified_findings)
        knowledge_by_id = {
            item.reference_id: item for item in context.knowledge_references
        }
        cited_knowledge = [
            knowledge_by_id[item]
            for item in proposal.evidence_references
            if item in knowledge_by_id
        ]
        predicted_parameters = set(predicted_subspace)
        _require_relevant_knowledge(cited_knowledge, predicted_parameters)
        if cited_knowledge:
            grounding = GroundingStatus.KNOWLEDGE_GROUNDED
            cap = min(item.prior_weight_cap for item in cited_knowledge)
        elif set(proposal.evidence_references) & prior_refs:
            grounding = GroundingStatus.PRIOR_RESULTS_GROUNDED
            cap = 0.8
        elif context.contract.contract_id in proposal.evidence_references:
            grounding = GroundingStatus.CONTRACT_GROUNDED
            cap = 0.6
        else:
            grounding = GroundingStatus.UNGROUNDED
            cap = 0.3
        return Hypothesis(
            hypothesis_id=_stable_id(
                "hyp",
                context.run_id,
                str(context.cycle),
                prompt_version,
                context.context_hash,
            ),
            statement=proposal.statement,
            rationale=proposal.rationale,
            predicted_subspace=predicted_subspace,
            expected_observation=proposal.expected_observation,
            falsification_condition=proposal.falsification_condition,
            evidence_references=proposal.evidence_references,
            prior_weight=min(proposal.confidence, cap),
            status=HypothesisStatus.OPEN,
            provenance=ProvenanceKind.MOCK,
            proposal_source=ProposalSource.MODEL_GENERATED,
            grounding_status=grounding,
            agent_call_id=call_id,
            prompt_version=prompt_version,
        )


class PlannerReconciler:
    def __init__(self, task: ResearchTask, contract: ResearchContract) -> None:
        self._task = task
        self._contract = contract

    def reconcile(
        self,
        proposal: PlannerProposal,
        context: PlannerAgentContext,
        *,
        call_id: str,
        prompt_version: str,
    ) -> SearchRequest:
        search_type = proposal.search_type
        permitted = set(context.permitted_evidence_reference_ids)
        if set(proposal.evidence_references) - permitted:
            raise ReconciliationError("unknown_evidence_reference")
        if search_type not in context.installed_search_capabilities:
            raise ReconciliationError("search_type_not_installed")
        if search_type not in context.contract.allowed_search_types:
            raise ReconciliationError("search_type_not_allowed")
        if search_type not in context.task.available_search_types:
            raise ReconciliationError("search_type_not_supported_by_task")
        if proposal.requested_experiment_budget > context.remaining_experiment_budget:
            raise ReconciliationError("requested_budget_exceeds_remaining_budget")
        search_space = dict(proposal.proposed_search_space)
        provisional = SearchRequest(
            request_id="proposal-validation",
            hypothesis_id=context.hypothesis.hypothesis_id,
            search_type=search_type,
            target=proposal.target,
            search_space=search_space,
            experiment_budget=proposal.requested_experiment_budget,
            rationale=proposal.rationale,
        )
        if search_type == SearchType.DIRECT:
            if proposal.requested_experiment_budget != 1:
                raise ReconciliationError("direct_requires_single_experiment")
            try:
                search_space = self._task.normalise_configuration(search_space)
            except (TypeError, ValueError) as exc:
                raise ReconciliationError("invalid_direct_configuration") from exc
        elif search_type == SearchType.OPTUNA:
            if not isinstance(self._task, OptunaCapableTask):
                raise ReconciliationError("task_not_optuna_capable")
            try:
                self._task.create_optuna_study_spec(
                    self._contract,
                    provisional,
                )
            except (TypeError, ValueError) as exc:
                raise ReconciliationError("invalid_optuna_narrowing") from exc
        elif search_type == SearchType.OPENEVOLVE:
            if not isinstance(self._task, OpenEvolveCapableTask):
                raise ReconciliationError("task_not_openevolve_capable")
            if not isinstance(search_space.get("openevolve"), dict):
                raise ReconciliationError("invalid_openevolve_configuration")
        else:
            raise ReconciliationError("unsupported_pr4_search_type")
        knowledge_by_id = {
            item.reference_id: item for item in context.knowledge_references
        }
        cited_knowledge = [
            knowledge_by_id[item]
            for item in proposal.evidence_references
            if item in knowledge_by_id
        ]
        _require_relevant_knowledge(cited_knowledge, set(search_space))
        prior_refs = _prior_reference_ids(context.prior_verified_findings)
        if cited_knowledge:
            grounding = GroundingStatus.KNOWLEDGE_GROUNDED
        elif set(proposal.evidence_references) & prior_refs:
            grounding = GroundingStatus.PRIOR_RESULTS_GROUNDED
        elif context.contract.contract_id in proposal.evidence_references:
            grounding = GroundingStatus.CONTRACT_GROUNDED
        else:
            grounding = GroundingStatus.UNGROUNDED
        return SearchRequest(
            request_id=_stable_id(
                "search",
                context.run_id,
                str(context.cycle),
                context.hypothesis.hypothesis_id,
                prompt_version,
                context.context_hash,
            ),
            hypothesis_id=context.hypothesis.hypothesis_id,
            search_type=search_type,
            target=proposal.target,
            search_space=search_space,
            experiment_budget=proposal.requested_experiment_budget,
            rationale=proposal.rationale,
            evidence_references=proposal.evidence_references,
            requires_human_approval=(
                proposal.recommends_human_approval
                or search_type in context.approval_requirements
            ),
            proposal_source=ProposalSource.MODEL_GENERATED,
            grounding_status=grounding,
            agent_call_id=call_id,
            prompt_version=prompt_version,
        )
