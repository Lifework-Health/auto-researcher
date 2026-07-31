from __future__ import annotations

import pytest

from auto_researcher.agents.models import (
    HypothesisProposal,
    PlannerProposal,
    PriorResearchSummary,
)
from auto_researcher.agents.context import AgentContextAssembler, AgentContextLimits
from auto_researcher.agents.reconciliation import (
    HypothesisReconciler,
    PlannerReconciler,
    ReconciliationError,
)
from auto_researcher.contracts.enums import (
    EvidenceStatus,
    GroundingStatus,
    ProposalSource,
    SearchType,
)
from auto_researcher.contracts.models import BudgetState
from auto_researcher.graph.nodes.supervisor import supervisor_prepare
from auto_researcher.knowledge.models import (
    KnowledgeContextReference,
    KnowledgeTrustTier,
)
from auto_researcher.runtime.dependencies import task_memory_dependencies
from auto_researcher.tasks.models import TaskRuntimeContext
from auto_researcher.tasks.synthetic import (
    SyntheticTask,
    default_synthetic_configuration,
    default_synthetic_contract,
)


def _contexts():
    contract = default_synthetic_contract()
    dependencies = task_memory_dependencies(
        SyntheticTask(),
        TaskRuntimeContext(),
        contract,
        default_synthetic_configuration(),
    )
    initial = {
        "run_id": "context-run",
        "thread_id": "context-thread",
        "contract": contract,
        "status": "RUNNING",
        "cycle": 0,
        "budget": BudgetState(
            maximum_cycles=1,
            maximum_experiments=1,
            maximum_cost=1,
        ),
        "decision_event_ids": [],
        "errors": [],
        "executed_nodes": [],
    }
    state = {**initial, **supervisor_prepare(initial)}
    hypothesis_context = dependencies.agent_context_assembler.hypothesis_context(
        state,
        dependencies.task_agent_context,
    )
    return dependencies, state, hypothesis_context


def test_safe_context_has_no_paths_or_unbounded_state():
    dependencies, _, context = _contexts()
    dumped = context.model_dump_json()
    assert "/Users/" not in dumped
    assert "ANTHROPIC_API_KEY" not in dumped
    assert (
        len(dumped)
        < dependencies.agent_context_assembler.limits.maximum_context_characters
    )
    assert context.contract.contract_id in context.permitted_evidence_reference_ids
    assert SearchType.OPENEVOLVE not in context.task.available_search_types


def test_context_size_limit_fails_before_a_model_call():
    dependencies, state, _ = _contexts()
    assembler = AgentContextAssembler(
        dependencies.provenance_store,
        limits=AgentContextLimits(maximum_context_characters=10),
    )
    with pytest.raises(ValueError, match="agent_context_too_large"):
        assembler.hypothesis_context(state, dependencies.task_agent_context)


def test_hypothesis_reconciliation_derives_grounding_and_caps_weight():
    _, _, context = _contexts()
    hypothesis = HypothesisReconciler().reconcile(
        HypothesisProposal(
            statement="Tree complexity may improve the bounded objective.",
            rationale="Contract-bounded test.",
            predicted_subspace={"complexity": [3, 5]},
            expected_observation="objective_score increases",
            falsification_condition="objective_score does not increase",
            evidence_references=(context.contract.contract_id,),
            confidence=0.95,
        ),
        context,
        call_id="call-1",
        prompt_version="1.0.0",
    )
    assert hypothesis.proposal_source == ProposalSource.MODEL_GENERATED
    assert hypothesis.grounding_status == GroundingStatus.CONTRACT_GROUNDED
    assert hypothesis.prior_weight == 0.6

    with pytest.raises(ReconciliationError, match="unknown_evidence_reference"):
        HypothesisReconciler().reconcile(
            HypothesisProposal(
                statement="Complexity may alter the objective.",
                rationale="Test.",
                predicted_subspace={"complexity": [3, 5]},
                expected_observation="objective_score changes",
                falsification_condition="objective_score does not change",
                evidence_references=("PMID:invented",),
                confidence=0.5,
            ),
            context,
            call_id="call-2",
            prompt_version="1.0.0",
        )


def test_planner_reconciliation_rejects_clipping_and_unknown_direct_fields():
    dependencies, state, hypothesis_context = _contexts()
    hypothesis = HypothesisReconciler().reconcile(
        HypothesisProposal(
            statement="Complexity may alter the objective.",
            rationale="Test.",
            predicted_subspace={"complexity": [3, 5]},
            expected_observation="objective_score changes",
            falsification_condition="objective_score does not change",
            confidence=0.2,
        ),
        hypothesis_context,
        call_id="call-1",
        prompt_version="1.0.0",
    )
    state["active_hypothesis"] = hypothesis
    planner_context = dependencies.agent_context_assembler.planner_context(
        state,
        dependencies.task_agent_context,
        dependencies.search_capabilities,
    )
    reconciler = PlannerReconciler(dependencies.task, state["contract"])
    with pytest.raises(ReconciliationError, match="exceeds"):
        reconciler.reconcile(
            PlannerProposal(
                search_type=SearchType.DIRECT,
                target="objective_score",
                proposed_search_space=default_synthetic_configuration(),
                requested_experiment_budget=2,
                rationale="Too large.",
            ),
            planner_context,
            call_id="call-2",
            prompt_version="1.0.0",
        )


def test_planner_reconciliation_applies_contract_approval_and_stable_id():
    dependencies, state, hypothesis_context = _contexts()
    contract = state["contract"].model_copy(
        update={"requires_approval_for": frozenset({SearchType.DIRECT})}
    )
    state["contract"] = contract
    hypothesis = HypothesisReconciler().reconcile(
        HypothesisProposal(
            statement="Complexity may alter the objective.",
            rationale="Test.",
            predicted_subspace={"complexity": [3, 5]},
            expected_observation="objective_score changes",
            falsification_condition="objective_score does not change",
            confidence=0.2,
        ),
        hypothesis_context,
        call_id="call-1",
        prompt_version="1.0.0",
    )
    state["active_hypothesis"] = hypothesis
    planner_context = dependencies.agent_context_assembler.planner_context(
        state,
        dependencies.task_agent_context,
        dependencies.search_capabilities,
    )
    proposal = PlannerProposal(
        search_type=SearchType.DIRECT,
        target="objective_score",
        proposed_search_space=default_synthetic_configuration(),
        requested_experiment_budget=1,
        rationale="Bounded test.",
    )
    reconciler = PlannerReconciler(dependencies.task, contract)
    first = reconciler.reconcile(
        proposal,
        planner_context,
        call_id="call-2",
        prompt_version="1.0.0",
    )
    second = reconciler.reconcile(
        proposal,
        planner_context,
        call_id="call-2",
        prompt_version="1.0.0",
    )
    assert first == second
    assert first.requires_human_approval is True
    with pytest.raises(ReconciliationError, match="invalid_direct_configuration"):
        reconciler.reconcile(
            PlannerProposal(
                search_type=SearchType.DIRECT,
                target="objective_score",
                proposed_search_space={
                    **default_synthetic_configuration(),
                    "unknown": True,
                },
                requested_experiment_budget=1,
                rationale="Unknown field.",
            ),
            planner_context,
            call_id="call-3",
            prompt_version="1.0.0",
        )


def test_empty_relevance_cannot_ground_hypothesis_or_plan():
    dependencies, state, hypothesis_context = _contexts()
    reference = KnowledgeContextReference(
        reference_id="knowledge-ref-empty-relevance",
        concise_claim="A registered entity exists.",
        citation_label="[fixture]",
        trust_tier=KnowledgeTrustTier.CURATED,
        confidence=1.0,
        entity_curies=("ENTITY:1", "ENTITY:1"),
        source_ids=("source:fixture",),
        bundle_id="bundle-fixture",
        relevant_parameters=(),
        prior_weight_cap=0.9,
    )
    hypothesis_context = hypothesis_context.model_copy(
        update={
            "permitted_evidence_reference_ids": (
                *hypothesis_context.permitted_evidence_reference_ids,
                reference.reference_id,
            ),
            "knowledge_references": (reference,),
        }
    )
    proposal = HypothesisProposal(
        statement="Complexity may alter the objective.",
        rationale="Test.",
        predicted_subspace={"complexity": [3, 5]},
        expected_observation="objective_score changes",
        falsification_condition="objective_score does not change",
        evidence_references=(reference.reference_id,),
        confidence=0.5,
    )
    with pytest.raises(ReconciliationError, match="knowledge_reference_not_relevant"):
        HypothesisReconciler().reconcile(
            proposal,
            hypothesis_context,
            call_id="call-empty-hypothesis",
            prompt_version="2.0.0",
        )

    state["active_hypothesis"] = HypothesisReconciler().reconcile(
        proposal.model_copy(update={"evidence_references": ()}),
        hypothesis_context,
        call_id="call-ungrounded-hypothesis",
        prompt_version="2.0.0",
    )
    planner_context = dependencies.agent_context_assembler.planner_context(
        state,
        dependencies.task_agent_context,
        dependencies.search_capabilities,
    ).model_copy(
        update={
            "permitted_evidence_reference_ids": (
                reference.reference_id,
                *hypothesis_context.permitted_evidence_reference_ids,
            ),
            "knowledge_references": (reference,),
        }
    )
    with pytest.raises(ReconciliationError, match="knowledge_reference_not_relevant"):
        PlannerReconciler(dependencies.task, state["contract"]).reconcile(
            PlannerProposal(
                search_type=SearchType.DIRECT,
                target="objective_score",
                proposed_search_space=default_synthetic_configuration(),
                requested_experiment_budget=1,
                rationale="Test.",
                evidence_references=(reference.reference_id,),
            ),
            planner_context,
            call_id="call-empty-planner",
            prompt_version="2.0.0",
        )


def test_planner_safe_artefact_reference_is_prior_results_grounded():
    dependencies, state, hypothesis_context = _contexts()
    state["active_hypothesis"] = HypothesisReconciler().reconcile(
        HypothesisProposal(
            statement="Complexity may alter the objective.",
            rationale="Test.",
            predicted_subspace={"complexity": [3, 5]},
            expected_observation="objective_score changes",
            falsification_condition="objective_score does not change",
            confidence=0.2,
        ),
        hypothesis_context,
        call_id="call-prior-hypothesis",
        prompt_version="2.0.0",
    )
    artefact_reference = "artefact:verified-result.json"
    finding = PriorResearchSummary(
        hypothesis_reference="hypothesis:prior",
        experiment_reference="experiment:prior",
        search_type=SearchType.DIRECT,
        primary_score=0.8,
        evidence_status=EvidenceStatus.SUPPORTED,
        constraint_compliant=True,
        concise_verified_finding="The prior result passed verification.",
        safe_artefact_references=(artefact_reference,),
    )
    planner_context = dependencies.agent_context_assembler.planner_context(
        state,
        dependencies.task_agent_context,
        dependencies.search_capabilities,
    ).model_copy(
        update={
            "prior_verified_findings": (finding,),
            "permitted_evidence_reference_ids": (artefact_reference,),
        }
    )
    request = PlannerReconciler(dependencies.task, state["contract"]).reconcile(
        PlannerProposal(
            search_type=SearchType.DIRECT,
            target="objective_score",
            proposed_search_space=default_synthetic_configuration(),
            requested_experiment_budget=1,
            rationale="Use the verified prior result.",
            evidence_references=(artefact_reference,),
        ),
        planner_context,
        call_id="call-prior-planner",
        prompt_version="2.0.0",
    )
    assert request.grounding_status == GroundingStatus.PRIOR_RESULTS_GROUNDED
