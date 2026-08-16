from __future__ import annotations

from dataclasses import replace

from auto_researcher.agents.context import AgentContextAssemblyError
from auto_researcher.agents.live.base import LiveAgentExecutionError
from auto_researcher.contracts.enums import (
    EventType,
    GroundingStatus,
    HypothesisStatus,
    ProposalSource,
    ProvenanceKind,
    RunStatus,
    SearchType,
)
from auto_researcher.contracts.models import (
    BudgetState,
    DecisionEvent,
    Hypothesis,
    SearchRequest,
)
from auto_researcher.graph.nodes.hypothesis import generate_hypothesis
from auto_researcher.graph.nodes.planner import plan_search
from auto_researcher.graph.nodes.provenance import record_provenance
from auto_researcher.tasks.feta_unet_search import FeTAUNetSearchTask
from auto_researcher.tasks.models import TaskRuntimeContext
from auto_researcher.tasks.synthetic import default_synthetic_configuration


class _OversizedPlannerContext:
    def planner_context(self, *_args, **_kwargs):
        raise AgentContextAssemblyError("agent_context_too_large")


class _OversizedHypothesisContext:
    def hypothesis_context(self, *_args, **_kwargs):
        raise AgentContextAssemblyError("agent_context_too_large")


class _PlannerMustNotRun:
    def plan(self, _context):
        raise AssertionError("planner model must not be called")


class _InvalidBudgetPlanner:
    def plan(self, context):
        return SearchRequest(
            request_id="search-invalid-budget",
            hypothesis_id=context.hypothesis.hypothesis_id,
            search_type=SearchType.DIRECT,
            target=context.contract.primary_metric,
            search_space=dict(context.hypothesis.predicted_subspace),
            experiment_budget=context.contract.maximum_experiments + 1,
            rationale="Deliberately exceeds the campaign contract.",
            proposal_source=ProposalSource.MODEL_GENERATED,
            grounding_status=context.hypothesis.grounding_status,
        )


class _FailingLiveHypothesisAgent:
    def generate(self, _context):
        raise LiveAgentExecutionError("invalid_structured_output")


def test_valid_hypothesis_uses_identity_stable_direct_fallback(
    contract_factory,
    deterministic_dependencies,
):
    contract = contract_factory(
        allowed=frozenset({SearchType.DIRECT, SearchType.OPTUNA}),
        maximum_experiments=4,
    )
    cycle_four_configuration = {
        "maximum_epochs": 150,
        "learning_rate": 0.0004,
        "weight_decay": 0.000003,
        "dropout": 0.0,
        "dice_weight": 1.0,
        "positive_negative_ratio": "1:1",
        "augmentation_strength": "baseline",
    }
    hypothesis = Hypothesis(
        hypothesis_id="hyp-cycle-four",
        statement="The exact bounded configuration may improve the score.",
        rationale="Cycle-four hypothesis.",
        predicted_subspace=cycle_four_configuration,
        expected_observation="objective_score increases",
        falsification_condition="objective_score does not increase",
        prior_weight=0.5,
        status=HypothesisStatus.OPEN,
        provenance=ProvenanceKind.MOCK,
        proposal_source=ProposalSource.MODEL_GENERATED,
        grounding_status=GroundingStatus.PRIOR_RESULTS_GROUNDED,
    )
    state = {
        "run_id": "run-cycle-four",
        "thread_id": "thread-cycle-four",
        "contract": contract,
        "status": RunStatus.RUNNING,
        "cycle": 4,
        "budget": BudgetState(
            maximum_cycles=12,
            maximum_experiments=4,
            maximum_cost=20,
            cycles_used=4,
            experiments_used=3,
        ),
        "active_hypothesis": hypothesis,
        "decision_event_ids": [],
        "errors": [],
        "executed_nodes": [],
    }
    dependencies = replace(
        deterministic_dependencies,
        agent_context_assembler=_OversizedPlannerContext(),
        planner_agent=_PlannerMustNotRun(),
        task=FeTAUNetSearchTask(),
    )

    first = plan_search(state, dependencies)
    second = plan_search(state, dependencies)

    assert "status" not in first
    assert first["planner_fallback_code"] == "agent_context_too_large"
    assert first["planner_failure_stage"] == "context_assembly"
    assert first["search_request"].search_type == SearchType.DIRECT
    assert first["search_request"].proposal_source == ProposalSource.DETERMINISTIC
    assert first["search_request"].search_space == cycle_four_configuration
    assert first["search_request"].request_id == second["search_request"].request_id
    assert first["errors"] == []

    record_provenance({**state, **first}, dependencies)
    planned = [
        event
        for event in dependencies.provenance_store.list_events(state["run_id"])
        if event.event_type == EventType.SEARCH_PLANNED
    ]
    assert len(planned) == 1
    assert "fallback:agent_context_too_large" in planned[0].output_references


def test_unusable_hypothesis_records_specific_safe_planner_failure(
    contract_factory,
    deterministic_dependencies,
):
    contract = contract_factory(allowed=frozenset({SearchType.DIRECT}))
    hypothesis = Hypothesis(
        hypothesis_id="hyp-invalid-fallback",
        statement="An invalid field cannot become an experiment.",
        rationale="Exercise the fail-closed boundary.",
        predicted_subspace={"unknown_parameter": 1},
        expected_observation="objective_score changes",
        falsification_condition="objective_score does not change",
        prior_weight=0.5,
        status=HypothesisStatus.OPEN,
        provenance=ProvenanceKind.MOCK,
        proposal_source=ProposalSource.MODEL_GENERATED,
        grounding_status=GroundingStatus.CONTRACT_GROUNDED,
    )
    state = {
        "run_id": "run-invalid-fallback",
        "thread_id": "thread-invalid-fallback",
        "contract": contract,
        "status": RunStatus.RUNNING,
        "cycle": 4,
        "budget": BudgetState(
            maximum_cycles=12,
            maximum_experiments=4,
            maximum_cost=20,
            cycles_used=4,
            experiments_used=3,
        ),
        "active_hypothesis": hypothesis,
        "decision_event_ids": [],
        "errors": [],
        "executed_nodes": [],
    }
    dependencies = replace(
        deterministic_dependencies,
        agent_context_assembler=_OversizedPlannerContext(),
        planner_agent=_PlannerMustNotRun(),
        task=FeTAUNetSearchTask(),
    )

    update = plan_search(state, dependencies)

    assert update["status"] == RunStatus.FAILED
    assert update["stop_reason"] == "agent_context_too_large"
    assert update["errors"] == ["agent_context_too_large"]
    record_provenance({**state, **update}, dependencies)
    stopped = [
        event
        for event in dependencies.provenance_store.list_events(state["run_id"])
        if event.event_type == EventType.RUN_STOPPED
    ]
    assert len(stopped) == 1
    assert stopped[0].output_references == (
        "error_code:agent_context_too_large",
        "failure_stage:context_assembly",
    )


def test_invalid_live_planner_request_uses_validated_direct_fallback(
    contract_factory,
    deterministic_dependencies,
):
    contract = contract_factory(
        allowed=frozenset({SearchType.DIRECT, SearchType.OPTUNA}),
        maximum_experiments=4,
    )
    configuration = default_synthetic_configuration()
    hypothesis = Hypothesis(
        hypothesis_id="hyp-invalid-live-plan",
        statement="The bounded configuration should remain executable.",
        rationale="Exercise post-model request validation.",
        predicted_subspace=configuration,
        expected_observation="objective_score increases",
        falsification_condition="objective_score does not increase",
        prior_weight=0.5,
        status=HypothesisStatus.OPEN,
        provenance=ProvenanceKind.MOCK,
        proposal_source=ProposalSource.MODEL_GENERATED,
        grounding_status=GroundingStatus.PRIOR_RESULTS_GROUNDED,
    )
    state = {
        "run_id": "run-invalid-live-plan",
        "thread_id": "thread-invalid-live-plan",
        "contract": contract,
        "status": RunStatus.RUNNING,
        "cycle": 2,
        "budget": BudgetState(
            maximum_cycles=4,
            maximum_experiments=4,
            maximum_cost=20,
            cycles_used=2,
            experiments_used=1,
        ),
        "active_hypothesis": hypothesis,
        "decision_event_ids": [],
        "errors": [],
        "executed_nodes": [],
    }
    dependencies = replace(
        deterministic_dependencies,
        planner_agent=_InvalidBudgetPlanner(),
    )

    update = plan_search(state, dependencies)

    assert "status" not in update
    assert update["errors"] == []
    assert update["planner_fallback_code"] == "planner_request_invalid"
    assert update["planner_failure_stage"] == "request_validation"
    assert update["search_request"].search_type == SearchType.DIRECT
    assert update["search_request"].experiment_budget == 1
    assert update["search_request"].search_space == configuration


def test_live_hypothesis_failure_reuses_best_verified_prior_result(
    contract_factory,
    deterministic_dependencies,
):
    contract = contract_factory(
        allowed=frozenset({SearchType.DIRECT}),
        maximum_cycles=4,
        maximum_experiments=4,
        maximum_cost=50,
    )
    configuration = default_synthetic_configuration()
    deterministic_dependencies.provenance_store.append_event(
        DecisionEvent(
            event_id="prior-evidence",
            run_id="run-hypothesis-fallback",
            cycle=1,
            event_type=EventType.EVIDENCE_VERIFIED,
            actor="verifier",
            input_references=("experiment-prior",),
            output_references=(
                "evidence:SUPPORTED",
                "verified:true",
                "constraints:true",
                "score:0.82",
                "search_type:DIRECT",
                "hypothesis:hypothesis-prior",
            ),
            rationale="Verified prior result.",
            timestamp=deterministic_dependencies.clock(),
            code_version="test",
            provenance=ProvenanceKind.REAL,
            safe_payload={"configuration": configuration},
        )
    )
    state = {
        "run_id": "run-hypothesis-fallback",
        "thread_id": "thread-hypothesis-fallback",
        "contract": contract,
        "status": RunStatus.RUNNING,
        "cycle": 2,
        "budget": BudgetState(
            maximum_cycles=4,
            maximum_experiments=4,
            maximum_cost=50,
            cycles_used=2,
            experiments_used=1,
        ),
        "decision_event_ids": [],
        "errors": [],
        "executed_nodes": [],
    }
    dependencies = replace(
        deterministic_dependencies,
        hypothesis_agent=_FailingLiveHypothesisAgent(),
    )

    update = generate_hypothesis(state, dependencies)

    hypothesis = update["active_hypothesis"]
    assert "status" not in update
    assert update["hypothesis_fallback_code"] == "invalid_structured_output"
    assert hypothesis.proposal_source == ProposalSource.DETERMINISTIC
    assert hypothesis.predicted_subspace == configuration
    assert hypothesis.evidence_references == (
        "hypothesis-prior",
        "experiment-prior",
    )
    record_provenance({**state, **update}, dependencies)
    proposed = [
        event
        for event in dependencies.provenance_store.list_events(state["run_id"])
        if event.event_type == EventType.HYPOTHESIS_PROPOSED
    ]
    assert "fallback:invalid_structured_output" in proposed[-1].output_references


def test_first_cycle_context_failure_uses_configured_incumbent(
    contract_factory,
    deterministic_dependencies,
):
    contract = contract_factory(
        allowed=frozenset({SearchType.DIRECT, SearchType.OPTUNA}),
        maximum_cycles=12,
        maximum_experiments=30,
        maximum_cost=50,
    )
    incumbent = {
        "maximum_epochs": 150,
        "learning_rate": 0.0003,
        "weight_decay": 0.000003,
        "dropout": 0.0,
        "dice_weight": 1.0,
        "positive_negative_ratio": "1:1",
        "augmentation_strength": "baseline",
    }
    state = {
        "run_id": "run-first-cycle-fallback",
        "thread_id": "thread-first-cycle-fallback",
        "contract": contract,
        "status": RunStatus.RUNNING,
        "cycle": 1,
        "budget": BudgetState(
            maximum_cycles=12,
            maximum_experiments=30,
            maximum_cost=50,
            cycles_used=1,
            experiments_used=0,
        ),
        "decision_event_ids": [],
        "errors": [],
        "executed_nodes": [],
    }
    dependencies = replace(
        deterministic_dependencies,
        agent_context_assembler=_OversizedHypothesisContext(),
        task=FeTAUNetSearchTask(),
        runtime_context=TaskRuntimeContext(
            task_options={"initial_incumbent_configuration": incumbent}
        ),
    )

    first = generate_hypothesis(state, dependencies)
    second = generate_hypothesis(state, dependencies)

    hypothesis = first["active_hypothesis"]
    assert "status" not in first
    assert first["hypothesis_fallback_code"] == "agent_context_too_large"
    assert first["hypothesis_failure_stage"] == "context_assembly"
    assert hypothesis.proposal_source == ProposalSource.DETERMINISTIC
    assert hypothesis.grounding_status == GroundingStatus.CONTRACT_GROUNDED
    assert hypothesis.predicted_subspace == incumbent
    assert hypothesis.evidence_references == (contract.contract_id,)
    assert hypothesis.hypothesis_id == second["active_hypothesis"].hypothesis_id

    record_provenance({**state, **first}, dependencies)
    proposed = [
        event
        for event in dependencies.provenance_store.list_events(state["run_id"])
        if event.event_type == EventType.HYPOTHESIS_PROPOSED
    ]
    assert "fallback:agent_context_too_large" in proposed[-1].output_references
