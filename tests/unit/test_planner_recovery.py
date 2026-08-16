from __future__ import annotations

from dataclasses import replace

from auto_researcher.agents.context import AgentContextAssemblyError
from auto_researcher.contracts.enums import (
    EventType,
    GroundingStatus,
    HypothesisStatus,
    ProposalSource,
    ProvenanceKind,
    RunStatus,
    SearchType,
)
from auto_researcher.contracts.models import BudgetState, Hypothesis
from auto_researcher.graph.nodes.planner import plan_search
from auto_researcher.graph.nodes.provenance import record_provenance
from auto_researcher.tasks.feta_unet_search import FeTAUNetSearchTask


class _OversizedPlannerContext:
    def planner_context(self, *_args, **_kwargs):
        raise AgentContextAssemblyError("agent_context_too_large")


class _PlannerMustNotRun:
    def plan(self, _context):
        raise AssertionError("planner model must not be called")


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
    assert "errors" not in first

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
