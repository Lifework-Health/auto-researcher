from __future__ import annotations

from datetime import UTC, datetime

import pytest

from auto_researcher.agents.models import (
    AgentBudgetPolicy,
    ModelCallConfig,
    ModelPricing,
)
from auto_researcher.agents.provenance import append_model_call_events
from auto_researcher.contracts.enums import (
    AgentCallStatus,
    EventType,
    ProposalSource,
    ProvenanceKind,
    RunStatus,
    SearchType,
)
from auto_researcher.contracts.models import ResearchContract
from auto_researcher.graph.builder import build_graph
from auto_researcher.runtime.dependencies import task_memory_dependencies
from auto_researcher.tasks.icca_nbs import ICCANBSTask
from auto_researcher.tasks.models import TaskRuntimeContext
from auto_researcher.tasks.synthetic import (
    SyntheticTask,
    default_synthetic_configuration,
    default_synthetic_contract,
)
from tests.fakes_agents import FakeStructuredModelClient
from tests.fakes_icca import make_fake_icca_bindings

FIXED_TIME = datetime(2026, 7, 30, 12, 0, tzinfo=UTC)


def _call_config() -> ModelCallConfig:
    return ModelCallConfig(
        provider="fake",
        model_id="fake-model-2026-07-30",
        temperature=0,
        maximum_output_tokens=512,
        timeout_seconds=10,
        maximum_attempts=2,
        maximum_cost_per_call=0.1,
        pricing=ModelPricing(
            version="fake-v1",
            input_cost_per_million_tokens=1,
            output_cost_per_million_tokens=2,
            currency="USD",
        ),
        prompt_version="1.0.0",
    )


def _invoke(dependencies, contract, run_id):
    return build_graph(dependencies).invoke(
        {
            "run_id": run_id,
            "thread_id": f"{run_id}-thread",
            "contract": contract,
        },
        {"configurable": {"thread_id": f"{run_id}-thread"}},
    )


@pytest.mark.parametrize("search_type", [SearchType.DIRECT, SearchType.OPTUNA])
def test_fake_live_synthetic_direct_and_optuna(search_type, tmp_path):
    budget = 1 if search_type == SearchType.DIRECT else 2
    contract = default_synthetic_contract(
        search_types=frozenset({search_type}),
        maximum_experiments=budget,
    )
    hypothesis = {
        "statement": "Bounded complexity may improve the synthetic objective.",
        "rationale": "Test the contract-bounded candidate region.",
        "predicted_subspace": {"complexity": [3, 6]},
        "expected_observation": "objective_score increases",
        "falsification_condition": "objective_score does not increase",
        "evidence_references": [contract.contract_id],
        "confidence": 0.7,
    }
    direct = default_synthetic_configuration()
    planner = {
        "search_type": search_type.value,
        "target": "objective_score",
        "proposed_search_space": (
            direct
            if search_type == SearchType.DIRECT
            else {"trial_budget": budget, "seed": 7}
        ),
        "requested_experiment_budget": budget,
        "rationale": "Run the bounded registered search.",
        "recommends_human_approval": False,
    }
    client = FakeStructuredModelClient(hypothesis, planner)
    dependencies = task_memory_dependencies(
        SyntheticTask(),
        TaskRuntimeContext(
            run_id=f"live-synthetic-{search_type.value.lower()}",
            output_dir=tmp_path,
            manifest_created_at=FIXED_TIME,
        ),
        contract,
        direct if search_type == SearchType.DIRECT else {"trial_budget": budget},
        model_client=client,
        hypothesis_call_config=_call_config(),
        planner_call_config=_call_config(),
        agent_budget_policy=AgentBudgetPolicy(maximum_total_model_calls=6),
        search_type=search_type,
        clock=lambda: FIXED_TIME,
    )
    final = _invoke(
        dependencies,
        contract,
        f"live-synthetic-{search_type.value.lower()}",
    )
    assert final["status"] == RunStatus.COMPLETED
    assert final["active_hypothesis"].proposal_source == ProposalSource.MODEL_GENERATED
    assert final["search_request"].proposal_source == ProposalSource.MODEL_GENERATED
    assert final["budget"].model_calls_used == 2
    assert final["budget"].model_cost_used > 0
    assert len(client.calls) == 2
    events = dependencies.provenance_store.list_events(final["run_id"])
    assert sum(event.event_type == EventType.MODEL_CALL_RESERVED for event in events) == 2
    assert sum(event.event_type == EventType.MODEL_CALL_COMPLETED for event in events) == 2
    assert all("task-context" not in event.rationale for event in events)
    hypothesis_event = next(
        event for event in events if event.event_type == EventType.HYPOTHESIS_PROPOSED
    )
    plan_event = next(
        event for event in events if event.event_type == EventType.SEARCH_PLANNED
    )
    assert final["active_hypothesis"].agent_call_id in hypothesis_event.input_references
    assert final["search_request"].agent_call_id in plan_event.input_references
    append_model_call_events(
        dependencies.provenance_store,
        dependencies.agent_call_store,
        run_id=final["run_id"],
        cycle=final["cycle"],
    )
    replayed_events = dependencies.provenance_store.list_events(final["run_id"])
    assert sum(
        event.event_type
        in {
            EventType.MODEL_CALL_RESERVED,
            EventType.MODEL_CALL_COMPLETED,
            EventType.MODEL_CALL_FAILED,
        }
        for event in replayed_events
    ) == 4
    mock_dependencies = task_memory_dependencies(
        SyntheticTask(),
        TaskRuntimeContext(output_dir=tmp_path / "mock"),
        contract,
        direct if search_type == SearchType.DIRECT else {"trial_budget": budget},
        search_type=search_type,
    )
    assert (
        build_graph(dependencies).get_graph().draw_mermaid()
        == build_graph(mock_dependencies).get_graph().draw_mermaid()
    )


def _icca_contract(search_type: SearchType, budget: int) -> ResearchContract:
    return ResearchContract(
        contract_id=f"live-icca-{search_type.value.lower()}",
        schema_version="1.0",
        task_id="icca_nbs",
        task_version="1.0",
        objective_version="0.9",
        primary_metric="stability_objective",
        task_constraints_version="0.9",
        question="Which bounded iCCA configuration satisfies the eligibility gates?",
        objective="maximise the imported stability objective",
        constraints={},
        allowed_search_types=frozenset({search_type}),
        evaluator_id="icca-nbs-v2-evaluator",
        verifier_id="deterministic-verifier",
        maximum_cycles=1,
        maximum_experiments=budget,
        maximum_cost=2,
        provenance=ProvenanceKind.REAL,
    )


@pytest.mark.parametrize("search_type", [SearchType.DIRECT, SearchType.OPTUNA])
def test_fake_live_icca_direct_and_optuna(search_type, tmp_path):
    for filename in ("Combined_binary_matrix.csv", "Combined_clinical.csv"):
        (tmp_path / filename).write_text(
            "private-patient,secret-raw-value\n",
            encoding="utf-8",
        )
    budget = 1 if search_type == SearchType.DIRECT else 2
    contract = _icca_contract(search_type, budget)
    hypothesis = {
        "statement": "A bounded alpha region may improve aggregate NBS stability.",
        "rationale": "Test only the task-registered mutation-only space.",
        "predicted_subspace": {"alpha": [0.4, 0.8]},
        "expected_observation": "stability_objective increases",
        "falsification_condition": "stability_objective does not increase",
        "evidence_references": [contract.contract_id],
        "confidence": 0.6,
    }
    direct = {
        "network": "Ideker",
        "alignment": "Intersect",
        "alpha": 0.7,
        "K": 5,
        "r": 10,
    }
    optuna = {
        "trial_budget": budget,
        "seed": 5,
        "fixed": {"network": "Ideker", "alignment": "Intersect", "r": 10},
    }
    planner = {
        "search_type": search_type.value,
        "target": "stability_objective",
        "proposed_search_space": direct if search_type == SearchType.DIRECT else optuna,
        "requested_experiment_budget": budget,
        "rationale": "Run only the task-registered bounded space.",
        "recommends_human_approval": False,
    }
    client = FakeStructuredModelClient(hypothesis, planner)
    bindings, _ = make_fake_icca_bindings()
    dependencies = task_memory_dependencies(
        ICCANBSTask(bindings),
        TaskRuntimeContext(
            run_id=contract.contract_id,
            data_dir=tmp_path,
            workspace_dir=tmp_path,
            output_dir=tmp_path / "output",
            manifest_created_at=FIXED_TIME,
        ),
        contract,
        direct if search_type == SearchType.DIRECT else optuna,
        model_client=client,
        hypothesis_call_config=_call_config(),
        planner_call_config=_call_config(),
        search_type=search_type,
        clock=lambda: FIXED_TIME,
    )
    final = _invoke(dependencies, contract, contract.contract_id)
    assert final["status"] == RunStatus.COMPLETED
    assert final["budget"].model_calls_used == 2
    assert len(client.calls) == 2
    rendered = "\n".join(call["user_prompt"] for call in client.calls)
    assert "private-patient" not in rendered
    assert "secret-raw-value" not in rendered
    assert str(tmp_path) not in rendered


def test_invalid_live_plan_fails_closed_before_experiment(tmp_path):
    contract = default_synthetic_contract()
    client = FakeStructuredModelClient(
        {
            "statement": "Complexity may change the objective.",
            "rationale": "Bounded test.",
            "predicted_subspace": {"complexity": [3, 6]},
            "expected_observation": "objective_score changes",
            "falsification_condition": "objective_score does not change",
            "evidence_references": [],
            "confidence": 0.2,
        },
        {
            "search_type": "DIRECT",
            "target": "objective_score",
            "proposed_search_space": {
                **default_synthetic_configuration(),
                "invented": True,
            },
            "requested_experiment_budget": 1,
            "rationale": "Invalid field.",
            "recommends_human_approval": False,
        },
    )
    dependencies = task_memory_dependencies(
        SyntheticTask(),
        TaskRuntimeContext(run_id="invalid-plan", output_dir=tmp_path),
        contract,
        default_synthetic_configuration(),
        model_client=client,
        hypothesis_call_config=_call_config(),
        planner_call_config=_call_config(),
    )
    final = _invoke(dependencies, contract, "invalid-plan")
    assert final["status"] == RunStatus.FAILED
    assert final.get("experiment_spec") is None
    assert "INVALID_STRUCTURED_OUTPUT" in final["errors"]
    assert len(client.calls) == 3  # hypothesis plus two bounded planner attempts
    assert final["budget"].model_calls_used == 3
    events = dependencies.provenance_store.list_events("invalid-plan")
    assert any(event.event_type == EventType.MODEL_CALL_FAILED for event in events)
    assert not any(event.event_type == EventType.EXPERIMENT_PREPARED for event in events)


def test_live_budget_precheck_prevents_paid_call(tmp_path):
    contract = default_synthetic_contract().model_copy(
        update={"maximum_cost": 0.05}
    )
    client = FakeStructuredModelClient(
        {
            "statement": "Complexity may change the objective.",
            "rationale": "Bounded test.",
            "predicted_subspace": {"complexity": [3, 6]},
            "expected_observation": "objective_score changes",
            "falsification_condition": "objective_score does not change",
            "evidence_references": [],
            "confidence": 0.2,
        },
        {},
    )
    dependencies = task_memory_dependencies(
        SyntheticTask(),
        TaskRuntimeContext(run_id="cost-precheck", output_dir=tmp_path),
        contract,
        default_synthetic_configuration(),
        model_client=client,
        hypothesis_call_config=_call_config(),
        planner_call_config=_call_config(),
    )
    final = _invoke(dependencies, contract, "cost-precheck")
    assert final["status"] == RunStatus.FAILED
    assert final["errors"] == ["insufficient_remaining_cost_budget"]
    assert client.calls == []
    assert dependencies.agent_call_store.list_records("cost-precheck") == ()


def test_valid_hypothesis_over_call_cost_limit_is_persisted_and_stops(tmp_path):
    contract = default_synthetic_contract()
    client = FakeStructuredModelClient(
        {
            "statement": "Complexity may change the objective.",
            "rationale": "Bounded test.",
            "predicted_subspace": {"complexity": [3, 6]},
            "expected_observation": "objective_score changes",
            "falsification_condition": "objective_score does not change",
            "evidence_references": [],
            "confidence": 0.2,
        },
        {},
    )
    low_cost_limit = _call_config().model_copy(
        update={"maximum_cost_per_call": 0.0001}
    )
    dependencies = task_memory_dependencies(
        SyntheticTask(),
        TaskRuntimeContext(run_id="hypothesis-cost-overrun", output_dir=tmp_path),
        contract,
        default_synthetic_configuration(),
        model_client=client,
        hypothesis_call_config=low_cost_limit,
        planner_call_config=_call_config(),
    )
    final = _invoke(dependencies, contract, "hypothesis-cost-overrun")
    assert final["status"] == RunStatus.STOPPED
    assert final["stop_reason"] == "maximum_agent_call_cost_exceeded"
    assert final["active_hypothesis"] is not None
    assert final.get("search_request") is None
    assert final.get("experiment_spec") is None
    assert len(client.calls) == 1
    assert final["budget"].model_cost_used == pytest.approx(0.0002)
    records = dependencies.agent_call_store.list_records(
        "hypothesis-cost-overrun"
    )
    assert records[-1].status == AgentCallStatus.COMPLETED


def test_valid_plan_over_call_cost_limit_is_persisted_and_stops(tmp_path):
    contract = default_synthetic_contract()
    client = FakeStructuredModelClient(
        {
            "statement": "Complexity may change the objective.",
            "rationale": "Bounded test.",
            "predicted_subspace": {"complexity": [3, 6]},
            "expected_observation": "objective_score changes",
            "falsification_condition": "objective_score does not change",
            "evidence_references": [],
            "confidence": 0.2,
        },
        {
            "search_type": "DIRECT",
            "target": "objective_score",
            "proposed_search_space": default_synthetic_configuration(),
            "requested_experiment_budget": 1,
            "rationale": "Valid but more expensive than the call ceiling.",
            "recommends_human_approval": False,
        },
    )
    low_cost_limit = _call_config().model_copy(
        update={"maximum_cost_per_call": 0.0001}
    )
    dependencies = task_memory_dependencies(
        SyntheticTask(),
        TaskRuntimeContext(run_id="planner-cost-overrun", output_dir=tmp_path),
        contract,
        default_synthetic_configuration(),
        model_client=client,
        hypothesis_call_config=_call_config(),
        planner_call_config=low_cost_limit,
    )
    final = _invoke(dependencies, contract, "planner-cost-overrun")
    assert final["status"] == RunStatus.STOPPED
    assert final["stop_reason"] == "maximum_agent_call_cost_exceeded"
    assert final["search_request"] is not None
    assert final.get("experiment_spec") is None
    assert len(client.calls) == 2
    assert final["budget"].model_cost_used == pytest.approx(0.0004)
    records = dependencies.agent_call_store.list_records("planner-cost-overrun")
    completed = [
        record for record in records if record.status == AgentCallStatus.COMPLETED
    ]
    assert len(completed) == 2


def test_total_model_call_limit_stops_before_planner_provider_request(tmp_path):
    contract = default_synthetic_contract()
    client = FakeStructuredModelClient(
        {
            "statement": "Complexity may change the objective.",
            "rationale": "Bounded test.",
            "predicted_subspace": {"complexity": [3, 6]},
            "expected_observation": "objective_score changes",
            "falsification_condition": "objective_score does not change",
            "evidence_references": [],
            "confidence": 0.2,
        },
        {
            "search_type": "DIRECT",
            "target": "objective_score",
            "proposed_search_space": default_synthetic_configuration(),
            "requested_experiment_budget": 1,
            "rationale": "Would be valid if budget remained.",
            "recommends_human_approval": False,
        },
    )
    dependencies = task_memory_dependencies(
        SyntheticTask(),
        TaskRuntimeContext(run_id="call-limit", output_dir=tmp_path),
        contract,
        default_synthetic_configuration(),
        model_client=client,
        hypothesis_call_config=_call_config(),
        planner_call_config=_call_config(),
        agent_budget_policy=AgentBudgetPolicy(maximum_total_model_calls=1),
    )
    final = _invoke(dependencies, contract, "call-limit")
    assert final["status"] == RunStatus.FAILED
    assert final["errors"] == ["maximum_total_model_calls_reached"]
    assert len(client.calls) == 1
    assert final["budget"].model_calls_used == 1
