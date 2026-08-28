from __future__ import annotations

from datetime import UTC, datetime

from auto_researcher.agents.call_store import (
    InMemoryAgentCallStore,
    SQLiteAgentCallStore,
)
from auto_researcher.agents.live.research_director import LiveResearchDirectorAgent
from auto_researcher.agents.models import (
    AgentBudgetPolicy,
    ModelCallConfig,
    ModelPricing,
    ResearchDirective,
    ResearchLandscapeEvidence,
)
from auto_researcher.agents.research_director_policy import (
    next_research_director_trigger,
)
from auto_researcher.agents.research_director_shadow import (
    ResearchDirectorShadowPolicy,
    evaluate_shadow_directive,
)
from auto_researcher.contracts.enums import EventType, ProvenanceKind
from auto_researcher.contracts.models import DecisionEvent
from auto_researcher.contracts.models import SearchRequest
from auto_researcher.graph.nodes.planner import _apply_research_directive
from auto_researcher.contracts.enums import SearchType
from auto_researcher.contracts.models import BudgetState
from auto_researcher.graph.nodes.research_director import research_director_decide
from auto_researcher.graph.nodes.supervisor import supervisor_prepare
from auto_researcher.runtime.identity import payload_hash
from auto_researcher.runtime.dependencies import task_memory_dependencies
from auto_researcher.tasks.models import TaskRuntimeContext
from auto_researcher.tasks.synthetic import (
    SyntheticTask,
    default_synthetic_configuration,
    default_synthetic_contract,
)
from tests.fakes_agents import FakeStructuredModelClient

NOW = datetime(2026, 8, 22, 16, 0, tzinfo=UTC)


def _config() -> ModelCallConfig:
    return ModelCallConfig(
        provider="fake",
        model_id="claude-opus-5",
        temperature=None,
        thinking={"type": "adaptive"},
        effort="xhigh",
        maximum_output_tokens=64_000,
        timeout_seconds=600,
        maximum_attempts=2,
        maximum_cost_per_call=5.0,
        pricing=ModelPricing(
            version="opus-test-v1",
            input_cost_per_million_tokens=5,
            output_cost_per_million_tokens=25,
            currency="USD",
        ),
        prompt_version="2.0.0",
    )


def _proposal(contract_id: str) -> dict:
    return {
        "mechanism_hypothesis": "Local optimisation of complexity may improve score.",
        "rationale": "Distinguish a bounded mechanism before allocating deeper work.",
        "parent_references": [contract_id],
        "selected_operators": ["DIRECT"],
        "experiment_allocation": {"DIRECT": 1},
        "targeted_dimensions": ["complexity"],
        "expected_observation": "objective_score improves over the parent.",
        "falsification_condition": "objective_score does not improve over the parent.",
        "alternative_explanations": ["learning-rate interaction"],
        "evidence_references": [contract_id],
        "confidence": 0.6,
    }


def _state(contract):
    initial = {
        "run_id": "director-run",
        "thread_id": "director-thread",
        "contract": contract,
        "status": "RUNNING",
        "cycle": 0,
        "budget": BudgetState(
            maximum_cycles=4,
            maximum_experiments=4,
            maximum_cost=50,
        ),
        "decision_event_ids": [],
        "errors": [],
        "executed_nodes": [],
    }
    return {**initial, **supervisor_prepare(initial)}


def test_research_director_is_typed_checkpointed_and_replay_safe():
    contract = default_synthetic_contract(
        search_types=frozenset({SearchType.DIRECT}),
        maximum_experiments=4,
    )
    store = InMemoryAgentCallStore()
    client = FakeStructuredModelClient({}, _proposal(contract.contract_id))
    client.model_id = "claude-opus-5"
    dependencies = task_memory_dependencies(
        SyntheticTask(),
        TaskRuntimeContext(task_options={"campaign_finalisation_reserve_seconds": 60}),
        contract,
        default_synthetic_configuration(),
        clock=lambda: NOW,
    )
    director = LiveResearchDirectorAgent(
        client=client,
        call_config=_config(),
        budget_policy=AgentBudgetPolicy(
            maximum_input_context_size=128_000,
            maximum_research_director_output_tokens=64_000,
            maximum_research_director_cost_per_call=5.0,
        ),
        call_store=store,
        clock=lambda: NOW,
    )
    object.__setattr__(dependencies, "research_director_agent", director)
    state = _state(contract)

    first = research_director_decide(state, dependencies)
    directive = first["active_research_directive"]
    assert directive.trigger == "campaign_start"
    assert directive.selected_operators == (SearchType.DIRECT,)
    assert directive.targeted_dimensions == ("complexity",)
    assert len(client.calls) == 1

    replay_state = {
        **state,
        "budget": first["budget"],
        "active_research_directive": directive,
    }
    second = research_director_decide(replay_state, dependencies)
    assert second["research_director_trigger_history"] == ("campaign_start",)
    assert len(client.calls) == 1


def test_research_director_accepts_dict_restored_active_directive():
    contract = default_synthetic_contract(
        search_types=frozenset({SearchType.DIRECT}),
        maximum_experiments=4,
    )
    store = InMemoryAgentCallStore()
    client = FakeStructuredModelClient({}, _proposal(contract.contract_id))
    client.model_id = "claude-opus-5"
    dependencies = task_memory_dependencies(
        SyntheticTask(),
        TaskRuntimeContext(),
        contract,
        default_synthetic_configuration(),
        clock=lambda: NOW,
    )
    director = LiveResearchDirectorAgent(
        client=client,
        call_config=_config(),
        budget_policy=AgentBudgetPolicy(maximum_input_context_size=128_000),
        call_store=store,
        clock=lambda: NOW,
    )
    object.__setattr__(dependencies, "research_director_agent", director)
    state = _state(contract)
    first = research_director_decide(state, dependencies)
    directive = first["active_research_directive"]

    replay_state = {
        **state,
        "budget": first["budget"],
        "active_research_directive": directive.model_dump(mode="json"),
    }
    second = research_director_decide(replay_state, dependencies)

    assert second["research_director_trigger_history"] == ("campaign_start",)
    assert len(client.calls) == 1


def test_research_director_reuses_last_directive_after_transient_failure():
    contract = default_synthetic_contract(
        search_types=frozenset({SearchType.DIRECT}),
        maximum_experiments=4,
    )
    dependencies = task_memory_dependencies(
        SyntheticTask(),
        TaskRuntimeContext(),
        contract,
        default_synthetic_configuration(),
        clock=lambda: NOW,
    )

    class FailingDirector:
        def decide(self, _context):
            raise RuntimeError("provider details must not escape")

    object.__setattr__(dependencies, "research_director_agent", FailingDirector())
    prior_client = FakeStructuredModelClient({}, _proposal(contract.contract_id))
    prior_client.model_id = "claude-opus-5"
    prior = LiveResearchDirectorAgent(
        client=prior_client,
        call_config=_config(),
        budget_policy=AgentBudgetPolicy(
            maximum_input_context_size=128_000,
            maximum_research_director_output_tokens=64_000,
            maximum_research_director_cost_per_call=5.0,
        ),
        call_store=InMemoryAgentCallStore(),
        clock=lambda: NOW,
    )
    context = dependencies.agent_context_assembler.research_director_context(
        _state(contract),
        dependencies.task_agent_context,
        dependencies.search_capabilities,
        trigger="campaign_start",
        finalisation_reserve_seconds=0,
    )
    active = prior.decide(context)
    state = {
        **_state(contract),
        "active_research_directive": active,
        "research_director_trigger": "after_25",
    }
    update = research_director_decide(state, dependencies)
    assert "status" not in update
    assert update["research_director_failure_code"] == "research_director_failed"


def _verified_event(event_id: str, epochs: int, score: float) -> DecisionEvent:
    return DecisionEvent(
        event_id=event_id,
        run_id="director-run",
        cycle=0,
        event_type=EventType.EVIDENCE_VERIFIED,
        actor="verifier",
        output_references=(
            "evidence:SUPPORTED",
            "verified:true",
            "constraints:true",
            f"score:{score}",
        ),
        rationale="verified",
        timestamp=NOW,
        code_version="test",
        provenance=ProvenanceKind.REAL,
        safe_payload={"configuration": {"maximum_epochs": epochs}},
    )


def test_research_director_cadence_is_deterministic_and_bounded():
    events = [_verified_event("event-10", 10, 0.7)]
    assert next_research_director_trigger(events, ()) == "campaign_start"
    assert (
        next_research_director_trigger(events, ("campaign_start",))
        == "first_verified_10ep"
    )
    assert (
        next_research_director_trigger(
            events,
            ("campaign_start", "first_verified_10ep"),
        )
        is None
    )


def test_research_director_detects_one_replay_stable_score_stall():
    events = [
        _verified_event(f"event-{index}", 25, 0.75 + min(index, 2) * 0.001)
        for index in range(6)
    ]
    scheduled = ("campaign_start", "first_verified_25ep")
    trigger = next_research_director_trigger(events, scheduled)
    assert trigger == "score_stall:event-5"
    assert next_research_director_trigger(events, (*scheduled, trigger)) is None


def test_research_landscape_must_match_bound_manifest():
    contract = default_synthetic_contract(
        search_types=frozenset({SearchType.DIRECT}),
        maximum_experiments=4,
    )
    evidence = ResearchLandscapeEvidence(
        evidence_id="v7-final",
        evidence_type="V7",
        evidence_hash="a" * 64,
        source_reference="evidence:v7-final",
        summary="Verified V7 final evidence.",
        reference_ids=("experiment:v7-final",),
        safe_payload={"primary_score": 0.82},
    )
    payload = [evidence.model_dump(mode="json")]
    dependencies = task_memory_dependencies(
        SyntheticTask(),
        TaskRuntimeContext(
            task_options={
                "research_director_evidence": payload,
                "research_director_evidence_manifest_sha256": payload_hash(payload),
            }
        ),
        contract,
        default_synthetic_configuration(),
        clock=lambda: NOW,
    )
    context = dependencies.agent_context_assembler.research_director_context(
        _state(contract),
        dependencies.task_agent_context,
        dependencies.search_capabilities,
        trigger="campaign_start",
        finalisation_reserve_seconds=0,
    )
    assert context.research_landscape == (evidence,)


def test_research_director_shadow_report_is_hash_bound():
    contract = default_synthetic_contract(
        search_types=frozenset({SearchType.DIRECT}),
        maximum_experiments=4,
    )
    client = FakeStructuredModelClient({}, _proposal(contract.contract_id))
    client.model_id = "claude-opus-5"
    dependencies = task_memory_dependencies(
        SyntheticTask(),
        TaskRuntimeContext(),
        contract,
        default_synthetic_configuration(),
        clock=lambda: NOW,
    )
    agent = LiveResearchDirectorAgent(
        client=client,
        call_config=_config(),
        budget_policy=AgentBudgetPolicy(maximum_input_context_size=128_000),
        call_store=InMemoryAgentCallStore(),
        clock=lambda: NOW,
    )
    context = dependencies.agent_context_assembler.research_director_context(
        _state(contract),
        dependencies.task_agent_context,
        dependencies.search_capabilities,
        trigger="campaign_start",
        finalisation_reserve_seconds=0,
    )
    directive = agent.decide(context)
    report = evaluate_shadow_directive(
        directive,
        ResearchDirectorShadowPolicy(
            policy_id="v8-shadow-v1",
            allowed_operators=frozenset({SearchType.DIRECT}),
            allowed_dimensions=frozenset({"complexity"}),
            maximum_allocation_by_operator={SearchType.DIRECT: 1},
            maximum_total_allocation=1,
        ),
    )
    assert report.passed is True
    assert len(report.report_sha256) == 64


def test_research_director_sqlite_reopen_replays_without_provider_call(tmp_path):
    contract = default_synthetic_contract(
        search_types=frozenset({SearchType.DIRECT}),
        maximum_experiments=4,
    )
    dependencies = task_memory_dependencies(
        SyntheticTask(),
        TaskRuntimeContext(),
        contract,
        default_synthetic_configuration(),
        clock=lambda: NOW,
    )
    context = dependencies.agent_context_assembler.research_director_context(
        _state(contract),
        dependencies.task_agent_context,
        dependencies.search_capabilities,
        trigger="campaign_start",
        finalisation_reserve_seconds=0,
    )
    database = tmp_path / "agent-calls.sqlite"
    first_client = FakeStructuredModelClient({}, _proposal(contract.contract_id))
    first_client.model_id = "claude-opus-5"
    first_store = SQLiteAgentCallStore(database)
    first = LiveResearchDirectorAgent(
        client=first_client,
        call_config=_config(),
        budget_policy=AgentBudgetPolicy(maximum_input_context_size=128_000),
        call_store=first_store,
        clock=lambda: NOW,
    ).decide(context)
    first_store.close()

    replay_client = FakeStructuredModelClient({}, _proposal(contract.contract_id))
    replay_client.model_id = "claude-opus-5"
    replay_store = SQLiteAgentCallStore(database)
    replayed = LiveResearchDirectorAgent(
        client=replay_client,
        call_config=_config(),
        budget_policy=AgentBudgetPolicy(maximum_input_context_size=128_000),
        call_store=replay_store,
        clock=lambda: NOW,
    ).decide(context)
    replay_store.close()
    assert replayed == first
    assert replay_client.calls == []


def test_research_directive_is_projected_into_openevolve_mutation_context():
    contract = default_synthetic_contract(
        search_types=frozenset({SearchType.DIRECT, SearchType.OPENEVOLVE}),
        maximum_experiments=4,
    )
    client = FakeStructuredModelClient({}, _proposal(contract.contract_id))
    client.model_id = "claude-opus-5"
    dependencies = task_memory_dependencies(
        SyntheticTask(),
        TaskRuntimeContext(),
        contract,
        default_synthetic_configuration(),
        clock=lambda: NOW,
    )
    context = dependencies.agent_context_assembler.research_director_context(
        _state(contract),
        dependencies.task_agent_context,
        dependencies.search_capabilities,
        trigger="campaign_start",
        finalisation_reserve_seconds=0,
    )
    directive = LiveResearchDirectorAgent(
        client=client,
        call_config=_config(),
        budget_policy=AgentBudgetPolicy(maximum_input_context_size=128_000),
        call_store=InMemoryAgentCallStore(),
        clock=lambda: NOW,
    ).decide(context)
    request = SearchRequest(
        request_id="search-1",
        hypothesis_id="hypothesis-1",
        search_type=SearchType.OPENEVOLVE,
        target="objective_score",
        search_space={"campaign_context": {"locked": True}},
        experiment_budget=1,
        rationale="bounded mutation",
    )
    projected = _apply_research_directive(
        request,
        {**_state(contract), "active_research_directive": directive},
    )
    mutation_context = projected.search_space["campaign_context"]
    assert mutation_context["locked"] is True
    assert mutation_context["research_directive"]["directive_id"] == (
        directive.directive_id
    )
    assert f"research-directive:{directive.directive_id}" in (
        projected.evidence_references
    )
    projected_from_legacy_dict = _apply_research_directive(
        request,
        {
            **_state(contract),
            "active_research_directive": directive.model_dump(mode="json"),
        },
    )
    assert (
        projected_from_legacy_dict.search_space["campaign_context"]
        ["research_directive"]["directive_id"]
        == directive.directive_id
    )


def test_research_directive_projection_preserves_safe_metadata_only_context():
    contract = default_synthetic_contract(
        search_types=frozenset({SearchType.DIRECT, SearchType.OPENEVOLVE}),
        maximum_experiments=4,
    )
    state = _state(contract)
    directive = ResearchDirective(
        directive_id="directive-safe-projection",
        trigger="campaign_start",
        mechanism_hypothesis="A bounded structural mutation may improve Dice.",
        rationale="Keep the sealed holdout unchanged while screening candidates.",
        parent_references=(),
        selected_operators=(SearchType.OPENEVOLVE,),
        experiment_allocation={"OPENEVOLVE": 1},
        targeted_dimensions=("feature_width",),
        expected_observation=(
            "objective score improves across 14 validation subjects at the "
            "screening rung"
        ),
        falsification_condition="objective score does not improve",
        alternative_explanations=(),
        evidence_references=(
            "artifact:req11-panel:abc123",
            "checkpoint:abc123",
        ),
        confidence=0.7,
        agent_call_id="model-call-safe-projection",
        prompt_version="2.0.0",
        context_hash="context-safe-projection",
    )
    request = SearchRequest(
        request_id="search-safe-projection",
        hypothesis_id="hypothesis-1",
        search_type=SearchType.OPENEVOLVE,
        target="objective_score",
        search_space={"campaign_context": {"locked": True}},
        experiment_budget=1,
        rationale="bounded mutation",
    )

    projected = _apply_research_directive(
        request,
        {**state, "active_research_directive": directive},
    )

    context = projected.search_space["campaign_context"]["research_directive"]
    assert context["mechanism_hypothesis"] == directive.mechanism_hypothesis
    assert context["evidence_references"] == ["artifact:req11-panel:abc123"]
    assert "expected_observation" not in context
    assert "rationale" not in context
    assert f"research-directive:{directive.directive_id}" in (
        projected.evidence_references
    )
