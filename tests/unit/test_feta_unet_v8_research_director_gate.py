from __future__ import annotations

from datetime import UTC, datetime

from auto_researcher.agents.models import (
    AgentBudgetPolicy,
    ModelCallConfig,
    ModelPricing,
    ResearchDirectorContext,
    TaskAgentContext,
)
from auto_researcher.agents.context import stable_context_hash
from auto_researcher.contracts.enums import SearchType
from auto_researcher.runtime.identity import payload_hash
from auto_researcher.tasks.feta_unet_search.v8_research_director_gate import (
    decide_shadow_and_replay,
)
from auto_researcher.tasks.feta_unet_search.v8_preflight import (
    _research_director_gate_valid,
)
from tests.fakes_agents import FakeStructuredModelClient

NOW = datetime(2026, 8, 23, 12, 0, tzinfo=UTC)


def _call_config() -> ModelCallConfig:
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
            version="test",
            input_cost_per_million_tokens=5,
            output_cost_per_million_tokens=25,
            currency="USD",
        ),
        prompt_version="2.0.0",
    )


def _context() -> ResearchDirectorContext:
    task = TaskAgentContext(
        task_id="feta_unet_search",
        task_version="1.0",
        display_name="V8 test",
        domain="segmentation",
        task_description="Bounded V8 search.",
        safe_scientific_vocabulary=("macro Dice",),
        primary_metric_description="Mean subject macro Dice.",
        scientific_constraint_summary=("holdout sealed",),
        dataset_summary={"holdout_subjects_evaluated": 0},
        available_search_types=(
            SearchType.DIRECT,
            SearchType.OPTUNA,
            SearchType.OPENEVOLVE,
        ),
        direct_configuration_schema={"model_variant": ["dynunet"]},
        optuna_space_summary={"learning_rate": [1e-5, 1e-3]},
        openevolve_space_summary={"mutable_policy": {"kernel_profile": ["standard"]}},
        fixed_scientific_context={"fold": 0},
        task_limitations=("no holdout",),
        safety_notes=("metadata only",),
    )
    payload = {
        "run_id": "v8-director-gate-test",
        "contract": {
            "contract_id": "v8-contract",
            "task_id": "feta_unet_search",
            "task_version": "1.0",
            "objective_version": "v8-objective",
            "question": "Which bounded mechanism should V8 test next?",
            "objective": "Improve the development macro Dice.",
            "primary_metric": "mean_subject_macro_dice",
            "maximum_experiments": 128,
            "allowed_search_types": ["DIRECT", "OPTUNA", "OPENEVOLVE"],
            "requires_approval_for": [],
            "constraints": {"holdout_policy": "sealed-no-evaluation"},
        },
        "task": task,
        "cycle": 1,
        "trigger": "campaign_start",
        "installed_search_capabilities": (
            SearchType.DIRECT,
            SearchType.OPTUNA,
            SearchType.OPENEVOLVE,
        ),
        "remaining_experiment_budget": 128,
        "remaining_cost_budget": 150.0,
        "remaining_time_seconds": 115_200.0,
        "model_calls_used": 0,
        "permitted_evidence_reference_ids": ("v8-contract",),
        "permitted_target_dimensions": (
            "kernel_profile",
            "learning_rate",
            "model_variant",
        ),
        "finalisation_reserve_seconds": 21_600.0,
    }
    serialisable = {
        key: (
            value.model_dump(mode="json")
            if hasattr(value, "model_dump")
            else [getattr(item, "value", item) for item in value]
            if isinstance(value, tuple)
            else value
        )
        for key, value in payload.items()
    }
    return ResearchDirectorContext(
        **payload,
        context_hash=stable_context_hash(serialisable),
    )


def test_v8_director_gate_calls_once_and_replays_without_provider(tmp_path):
    context = _context()
    client = FakeStructuredModelClient(
        {},
        {
            "mechanism_hypothesis": "Refine promising bounded mechanisms.",
            "rationale": "Use complementary operators within the frozen envelope.",
            "parent_references": ["v8-contract"],
            "selected_operators": ["DIRECT", "OPTUNA", "OPENEVOLVE"],
            "experiment_allocation": {"DIRECT": 8, "OPTUNA": 26, "OPENEVOLVE": 10},
            "targeted_dimensions": [
                "kernel_profile",
                "learning_rate",
                "model_variant",
            ],
            "expected_observation": "mean subject macro dice improves.",
            "falsification_condition": "mean subject macro dice does not improve.",
            "alternative_explanations": ["optimisation variance"],
            "evidence_references": ["v8-contract"],
            "confidence": 0.7,
        },
    )
    client.model_id = "claude-opus-5"
    directive, shadow, first, replay, records = decide_shadow_and_replay(
        context=context,
        client=client,
        call_config=_call_config(),
        budget_policy=AgentBudgetPolicy(
            maximum_input_context_size=128_000,
            maximum_research_director_output_tokens=64_000,
            maximum_research_director_cost_per_call=5.0,
        ),
        agent_calls_path=tmp_path / "calls.sqlite",
        clock=lambda: NOW,
    )
    assert shadow.passed is True
    assert shadow.total_allocation == 44
    assert first["replayed"] is False
    assert replay["replayed"] is True
    assert len(client.calls) == 1
    assert len(records) == 2
    assert directive.context_hash == context.context_hash


def test_v8_director_gate_rejects_allocation_outside_locked_envelope(tmp_path):
    context = _context()
    client = FakeStructuredModelClient(
        {},
        {
            "mechanism_hypothesis": "Overallocate Direct candidates.",
            "rationale": "This should be rejected by the shadow policy.",
            "parent_references": ["v8-contract"],
            "selected_operators": ["DIRECT"],
            "experiment_allocation": {"DIRECT": 9},
            "targeted_dimensions": ["model_variant"],
            "expected_observation": "mean subject macro dice improves.",
            "falsification_condition": "mean subject macro dice does not improve.",
            "alternative_explanations": [],
            "evidence_references": ["v8-contract"],
            "confidence": 0.4,
        },
    )
    client.model_id = "claude-opus-5"
    _directive, shadow, _first, replay, _records = decide_shadow_and_replay(
        context=context,
        client=client,
        call_config=_call_config(),
        budget_policy=AgentBudgetPolicy(maximum_input_context_size=128_000),
        agent_calls_path=tmp_path / "calls.sqlite",
        clock=lambda: NOW,
    )
    assert shadow.passed is False
    assert "operator_allocation_exceeds_locked_envelope" in shadow.violations
    assert replay["replayed"] is True


def _bound_gate_report() -> dict:
    context_hash = "a" * 64
    directive = {"experiment_allocation": {"DIRECT": 8, "OPTUNA": 26, "OPENEVOLVE": 10}}
    shadow_base = {
        "policy_id": "feta-unet-v8-operator-envelope-v1",
        "directive_id": "directive-v8",
        "passed": True,
        "violations": [],
        "total_allocation": 44,
    }
    telemetry = {
        "call_id": "model-call-v8",
        "role": "RESEARCH_DIRECTOR",
        "provider": "anthropic",
        "model_id": "claude-opus-5",
        "context_hash": context_hash,
        "provider_attempts": 1,
        "failed": False,
    }
    base = {
        "schema_version": "feta-unet-v8-research-director-gate-v1",
        "research_director_evidence_manifest_sha256": "b" * 64,
        "model": {
            "provider": "anthropic",
            "model_id": "claude-opus-5",
            "thinking": {"type": "adaptive"},
            "effort": "xhigh",
        },
        "context_hash": context_hash,
        "operator_limits": {"DIRECT": 8, "OPTUNA": 26, "OPENEVOLVE": 10},
        "maximum_total_allocation": 44,
        "directive": directive,
        "shadow_report": {
            **shadow_base,
            "report_sha256": payload_hash(shadow_base),
        },
        "first_call_telemetry": {**telemetry, "replayed": False},
        "replay_telemetry": {**telemetry, "replayed": True},
        "durable_call_records": [
            {
                "provider": "anthropic",
                "model_id": "claude-opus-5",
                "status": "RESERVED",
            },
            {
                "provider": "anthropic",
                "model_id": "claude-opus-5",
                "status": "COMPLETED",
            },
        ],
        "replay_provider_calls": 0,
        "directive_replayed_exactly": True,
        "experiments_dispatched": 0,
        "holdout_subjects_evaluated": 0,
        "passed": True,
    }
    return {**base, "report_sha256": payload_hash(base)}


def test_v8_preflight_accepts_only_hash_bound_live_director_gate():
    report = _bound_gate_report()
    options = {
        "research_director_evidence_manifest_sha256": "b" * 64,
        "research_director_gate": report,
        "research_director_gate_sha256": payload_hash(report),
    }
    assert _research_director_gate_valid(options) is True

    options["research_director_gate"]["replay_provider_calls"] = 1
    assert _research_director_gate_valid(options) is False
