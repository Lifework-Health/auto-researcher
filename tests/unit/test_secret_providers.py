from __future__ import annotations

import json
import logging
import pickle
import sys
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from auto_researcher.agents.models import AgentCallRecord, ModelCallConfig, ModelPricing
from auto_researcher.agents.prompts import load_prompt
from auto_researcher.contracts.enums import (
    AgentCallStatus,
    AgentRole,
    EventType,
    ProvenanceKind,
    SearchType,
)
from auto_researcher.contracts.models import ApprovalRequest, BudgetState, DecisionEvent
from auto_researcher.graph.nodes.supervisor import supervisor_prepare
from auto_researcher.providers.anthropic import create_anthropic_client
from auto_researcher.runtime.checkpoints import checkpoint_serializer
from auto_researcher.runtime.dependencies import task_memory_dependencies
from auto_researcher.runtime.execution import execution_identity
from auto_researcher.secrets import (
    EnvironmentSecretProvider,
    GoogleSecretManagerProvider,
    ResolvedSecret,
    SecretProviderKind,
    SecretReference,
    SecretResolutionError,
    SecretResolutionErrorCode,
    parse_secret_reference,
)
from auto_researcher.tasks.models import TaskRuntimeContext
from auto_researcher.tasks.synthetic import (
    SyntheticTask,
    default_synthetic_configuration,
    default_synthetic_contract,
)


SECRET = "test-secret-value-that-must-never-leak"


def _environment_reference(*, required: bool = True) -> SecretReference:
    return SecretReference(
        logical_name="anthropic_api_key",
        provider=SecretProviderKind.ENVIRONMENT,
        provider_identifier="ANTHROPIC_API_KEY",
        required=required,
    )


def _google_reference(
    *, version: str | None = None, required: bool = True
) -> SecretReference:
    return SecretReference(
        logical_name="anthropic_api_key",
        provider=SecretProviderKind.GOOGLE_SECRET_MANAGER,
        provider_identifier=("projects/auto-researcherv22/secrets/anthropic-api-key"),
        version=version,
        required=required,
    )


class FakeGoogleClient:
    def __init__(self, *, value: str = SECRET, error: Exception | None = None) -> None:
        self.value = value
        self.error = error
        self.requests: list[dict] = []
        self.timeouts: list[float] = []

    def access_secret_version(self, *, request: dict, timeout: float):
        self.requests.append(request)
        self.timeouts.append(timeout)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(payload=SimpleNamespace(data=self.value.encode()))


def _named_error(name: str) -> Exception:
    error_type = type(name, (Exception,), {})
    return error_type(f"provider accidentally echoed {SECRET}")


def _model_config() -> ModelCallConfig:
    return ModelCallConfig(
        provider="anthropic",
        model_id="explicit-model-2026-07-30",
        temperature=0,
        maximum_output_tokens=100,
        timeout_seconds=10,
        maximum_attempts=1,
        maximum_cost_per_call=0.1,
        pricing=ModelPricing(
            version="test-v1",
            input_cost_per_million_tokens=1,
            output_cost_per_million_tokens=2,
            currency="USD",
        ),
        prompt_version="2.0.0",
    )


def test_environment_provider_resolves_successfully():
    resolved = EnvironmentSecretProvider({"ANTHROPIC_API_KEY": SECRET}).resolve(
        _environment_reference()
    )
    assert resolved is not None
    assert resolved.reveal() == SECRET


def test_environment_provider_missing_required_fails_closed():
    with pytest.raises(SecretResolutionError) as caught:
        EnvironmentSecretProvider({}).resolve(_environment_reference())
    assert caught.value.code is SecretResolutionErrorCode.MISSING
    assert SECRET not in str(caught.value)


def test_environment_provider_missing_optional_returns_none():
    assert (
        EnvironmentSecretProvider({}).resolve(_environment_reference(required=False))
        is None
    )


def test_google_provider_uses_latest_and_explicit_versions():
    client = FakeGoogleClient()
    provider = GoogleSecretManagerProvider(client=client)

    latest = provider.resolve(_google_reference())
    explicit = provider.resolve(_google_reference(version="7"))

    assert latest is not None and latest.reveal() == SECRET
    assert explicit is not None and explicit.reveal() == SECRET
    assert client.requests == [
        {
            "name": "projects/auto-researcherv22/secrets/anthropic-api-key/versions/latest"
        },
        {"name": "projects/auto-researcherv22/secrets/anthropic-api-key/versions/7"},
    ]
    assert client.timeouts == [10.0, 10.0]


@pytest.mark.parametrize(
    ("exception_name", "code"),
    [
        ("DefaultCredentialsError", SecretResolutionErrorCode.AUTHENTICATION_FAILED),
        ("PermissionDenied", SecretResolutionErrorCode.PERMISSION_DENIED),
        ("FailedPrecondition", SecretResolutionErrorCode.API_UNAVAILABLE),
        ("NotFound", SecretResolutionErrorCode.NOT_FOUND),
        ("DeadlineExceeded", SecretResolutionErrorCode.TIMEOUT),
        ("ServiceUnavailable", SecretResolutionErrorCode.UNAVAILABLE),
    ],
)
def test_google_failures_are_bounded_and_never_leak(
    exception_name: str,
    code: SecretResolutionErrorCode,
):
    provider = GoogleSecretManagerProvider(
        client=FakeGoogleClient(error=_named_error(exception_name))
    )
    with pytest.raises(SecretResolutionError) as caught:
        provider.resolve(_google_reference())
    assert caught.value.code is code
    assert SECRET not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_google_authentication_failure_during_client_creation_is_safe():
    def fail_authentication():
        raise _named_error("DefaultCredentialsError")

    provider = GoogleSecretManagerProvider(client_factory=fail_authentication)
    with pytest.raises(SecretResolutionError) as caught:
        provider.resolve(_google_reference())
    assert caught.value.code is SecretResolutionErrorCode.AUTHENTICATION_FAILED
    assert SECRET not in repr(caught.value)


def test_google_dependency_remains_optional_and_failure_is_safe(monkeypatch):
    def missing_dependency(_name: str):
        raise ImportError(SECRET)

    monkeypatch.setattr(
        "auto_researcher.secrets.providers.import_module",
        missing_dependency,
    )
    with pytest.raises(SecretResolutionError) as caught:
        GoogleSecretManagerProvider().resolve(_google_reference())
    assert caught.value.code is SecretResolutionErrorCode.DEPENDENCY_UNAVAILABLE
    assert SECRET not in str(caught.value)


def test_google_service_disabled_reason_is_classified_without_reading_message():
    class PermissionDenied(Exception):
        reason = "SERVICE_DISABLED"

    provider = GoogleSecretManagerProvider(
        client=FakeGoogleClient(error=PermissionDenied(SECRET))
    )
    with pytest.raises(SecretResolutionError) as caught:
        provider.resolve(_google_reference())
    assert caught.value.code is SecretResolutionErrorCode.API_UNAVAILABLE
    assert SECRET not in str(caught.value)


def test_google_optional_not_found_returns_none():
    provider = GoogleSecretManagerProvider(
        client=FakeGoogleClient(error=_named_error("NotFound"))
    )
    assert provider.resolve(_google_reference(required=False)) is None


def test_reference_forbids_plaintext_and_resolved_value_is_nonserialisable():
    with pytest.raises(ValidationError):
        SecretReference.model_validate(
            {
                **_environment_reference().model_dump(mode="json"),
                "value": SECRET,
            }
        )
    resolved = ResolvedSecret(SECRET)
    assert SECRET not in repr(resolved)
    assert SECRET not in str(resolved)
    assert SECRET not in json.dumps({"credential": resolved}, default=str)
    with pytest.raises(TypeError, match="cannot be serialised"):
        pickle.dumps(resolved)


def test_untrusted_reference_parse_and_cli_rejections_do_not_echo_plaintext():
    with pytest.raises(ValueError) as reference_error:
        parse_secret_reference(
            {
                **_environment_reference().model_dump(mode="json"),
                "value": SECRET,
            }
        )
    assert SECRET not in str(reference_error.value)

    live_payload = {
        "agents": {
            "mode": "live",
            "provider": "anthropic",
            "model_id": "explicit-model-2026-07-30",
            "api_key": SECRET,
        }
    }
    from auto_researcher.cli import _load_live_agents

    with pytest.raises(ValueError) as cli_error:
        _load_live_agents(live_payload)
    assert SECRET not in str(cli_error.value)


def test_secret_does_not_leak_through_logs_or_safe_domain_serialisation(caplog):
    reference = _google_reference(version="9")
    resolved = ResolvedSecret(SECRET)
    with caplog.at_level(logging.DEBUG):
        logging.getLogger("test.secrets").debug(
            "provider=%r reference=%r credential=%r",
            GoogleSecretManagerProvider(client=FakeGoogleClient()),
            reference,
            resolved,
        )
    event = DecisionEvent(
        event_id="event-1",
        run_id="run-1",
        cycle=0,
        event_type=EventType.RUN_INITIALISED,
        actor="runtime",
        rationale="credential identity is runtime-only",
        timestamp=datetime(2026, 8, 13, tzinfo=UTC),
        code_version="test",
        provenance=ProvenanceKind.REAL,
    )
    approval = ApprovalRequest(
        request_id="approval-1",
        run_id="run-1",
        cycle=1,
        search_request_id="search-1",
        search_type=SearchType.DIRECT,
        target="objective_score",
        rationale="bounded approval",
    )
    call_record = AgentCallRecord(
        record_id="call-1:1:reserved",
        call_id="call-1",
        run_id="run-1",
        cycle=1,
        role=AgentRole.HYPOTHESIS,
        provider="anthropic",
        model_id="explicit-model-2026-07-30",
        prompt_name="hypothesis",
        prompt_version="2.0.0",
        prompt_hash="prompt-hash",
        context_hash="context-hash",
        response_schema_version="schema-hash",
        status=AgentCallStatus.RESERVED,
        pricing=_model_config().pricing,
        pricing_version="test-v1",
        pricing_currency="USD",
        created_at=datetime(2026, 8, 13, tzinfo=UTC),
    )
    contract = default_synthetic_contract(1)
    dependencies = task_memory_dependencies(
        SyntheticTask(),
        TaskRuntimeContext(),
        contract,
        default_synthetic_configuration(),
    )
    initial_state = {
        "run_id": "run-1",
        "thread_id": "thread-1",
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
    state = {**initial_state, **supervisor_prepare(initial_state)}
    agent_context = dependencies.agent_context_assembler.hypothesis_context(
        state,
        dependencies.task_agent_context,
    )
    system_prompt, user_prompt = load_prompt("hypothesis", "2.0.0").render(
        context_json=agent_context.model_dump_json()
    )
    serialised = json.dumps(
        {
            "task_configuration": {"credential": reference.model_dump(mode="json")},
            "research_contract": contract.model_dump(mode="json"),
            "provenance": event.model_dump(mode="json"),
            "model_call_record": call_record.model_dump(mode="json"),
            "model_call_configuration": _model_config().model_dump(mode="json"),
            "approval": approval.model_dump(mode="json"),
            "checkpoint": {
                "run_id": "run-1",
                "thread_id": "thread-1",
                "contract": contract.model_dump(mode="json"),
                "pending_human_request": approval.model_dump(mode="json"),
            },
            "model_context": agent_context.model_dump(mode="json"),
            "model_prompts": [system_prompt, user_prompt],
        },
        sort_keys=True,
        default=str,
    )
    assert SECRET not in caplog.text
    assert SECRET not in serialised
    assert "reveal" not in serialised


def test_resolved_value_is_rejected_by_checkpoint_serializer():
    with pytest.raises(TypeError) as caught:
        checkpoint_serializer().dumps_typed(ResolvedSecret(SECRET))
    assert SECRET not in str(caught.value)


def test_environment_and_version_rotation_do_not_change_scientific_identity():
    environment_reference = _environment_reference()
    first = EnvironmentSecretProvider({"ANTHROPIC_API_KEY": "rotated-one"})
    second = EnvironmentSecretProvider({"ANTHROPIC_API_KEY": "rotated-two"})
    assert (
        first.resolve(environment_reference).reveal()
        != second.resolve(environment_reference).reveal()
    )
    version_seven = _google_reference(version="7")
    version_eight = _google_reference(version="8")
    google = GoogleSecretManagerProvider(client=FakeGoogleClient())
    google.resolve(version_seven)
    google.resolve(version_eight)
    initial = {
        "run_id": "run-1",
        "thread_id": "thread-1",
        "contract": default_synthetic_contract(1),
    }
    configuration = {"configurable": {"thread_id": "thread-1"}}
    environment_identity = execution_identity(initial, configuration)
    version_seven_identity = execution_identity(initial, configuration)
    version_eight_identity = execution_identity(initial, configuration)
    assert environment_identity == version_seven_identity == version_eight_identity


def test_existing_anthropic_environment_loading_remains_compatible(
    monkeypatch,
):
    captured: dict = {}

    class FakeChatAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    monkeypatch.setitem(
        sys.modules,
        "langchain_anthropic",
        SimpleNamespace(ChatAnthropic=FakeChatAnthropic),
    )
    client = create_anthropic_client(_model_config())
    assert client.provider == "anthropic"
    assert captured["api_key"].get_secret_value() == SECRET
    assert SECRET not in repr(captured["api_key"])
    assert SECRET not in repr(client)


def test_runtime_uses_configured_google_reference_at_anthropic_boundary(monkeypatch):
    import auto_researcher.secrets as secrets_module
    from auto_researcher.cli import _load_live_agents

    client = FakeGoogleClient()
    captured: list[dict] = []

    class FakeChatAnthropic:
        def __init__(self, **kwargs):
            captured.append(kwargs)

    monkeypatch.setattr(
        secrets_module,
        "provider_for_reference",
        lambda _reference: GoogleSecretManagerProvider(client=client),
    )
    monkeypatch.setitem(
        sys.modules,
        "langchain_anthropic",
        SimpleNamespace(ChatAnthropic=FakeChatAnthropic),
    )
    role = {
        "maximum_output_tokens": 100,
        "timeout_seconds": 10,
        "maximum_attempts": 1,
        "maximum_cost_per_call": 0.1,
    }
    loaded = _load_live_agents(
        {
            "agents": {
                "mode": "live",
                "provider": "anthropic",
                "model_id": "explicit-model-2026-07-30",
                "credential": _google_reference(version="11").model_dump(mode="json"),
                "pricing": {
                    "version": "test-v1",
                    "input_cost_per_million_tokens": 1,
                    "output_cost_per_million_tokens": 2,
                    "currency": "USD",
                },
                "hypothesis": role,
                "planner": role,
            }
        }
    )
    assert loaded[-1] == "live"
    assert len(captured) == 2
    assert all(item["api_key"].get_secret_value() == SECRET for item in captured)
    assert client.requests == [
        {"name": "projects/auto-researcherv22/secrets/anthropic-api-key/versions/11"},
        {"name": "projects/auto-researcherv22/secrets/anthropic-api-key/versions/11"},
    ]


def test_anthropic_initialisation_error_never_echoes_resolved_value(monkeypatch):
    class FailingChatAnthropic:
        def __init__(self, **kwargs):
            raise ValueError(kwargs["api_key"].get_secret_value())

    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    monkeypatch.setitem(
        sys.modules,
        "langchain_anthropic",
        SimpleNamespace(ChatAnthropic=FailingChatAnthropic),
    )
    with pytest.raises(RuntimeError) as caught:
        create_anthropic_client(_model_config())
    assert SECRET not in str(caught.value)
    assert caught.value.__context__ is None
