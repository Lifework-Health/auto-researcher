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
from auto_researcher.contracts.models import (
    ApprovalRequest,
    BudgetState,
    DecisionEvent,
    ExperimentSpec,
    SearchRequest,
)
from auto_researcher.graph.nodes.evaluate import _evaluation_identity
from auto_researcher.graph.nodes.supervisor import supervisor_prepare
from auto_researcher.providers.anthropic import create_anthropic_client
from auto_researcher.research_state import SQLiteResearchStateStore
from auto_researcher.runtime.checkpoints import checkpoint_serializer
from auto_researcher.runtime.dependencies import task_memory_dependencies
from auto_researcher.runtime.execution import execution_identity
from auto_researcher.runtime.identity import payload_hash
from auto_researcher.resources import (
    CourtesyResourceAdmissionPolicy,
    ResourceBroker,
    ResourceCandidate,
    ResourceRequest,
    ResourceRequirement,
)
from auto_researcher.search.optuna.naming import build_study_identity
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
from tests.unit.test_research_state import (
    _cards,
    _decision,
    _experiment,
    _external,
    _hypotheses,
    _inference,
    _observation,
    _programme,
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
    def __init__(
        self,
        *,
        value: object = SECRET.encode(),
        values: list[object] | None = None,
        error: Exception | None = None,
        response: object | None = None,
    ) -> None:
        self.value = value
        self.values = values
        self.error = error
        self.response = response
        self.requests: list[dict] = []
        self.timeouts: list[float] = []

    def access_secret_version(self, *, request: dict, timeout: float):
        self.requests.append(request)
        self.timeouts.append(timeout)
        if self.error is not None:
            raise self.error
        if self.response is not None:
            return self.response
        value = self.values.pop(0) if self.values is not None else self.value
        return SimpleNamespace(payload=SimpleNamespace(data=value))


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


def test_environment_provider_never_falls_back_to_logical_name():
    invalid = SecretReference.model_construct(
        logical_name="ANTHROPIC_API_KEY",
        provider=SecretProviderKind.ENVIRONMENT,
        provider_identifier=None,
        version=None,
        required=True,
    )
    with pytest.raises(SecretResolutionError) as caught:
        EnvironmentSecretProvider({"ANTHROPIC_API_KEY": SECRET}).resolve(invalid)
    assert caught.value.code is SecretResolutionErrorCode.INVALID_REFERENCE
    assert SECRET not in str(caught.value)


@pytest.mark.parametrize(
    "payload",
    [
        {
            "logical_name": "anthropic_api_key",
            "provider": "environment",
        },
        {
            "logical_name": "anthropic_api_key",
            "provider": "google_secret_manager",
        },
        {
            "logical_name": "anthropic_api_key",
            "provider": "google_secret_manager",
            "provider_identifier": "anthropic-api-key",
        },
    ],
)
def test_standard_references_require_explicit_unambiguous_identifiers(payload):
    with pytest.raises(ValueError, match="secret_reference_invalid") as caught:
        parse_secret_reference(payload)
    assert SECRET not in str(caught.value)


@pytest.mark.parametrize("version", [None, "7"])
def test_every_accepted_google_reference_forms_exact_resource(monkeypatch, version):
    import auto_researcher.secrets.providers as provider_module

    client = FakeGoogleClient()
    monkeypatch.setattr(provider_module, "_google_client_factory", lambda: client)
    reference = parse_secret_reference(
        _google_reference(version=version).model_dump(mode="json")
    )

    resolved = provider_module.provider_for_reference(reference).resolve(reference)

    assert resolved is not None
    assert client.requests == [
        {
            "name": "projects/auto-researcherv22/secrets/"
            f"anthropic-api-key/versions/{version or 'latest'}"
        }
    ]


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
    "client",
    [
        FakeGoogleClient(value=b""),
        FakeGoogleClient(value=b"\xff\xfe"),
        FakeGoogleClient(value=object()),
        FakeGoogleClient(response=SimpleNamespace()),
    ],
    ids=("empty", "non-utf8", "non-bytes", "malformed"),
)
def test_google_invalid_payloads_fail_closed_as_invalid_value(client):
    with pytest.raises(SecretResolutionError) as caught:
        GoogleSecretManagerProvider(client=client).resolve(_google_reference())
    assert caught.value.code is SecretResolutionErrorCode.INVALID_VALUE
    assert SECRET not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_google_invalid_optional_payload_is_still_an_invalid_value_error():
    with pytest.raises(SecretResolutionError) as caught:
        GoogleSecretManagerProvider(client=FakeGoogleClient(value=b"\xff")).resolve(
            _google_reference(required=False)
        )
    assert caught.value.code is SecretResolutionErrorCode.INVALID_VALUE


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


def test_live_anthropic_rejects_optional_credentials_before_resolution():
    from auto_researcher.cli import _load_live_agents

    role = {
        "maximum_output_tokens": 100,
        "timeout_seconds": 10,
        "maximum_attempts": 1,
        "maximum_cost_per_call": 0.1,
    }
    with pytest.raises(ValueError, match="credentials must be required"):
        _load_live_agents(
            {
                "agents": {
                    "mode": "live",
                    "provider": "anthropic",
                    "model_id": "explicit-model-2026-07-30",
                    "credential": _environment_reference(required=False).model_dump(
                        mode="json"
                    ),
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


def test_secret_does_not_leak_through_current_architecture_models(caplog, tmp_path):
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
    programme = _programme()
    card = _cards()[0]
    external = _external(programme, card)
    observation = _observation(programme)
    left, right, _, _ = _hypotheses(programme, card.evidence_id)
    experiment = _experiment(programme, card.evidence_id)
    inference = _inference(programme, card.evidence_id)
    decision = _decision(programme, card.evidence_id)
    state_path = tmp_path / "research-state.sqlite3"
    research_store = SQLiteResearchStateStore(state_path)
    research_store.create_programme(programme)
    research_store.append_many(
        (external, observation, left, right, experiment, inference, decision)
    )
    research_state = research_store.load_state(programme.programme_id)
    research_store.close()

    resource_request = ResourceRequest(
        request_id="secret-boundary-resource-check",
        requirements=(ResourceRequirement(resource_type="gpu"),),
    )
    resource_candidate = ResourceCandidate(
        resource_id="gpu-0",
        resource_type="gpu",
    )

    class FixedResourceProvider:
        def candidates(self, _request):
            return (resource_candidate,)

    resource_admission = ResourceBroker(
        FixedResourceProvider(),
        CourtesyResourceAdmissionPolicy(),
        clock=lambda: 0.0,
    ).wait_for_admission(resource_request)

    search_request = SearchRequest(
        request_id="search-secret-boundary-check",
        hypothesis_id="hypothesis-1",
        search_type=SearchType.DIRECT,
        target="objective_score",
        search_space={},
        experiment_budget=1,
        rationale="Exercise the real search contract boundary.",
    )
    experiment_spec = ExperimentSpec(
        experiment_id="experiment-secret-boundary-check",
        hypothesis_id=search_request.hypothesis_id,
        search_request_id=search_request.request_id,
        configuration=default_synthetic_configuration(),
        evaluator_id="synthetic-evaluator",
        code_version="test-code-v1",
        dataset_version="synthetic-data-v1",
        provenance=ProvenanceKind.SIMULATED,
    )

    for model in (
        contract,
        search_request,
        experiment_spec,
        event,
        approval,
        call_record,
        _model_config(),
        TaskRuntimeContext(),
        agent_context,
        card,
        research_state,
        resource_request,
        resource_candidate,
        resource_admission,
    ):
        with pytest.raises(ValidationError) as caught:
            type(model).model_validate(
                {**model.model_dump(mode="python"), "credential": resolved}
            )
        assert SECRET not in str(caught.value)

    serialised = json.dumps(
        {
            "task_configuration": {"credential": reference.model_dump(mode="json")},
            "research_contract": contract.model_dump(mode="json"),
            "search_request": search_request.model_dump(mode="json"),
            "experiment_spec": experiment_spec.model_dump(mode="json"),
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
            "research_intelligence_card": card.model_dump(mode="json"),
            "research_state": research_state.model_dump(mode="json"),
            "resource_admission": resource_admission.model_dump(mode="json"),
        },
        sort_keys=True,
        default=str,
    )
    assert SECRET not in caplog.text
    assert SECRET not in serialised
    assert SECRET.encode() not in state_path.read_bytes()
    assert "reveal" not in serialised


def test_resolved_value_is_rejected_by_checkpoint_serializer():
    with pytest.raises(TypeError) as caught:
        checkpoint_serializer().dumps_typed(ResolvedSecret(SECRET))
    assert SECRET not in str(caught.value)


def test_secret_rotation_does_not_change_any_scientific_or_reuse_identity():
    environment_reference = _environment_reference()
    version_seven = _google_reference(version="7")
    version_eight = _google_reference(version="8")
    operational_credentials = (
        (
            environment_reference,
            EnvironmentSecretProvider({"ANTHROPIC_API_KEY": "rotated-env"}).resolve(
                environment_reference
            ),
        ),
        (
            version_seven,
            GoogleSecretManagerProvider(
                client=FakeGoogleClient(value=b"rotated-gsm-seven")
            ).resolve(version_seven),
        ),
        (
            version_eight,
            GoogleSecretManagerProvider(
                client=FakeGoogleClient(value=b"rotated-gsm-eight")
            ).resolve(version_eight),
        ),
    )
    assert (
        len({reference.model_dump_json() for reference, _ in operational_credentials})
        == 3
    )
    assert len({credential.reveal() for _, credential in operational_credentials}) == 3

    contract = default_synthetic_contract(1)
    request = SearchRequest(
        request_id="search-rotation-invariant",
        hypothesis_id="hypothesis-1",
        search_type=SearchType.OPTUNA,
        target="objective_score",
        search_space={},
        experiment_budget=2,
        rationale="Test the bounded synthetic space.",
    )
    task = SyntheticTask()
    runtime_context = TaskRuntimeContext(
        run_id="run-1",
        manifest_created_at=datetime(2026, 8, 13, tzinfo=UTC),
    )
    metadata = task.experiment_metadata(runtime_context)
    study_spec = task.create_optuna_study_spec(contract, request)
    experiment = ExperimentSpec(
        experiment_id="experiment-1",
        hypothesis_id=request.hypothesis_id,
        search_request_id=request.request_id,
        configuration=default_synthetic_configuration(),
        evaluator_id=metadata.evaluator_id,
        code_version=metadata.code_version,
        dataset_version=metadata.dataset_version,
        provenance=metadata.provenance,
    )
    dependencies = task_memory_dependencies(
        task,
        runtime_context,
        contract,
        default_synthetic_configuration(),
    )
    initial = {
        "run_id": "run-1",
        "thread_id": "thread-1",
        "contract": contract,
    }
    configuration = {"configurable": {"thread_id": "thread-1"}}

    identities = []
    for _reference, credential in operational_credentials:
        assert credential is not None
        identities.append(
            (
                payload_hash(contract),
                execution_identity(initial, configuration),
                build_study_identity(
                    run_id="run-1",
                    contract=contract,
                    request=request,
                    metadata=metadata,
                    spec=study_spec,
                ),
                payload_hash(experiment),
                _evaluation_identity(
                    {"run_id": "run-1", "experiment_spec": experiment},
                    dependencies,
                ),
            )
        )

    assert identities[0] == identities[1] == identities[2]


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


def test_runtime_resolves_once_per_assembly_and_refreshes_on_next_assembly(monkeypatch):
    import auto_researcher.secrets as secrets_module
    from auto_researcher.cli import _load_live_agents

    client = FakeGoogleClient(values=[b"assembly-one", b"assembly-two"])
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
    payload = {
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
    first = _load_live_agents(payload)
    second = _load_live_agents(payload)

    assert first[-1] == second[-1] == "live"
    assert [item["api_key"].get_secret_value() for item in captured] == [
        "assembly-one",
        "assembly-one",
        "assembly-two",
        "assembly-two",
    ]
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
