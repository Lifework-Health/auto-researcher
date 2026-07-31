from __future__ import annotations

import pytest

from auto_researcher.contracts.enums import KnowledgeRetrievalStatus
from auto_researcher.contracts.models import KnowledgeGroundingRequirement
from auto_researcher.knowledge.identity import content_hash, query_plan_hash
from auto_researcher.knowledge.models import (
    KnowledgeErrorCode,
    KnowledgeProviderConfiguration,
    KnowledgeRetrievalRecord,
    KnowledgeRetrievalRequest,
)
from auto_researcher.knowledge.providers.static import StaticKnowledgeProvider
from auto_researcher.knowledge.runtime import (
    KnowledgeRetrievalCoordinator,
    KnowledgeRetrievalExecutionError,
)
from auto_researcher.knowledge.store import (
    InMemoryKnowledgeRetrievalStore,
    SQLiteKnowledgeRetrievalStore,
)
from auto_researcher.knowledge.validation import KnowledgeBundleValidator
from auto_researcher.knowledge.templates import default_template_registry
from auto_researcher.tasks.models import TaskRuntimeContext
from auto_researcher.tasks.synthetic import default_synthetic_contract
from auto_researcher.tasks.synthetic.knowledge import (
    synthetic_grounding_policy,
    synthetic_query_plan,
)
from tests.conftest import fixed_clock


def _runtime(tmp_path, store=None):
    requirement = KnowledgeGroundingRequirement(
        mode="OPTIONAL",
        permitted_providers=frozenset({"static"}),
        maximum_query_records=10,
        knowledge_schema_version="synthetic-v1",
        knowledge_content_version="fixture-v1",
    )
    contract = default_synthetic_contract().model_copy(
        update={"grounding": requirement}
    )
    plan = synthetic_query_plan(contract, TaskRuntimeContext(), {})
    configuration = KnowledgeProviderConfiguration(
        provider_id="static",
        graph_alias="fixture",
        database="static",
        schema_version="synthetic-v1",
        content_version="fixture-v1",
        maximum_records=10,
    )
    policy = synthetic_grounding_policy(contract)
    request = KnowledgeRetrievalRequest(
        retrieval_id="retrieval-replay",
        run_id="run-replay",
        cycle=1,
        provider_id="static",
        graph_alias="fixture",
        schema_version="synthetic-v1",
        content_version="fixture-v1",
        query_plan=plan,
        query_plan_hash=query_plan_hash(plan),
        grounding_policy_hash=content_hash(policy),
        template_hashes={
            "generic.entity_lookup@1.0.0": default_template_registry()
            .get("generic.entity_lookup", "1.0.0")
            .cypher_sha256
        },
        task_id="synthetic",
        contract_id=contract.contract_id,
    )
    selected_store = store or InMemoryKnowledgeRetrievalStore()
    coordinator = KnowledgeRetrievalCoordinator(
        store=selected_store,
        validator=KnowledgeBundleValidator(),
        runtime_context=TaskRuntimeContext(
            run_id=request.run_id,
            output_dir=tmp_path,
        ),
        clock=fixed_clock,
    )
    provider = StaticKnowledgeProvider(configuration, clock=fixed_clock)
    return (
        coordinator,
        selected_store,
        provider,
        configuration,
        request,
        policy,
    )


def test_completed_bundle_replay_never_calls_provider_again(tmp_path):
    coordinator, store, provider, config, request, policy = _runtime(tmp_path)
    first, replayed = coordinator.run(request, provider, config, policy)
    second, replayed_again = coordinator.run(request, provider, config, policy)
    assert not replayed
    assert replayed_again
    assert first == second
    assert provider.calls == 1
    assert [
        item.status for item in store.records_for_retrieval(request.retrieval_id)
    ] == [
        KnowledgeRetrievalStatus.RESERVED,
        KnowledgeRetrievalStatus.COMPLETED,
    ]
    artefact_dir = (
        tmp_path / "runs" / request.run_id / "knowledge" / request.retrieval_id
    )
    assert {item.name for item in artefact_dir.iterdir()} == {
        "retrieval_request.json",
        "query_plan.json",
        "graph_snapshot.json",
        "knowledge_bundle.json",
        "validation_summary.json",
    }


def test_reserved_external_read_requires_explicit_retry(tmp_path):
    coordinator, store, provider, config, request, policy = _runtime(tmp_path)
    store.append(
        KnowledgeRetrievalRecord(
            record_id="reserved-before-crash",
            retrieval_id=request.retrieval_id,
            run_id=request.run_id,
            cycle=1,
            status=KnowledgeRetrievalStatus.RESERVED,
            request=request,
            provider_request_started=True,
            created_at=fixed_clock(),
        )
    )
    try:
        coordinator.run(request, provider, config, policy)
    except KnowledgeRetrievalExecutionError as exc:
        assert exc.code == KnowledgeErrorCode.RETRIEVAL_INDETERMINATE.value
    else:
        raise AssertionError("an ambiguous external read must not be replayed")
    assert provider.calls == 0
    assert (
        store.latest(request.retrieval_id).status
        == KnowledgeRetrievalStatus.INDETERMINATE
    )
    retry = store.create_retry(request.retrieval_id, created_at=fixed_clock())
    bundle, replayed = coordinator.run(request, provider, config, policy)
    assert not replayed
    assert provider.calls == 1
    assert bundle.retrieval_id == retry.retrieval_id
    assert store.latest(retry.retrieval_id).status == KnowledgeRetrievalStatus.COMPLETED


def test_sqlite_store_is_append_only_and_retry_is_idempotent(tmp_path):
    path = tmp_path / "knowledge.sqlite"
    store = SQLiteKnowledgeRetrievalStore(path)
    coordinator, _, _, _, request, _ = _runtime(tmp_path, store=store)
    del coordinator
    record = KnowledgeRetrievalRecord(
        record_id="reserved",
        retrieval_id=request.retrieval_id,
        run_id=request.run_id,
        cycle=1,
        status=KnowledgeRetrievalStatus.RESERVED,
        request=request,
        provider_request_started=True,
        created_at=fixed_clock(),
    )
    store.append(record)
    store.append(record)
    assert len(store.records_for_retrieval(request.retrieval_id)) == 1
    store.close()
    reopened = SQLiteKnowledgeRetrievalStore(path)
    try:
        assert reopened.latest(request.retrieval_id) == record
    finally:
        reopened.close()


def test_second_indeterminate_attempt_can_be_explicitly_retried(tmp_path):
    coordinator, store, provider, config, request, policy = _runtime(tmp_path)
    initial = KnowledgeRetrievalRecord(
        record_id="initial:reserved",
        retrieval_id=request.retrieval_id,
        run_id=request.run_id,
        cycle=1,
        status=KnowledgeRetrievalStatus.RESERVED,
        request=request,
        provider_request_started=True,
        created_at=fixed_clock(),
    )
    store.append(initial)
    with pytest.raises(KnowledgeRetrievalExecutionError):
        coordinator.run(request, provider, config, policy)
    child = store.create_retry(request.retrieval_id, created_at=fixed_clock())
    store.append(
        child.model_copy(
            update={
                "record_id": f"{child.retrieval_id}:started",
                "provider_request_started": True,
            }
        )
    )
    with pytest.raises(KnowledgeRetrievalExecutionError):
        coordinator.run(request, provider, config, policy)
    assert (
        store.latest(child.retrieval_id).status
        == KnowledgeRetrievalStatus.INDETERMINATE
    )
    grandchild = store.create_retry(child.retrieval_id, created_at=fixed_clock())
    bundle, replayed = coordinator.run(request, provider, config, policy)
    assert not replayed
    assert provider.calls == 1
    assert bundle.retrieval_id == grandchild.retrieval_id
