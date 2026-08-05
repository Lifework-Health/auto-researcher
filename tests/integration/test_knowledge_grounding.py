from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from datetime import UTC, datetime

from auto_researcher.agents.models import (
    ModelCallConfig,
    ModelPricing,
    StructuredModelResponse,
)
from auto_researcher.contracts.enums import (
    EvidenceStatus,
    GroundingStatus,
    KnowledgeGroundingMode,
    KnowledgeRetrievalStatus,
    ProvenanceKind,
    ReadSafetyMode,
    RunStatus,
    SearchType,
)
from auto_researcher.contracts.models import (
    KnowledgeGroundingRequirement,
    ResearchContract,
)
from auto_researcher.graph.builder import build_graph
from auto_researcher.graph.nodes.knowledge import retrieve_knowledge
from auto_researcher.knowledge.models import KnowledgeProviderConfiguration
from auto_researcher.knowledge.provenance import append_knowledge_retrieval_events
from auto_researcher.knowledge.providers.neo4j import Neo4jKnowledgeProvider
from auto_researcher.knowledge.providers.static import StaticKnowledgeProvider
from auto_researcher.knowledge.read_safety import ReadSafetyAttestation
from auto_researcher.knowledge.templates import default_template_registry
from auto_researcher.runtime.dependencies import task_memory_dependencies
from auto_researcher.tasks.icca_nbs import ICCANBSTask
from auto_researcher.tasks.models import TaskRuntimeContext
from auto_researcher.tasks.synthetic import (
    SyntheticTask,
    default_synthetic_configuration,
    default_synthetic_contract,
)
from tests.conftest import fixed_clock
from tests.fakes_icca import make_fake_icca_bindings
from tests.helpers_read_safety import operator_configuration
from tests.unit.test_neo4j_knowledge_provider import FakeDriver, _row


class ContextAwareFakeClient:
    provider = "fake"
    model_id = "fake-model-2026-07-31"

    def __init__(self, *, icca: bool = False, cite_knowledge: bool = True):
        self.icca = icca
        self.cite_knowledge = cite_knowledge
        self.calls = []

    def generate_structured(
        self,
        *,
        call_id,
        system_prompt,
        user_prompt,
        response_model,
        call_config,
        context_hash,
    ):
        references = re.findall(
            r'"reference_id":"(knowledge-ref-[0-9a-f]+)"',
            user_prompt,
        )
        cited = references[:1] if self.cite_knowledge else []
        if response_model.__name__ == "HypothesisProposal":
            output = (
                {
                    "statement": "A bounded alpha region may improve iCCA stability.",
                    "rationale": "Test a source-backed, task-bounded prior.",
                    "predicted_subspace": {"alpha": [0.4, 0.8]},
                    "expected_observation": "stability_objective increases",
                    "falsification_condition": "stability_objective does not increase",
                    "evidence_references": cited,
                    "confidence": 0.95,
                }
                if self.icca
                else {
                    "statement": "Bounded complexity may improve the objective.",
                    "rationale": "Test a source-backed, task-bounded prior.",
                    "predicted_subspace": {"complexity": [3, 6]},
                    "expected_observation": "objective_score increases",
                    "falsification_condition": "objective_score does not increase",
                    "evidence_references": cited,
                    "confidence": 0.95,
                }
            )
        else:
            output = {
                "search_type": "DIRECT",
                "target": ("stability_objective" if self.icca else "objective_score"),
                "proposed_search_space": (
                    {
                        "network": "Ideker",
                        "alignment": "Intersect",
                        "alpha": 0.7,
                        "K": 5,
                        "r": 10,
                    }
                    if self.icca
                    else default_synthetic_configuration()
                ),
                "requested_experiment_budget": 1,
                "rationale": "Run one bounded registered experiment.",
                "evidence_references": cited,
                "recommends_human_approval": False,
            }
        self.calls.append(
            {
                "role": response_model.__name__,
                "system": system_prompt,
                "user": user_prompt,
            }
        )
        encoded = json.dumps(output, sort_keys=True).encode()
        return StructuredModelResponse(
            call_id=call_id,
            provider=self.provider,
            model_id=self.model_id,
            structured_output=output,
            input_tokens=100,
            output_tokens=50,
            estimated_cost=0.0002,
            latency_ms=1,
            prompt_version=call_config.prompt_version,
            context_hash=context_hash,
            response_hash=hashlib.sha256(encoded).hexdigest(),
        )


def _call_config():
    return ModelCallConfig(
        provider="fake",
        model_id="fake-model-2026-07-31",
        temperature=0,
        maximum_output_tokens=512,
        timeout_seconds=10,
        maximum_attempts=1,
        maximum_cost_per_call=0.1,
        pricing=ModelPricing(
            version="fixture-v1",
            input_cost_per_million_tokens=1,
            output_cost_per_million_tokens=2,
            currency="USD",
        ),
        prompt_version="2.0.0",
    )


def _synthetic_contract(mode):
    return default_synthetic_contract().model_copy(
        update={
            "grounding": KnowledgeGroundingRequirement(
                mode=mode,
                permitted_providers=(
                    frozenset({"static"})
                    if mode != KnowledgeGroundingMode.DISABLED
                    else frozenset()
                ),
                maximum_query_records=10,
                knowledge_schema_version=(
                    "synthetic-v1"
                    if mode != KnowledgeGroundingMode.DISABLED
                    else "none"
                ),
                knowledge_content_version=(
                    "fixture-v1" if mode != KnowledgeGroundingMode.DISABLED else "none"
                ),
            )
        }
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


def test_static_fake_live_grounding_is_cited_bounded_and_not_experimental_proof(
    tmp_path,
):
    contract = _synthetic_contract(KnowledgeGroundingMode.OPTIONAL)
    configuration = KnowledgeProviderConfiguration(
        provider_id="static",
        graph_alias="synthetic-fixture",
        database="static",
        schema_version="synthetic-v1",
        content_version="fixture-v1",
        maximum_records=10,
    )
    provider = StaticKnowledgeProvider(configuration, clock=fixed_clock)
    client = ContextAwareFakeClient()
    dependencies = task_memory_dependencies(
        SyntheticTask(),
        TaskRuntimeContext(
            run_id="knowledge-live",
            output_dir=tmp_path,
            manifest_created_at=fixed_clock(),
        ),
        contract,
        default_synthetic_configuration(),
        model_client=client,
        hypothesis_call_config=_call_config(),
        planner_call_config=_call_config(),
        knowledge_provider=provider,
        knowledge_configuration=configuration,
        clock=fixed_clock,
    )
    final = _invoke(dependencies, contract, "knowledge-live")
    assert final["status"] == RunStatus.COMPLETED
    assert final["knowledge_retrieval_status"] == KnowledgeRetrievalStatus.COMPLETED
    assert provider.calls == 1
    assert (
        final["active_hypothesis"].grounding_status
        == GroundingStatus.KNOWLEDGE_GROUNDED
    )
    assert final["active_hypothesis"].prior_weight == 0.9
    assert (
        final["search_request"].grounding_status == GroundingStatus.KNOWLEDGE_GROUNDED
    )
    assert final["active_hypothesis"].evidence_references
    assert final["verification_result"].evidence_status != EvidenceStatus.SUPPORTED
    rendered = "\n".join(item["user"] for item in client.calls)
    assert "knowledge-ref-" in rendered
    assert "unverified diagnostic assertion" not in rendered
    assert "/Users/" not in rendered
    events = dependencies.provenance_store.list_events("knowledge-live")
    assert [item.event_type.value for item in events[:3]] == [
        "KNOWLEDGE_RETRIEVAL_RESERVED",
        "KNOWLEDGE_RETRIEVAL_COMPLETED",
        "KNOWLEDGE_BUNDLE_VALIDATED",
    ]
    hypothesis_event = next(
        item for item in events if item.event_type.value == "HYPOTHESIS_PROPOSED"
    )
    assert (
        final["knowledge_bundle_reference"].bundle_id
        in hypothesis_event.input_references
    )
    assert any(
        reference.endswith(final["active_hypothesis"].evidence_references[0])
        for reference in hypothesis_event.output_references
    )
    count = len(events)
    append_knowledge_retrieval_events(
        dependencies.provenance_store,
        dependencies.knowledge_retrieval_store,
        run_id="knowledge-live",
        cycle=1,
    )
    assert len(dependencies.provenance_store.list_events("knowledge-live")) == count


def test_uncited_bundle_does_not_claim_knowledge_grounding(tmp_path):
    contract = _synthetic_contract(KnowledgeGroundingMode.OPTIONAL)
    configuration = KnowledgeProviderConfiguration(
        provider_id="static",
        graph_alias="fixture",
        database="static",
        schema_version="synthetic-v1",
        content_version="fixture-v1",
        maximum_records=10,
    )
    provider = StaticKnowledgeProvider(configuration, clock=fixed_clock)
    client = ContextAwareFakeClient(cite_knowledge=False)
    dependencies = task_memory_dependencies(
        SyntheticTask(),
        TaskRuntimeContext(run_id="uncited", output_dir=tmp_path),
        contract,
        default_synthetic_configuration(),
        model_client=client,
        hypothesis_call_config=_call_config(),
        planner_call_config=_call_config(),
        knowledge_provider=provider,
        knowledge_configuration=configuration,
        clock=fixed_clock,
    )
    final = _invoke(dependencies, contract, "uncited")
    assert final["knowledge_bundle_reference"].reference_ids
    assert final["active_hypothesis"].grounding_status == GroundingStatus.UNGROUNDED
    assert final["search_request"].grounding_status == GroundingStatus.UNGROUNDED


def test_optional_unavailable_continues_but_required_stops_before_model_call(tmp_path):
    optional = _synthetic_contract(KnowledgeGroundingMode.OPTIONAL)
    optional_client = ContextAwareFakeClient(cite_knowledge=False)
    optional_dependencies = task_memory_dependencies(
        SyntheticTask(),
        TaskRuntimeContext(run_id="optional", output_dir=tmp_path / "optional"),
        optional,
        default_synthetic_configuration(),
        model_client=optional_client,
        hypothesis_call_config=_call_config(),
        planner_call_config=_call_config(),
        clock=fixed_clock,
    )
    optional_final = _invoke(optional_dependencies, optional, "optional")
    assert optional_final["status"] == RunStatus.COMPLETED
    assert len(optional_client.calls) == 2
    assert (
        optional_final["knowledge_retrieval_status"] == KnowledgeRetrievalStatus.FAILED
    )

    required = _synthetic_contract(KnowledgeGroundingMode.REQUIRED)
    required_client = ContextAwareFakeClient()
    required_dependencies = task_memory_dependencies(
        SyntheticTask(),
        TaskRuntimeContext(run_id="required", output_dir=tmp_path / "required"),
        required,
        default_synthetic_configuration(),
        model_client=required_client,
        hypothesis_call_config=_call_config(),
        planner_call_config=_call_config(),
        clock=fixed_clock,
    )
    required_final = _invoke(required_dependencies, required, "required")
    assert required_final["status"] == RunStatus.STOPPED
    assert required_final["stop_reason"] == "required_knowledge_unavailable"
    assert required_client.calls == []


def test_optional_unverified_continues_ungrounded_without_provider_call(tmp_path):
    contract = _synthetic_contract(KnowledgeGroundingMode.OPTIONAL)
    contract = contract.model_copy(
        update={
            "grounding": contract.grounding.model_copy(
                update={
                    "permitted_read_safety_modes": frozenset(
                        {ReadSafetyMode.UNVERIFIED}
                    )
                }
            )
        }
    )
    configuration = KnowledgeProviderConfiguration(
        provider_id="static",
        graph_alias="unverified-fixture",
        database="static",
        schema_version="synthetic-v1",
        content_version="fixture-v1",
        maximum_records=10,
        read_safety_mode=ReadSafetyMode.UNVERIFIED,
    )
    provider = StaticKnowledgeProvider(configuration, clock=fixed_clock)
    client = ContextAwareFakeClient(cite_knowledge=False)
    dependencies = task_memory_dependencies(
        SyntheticTask(),
        TaskRuntimeContext(run_id="unverified", output_dir=tmp_path),
        contract,
        default_synthetic_configuration(),
        model_client=client,
        hypothesis_call_config=_call_config(),
        planner_call_config=_call_config(),
        knowledge_provider=provider,
        knowledge_configuration=configuration,
        clock=fixed_clock,
    )

    final = _invoke(dependencies, contract, "unverified")
    assert final["status"] == RunStatus.COMPLETED
    assert final["knowledge_retrieval_status"] == KnowledgeRetrievalStatus.FAILED
    assert final["knowledge_errors"] == ["READ_ONLY_NOT_VERIFIED"]
    assert provider.calls == 0
    assert len(client.calls) == 2


def test_disabled_grounding_makes_no_provider_call_and_topology_is_unchanged(tmp_path):
    contract = _synthetic_contract(KnowledgeGroundingMode.DISABLED)
    configuration = KnowledgeProviderConfiguration(
        provider_id="static",
        graph_alias="disabled",
        database="static",
        schema_version="none",
        content_version="none",
        maximum_records=10,
    )
    provider = StaticKnowledgeProvider(configuration, clock=fixed_clock)
    with_provider = task_memory_dependencies(
        SyntheticTask(),
        TaskRuntimeContext(run_id="disabled", output_dir=tmp_path),
        contract,
        default_synthetic_configuration(),
        knowledge_provider=provider,
        knowledge_configuration=configuration,
        clock=fixed_clock,
    )
    without_provider = task_memory_dependencies(
        SyntheticTask(),
        TaskRuntimeContext(run_id="disabled-2", output_dir=tmp_path),
        contract,
        default_synthetic_configuration(),
        clock=fixed_clock,
    )
    final = _invoke(with_provider, contract, "disabled")
    assert final["knowledge_retrieval_status"] == KnowledgeRetrievalStatus.DISABLED
    assert provider.calls == 0
    assert (
        build_graph(with_provider).get_graph().draw_mermaid()
        == build_graph(without_provider).get_graph().draw_mermaid()
    )


def test_fake_icca_neo4j_profile_grounding(tmp_path):
    for filename in ("Combined_binary_matrix.csv", "Combined_clinical.csv"):
        (tmp_path / filename).write_text("fixture,data\n", encoding="utf-8")
    requirement = KnowledgeGroundingRequirement(
        mode="REQUIRED",
        permitted_providers=frozenset({"neo4j"}),
        permitted_read_safety_modes=frozenset({ReadSafetyMode.OPERATOR_ATTESTED}),
        maximum_query_records=10,
        knowledge_schema_version="knowledge-graph-auto-v0.1",
        knowledge_content_version="backbone-test",
    )
    contract = ResearchContract(
        contract_id="icca-knowledge-fixture",
        schema_version="1.0",
        task_id="icca_nbs",
        task_version="1.0",
        objective_version="0.9",
        primary_metric="stability_objective",
        task_constraints_version="1.0",
        question="Which bounded configuration is eligible?",
        objective="maximise stability",
        constraints={},
        allowed_search_types=frozenset({SearchType.DIRECT}),
        evaluator_id="icca-nbs-v2-evaluator",
        verifier_id="deterministic-verifier",
        maximum_cycles=1,
        maximum_experiments=1,
        maximum_cost=2,
        grounding=requirement,
        provenance=ProvenanceKind.REAL,
    )
    configuration = operator_configuration()
    driver = FakeDriver(_row())
    provider = Neo4jKnowledgeProvider(
        configuration,
        default_template_registry(),
        driver=driver,
        clock=fixed_clock,
        query_factory=lambda text, timeout: text,
    )
    client = ContextAwareFakeClient(icca=True)
    bindings, _ = make_fake_icca_bindings()
    dependencies = task_memory_dependencies(
        ICCANBSTask(bindings),
        TaskRuntimeContext(
            run_id="icca-knowledge",
            data_dir=tmp_path,
            workspace_dir=tmp_path,
            output_dir=tmp_path / "output",
            task_options={
                "grounding": {
                    "include_network_catalog": False,
                    "gene_curies": ["HGNC:11998"],
                    "gene_seed_provenance": "CURATED",
                    "include_pathways": True,
                }
            },
            manifest_created_at=fixed_clock(),
        ),
        contract,
        {
            "network": "Ideker",
            "alignment": "Intersect",
            "alpha": 0.7,
            "K": 5,
            "r": 10,
        },
        model_client=client,
        hypothesis_call_config=_call_config(),
        planner_call_config=_call_config(),
        knowledge_provider=provider,
        knowledge_configuration=configuration,
        clock=fixed_clock,
    )
    final = _invoke(dependencies, contract, "icca-knowledge")
    assert final["status"] == RunStatus.COMPLETED, (
        final["knowledge_errors"],
        final["stop_reason"],
        driver.queries,
    )
    assert (
        final["active_hypothesis"].grounding_status
        == GroundingStatus.KNOWLEDGE_GROUNDED
    )
    assert final["knowledge_bundle_reference"].reference_ids
    assert any("nested_labels" in query for query, _ in driver.queries)
    assert "retrieve_knowledge" in final["executed_nodes"]
    knowledge_event = next(
        event
        for event in dependencies.provenance_store.list_events("icca-knowledge")
        if event.event_type.value == "KNOWLEDGE_RETRIEVAL_COMPLETED"
    )
    provenance_text = "\n".join(knowledge_event.output_references)
    assert "read_safety_mode:OPERATOR_ATTESTED" in provenance_text
    assert "credential_class:MANAGED_INSTANCE_PRIMARY" in provenance_text
    assert "attestation_hash_algorithm:canonical-json-sha256-v1" in provenance_text
    assert "configuration_hash_algorithm:canonical-json-sha256-v1" in provenance_text
    assert "zero_updates_confirmed:true" in provenance_text
    assert "zero_system_updates_confirmed:true" in provenance_text
    assert "password" not in provenance_text.casefold()
    assert "neo4j+s://" not in provenance_text
    assert "@lifework" not in provenance_text
    queries_before_replay = len(driver.queries)
    connectivity_before_replay = driver.connectivity_calls
    replayed = retrieve_knowledge(
        {"run_id": "icca-knowledge", "cycle": 1, "contract": contract},
        dependencies,
    )
    assert replayed["knowledge_retrieval_status"] == KnowledgeRetrievalStatus.COMPLETED
    assert len(driver.queries) == queries_before_replay
    assert driver.connectivity_calls == connectivity_before_replay

    restored_attestation = ReadSafetyAttestation.model_validate_json(
        configuration.read_safety_attestation.model_dump_json()
    )
    restored_configuration = configuration.model_copy(
        update={"read_safety_attestation": restored_attestation}
    )
    replay_driver = FakeDriver(_row())
    reconstructed = replace(
        dependencies,
        knowledge_configuration=restored_configuration,
        knowledge_provider=Neo4jKnowledgeProvider(
            restored_configuration,
            default_template_registry(),
            driver=replay_driver,
            clock=fixed_clock,
            query_factory=lambda text, timeout: text,
        ),
    )
    replayed_after_round_trip = retrieve_knowledge(
        {"run_id": "icca-knowledge", "cycle": 1, "contract": contract},
        reconstructed,
    )
    assert (
        replayed_after_round_trip["knowledge_bundle_reference"].retrieval_id
        == final["knowledge_bundle_reference"].retrieval_id
    )
    assert replay_driver.queries == []


def test_required_expired_operator_attestation_stops_before_models_and_evaluator(
    tmp_path,
):
    for filename in ("Combined_binary_matrix.csv", "Combined_clinical.csv"):
        (tmp_path / filename).write_text("fixture,data\n", encoding="utf-8")
    requirement = KnowledgeGroundingRequirement(
        mode="REQUIRED",
        permitted_providers=frozenset({"neo4j"}),
        permitted_read_safety_modes=frozenset({ReadSafetyMode.OPERATOR_ATTESTED}),
        maximum_query_records=10,
        knowledge_schema_version="knowledge-graph-auto-v0.1",
        knowledge_content_version="backbone-test",
    )
    contract = ResearchContract(
        contract_id="icca-expired-attestation",
        schema_version="1.0",
        task_id="icca_nbs",
        task_version="1.0",
        objective_version="0.9",
        primary_metric="stability_objective",
        task_constraints_version="1.0",
        question="Which bounded configuration is eligible?",
        objective="maximise stability",
        constraints={},
        allowed_search_types=frozenset({SearchType.DIRECT}),
        evaluator_id="icca-nbs-v2-evaluator",
        verifier_id="deterministic-verifier",
        maximum_cycles=1,
        maximum_experiments=1,
        maximum_cost=2,
        grounding=requirement,
        provenance=ProvenanceKind.REAL,
    )
    configuration = operator_configuration(
        expires_at=datetime(2026, 7, 30, 11, 0, tzinfo=UTC)
    )
    provider = Neo4jKnowledgeProvider(
        configuration,
        default_template_registry(),
        driver=FakeDriver(_row()),
        clock=fixed_clock,
        query_factory=lambda text, timeout: text,
    )
    client = ContextAwareFakeClient(icca=True)
    bindings, evaluator_calls = make_fake_icca_bindings()
    dependencies = task_memory_dependencies(
        ICCANBSTask(bindings),
        TaskRuntimeContext(
            run_id="expired-attestation",
            data_dir=tmp_path,
            workspace_dir=tmp_path,
            output_dir=tmp_path / "output-expired",
            task_options={
                "grounding": {
                    "include_network_catalog": False,
                    "gene_curies": ["HGNC:11998"],
                    "gene_seed_provenance": "CURATED",
                    "include_pathways": True,
                }
            },
            manifest_created_at=fixed_clock(),
        ),
        contract,
        {
            "network": "Ideker",
            "alignment": "Intersect",
            "alpha": 0.7,
            "K": 5,
            "r": 10,
        },
        model_client=client,
        hypothesis_call_config=_call_config(),
        planner_call_config=_call_config(),
        knowledge_provider=provider,
        knowledge_configuration=configuration,
        clock=fixed_clock,
    )

    final = _invoke(dependencies, contract, "expired-attestation")
    assert final["status"] == RunStatus.STOPPED
    assert final["stop_reason"] == "required_knowledge_unavailable"
    assert final["knowledge_errors"] == ["ATTESTATION_INVALID"]
    assert client.calls == []
    assert evaluator_calls["evaluate"] == 0
