from __future__ import annotations

import pytest
from pydantic import ValidationError

from auto_researcher.cli import _load_grounding
from auto_researcher.contracts.enums import KnowledgeGroundingMode
from auto_researcher.contracts.models import KnowledgeGroundingRequirement
from auto_researcher.knowledge.identity import (
    assertion_id,
    entity_id,
    retrieval_id,
)
from auto_researcher.knowledge.models import (
    KnowledgeEntity,
    KnowledgeProviderConfiguration,
    KnowledgeTemplateRequest,
)
from auto_researcher.knowledge.registry import KnowledgeProviderRegistry
from auto_researcher.knowledge.schemas.knowledge_graph_auto_v0_1 import (
    KnowledgeGraphAutoProfile,
)
from auto_researcher.knowledge.templates import (
    KnowledgeQueryTemplate,
    KnowledgeQueryTemplateRegistry,
    default_template_registry,
    lint_read_only_cypher,
)
from auto_researcher.tasks.synthetic import default_synthetic_contract
from auto_researcher.tasks.icca_nbs.knowledge import icca_query_plan
from auto_researcher.tasks.models import TaskRuntimeContext
from tests.fakes_icca import make_fake_icca_bindings


def _grounded_contract(mode=KnowledgeGroundingMode.OPTIONAL):
    requirement = KnowledgeGroundingRequirement(
        mode=mode,
        permitted_providers=frozenset({"static", "neo4j"}),
        maximum_query_records=10,
        maximum_retrieval_duration=10,
        knowledge_schema_version="synthetic-v1",
        knowledge_content_version="fixture-v1",
    )
    return default_synthetic_contract().model_copy(update={"grounding": requirement})


def test_configuration_rejects_credentials_and_runtime_cannot_weaken_contract():
    with pytest.raises(ValidationError):
        KnowledgeProviderConfiguration(
            provider_id="neo4j",
            graph_alias="safe-alias",
            database="neo4j",
            schema_version="synthetic-v1",
            content_version="fixture-v1",
            password="secret",
        )
    with pytest.raises(ValueError, match="environment"):
        _load_grounding(
            {
                "grounding": {
                    "mode": "OPTIONAL",
                    "provider": "neo4j",
                    "password": "secret",
                }
            },
            _grounded_contract(),
        )
    with pytest.raises(ValueError, match="environment"):
        _load_grounding(
            {
                "grounding": {
                    "mode": "OPTIONAL",
                    "provider": "neo4j",
                    "connection": {"username": "must-not-be-configured"},
                }
            },
            _grounded_contract(),
        )
    with pytest.raises(ValueError, match="weakens"):
        _load_grounding(
            {
                "grounding": {
                    "mode": "OPTIONAL",
                    "provider": "static",
                    "minimum_assertion_confidence": 0.1,
                }
            },
            _grounded_contract(),
        )


def test_identifiers_are_stable_and_internal_properties_are_rejected():
    first = retrieval_id(
        run_id="run",
        cycle=1,
        task_id="synthetic",
        task_version="1",
        contract_id="contract",
        provider_id="static",
        provider_version="1",
        graph_alias="fixture",
        schema_version="1",
        content_version="1",
        query_plan_version="1",
        plan_hash="a" * 64,
    )
    second = retrieval_id(
        run_id="run",
        cycle=1,
        task_id="synthetic",
        task_version="1",
        contract_id="contract",
        provider_id="static",
        provider_version="1",
        graph_alias="fixture",
        schema_version="1",
        content_version="1",
        query_plan_version="1",
        plan_hash="a" * 64,
    )
    assert first == second
    assert entity_id("HGNC:1", "Gene") == entity_id("HGNC:1", "Gene")
    assert assertion_id("HGNC:1", "PARTICIPATES_IN", "GO:1", ("source:1",))
    with pytest.raises(ValidationError, match="prohibited"):
        KnowledgeEntity(
            entity_id="entity",
            curie="HGNC:1",
            entity_type="Gene",
            name="A",
            safe_properties={"element_id": "neo4j-internal"},
            source_references=("source:1",),
        )
    with pytest.raises(ValidationError, match="CURIE"):
        KnowledgeEntity(
            entity_id="entity",
            curie="not-a-curie",
            entity_type="Gene",
            name="A",
            source_references=("source:1",),
        )


@pytest.mark.parametrize(
    "query",
    [
        "MATCH (n) DELETE n ORDER BY n.x LIMIT $limit",
        "MATCH (n) SET n.x = 1 ORDER BY n.x LIMIT $limit",
        "CALL db.labels() YIELD label RETURN label ORDER BY label LIMIT $limit",
        "MATCH (n) RETURN n LIMIT $limit",
    ],
)
def test_template_lint_rejects_writes_calls_and_unordered_results(query):
    with pytest.raises(ValueError):
        lint_read_only_cypher(query)


def test_template_registry_is_versioned_bounded_and_task_compatible():
    registry = default_template_registry()
    request = KnowledgeTemplateRequest(
        template_id="generic.entity_lookup",
        template_version="1.0.0",
        parameters={"curie": "HGNC:1", "limit": 2},
        maximum_records=2,
        rationale="test",
    )
    template, parameters = registry.validate_request(
        request,
        task_id="icca_nbs",
        schema_version="knowledge-graph-auto-v0.1",
    )
    assert parameters == {"curie": "HGNC:1", "limit": 2}
    assert len(template.cypher_sha256) == 64
    with pytest.raises(ValueError, match="compatible"):
        registry.validate_request(
            request,
            task_id="unregistered",
            schema_version="knowledge-graph-auto-v0.1",
        )
    with pytest.raises(ValueError, match="row limit"):
        registry.validate_request(
            request.model_copy(update={"maximum_records": 101}),
            task_id="icca_nbs",
            schema_version="knowledge-graph-auto-v0.1",
        )


def test_provider_registry_and_schema_profile_fail_closed(tmp_path):
    registry = KnowledgeProviderRegistry()
    registry.register("fixture", lambda: object())
    assert registry.contains("fixture")
    assert registry.list_providers() == ("fixture",)
    with pytest.raises(ValueError, match="already"):
        registry.register("fixture", lambda: object())
    with pytest.raises(KeyError, match="unknown"):
        registry.get("missing")

    profile = KnowledgeGraphAutoProfile()
    result = profile.preflight(
        labels={"Gene", "Signature"},
        relationships={"MEMBER_OF"},
        curie_coverage={"Gene": (1, 2)},
        graph_counts={"Gene": 2},
        required_labels={"Gene", "Pathway"},
        required_relationships={"INCLUDES"},
    )
    assert not result.passed
    assert result.missing_labels == ("Pathway",)
    assert result.missing_relationships == ("INCLUDES",)
    assert any("coverage" in warning for warning in result.warnings)
    assert any("MEMBER_OF" in warning for warning in result.warnings)


def test_custom_registry_rejects_duplicate_template(tmp_path):
    path = tmp_path / "read.cypher"
    path.write_text(
        "MATCH (n) RETURN n ORDER BY n.name LIMIT $limit",
        encoding="utf-8",
    )
    base = default_template_registry().get("generic.entity_lookup", "1.0.0")
    template = KnowledgeQueryTemplate(
        template_id="test.read",
        version="1.0.0",
        cypher_path=path,
        parameter_model=base.parameter_model,
        output_schema_version="test-row-v1",
        allowed_labels=frozenset(),
        allowed_relationships=frozenset(),
        maximum_hops=0,
        maximum_rows=1,
        task_compatibility=frozenset({"synthetic"}),
        schema_compatibility=frozenset({"synthetic-v1"}),
    )
    registry = KnowledgeQueryTemplateRegistry()
    registry.register(template)
    with pytest.raises(ValueError, match="already"):
        registry.register(template)


def test_icca_gene_seeds_require_explicit_non_patient_provenance():
    contract = _grounded_contract().model_copy(
        update={
            "task_id": "icca_nbs",
            "grounding": KnowledgeGroundingRequirement(
                mode="OPTIONAL",
                permitted_providers=frozenset({"neo4j"}),
                knowledge_schema_version="knowledge-graph-auto-v0.1",
                knowledge_content_version="fixture-v1",
            ),
        }
    )
    bindings, _ = make_fake_icca_bindings()
    with pytest.raises(ValueError, match="non-patient"):
        icca_query_plan(
            contract,
            TaskRuntimeContext(
                task_options={
                    "grounding": {
                        "include_network_catalog": False,
                        "gene_curies": ["HGNC:11998"],
                    }
                }
            ),
            {},
            bindings,
        )
