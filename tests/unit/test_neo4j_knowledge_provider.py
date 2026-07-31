from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from auto_researcher.knowledge.identity import query_plan_hash
from auto_researcher.knowledge.models import (
    KnowledgeProviderConfiguration,
    KnowledgeQueryPlan,
    KnowledgeRetrievalRequest,
    KnowledgeTemplateRequest,
)
from auto_researcher.knowledge.protocols import KnowledgeProviderError
from auto_researcher.knowledge.providers.neo4j import (
    Neo4jKnowledgeProvider,
    _safe_plain,
)
from auto_researcher.knowledge.templates import default_template_registry
from tests.conftest import fixed_clock


class FakeResult:
    def __init__(self, rows, *, updates=False):
        self.rows = rows
        self.summary = SimpleNamespace(
            counters=SimpleNamespace(
                contains_updates=updates,
                contains_system_updates=False,
            )
        )

    def __iter__(self):
        return iter(self.rows)

    def consume(self):
        return self.summary


class FakeTransaction:
    def __init__(self, driver):
        self.driver = driver

    def run(self, query, parameters):
        self.driver.queries.append((str(query), parameters))
        if "nested_labels" in str(query):
            return FakeResult(
                [
                    {
                        "labels": ["Gene", "Signature", "Pathway"],
                        "relationships": [
                            "INCLUDES",
                            "PARTICIPATES_IN",
                            "PART_OF",
                        ],
                    }
                ]
            )
        rows = (
            self.driver.row if isinstance(self.driver.row, list) else [self.driver.row]
        )
        return FakeResult(rows, updates=self.driver.report_updates)


class FakeSession:
    def __init__(self, driver):
        self.driver = driver

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def run(self, query):
        self.driver.queries.append((query, {}))
        return FakeResult([{"access": "GRANTED", "action": "MATCH"}])

    def execute_read(self, work):
        return work(FakeTransaction(self.driver))


class FakeDriver:
    def __init__(self, row, *, report_updates=False):
        self.row = row
        self.report_updates = report_updates
        self.queries = []
        self.databases = []
        self.closed = False
        self.connectivity_calls = 0

    def verify_connectivity(self):
        self.connectivity_calls += 1
        return None

    def session(self, *, database):
        self.databases.append(database)
        return FakeSession(self)

    def close(self):
        self.closed = True


def _configuration():
    return KnowledgeProviderConfiguration(
        provider_id="neo4j",
        graph_alias="cell-biology",
        database="neo4j",
        schema_version="knowledge-graph-auto-v0.1",
        content_version="backbone-test",
        query_timeout_seconds=5,
        maximum_records=10,
    )


def _request():
    templates = default_template_registry()
    plan = KnowledgeQueryPlan(
        task_id="icca_nbs",
        task_version="1.0",
        schema_version="knowledge-graph-auto-v0.1",
        query_plan_version="1.0.0",
        template_requests=(
            KnowledgeTemplateRequest(
                template_id="icca_nbs.gene_signature_pathway",
                template_version="1.0.0",
                parameters={"gene_curies": ["HGNC:11998"], "limit": 5},
                maximum_records=5,
                rationale="fixture",
            ),
        ),
        grounding_policy_id="icca-nbs-grounding-v1",
        maximum_total_records=10,
        maximum_references=10,
    )
    return KnowledgeRetrievalRequest(
        retrieval_id="neo4j-retrieval",
        run_id="neo4j-run",
        cycle=1,
        provider_id="neo4j",
        graph_alias="cell-biology",
        schema_version="knowledge-graph-auto-v0.1",
        content_version="backbone-test",
        query_plan=plan,
        query_plan_hash=query_plan_hash(plan),
        grounding_policy_hash="0" * 64,
        template_hashes={
            "generic.schema_preflight@1.0.0": templates.get(
                "generic.schema_preflight",
                "1.0.0",
            ).cypher_sha256,
            "icca_nbs.gene_signature_pathway@1.0.0": templates.get(
                "icca_nbs.gene_signature_pathway",
                "1.0.0",
            ).cypher_sha256,
        },
        task_id="icca_nbs",
        contract_id="icca-contract",
    )


def _row():
    return {
        "source": {
            "source_id": "source:go:v1",
            "source_type": "ONTOLOGY_RELEASE",
            "title": "Gene Ontology",
            "version": "v1",
            "accession": "GO:0001",
            "publisher_or_database": "GO",
            "asserted_by": "curator",
        },
        "entities": [
            {
                "curie": "HGNC:11998",
                "entity_type": "Gene",
                "name": "TP53",
                "safe_properties": {"symbol": "TP53"},
            },
            {
                "curie": "GO:0001",
                "entity_type": "Pathway",
                "name": "fixture pathway",
                "safe_properties": {},
            },
        ],
        "assertion": {
            "subject_curie": "HGNC:11998",
            "predicate": "PARTICIPATES_IN",
            "object_curie": "GO:0001",
            "method": "curated ontology",
            "confidence": 1.0,
            "asserted_by": "curator",
            "trust_tier": "CURATED",
            "safe_properties": {},
        },
        "reference": {
            "reference_type": "GENE_CONTEXT",
            "concise_claim": "The gene participates in a registered pathway.",
            "relevant_parameters": ["alpha"],
        },
    }


def test_neo4j_provider_uses_explicit_database_preflight_and_execute_read():
    driver = FakeDriver(_row())
    configuration = _configuration()
    provider = Neo4jKnowledgeProvider(
        configuration,
        default_template_registry(),
        driver=driver,
        clock=fixed_clock,
        query_factory=lambda text, timeout: text,
    )
    readiness = provider.readiness(configuration)
    assert readiness.ready
    bundle = provider.retrieve(_request())
    assert len(bundle.references) == 1
    assert bundle.graph_snapshot.graph_alias == "cell-biology"
    assert all(database == "neo4j" for database in driver.databases)
    assert any("nested_labels" in query for query, _ in driver.queries)
    assert all(
        not any(token in query.upper() for token in (" CREATE ", " DELETE ", " MERGE "))
        for query, _ in driver.queries
    )
    provider.close()
    assert driver.closed


def test_default_live_query_path_uses_strings_in_managed_transactions():
    driver = FakeDriver(_row())
    configuration = _configuration()
    provider = Neo4jKnowledgeProvider(
        configuration,
        default_template_registry(),
        driver=driver,
        clock=fixed_clock,
    )

    bundle = provider.retrieve(_request())

    assert bundle.references
    assert driver.queries
    assert all(isinstance(query, str) for query, _ in driver.queries)


def test_schema_mismatch_and_reported_updates_fail_closed():
    configuration = _configuration()
    missing_schema = FakeDriver(_row())
    original_run = FakeTransaction.run

    def run_without_pathway(self, query, parameters):
        if "nested_labels" in str(query):
            return FakeResult(
                [{"labels": ["Gene"], "relationships": ["PARTICIPATES_IN"]}]
            )
        return original_run(self, query, parameters)

    FakeTransaction.run = run_without_pathway
    try:
        provider = Neo4jKnowledgeProvider(
            configuration,
            default_template_registry(),
            driver=missing_schema,
            clock=fixed_clock,
            query_factory=lambda text, timeout: text,
        )
        with pytest.raises(KnowledgeProviderError, match="SCHEMA_MISMATCH"):
            provider.retrieve(_request())
    finally:
        FakeTransaction.run = original_run

    update_driver = FakeDriver(_row(), report_updates=True)
    provider = Neo4jKnowledgeProvider(
        configuration,
        default_template_registry(),
        driver=update_driver,
        clock=fixed_clock,
        query_factory=lambda text, timeout: text,
    )
    with pytest.raises(KnowledgeProviderError, match="FORBIDDEN_WRITE_DETECTED"):
        provider.retrieve(_request())


def test_raw_neo4j_like_objects_are_plain_and_internal_ids_are_dropped():
    class NodeLike:
        def items(self):
            return {
                "element_id": "4:secret",
                "symbol": "TP53",
                "nested": {"id": "internal", "safe": True},
            }.items()

    assert _safe_plain(NodeLike()) == {
        "symbol": "TP53",
        "nested": {"safe": True},
    }


def test_unconfigured_provider_readiness_is_safe():
    configuration = _configuration()
    provider = Neo4jKnowledgeProvider(
        configuration,
        default_template_registry(),
        clock=fixed_clock,
    )
    result = provider.readiness(configuration)
    assert not result.ready
    assert "PROVIDER_NOT_CONFIGURED" in {item.value for item in result.errors}
    assert "bolt://" not in result.model_dump_json()


def test_only_transient_reads_receive_a_bounded_retry():
    class ServiceUnavailable(Exception):
        pass

    class TransientSession(FakeSession):
        def execute_read(self, work):
            if self.driver.failures:
                self.driver.failures -= 1
                raise ServiceUnavailable("safe transient fixture")
            return super().execute_read(work)

    class TransientDriver(FakeDriver):
        failures = 1

        def session(self, *, database):
            self.databases.append(database)
            return TransientSession(self)

    configuration = _configuration().model_copy(update={"maximum_attempts": 2})
    driver = TransientDriver(_row())
    provider = Neo4jKnowledgeProvider(
        configuration,
        default_template_registry(),
        driver=driver,
        clock=fixed_clock,
        query_factory=lambda text, timeout: text,
    )
    bundle = provider.retrieve(_request())
    assert bundle.references
    assert driver.failures == 0
    assert len(driver.databases) == 3  # two preflight attempts and one task read


def test_total_retrieval_deadline_applies_before_each_query():
    configuration = _configuration()
    moments = iter((0.0, 6.0))
    provider = Neo4jKnowledgeProvider(
        configuration,
        default_template_registry(),
        driver=FakeDriver(_row()),
        clock=fixed_clock,
        query_factory=lambda text, timeout: text,
        monotonic=lambda: next(moments),
    )
    with pytest.raises(KnowledgeProviderError, match="QUERY_TIMEOUT"):
        provider.retrieve(_request())


def test_repeated_entities_merge_safe_source_references():
    first = _row()
    second = deepcopy(first)
    second["source"]["source_id"] = "source:go:v2"
    second["source"]["version"] = "v2"
    second["source"]["accession"] = "GO:0002"
    second["entities"][1]["curie"] = "GO:0002"
    second["entities"][1]["name"] = "second pathway"
    second["assertion"]["object_curie"] = "GO:0002"
    provider = Neo4jKnowledgeProvider(
        _configuration(),
        default_template_registry(),
        driver=FakeDriver([first, second]),
        clock=fixed_clock,
        query_factory=lambda text, timeout: text,
    )
    bundle = provider.retrieve(_request())
    gene = next(item for item in bundle.entities if item.curie == "HGNC:11998")
    assert gene.source_references == ("source:go:v1", "source:go:v2")
    assert len(bundle.entities) == 3
