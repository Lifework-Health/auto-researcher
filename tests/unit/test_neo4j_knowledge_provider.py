from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from auto_researcher.knowledge.identity import query_plan_hash
from auto_researcher.knowledge.models import (
    KnowledgeErrorCode,
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
from tests.helpers_read_safety import operator_configuration


class FakeResult:
    def __init__(self, rows, *, updates=False, system_updates=False):
        self.rows = rows
        self.summary = SimpleNamespace(
            counters=SimpleNamespace(
                contains_updates=updates,
                contains_system_updates=system_updates,
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
        return FakeResult(
            rows,
            updates=self.driver.report_updates,
            system_updates=self.driver.report_system_updates,
        )


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
    def __init__(
        self,
        row,
        *,
        report_updates=False,
        report_system_updates=False,
    ):
        self.row = row
        self.report_updates = report_updates
        self.report_system_updates = report_system_updates
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


def test_operator_attested_readiness_and_query_audit_are_explicit():
    configuration = operator_configuration()
    driver = FakeDriver(_row())
    provider = Neo4jKnowledgeProvider(
        configuration,
        default_template_registry(),
        driver=driver,
        clock=fixed_clock,
        query_factory=lambda text, timeout: text,
    )

    readiness = provider.readiness(configuration)
    assert readiness.ready
    assert readiness.read_safety_mode.value == "OPERATOR_ATTESTED"
    assert not readiness.privilege_verified
    assert readiness.attestation_valid
    assert readiness.residual_risk == ("DATABASE_CREDENTIAL_NOT_ENFORCED_READ_ONLY")
    assert any("weaker" in warning for warning in readiness.warnings)
    assert not any("SHOW USER PRIVILEGES" in query for query, _ in driver.queries)

    bundle = provider.retrieve(_request())
    metadata = bundle.graph_snapshot.safe_graph_metadata
    assert metadata["read_safety_mode"] == "OPERATOR_ATTESTED"
    assert metadata["credential_class"] == "MANAGED_INSTANCE_PRIMARY"
    assert metadata["residual_risk"] == ("DATABASE_CREDENTIAL_NOT_ENFORCED_READ_ONLY")
    assert all(
        item["zero_updates_confirmed"] and item["zero_system_updates_confirmed"]
        for item in metadata["query_execution_audit"]
    )


def test_operator_attested_requires_current_attestation():
    with pytest.raises(ValidationError, match="requires Neo4j and an attestation"):
        _configuration().model_copy(
            update={
                "read_safety_mode": "OPERATOR_ATTESTED",
                "read_safety_attestation": None,
            }
        ).model_validate(
            {
                **_configuration().model_dump(mode="python"),
                "read_safety_mode": "OPERATOR_ATTESTED",
                "read_safety_attestation": None,
            }
        )

    expired = operator_configuration(
        expires_at=datetime(2026, 7, 30, 11, 0, tzinfo=UTC)
    )
    provider = Neo4jKnowledgeProvider(
        expired,
        default_template_registry(),
        driver=FakeDriver(_row()),
        clock=fixed_clock,
        query_factory=lambda text, timeout: text,
    )
    readiness = provider.readiness(expired)
    assert not readiness.ready
    assert not readiness.attestation_valid
    assert "ATTESTATION_INVALID" in {item.value for item in readiness.errors}


def test_operator_readiness_rejects_hash_or_runtime_configuration_drift():
    configuration = operator_configuration()
    attestation = configuration.read_safety_attestation
    assert attestation is not None
    tampered_attestation = attestation.model_copy(
        update={"residual_risk_statement": "Changed but credential-free risk."}
    )
    tampered = configuration.model_copy(
        update={"read_safety_attestation": tampered_attestation}
    )
    changed_runtime = configuration.model_copy(update={"maximum_records": 9})

    for changed in (tampered, changed_runtime):
        readiness = Neo4jKnowledgeProvider(
            changed,
            default_template_registry(),
            driver=FakeDriver(_row()),
            clock=fixed_clock,
            query_factory=lambda text, timeout: text,
        ).readiness(changed)
        assert not readiness.ready
        assert not readiness.attestation_valid
        assert KnowledgeErrorCode.ATTESTATION_INVALID in readiness.errors


def test_operator_attested_rejects_unattested_template_and_update_counters():
    preflight_only = operator_configuration(
        template_ids=("generic.schema_preflight@1.0.0",)
    )
    provider = Neo4jKnowledgeProvider(
        preflight_only,
        default_template_registry(),
        driver=FakeDriver(_row()),
        clock=fixed_clock,
        query_factory=lambda text, timeout: text,
    )
    assert provider.readiness(preflight_only).ready
    with pytest.raises(KnowledgeProviderError, match="ATTESTATION_INVALID"):
        provider.retrieve(_request())

    for driver in (
        FakeDriver(_row(), report_updates=True),
        FakeDriver(_row(), report_system_updates=True),
    ):
        configuration = operator_configuration()
        provider = Neo4jKnowledgeProvider(
            configuration,
            default_template_registry(),
            driver=driver,
            clock=fixed_clock,
            query_factory=lambda text, timeout: text,
        )
        with pytest.raises(
            KnowledgeProviderError,
            match="OPERATOR_ATTESTED_WRITE_BARRIER_VIOLATION",
        ):
            provider.retrieve(_request())


def test_privilege_verified_fails_when_privileges_unavailable_or_admin_like():
    class PrivilegeUnavailableSession(FakeSession):
        def run(self, query):
            raise RuntimeError("safe fixture")

    class PrivilegeUnavailableDriver(FakeDriver):
        def session(self, *, database):
            self.databases.append(database)
            return PrivilegeUnavailableSession(self)

    configuration = _configuration()
    unavailable = Neo4jKnowledgeProvider(
        configuration,
        default_template_registry(),
        driver=PrivilegeUnavailableDriver(_row()),
        clock=fixed_clock,
    ).readiness(configuration)
    assert not unavailable.ready
    assert not unavailable.privilege_verified

    class AdminPrivilegeSession(FakeSession):
        def run(self, query):
            return FakeResult([{"access": "GRANTED", "action": "create index"}])

    class AdminPrivilegeDriver(FakeDriver):
        def session(self, *, database):
            self.databases.append(database)
            return AdminPrivilegeSession(self)

    admin_like = Neo4jKnowledgeProvider(
        configuration,
        default_template_registry(),
        driver=AdminPrivilegeDriver(_row()),
        clock=fixed_clock,
    ).readiness(configuration)
    assert not admin_like.ready
    assert not admin_like.privilege_verified

    class AllPrivilegesSession(FakeSession):
        def run(self, query):
            return FakeResult(
                [{"access": "GRANTED", "action": "all database privileges"}]
            )

    class AllPrivilegesDriver(FakeDriver):
        def session(self, *, database):
            self.databases.append(database)
            return AllPrivilegesSession(self)

    all_privileges = Neo4jKnowledgeProvider(
        configuration,
        default_template_registry(),
        driver=AllPrivilegesDriver(_row()),
        clock=fixed_clock,
    ).readiness(configuration)
    assert not all_privileges.ready
    assert not all_privileges.privilege_verified
