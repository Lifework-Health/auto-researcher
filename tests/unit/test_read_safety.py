from __future__ import annotations

from datetime import UTC, datetime
import os
import subprocess
import sys

import pytest
from pydantic import ValidationError
import yaml

from auto_researcher.knowledge.models import KnowledgeProviderConfiguration
from auto_researcher.knowledge.read_safety import (
    CANONICAL_HASH_ALGORITHM,
    ProhibitedCapability,
    ReadSafetyAttestation,
    attestation_content_hash,
    read_safety_configuration_hash,
    seal_attestation,
    validate_operator_attestation,
)
from auto_researcher.knowledge.templates import default_template_registry
from tests.conftest import fixed_clock
from tests.helpers_read_safety import operator_configuration


def _validation_errors(configuration) -> set[str]:
    attestation = configuration.read_safety_attestation
    assert attestation is not None
    return set(
        validate_operator_attestation(
            attestation,
            configuration,
            default_template_registry(),
            now=fixed_clock(),
        )
    )


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("graph_alias", "different-graph", "ATTESTATION_GRAPH_ALIAS_MISMATCH"),
        ("schema_version", "different-schema", "ATTESTATION_SCHEMA_VERSION_MISMATCH"),
        (
            "content_version",
            "different-content",
            "ATTESTATION_CONTENT_VERSION_MISMATCH",
        ),
    ],
)
def test_attestation_identity_mismatches_are_rejected(field, value, expected):
    configuration = operator_configuration()
    attestation = configuration.read_safety_attestation
    assert attestation is not None
    changed = seal_attestation(attestation.model_copy(update={field: value}))
    changed_configuration = configuration.model_copy(
        update={"read_safety_attestation": changed}
    )

    assert expected in _validation_errors(changed_configuration)


def test_provider_tier_and_credential_vocabularies_are_closed():
    payload = operator_configuration().read_safety_attestation.model_dump(mode="python")
    for field, value in (
        ("provider_id", "other-provider"),
        ("service_tier", "UNSPECIFIED"),
        ("credential_class", "READ_ONLY"),
        ("identity_class", "VIEWER"),
    ):
        with pytest.raises(ValidationError):
            ReadSafetyAttestation.model_validate({**payload, field: value})


def test_template_set_and_configuration_hash_mismatches_are_rejected():
    configuration = operator_configuration()
    attestation = configuration.read_safety_attestation
    assert attestation is not None
    unknown_template = seal_attestation(
        attestation.model_copy(
            update={
                "permitted_query_template_ids": frozenset(
                    {
                        "generic.schema_preflight@1.0.0",
                        "missing.template@1.0.0",
                    }
                )
            }
        )
    )
    template_mismatch = configuration.model_copy(
        update={"read_safety_attestation": unknown_template}
    )
    assert "ATTESTATION_TEMPLATE_SET_MISMATCH" in _validation_errors(template_mismatch)

    wrong_hash = seal_attestation(
        attestation.model_copy(update={"configuration_hash": "1" * 64})
    )
    hash_mismatch = configuration.model_copy(
        update={"read_safety_attestation": wrong_hash}
    )
    assert "ATTESTATION_CONFIGURATION_HASH_MISMATCH" in _validation_errors(
        hash_mismatch
    )


def test_attestation_hash_is_deterministic_and_tamper_evident():
    first = operator_configuration().read_safety_attestation
    second = operator_configuration().read_safety_attestation
    assert first is not None and second is not None
    assert first.attestation_hash == second.attestation_hash
    assert first.attestation_hash == attestation_content_hash(first)

    tampered = first.model_copy(
        update={"residual_risk_statement": "A changed safe residual risk statement."}
    )
    configuration = operator_configuration().model_copy(
        update={"read_safety_attestation": tampered}
    )
    assert "ATTESTATION_HASH_MISMATCH" in _validation_errors(configuration)


def test_unordered_attestation_fields_and_self_hash_are_canonical():
    configuration = operator_configuration()
    attestation = configuration.read_safety_attestation
    assert attestation is not None
    reordered = attestation.model_copy(
        update={
            "evidence_references": frozenset(
                reversed(tuple(attestation.evidence_references))
            ),
            "permitted_query_template_ids": frozenset(
                reversed(tuple(attestation.permitted_query_template_ids))
            ),
            "prohibited_capabilities": frozenset(
                reversed(tuple(attestation.prohibited_capabilities))
            ),
            "attestation_hash": "f" * 64,
        }
    )

    assert attestation_content_hash(reordered) == attestation.attestation_hash


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("expires_at", datetime(2026, 8, 29, tzinfo=UTC)),
        ("graph_alias", "changed-graph"),
        ("content_version", "changed-content"),
    ],
)
def test_identity_bearing_attestation_fields_change_hash(field, value):
    attestation = operator_configuration().read_safety_attestation
    assert attestation is not None

    changed = attestation.model_copy(update={field: value})

    assert attestation_content_hash(changed) != attestation.attestation_hash


def test_changed_prohibited_capability_changes_hash():
    attestation = operator_configuration().read_safety_attestation
    assert attestation is not None
    changed = attestation.model_copy(
        update={
            "prohibited_capabilities": frozenset(
                set(ProhibitedCapability) - {ProhibitedCapability.GRAPH_WRITE}
            )
        }
    )

    assert attestation_content_hash(changed) != attestation.attestation_hash


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("query_timeout_seconds", 4),
        ("maximum_records", 9),
        ("maximum_graph_hops", 2),
    ],
)
def test_runtime_safety_caps_change_configuration_hash(field, value):
    configuration = operator_configuration()
    attestation = configuration.read_safety_attestation
    assert attestation is not None
    changed = configuration.model_copy(update={field: value})

    assert (
        read_safety_configuration_hash(
            changed,
            default_template_registry(),
            attestation,
        )
        != attestation.configuration_hash
    )


def test_template_hash_change_changes_configuration_hash():
    configuration = operator_configuration()
    attestation = configuration.read_safety_attestation
    assert attestation is not None
    registry = default_template_registry()

    class ChangedHashRegistry:
        def get(self, template_id, version):
            registry.get(template_id, version)
            return type("ChangedTemplate", (), {"cypher_sha256": "f" * 64})()

    assert (
        read_safety_configuration_hash(
            configuration,
            ChangedHashRegistry(),
            attestation,
        )
        != attestation.configuration_hash
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (
            "permitted_query_template_ids",
            [
                "generic.schema_preflight@1.0.0",
                "generic.schema_preflight@1.0.0",
            ],
        ),
        ("evidence_references", ["same-evidence", "same-evidence"]),
        ("prohibited_capabilities", ["GRAPH_WRITE", "GRAPH_WRITE"]),
    ],
)
def test_unordered_input_duplicates_fail_closed(field, value):
    payload = operator_configuration().read_safety_attestation.model_dump(mode="python")
    payload[field] = value
    with pytest.raises(
        ValidationError,
        match="ATTESTATION_DUPLICATE_UNORDERED_VALUE",
    ):
        ReadSafetyAttestation.model_validate(payload)


def test_naive_timestamps_fail_closed():
    payload = operator_configuration().read_safety_attestation.model_dump(mode="python")

    payload["reviewed_at"] = datetime(2026, 7, 29)
    with pytest.raises(ValidationError):
        ReadSafetyAttestation.model_validate(payload)


def test_hashes_are_identical_across_processes_and_pythonhashseed_values():
    command = (
        "from tests.helpers_read_safety import operator_configuration;"
        "a=operator_configuration().read_safety_attestation;"
        "print(a.configuration_hash, a.attestation_hash)"
    )
    results = set()
    for seed in ("1", "2", "7", "41", "999", "random"):
        environment = {**os.environ, "PYTHONHASHSEED": seed}
        completed = subprocess.run(
            [sys.executable, "-c", command],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        results.add(completed.stdout.strip())

    assert len(results) == 1, "ATTESTATION_HASH_NONDETERMINISTIC"


def test_yaml_orders_and_formats_recompute_same_hashes_across_processes(tmp_path):
    configuration = operator_configuration()
    attestation = configuration.read_safety_attestation
    assert attestation is not None
    base = attestation.model_dump(mode="json")
    seeds = ("1", "2", "7", "41", "999", "random")
    script = """
import sys
import yaml
from auto_researcher.contracts.enums import ReadSafetyMode
from auto_researcher.knowledge.models import KnowledgeProviderConfiguration
from auto_researcher.knowledge.read_safety import (
    attestation_content_hash,
    parse_read_safety_attestation,
    read_safety_configuration_hash,
)
from auto_researcher.knowledge.templates import default_template_registry
payload = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
attestation = parse_read_safety_attestation(payload)
configuration = KnowledgeProviderConfiguration(
    provider_id="neo4j",
    graph_alias="cell-biology",
    database="neo4j",
    schema_version="knowledge-graph-auto-v0.1",
    content_version="backbone-test",
    query_timeout_seconds=5,
    maximum_records=10,
    maximum_graph_hops=3,
    minimum_assertion_confidence=0.6,
    allowed_trust_tiers=frozenset({"CURATED", "CORPUS"}),
    read_safety_mode=ReadSafetyMode.OPERATOR_ATTESTED,
    read_safety_attestation=attestation,
)
print(
    read_safety_configuration_hash(
        configuration, default_template_registry(), attestation
    ),
    attestation_content_hash(attestation),
)
"""
    results = set()
    items = tuple(base.items())
    for index, seed in enumerate(seeds):
        rotated = items[index:] + items[:index]
        payload = dict(rotated)
        for field in (
            "evidence_references",
            "permitted_query_template_ids",
            "prohibited_capabilities",
        ):
            if index % 2:
                payload[field] = list(reversed(payload[field]))
        path = tmp_path / f"attestation-{index}.yaml"
        rendered = yaml.safe_dump(
            payload,
            sort_keys=index % 3 == 0,
            default_flow_style=index % 3 == 1,
            allow_unicode=True,
        )
        path.write_text(
            rendered if index % 2 else rendered.rstrip(),
            encoding="utf-8",
        )
        completed = subprocess.run(
            [sys.executable, "-c", script, str(path)],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        results.add(completed.stdout.strip())

    expected = f"{attestation.configuration_hash} {attestation.attestation_hash}"
    assert results == {expected}, "ATTESTATION_HASH_NONDETERMINISTIC"


def test_model_json_round_trip_preserves_canonical_hash():
    configuration = operator_configuration()
    attestation = configuration.read_safety_attestation
    assert attestation is not None
    restored = ReadSafetyAttestation.model_validate_json(attestation.model_dump_json())
    restored_configuration = KnowledgeProviderConfiguration.model_validate_json(
        configuration.model_dump_json()
    )

    assert restored.attestation_hash == attestation.attestation_hash
    assert attestation_content_hash(restored) == attestation.attestation_hash
    assert (
        read_safety_configuration_hash(
            restored_configuration,
            default_template_registry(),
            restored,
        )
        == attestation.configuration_hash
    )
    assert restored.attestation_hash_algorithm == CANONICAL_HASH_ALGORITHM
    assert restored.configuration_hash_algorithm == CANONICAL_HASH_ALGORITHM


def test_attestation_rejects_credentials_email_and_raw_uri():
    attestation = operator_configuration().read_safety_attestation
    assert attestation is not None
    payload = attestation.model_dump(mode="python")
    for addition in (
        {"username": "not-allowed"},
        {"password": "not-allowed"},
        {"uri": "not-allowed"},
    ):
        with pytest.raises(ValidationError):
            ReadSafetyAttestation.model_validate({**payload, **addition})
    with pytest.raises(ValidationError):
        ReadSafetyAttestation.model_validate(
            {**payload, "reviewer": "operator@example.test"}
        )
    with pytest.raises(ValidationError):
        ReadSafetyAttestation.model_validate(
            {
                **payload,
                "residual_risk_statement": (
                    "Connection details neo4j+s://example are prohibited."
                ),
            }
        )


def test_expiry_is_checked_against_an_aware_clock():
    configuration = operator_configuration(
        expires_at=datetime(2026, 7, 30, 11, 59, tzinfo=UTC)
    )
    assert "ATTESTATION_EXPIRED" in _validation_errors(configuration)
    with pytest.raises(ValueError, match="timezone-aware"):
        validate_operator_attestation(
            configuration.read_safety_attestation,
            configuration,
            default_template_registry(),
            now=datetime(2026, 7, 30),
        )


def test_future_review_cannot_be_used_before_it_becomes_valid():
    configuration = operator_configuration()
    attestation = configuration.read_safety_attestation
    assert attestation is not None
    future = seal_attestation(
        attestation.model_copy(
            update={
                "reviewed_at": datetime(2026, 8, 1, tzinfo=UTC),
                "expires_at": datetime(2026, 8, 30, tzinfo=UTC),
            }
        )
    )
    configuration = configuration.model_copy(update={"read_safety_attestation": future})

    assert "ATTESTATION_NOT_YET_VALID" in _validation_errors(configuration)


def test_attestation_only_applies_to_operator_mode():
    operator = operator_configuration()
    payload = operator.model_dump(mode="python")
    payload["read_safety_mode"] = "PRIVILEGE_VERIFIED"
    with pytest.raises(ValidationError, match="only in OPERATOR_ATTESTED"):
        KnowledgeProviderConfiguration.model_validate(payload)
