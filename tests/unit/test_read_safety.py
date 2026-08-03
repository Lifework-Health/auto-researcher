from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from auto_researcher.knowledge.models import KnowledgeProviderConfiguration
from auto_researcher.knowledge.read_safety import (
    ReadSafetyAttestation,
    attestation_content_hash,
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
        ("content_version", "different-content", "ATTESTATION_CONTENT_VERSION_MISMATCH"),
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
    payload = operator_configuration().read_safety_attestation.model_dump(
        mode="python"
    )
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
                "permitted_query_template_ids": (
                    "generic.schema_preflight@1.0.0",
                    "missing.template@1.0.0",
                )
            }
        )
    )
    template_mismatch = configuration.model_copy(
        update={"read_safety_attestation": unknown_template}
    )
    assert "ATTESTATION_TEMPLATE_SET_MISMATCH" in _validation_errors(
        template_mismatch
    )

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
    configuration = configuration.model_copy(
        update={"read_safety_attestation": future}
    )

    assert "ATTESTATION_NOT_YET_VALID" in _validation_errors(configuration)


def test_attestation_only_applies_to_operator_mode():
    operator = operator_configuration()
    payload = operator.model_dump(mode="python")
    payload["read_safety_mode"] = "PRIVILEGE_VERIFIED"
    with pytest.raises(ValidationError, match="only in OPERATOR_ATTESTED"):
        KnowledgeProviderConfiguration.model_validate(payload)
