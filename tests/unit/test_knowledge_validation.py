from __future__ import annotations

from auto_researcher.contracts.models import KnowledgeGroundingRequirement
from auto_researcher.knowledge.identity import content_hash, query_plan_hash
from auto_researcher.knowledge.models import (
    KnowledgeProviderConfiguration,
    KnowledgeRetrievalRequest,
    KnowledgeTrustTier,
)
from auto_researcher.knowledge.providers.static import StaticKnowledgeProvider
from auto_researcher.knowledge.templates import default_template_registry
from auto_researcher.knowledge.validation import KnowledgeBundleValidator
from auto_researcher.tasks.models import TaskRuntimeContext
from auto_researcher.tasks.synthetic import default_synthetic_contract
from auto_researcher.tasks.synthetic.knowledge import (
    synthetic_grounding_policy,
    synthetic_query_plan,
)
from tests.conftest import fixed_clock


def _draft_and_policy():
    requirement = KnowledgeGroundingRequirement(
        mode="OPTIONAL",
        permitted_providers=frozenset({"static"}),
        permitted_trust_tiers=frozenset({"CURATED", "CORPUS"}),
        minimum_assertion_confidence=0.6,
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
        graph_alias="synthetic-fixture",
        database="static",
        schema_version="synthetic-v1",
        content_version="fixture-v1",
        maximum_records=10,
    )
    policy = synthetic_grounding_policy(contract)
    request = KnowledgeRetrievalRequest(
        retrieval_id="retrieval-1",
        run_id="run-1",
        cycle=1,
        provider_id="static",
        graph_alias=configuration.graph_alias,
        schema_version=configuration.schema_version,
        content_version=configuration.content_version,
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
    draft = StaticKnowledgeProvider(
        configuration,
        clock=fixed_clock,
    ).retrieve(request)
    return draft, policy, configuration


def _validate(draft, policy, configuration):
    return KnowledgeBundleValidator().validate(
        draft,
        policy,
        provider_id=configuration.provider_id,
        schema_version=configuration.schema_version,
        content_version=configuration.content_version,
        maximum_records=configuration.maximum_records,
        query_plan_hash=draft.query_plan_hash,
    )


def test_static_bundle_accepts_only_source_backed_curated_grounding():
    draft, policy, configuration = _draft_and_policy()
    validated = _validate(draft, policy, configuration)
    assert validated.validation_result.passed
    assert validated.validation_result.accepted_reference_count == 2
    assert validated.validation_result.rejected_assertion_count == 1
    assert validated.validation_result.trust_tier_summary == {"CURATED": 2}
    assert len(validated.references) == 2
    assert validated.references[0].trust_tier == KnowledgeTrustTier.CURATED
    assert validated.references[0].prior_weight_cap == 0.9
    assert all(item.asserted_by != "llm" for item in validated.assertions)


def test_validation_fails_on_identity_version_patient_and_path_leaks():
    draft, policy, configuration = _draft_and_policy()
    bad_snapshot = draft.graph_snapshot.model_copy(update={"schema_version": "wrong"})
    bad_reference = draft.references[0].model_copy(
        update={
            "concise_claim": ("patient-123 was loaded from /Users/example/private.csv")
        }
    )
    bad = draft.model_copy(
        update={
            "graph_snapshot": bad_snapshot,
            "references": (bad_reference, *draft.references[1:]),
        }
    )
    validated = _validate(bad, policy, configuration)
    assert not validated.validation_result.passed
    reasons = set(validated.validation_result.reason_codes)
    assert "schema_version_mismatch" in reasons
    assert "patient_like_identifier" in reasons
    assert "absolute_runtime_path" in reasons


def test_missing_or_invented_source_never_becomes_a_reference():
    draft, policy, configuration = _draft_and_policy()
    assertion = draft.assertions[1].model_copy(
        update={"source_references": ("source:invented",)}
    )
    reference = draft.references[1].model_copy(
        update={"source_references": ("source:invented",)}
    )
    changed = draft.model_copy(
        update={
            "assertions": (assertion,),
            "references": (reference,),
        }
    )
    validated = _validate(changed, policy, configuration)
    assert validated.references == ()
    assert "missing_assertion_source" in validated.validation_result.reason_codes
