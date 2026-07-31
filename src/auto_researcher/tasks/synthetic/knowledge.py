"""Synthetic task knowledge query plan and evidence policy."""

from auto_researcher.contracts.enums import KnowledgeGroundingMode, SearchType
from auto_researcher.contracts.models import ResearchContract
from auto_researcher.knowledge.models import (
    KnowledgeGroundingPolicy,
    KnowledgeQueryPlan,
    KnowledgeSourceType,
    KnowledgeTemplateRequest,
    KnowledgeTrustTier,
)
from auto_researcher.search.protocols import SearchCapability
from auto_researcher.tasks.models import TaskRuntimeContext


def synthetic_query_plan(
    contract: ResearchContract,
    runtime_context: TaskRuntimeContext,
    search_capabilities: dict[SearchType, SearchCapability],
) -> KnowledgeQueryPlan:
    del runtime_context, search_capabilities
    limit = min(10, contract.grounding.maximum_query_records)
    return KnowledgeQueryPlan(
        task_id="synthetic",
        task_version="1.0",
        schema_version=contract.grounding.knowledge_schema_version,
        query_plan_version="1.0.0",
        template_requests=(
            KnowledgeTemplateRequest(
                template_id="generic.entity_lookup",
                template_version="1.0.0",
                parameters={"curie": "SYNTH:complexity", "limit": limit},
                maximum_records=limit,
                rationale="Retrieve the registered synthetic parameter identity.",
            ),
        ),
        grounding_policy_id="synthetic-grounding-v1",
        maximum_total_records=limit,
        maximum_references=contract.grounding.maximum_knowledge_references,
        required=contract.grounding.mode == KnowledgeGroundingMode.REQUIRED,
    )


def synthetic_grounding_policy(
    contract: ResearchContract,
) -> KnowledgeGroundingPolicy:
    tiers = frozenset(
        KnowledgeTrustTier(value) for value in contract.grounding.permitted_trust_tiers
    )
    return KnowledgeGroundingPolicy(
        policy_id="synthetic-grounding-v1",
        allowed_entity_types=frozenset({"Parameter", "Metric"}),
        allowed_predicates=frozenset({"BOUNDS", "ASSOCIATED_WITH"}),
        allowed_source_types=frozenset(
            {
                KnowledgeSourceType.ONTOLOGY_RELEASE,
                KnowledgeSourceType.LITERATURE,
                KnowledgeSourceType.LIVE_ASSERTION,
            }
        ),
        allowed_asserted_by=frozenset({"curator", "corpus", "llm"}),
        allowed_trust_tiers=tiers,
        minimum_assertion_confidence=(contract.grounding.minimum_assertion_confidence),
        maximum_references=contract.grounding.maximum_knowledge_references,
        maximum_assertions=20,
        maximum_entities=20,
    )
