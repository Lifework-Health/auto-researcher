"""Safe, task-owned iCCA knowledge plan and relevance policy."""

from __future__ import annotations

import re

from auto_researcher.contracts.enums import KnowledgeGroundingMode, SearchType
from auto_researcher.contracts.models import ResearchContract
from auto_researcher.knowledge.models import (
    CURIE_PATTERN,
    KnowledgeGroundingPolicy,
    KnowledgeQueryPlan,
    KnowledgeSourceType,
    KnowledgeTemplateRequest,
    KnowledgeTrustTier,
)
from auto_researcher.search.protocols import SearchCapability
from auto_researcher.tasks.icca_nbs.bindings import ICCABindings
from auto_researcher.tasks.models import TaskRuntimeContext


def icca_query_plan(
    contract: ResearchContract,
    runtime_context: TaskRuntimeContext,
    search_capabilities: dict[SearchType, SearchCapability],
    bindings: ICCABindings,
) -> KnowledgeQueryPlan:
    del search_capabilities
    configured = runtime_context.task_options.get("grounding", {})
    if not isinstance(configured, dict):
        raise ValueError("iCCA grounding task option must be a mapping")
    maximum = contract.grounding.maximum_query_records
    requests: list[KnowledgeTemplateRequest] = []
    if configured.get("include_network_catalog", True):
        requested = configured.get("network_codenames")
        codenames = (
            tuple(str(item) for item in requested)
            if isinstance(requested, list)
            else tuple(
                sorted(
                    str(getattr(item, "doc_name", getattr(item, "name", item)))
                    for item in bindings.network_type
                )
            )
        )
        requests.append(
            KnowledgeTemplateRequest(
                template_id="icca_nbs.network_catalog",
                template_version="1.0.0",
                parameters={
                    "codenames": list(codenames),
                    "limit": min(len(codenames), maximum),
                },
                maximum_records=min(maximum, max(1, len(codenames))),
                rationale="Ground only task-supported iCCA network choices.",
            )
        )
    gene_curies = _validated_curies(configured.get("gene_curies", ()), "gene")
    if gene_curies and configured.get("gene_seed_provenance") not in {
        "CURATED",
        "ONTOLOGY",
        "PUBLICATION",
    }:
        raise ValueError("iCCA gene seeds require non-patient curated provenance")
    if configured.get("include_pathways", bool(gene_curies)) and gene_curies:
        requests.append(
            KnowledgeTemplateRequest(
                template_id="icca_nbs.gene_signature_pathway",
                template_version="1.0.0",
                parameters={
                    "gene_curies": list(gene_curies),
                    "limit": maximum,
                },
                maximum_records=maximum,
                rationale="Ground configured non-patient gene seeds in registered signatures and pathways.",
            )
        )
    signature_curies = _validated_signature_ids(configured.get("signature_ids", ()))
    for signature_curie in signature_curies:
        requests.append(
            KnowledgeTemplateRequest(
                template_id="generic.entity_lookup",
                template_version="1.0.0",
                parameters={"curie": signature_curie, "limit": 1},
                maximum_records=1,
                rationale="Resolve one configured signature identifier exactly.",
            )
        )
    disease_curies = _validated_curies(
        configured.get("disease_curies", ()),
        "disease",
    )
    if (
        configured.get("include_disease_context", bool(disease_curies))
        and disease_curies
    ):
        requests.append(
            KnowledgeTemplateRequest(
                template_id="icca_nbs.disease_context",
                template_version="1.0.0",
                parameters={
                    "disease_curies": list(disease_curies),
                    "limit": maximum,
                },
                maximum_records=maximum,
                rationale="Ground only explicitly configured disease identifiers.",
            )
        )
    if configured.get("include_immune_bridge", False):
        if not disease_curies:
            raise ValueError("iCCA immune bridge requires stable disease CURIEs")
        requests.append(
            KnowledgeTemplateRequest(
                template_id="icca_nbs.immune_bridge",
                template_version="1.0.0",
                parameters={
                    "disease_curies": list(disease_curies),
                    "limit": maximum,
                },
                maximum_records=maximum,
                rationale="Retrieve the registered disease-to-immune bridge.",
            )
        )
    requests = _fit_query_budget(requests, maximum)
    return KnowledgeQueryPlan(
        task_id="icca_nbs",
        task_version="1.0",
        schema_version=contract.grounding.knowledge_schema_version,
        query_plan_version="1.0.0",
        template_requests=tuple(requests),
        grounding_policy_id="icca-nbs-grounding-v1",
        maximum_total_records=maximum,
        maximum_references=contract.grounding.maximum_knowledge_references,
        required=contract.grounding.mode == KnowledgeGroundingMode.REQUIRED,
    )


def _validated_curies(values, field: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"iCCA {field}_curies must be a list")
    result = tuple(sorted({str(item) for item in values}))
    if any(not CURIE_PATTERN.fullmatch(item) for item in result):
        raise ValueError(f"iCCA {field}_curies require stable CURIEs")
    return result


def _validated_signature_ids(values) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError("iCCA signature_ids must be a list")
    result = []
    for value in sorted({str(item) for item in values}):
        curie = value if ":" in value else f"MSIGDB:{value}"
        if not CURIE_PATTERN.fullmatch(curie):
            raise ValueError("iCCA signature_ids require stable identifiers")
        prefix, local = curie.split(":", 1)
        if prefix != "MSIGDB" or not re.fullmatch(r"[A-Za-z0-9_.-]+", local):
            raise ValueError("iCCA signature_ids must use the MSIGDB namespace")
        result.append(curie)
    return tuple(result)


def _fit_query_budget(
    requests: list[KnowledgeTemplateRequest],
    maximum: int,
) -> list[KnowledgeTemplateRequest]:
    if len(requests) > maximum:
        raise ValueError("knowledge query count exceeds the record budget")
    if not requests:
        return requests
    per_query = max(1, maximum // len(requests))
    fitted = []
    for request in requests:
        record_limit = min(request.maximum_records, per_query)
        parameters = dict(request.parameters)
        parameters["limit"] = min(int(parameters["limit"]), record_limit)
        fitted.append(
            request.model_copy(
                update={
                    "parameters": parameters,
                    "maximum_records": record_limit,
                }
            )
        )
    return fitted


def icca_grounding_policy(
    contract: ResearchContract,
) -> KnowledgeGroundingPolicy:
    return KnowledgeGroundingPolicy(
        policy_id="icca-nbs-grounding-v1",
        allowed_entity_types=frozenset(
            {
                "Gene",
                "Signature",
                "Pathway",
                "Network",
                "Disease",
                "Subtype",
                "ClinicalCovariate",
                "CellState",
            }
        ),
        allowed_predicates=frozenset(
            {
                "IDENTIFIES",
                "CATALOGUED_AS",
                "INCLUDES",
                "PARTICIPATES_IN",
                "PART_OF",
                "IS_A",
                "SUBTYPE_OF",
                "DEFINED_BY",
                "IMPLICATED_IN",
                "PROGNOSTIC_IN",
                "HAS_IMMUNE_PHENOTYPE",
            }
        ),
        allowed_source_types=frozenset(
            {
                KnowledgeSourceType.ONTOLOGY_RELEASE,
                KnowledgeSourceType.CURATED_DATABASE,
                KnowledgeSourceType.LITERATURE,
                KnowledgeSourceType.CURATED_ASSERTION,
                KnowledgeSourceType.CORPUS_ASSERTION,
            }
        ),
        allowed_asserted_by=frozenset({"curator", "corpus"}),
        allowed_trust_tiers=frozenset(
            KnowledgeTrustTier(value)
            for value in contract.grounding.permitted_trust_tiers
        ),
        minimum_assertion_confidence=(contract.grounding.minimum_assertion_confidence),
        maximum_references=contract.grounding.maximum_knowledge_references,
        maximum_assertions=contract.grounding.maximum_query_records,
        maximum_entities=contract.grounding.maximum_query_records * 2,
    )
