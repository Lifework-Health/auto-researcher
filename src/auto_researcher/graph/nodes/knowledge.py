"""Deterministic knowledge retrieval before any hypothesis model call."""

from auto_researcher.contracts.enums import (
    KnowledgeGroundingMode,
    KnowledgeRetrievalStatus,
    RunStatus,
    ReadSafetyMode,
)
from auto_researcher.graph.state import ResearchState
from auto_researcher.knowledge.identity import content_hash, retrieval_id
from auto_researcher.knowledge.models import (
    KnowledgeBundleReference,
    KnowledgeErrorCode,
    KnowledgeRetrievalRequest,
)
from auto_researcher.knowledge.runtime import KnowledgeRetrievalExecutionError
from auto_researcher.runtime.dependencies import RuntimeDependencies
from auto_researcher.tasks.protocols import KnowledgeGroundingCapableTask


def retrieve_knowledge(
    state: ResearchState,
    dependencies: RuntimeDependencies,
) -> dict:
    requirement = state["contract"].grounding
    if requirement.mode == KnowledgeGroundingMode.DISABLED:
        return {
            "knowledge_retrieval_status": KnowledgeRetrievalStatus.DISABLED,
            "knowledge_bundle_reference": KnowledgeBundleReference(
                status=KnowledgeRetrievalStatus.DISABLED
            ),
            "knowledge_errors": [],
            "knowledge_warnings": [KnowledgeErrorCode.KNOWLEDGE_DISABLED.value],
            "executed_nodes": ["retrieve_knowledge"],
        }
    required = requirement.mode == KnowledgeGroundingMode.REQUIRED
    if not isinstance(dependencies.task, KnowledgeGroundingCapableTask):
        if not required:
            return {
                "knowledge_retrieval_status": KnowledgeRetrievalStatus.DISABLED,
                "knowledge_bundle_reference": KnowledgeBundleReference(
                    status=KnowledgeRetrievalStatus.DISABLED
                ),
                "knowledge_errors": [],
                "knowledge_warnings": [
                    KnowledgeErrorCode.TASK_NOT_KNOWLEDGE_CAPABLE.value
                ],
                "executed_nodes": ["retrieve_knowledge"],
            }
        return _unavailable(
            required,
            KnowledgeErrorCode.TASK_NOT_KNOWLEDGE_CAPABLE.value,
        )
    configuration = dependencies.knowledge_configuration
    provider = dependencies.knowledge_provider
    if configuration is None or provider is None:
        return _unavailable(
            required,
            KnowledgeErrorCode.PROVIDER_NOT_CONFIGURED.value,
        )
    if configuration.provider_id not in requirement.permitted_providers:
        return _unavailable(
            required,
            KnowledgeErrorCode.PROVIDER_NOT_CONFIGURED.value,
        )
    if configuration.read_safety_mode not in requirement.permitted_read_safety_modes:
        return _unavailable(
            required,
            KnowledgeErrorCode.READ_SAFETY_MODE_NOT_PERMITTED.value,
        )
    if configuration.read_safety_mode == ReadSafetyMode.UNVERIFIED:
        return _unavailable(
            required,
            KnowledgeErrorCode.READ_ONLY_NOT_VERIFIED.value,
        )
    try:
        plan = dependencies.task.create_knowledge_query_plan(
            state["contract"],
            dependencies.runtime_context,
            dependencies.search_capabilities,
        )
        policy = dependencies.task.create_grounding_policy(state["contract"])
        policy = _apply_runtime_policy(policy, configuration)
        _validate_plan_and_policy(
            plan,
            policy,
            state["contract"],
            configuration,
        )
        template_hashes = {}
        for template_request in plan.template_requests:
            template, _ = dependencies.knowledge_template_registry.validate_request(
                template_request,
                task_id=plan.task_id,
                schema_version=plan.schema_version,
            )
            if template.maximum_hops > requirement.maximum_graph_hops:
                raise ValueError("knowledge template exceeds the graph-hop limit")
            template_hashes[f"{template.template_id}@{template.version}"] = (
                template.cypher_sha256
            )
        provider_templates = provider.execution_template_hashes()
        if set(template_hashes) & set(provider_templates):
            raise ValueError("provider and task template identities overlap")
        template_hashes.update(provider_templates)
        policy_digest = content_hash(policy)
        plan_digest = content_hash(
            {
                "query_plan": plan.model_dump(mode="json"),
                "template_hashes": template_hashes,
                "grounding_policy_hash": policy_digest,
                "read_safety": {
                    "mode": configuration.read_safety_mode.value,
                    "attestation_id": (
                        configuration.read_safety_attestation.attestation_id
                        if configuration.read_safety_attestation
                        else None
                    ),
                    "attestation_version": (
                        configuration.read_safety_attestation.attestation_version
                        if configuration.read_safety_attestation
                        else None
                    ),
                    "attestation_hash": (
                        configuration.read_safety_attestation.attestation_hash
                        if configuration.read_safety_attestation
                        else None
                    ),
                    "attestation_hash_algorithm": (
                        configuration.read_safety_attestation.attestation_hash_algorithm
                        if configuration.read_safety_attestation
                        else None
                    ),
                    "configuration_hash_algorithm": (
                        configuration.read_safety_attestation.configuration_hash_algorithm
                        if configuration.read_safety_attestation
                        else None
                    ),
                },
            }
        )
        request_id = retrieval_id(
            run_id=state["run_id"],
            cycle=state["cycle"],
            task_id=plan.task_id,
            task_version=plan.task_version,
            contract_id=state["contract"].contract_id,
            provider_id=provider.provider_id,
            provider_version=provider.provider_version,
            graph_alias=configuration.graph_alias,
            schema_version=configuration.schema_version,
            content_version=configuration.content_version,
            query_plan_version=plan.query_plan_version,
            plan_hash=plan_digest,
        )
        request = KnowledgeRetrievalRequest(
            retrieval_id=request_id,
            run_id=state["run_id"],
            cycle=state["cycle"],
            provider_id=provider.provider_id,
            graph_alias=configuration.graph_alias,
            schema_version=configuration.schema_version,
            content_version=configuration.content_version,
            query_plan=plan,
            query_plan_hash=plan_digest,
            grounding_policy_hash=policy_digest,
            template_hashes=template_hashes,
            task_id=plan.task_id,
            contract_id=state["contract"].contract_id,
        )
        bundle = dependencies.knowledge_coordinator.replay(request)
        if bundle is None:
            readiness = provider.readiness(configuration)
            if not readiness.ready:
                code = (
                    readiness.errors[0].value
                    if readiness.errors
                    else KnowledgeErrorCode.CONNECTIVITY_FAILED.value
                )
                update = _unavailable(required, code)
                update["knowledge_warnings"] = list(readiness.warnings)
                return update
            bundle, _ = dependencies.knowledge_coordinator.run(
                request,
                provider,
                configuration,
                policy,
            )
    except (ValueError, KeyError, KnowledgeRetrievalExecutionError) as exc:
        code = (
            exc.code
            if isinstance(exc, KnowledgeRetrievalExecutionError)
            else KnowledgeErrorCode.BUNDLE_VALIDATION_FAILED.value
        )
        return _unavailable(required, code)
    reference = KnowledgeBundleReference(
        bundle_id=bundle.bundle_id,
        bundle_hash=bundle.bundle_hash,
        retrieval_id=bundle.retrieval_id,
        provider_id=provider.provider_id,
        reference_ids=tuple(item.reference_id for item in bundle.references),
        trust_summary=bundle.validation_result.trust_tier_summary,
        artefact_reference=(
            bundle.artefact_references[3]
            if len(bundle.artefact_references) > 3
            else None
        ),
        status=KnowledgeRetrievalStatus.COMPLETED,
    )
    if not bundle.references:
        return {
            "status": RunStatus.STOPPED if required else RunStatus.RUNNING,
            "stop_reason": ("required_knowledge_unavailable" if required else None),
            "knowledge_retrieval_status": KnowledgeRetrievalStatus.COMPLETED,
            "knowledge_bundle_reference": reference,
            "knowledge_errors": (
                [KnowledgeErrorCode.EMPTY_GROUNDING_RESULT.value] if required else []
            ),
            "knowledge_warnings": (
                [] if required else [KnowledgeErrorCode.EMPTY_GROUNDING_RESULT.value]
            ),
            "executed_nodes": ["retrieve_knowledge"],
        }
    return {
        "knowledge_retrieval_status": KnowledgeRetrievalStatus.COMPLETED,
        "knowledge_bundle_reference": reference,
        "knowledge_errors": [],
        "knowledge_warnings": [],
        "executed_nodes": ["retrieve_knowledge"],
    }


def _unavailable(required: bool, code: str) -> dict:
    return {
        "status": RunStatus.STOPPED if required else RunStatus.RUNNING,
        "stop_reason": "required_knowledge_unavailable" if required else None,
        "knowledge_retrieval_status": KnowledgeRetrievalStatus.FAILED,
        "knowledge_bundle_reference": KnowledgeBundleReference(
            status=KnowledgeRetrievalStatus.FAILED
        ),
        "knowledge_errors": [code],
        "knowledge_warnings": [],
        "executed_nodes": ["retrieve_knowledge"],
    }


def _validate_plan_and_policy(plan, policy, contract, configuration) -> None:
    requirement = contract.grounding
    if plan.task_id != contract.task_id or plan.task_version != contract.task_version:
        raise ValueError("knowledge query plan task identity mismatch")
    if (
        plan.schema_version != requirement.knowledge_schema_version
        or plan.schema_version != configuration.schema_version
    ):
        raise ValueError("knowledge query plan schema mismatch")
    if plan.grounding_policy_id != policy.policy_id:
        raise ValueError("knowledge query plan policy identity mismatch")
    if plan.maximum_total_records > min(
        requirement.maximum_query_records,
        configuration.maximum_records,
    ):
        raise ValueError("knowledge query plan exceeds the record limit")
    if (
        sum(item.maximum_records for item in plan.template_requests)
        > plan.maximum_total_records
    ):
        raise ValueError("knowledge template limits exceed the plan total")
    if plan.maximum_references > requirement.maximum_knowledge_references:
        raise ValueError("knowledge query plan exceeds the reference limit")
    if plan.required != (requirement.mode == KnowledgeGroundingMode.REQUIRED):
        raise ValueError("knowledge query plan required mode mismatch")
    if policy.maximum_references > requirement.maximum_knowledge_references:
        raise ValueError("knowledge policy exceeds the reference limit")
    if policy.minimum_assertion_confidence < requirement.minimum_assertion_confidence:
        raise ValueError("knowledge policy weakens the confidence threshold")
    if not {item.value for item in policy.allowed_trust_tiers}.issubset(
        requirement.permitted_trust_tiers
    ):
        raise ValueError("knowledge policy includes a disallowed trust tier")


def _apply_runtime_policy(policy, configuration):
    updates = {}
    if configuration.minimum_assertion_confidence is not None:
        updates["minimum_assertion_confidence"] = max(
            policy.minimum_assertion_confidence,
            configuration.minimum_assertion_confidence,
        )
    if configuration.allowed_trust_tiers is not None:
        updates["allowed_trust_tiers"] = (
            policy.allowed_trust_tiers & configuration.allowed_trust_tiers
        )
    return policy.model_copy(update=updates) if updates else policy
