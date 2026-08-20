"""Task-driven runtime assembly; LangGraph receives only generic dependencies."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
import importlib.util
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from auto_researcher.agents.mock import (
    ConfiguredPlannerAgent,
    MockHypothesisAgent,
)
from auto_researcher.agents.protocols import HypothesisAgent, PlannerAgent
from auto_researcher.agents.call_store import (
    AgentCallStore,
    InMemoryAgentCallStore,
    SQLiteAgentCallStore,
)
from auto_researcher.agents.context import AgentContextAssembler, AgentContextLimits
from auto_researcher.agents.live import LiveHypothesisAgent, LivePlannerAgent
from auto_researcher.agents.models import (
    AgentBudgetPolicy,
    ModelCallConfig,
    TaskAgentContext,
)
from auto_researcher.providers.protocols import StructuredModelClient
from auto_researcher.knowledge.models import KnowledgeProviderConfiguration
from auto_researcher.knowledge.protocols import KnowledgeProvider
from auto_researcher.knowledge.registry import KnowledgeProviderRegistry
from auto_researcher.knowledge.runtime import KnowledgeRetrievalCoordinator
from auto_researcher.knowledge.store import (
    InMemoryKnowledgeRetrievalStore,
    KnowledgeRetrievalStore,
    SQLiteKnowledgeRetrievalStore,
)
from auto_researcher.knowledge.templates import (
    KnowledgeQueryTemplateRegistry,
    default_template_registry,
)
from auto_researcher.knowledge.validation import KnowledgeBundleValidator
from auto_researcher.contracts.enums import SearchType
from auto_researcher.contracts.models import ResearchContract, SearchRequest
from auto_researcher.evaluation.protocols import Evaluator
from auto_researcher.provenance.protocols import ProvenanceStore
from auto_researcher.provenance.sqlite_store import SQLiteProvenanceStore
from auto_researcher.runtime.checkpoints import memory_checkpointer, sqlite_checkpointer
from auto_researcher.search.direct import DirectSearchBackend
from auto_researcher.search.openevolve.backend import OpenEvolveBackend
from auto_researcher.search.openevolve.live_runtime import (
    MetadataOnlyLiveOpenEvolveRuntime,
    assemble_metadata_only_live_openevolve,
)
from auto_researcher.search.openevolve.native_engine import (
    ApprovedModel,
    ScientificCandidateIdentity,
)
from auto_researcher.search.openevolve.native_runtime import (
    StandardNativeOpenEvolveRuntime,
)
from auto_researcher.search.openevolve.protocols import MutationOperator
from auto_researcher.search.openevolve.mutation import DeterministicMutationOperator
from auto_researcher.search.openevolve.sandbox import LocalSandboxRunner
from auto_researcher.resources import (
    CourtesyResourceAdmissionPolicy,
    InMemoryResourceLeaseStore,
    NvidiaGPUResourceProvider,
    ResourceBroker,
    ResourceRequest,
    ResourceRequirement,
)
from auto_researcher.search.protocols import SearchBackend, SearchCapability
from auto_researcher.search.registry import SearchBackendRegistry
from auto_researcher.search.optuna.backend import OptunaAskTellBackend
from auto_researcher.search.optuna.storage import (
    OptunaStorageHandle,
    in_memory_storage,
    sqlite_storage,
)
from auto_researcher.tasks.models import (
    ArtefactPolicy,
    DatasetManifest,
    ExperimentMetadata,
    TaskDescriptor,
    TaskNotReadyError,
    TaskRuntimeContext,
)
from auto_researcher.tasks.protocols import (
    AgentContextCapableTask,
    IntermediateReportingEvaluator,
    OpenEvolveCapableTask,
    OptunaCapableTask,
    ResearchTask,
    VerificationPolicy,
)
from auto_researcher.verification.verifier import DeterministicVerifier, Verifier


@dataclass(frozen=True)
class RuntimeDependencies:
    hypothesis_agent: HypothesisAgent
    planner_agent: PlannerAgent
    direct_search_backend: SearchBackend
    evaluator: Evaluator
    verifier: Verifier
    provenance_store: ProvenanceStore
    agent_call_store: AgentCallStore
    knowledge_retrieval_store: KnowledgeRetrievalStore
    knowledge_provider_registry: KnowledgeProviderRegistry
    knowledge_provider: KnowledgeProvider | None
    knowledge_configuration: KnowledgeProviderConfiguration | None
    knowledge_template_registry: KnowledgeQueryTemplateRegistry
    knowledge_coordinator: KnowledgeRetrievalCoordinator
    agent_context_assembler: AgentContextAssembler
    task_agent_context: TaskAgentContext
    agent_budget_policy: AgentBudgetPolicy
    checkpointer: Any
    clock: Callable[[], datetime]
    id_generator: Callable[[str], str]
    task_descriptor: TaskDescriptor
    dataset_manifest: DatasetManifest
    artefact_policy: ArtefactPolicy
    verification_policy: VerificationPolicy
    task: ResearchTask
    runtime_context: TaskRuntimeContext
    experiment_metadata: ExperimentMetadata
    search_capabilities: dict[SearchType, SearchCapability]
    search_backend_registry: SearchBackendRegistry
    optuna_backend: OptunaAskTellBackend | None = None
    optuna_storage_handle: OptunaStorageHandle | None = None
    openevolve_backend: OpenEvolveBackend | None = None
    native_openevolve_runtime: StandardNativeOpenEvolveRuntime | None = None


def utc_now() -> datetime:
    return datetime.now(UTC)


def random_id(prefix: str) -> str:
    return f"{prefix}-{uuid4()}"


def _context_for_contract(
    context: TaskRuntimeContext,
    contract: ResearchContract,
    manifest_created_at: datetime,
) -> TaskRuntimeContext:
    options = dict(context.task_options)
    options["objective_version"] = contract.objective_version
    payload = context.model_dump(mode="python")
    payload.update(
        {
            "task_options": options,
            "manifest_created_at": context.manifest_created_at or manifest_created_at,
        }
    )
    return TaskRuntimeContext.model_validate(payload)


def _assemble_task_dependencies(
    *,
    task: ResearchTask,
    runtime_context: TaskRuntimeContext,
    contract: ResearchContract,
    experiment_configuration: dict,
    provenance_store: ProvenanceStore,
    checkpointer: Any,
    hypothesis_agent: HypothesisAgent | None,
    planner_agent: PlannerAgent | None,
    evaluator: Evaluator | None,
    verifier: Verifier | None,
    agent_call_store: AgentCallStore | None,
    model_client: StructuredModelClient | None,
    planner_model_client: StructuredModelClient | None,
    hypothesis_call_config: ModelCallConfig | None,
    planner_call_config: ModelCallConfig | None,
    agent_budget_policy: AgentBudgetPolicy | None,
    knowledge_provider: KnowledgeProvider | None,
    knowledge_configuration: KnowledgeProviderConfiguration | None,
    knowledge_retrieval_store: KnowledgeRetrievalStore | None,
    knowledge_template_registry: KnowledgeQueryTemplateRegistry | None,
    clock: Callable[[], datetime],
    id_generator: Callable[[str], str],
    search_type: SearchType = SearchType.DIRECT,
    optuna_storage_handle: OptunaStorageHandle | None = None,
    openevolve_mutation_operator: MutationOperator | None = None,
    openevolve_sandbox_runner: Any | None = None,
    openevolve_live_runtime: MetadataOnlyLiveOpenEvolveRuntime | None = None,
    native_openevolve_models: tuple[ApprovedModel, ...] = (),
    native_openevolve_resource_broker: ResourceBroker | None = None,
) -> RuntimeDependencies:
    task.validate_contract(contract)
    context = _context_for_contract(runtime_context, contract, clock())
    readiness = task.readiness(context)
    if not readiness.ready:
        detail = "; ".join(readiness.errors) or "unspecified readiness failure"
        raise TaskNotReadyError(
            f"task {task.task_id}@{task.task_version} is not ready: {detail}"
        )
    descriptor = task.descriptor()
    manifest = task.dataset_manifest(context)
    metadata = task.experiment_metadata(context)
    if metadata.evaluator_id != contract.evaluator_id:
        raise ValueError(
            "task experiment metadata does not match contract evaluator_id"
        )
    if metadata.provenance != contract.provenance:
        raise ValueError("task experiment metadata does not match contract provenance")
    if metadata.dataset_version != manifest.dataset_version:
        raise ValueError("task experiment metadata does not match dataset manifest")
    normalised = (
        task.normalise_configuration(experiment_configuration)
        if search_type == SearchType.DIRECT
        else experiment_configuration
    )
    policy = task.create_verification_policy(contract)
    if policy.policy_id != descriptor.verification_policy_id:
        raise ValueError("task verification policy does not match its descriptor")
    task_evaluator = evaluator or task.create_evaluator(context)
    if task_evaluator.evaluator_id != metadata.evaluator_id:
        raise ValueError("task evaluator does not match experiment metadata")
    selected_verifier = verifier or DeterministicVerifier(policy)
    optuna_installed = importlib.util.find_spec("optuna") is not None
    optuna_capable = isinstance(task, OptunaCapableTask)
    optuna_permitted = SearchType.OPTUNA in contract.allowed_search_types
    if search_type == SearchType.OPTUNA and optuna_capable and optuna_permitted:
        validation_request = SearchRequest(
            request_id="runtime-capability-validation",
            hypothesis_id="runtime-capability-validation",
            search_type=SearchType.OPTUNA,
            target="validate task-owned Optuna search space",
            search_space=experiment_configuration,
            experiment_budget=experiment_configuration.get(
                "trial_budget",
                contract.maximum_experiments,
            ),
            rationale="Validate the task capability before graph execution.",
        )
        validation_spec = cast(OptunaCapableTask, task).create_optuna_study_spec(
            contract,
            validation_request,
        )
        if validation_spec.pruner.type != "none" and not isinstance(
            task_evaluator, IntermediateReportingEvaluator
        ):
            raise ValueError("optuna_pruner_requires_intermediate_reporting_evaluator")
    if (
        search_type == SearchType.OPTUNA
        and optuna_installed
        and optuna_capable
        and optuna_permitted
        and optuna_storage_handle is None
    ):
        optuna_storage_handle = in_memory_storage()
    optuna_available = (
        optuna_installed
        and optuna_capable
        and optuna_permitted
        and optuna_storage_handle is not None
    )
    openevolve_capable = isinstance(task, OpenEvolveCapableTask)
    openevolve_permitted = SearchType.OPENEVOLVE in contract.allowed_search_types
    raw_openevolve = experiment_configuration.get("openevolve", {})
    if not isinstance(raw_openevolve, dict):
        raise ValueError("openevolve_finite_configuration_required")
    native_mode_value = raw_openevolve.get("native_controller")
    if native_mode_value is not None and type(native_mode_value) is not bool:
        raise ValueError("openevolve_native_controller_must_be_boolean")
    native_mode = search_type == SearchType.OPENEVOLVE and native_mode_value is True
    openevolve_backend: OpenEvolveBackend | None = None
    call_store = agent_call_store or InMemoryAgentCallStore()
    if openevolve_capable and openevolve_permitted:
        component = cast(OpenEvolveCapableTask, task).create_evolvable_component(
            contract,
            context,
        )
        verifier_identity = f"{selected_verifier.version}@{policy.policy_id}"
        workspace_root = (
            context.workspace_dir / "openevolve-sandboxes"
            if context.workspace_dir is not None
            else None
        )
        if openevolve_live_runtime is not None:
            if search_type != SearchType.OPENEVOLVE:
                raise ValueError("live_mutation_requires_openevolve_search")
            if (
                openevolve_mutation_operator is not None
                or openevolve_sandbox_runner is not None
            ):
                raise ValueError("live_mutation_runtime_injection_conflict")
            if context.workspace_dir is None:
                raise ValueError("live_mutation_workspace_required")
            if context.run_id is None:
                raise ValueError("live_mutation_runtime_run_required")
            (
                openevolve_mutation_operator,
                openevolve_sandbox_runner,
            ) = assemble_metadata_only_live_openevolve(
                runtime=openevolve_live_runtime,
                task=task,
                component=component,
                research_contract=contract,
                run_id=context.run_id,
                experiment_configuration=experiment_configuration,
                call_store=call_store,
                workspace_root=(
                    context.workspace_dir.expanduser().resolve()
                    / "openevolve-sandboxes"
                ),
                now=clock,
            )
        openevolve_backend = OpenEvolveBackend(
            component,
            metadata,
            verifier_identity,
            openevolve_mutation_operator or DeterministicMutationOperator(),
            openevolve_sandbox_runner or LocalSandboxRunner(workspace_root),
        )
        if search_type == SearchType.OPENEVOLVE:
            validation_request = SearchRequest(
                request_id="runtime-capability-validation",
                hypothesis_id="runtime-capability-validation",
                search_type=SearchType.OPENEVOLVE,
                target="validate task-owned OpenEvolve mutable surface",
                search_space=experiment_configuration,
                experiment_budget=int(
                    experiment_configuration.get(
                        "trial_budget", contract.maximum_experiments
                    )
                ),
                rationale="Validate finite OpenEvolve configuration before execution.",
            )
            openevolve_backend.create_search_contract(validation_request, contract)
    native_runtime: StandardNativeOpenEvolveRuntime | None = None
    if native_mode:
        if openevolve_backend is None or context.run_id is None:
            raise ValueError("native_openevolve_runtime_context_invalid")
        approved_bridge = getattr(openevolve_mutation_operator, "bridge", None)
        if approved_bridge is None and not native_openevolve_models:
            raise ValueError("native_openevolve_approved_model_bridge_required")
        resource_configuration = experiment_configuration.get("resources")
        request_factory = None
        if resource_configuration is not None:
            if not isinstance(resource_configuration, dict):
                raise ValueError("openevolve_resource_configuration_invalid")
            if resource_configuration.get("resource_type", "gpu") != "gpu":
                raise ValueError("openevolve_resource_type_not_supported")
            equivalence = frozenset(
                str(item)
                for item in resource_configuration.get(
                    "equivalence_requirements",
                    ("nvidia-cuda", "whole-physical-gpu"),
                )
            )
            native_openevolve_resource_broker = (
                native_openevolve_resource_broker
                or ResourceBroker(
                    NvidiaGPUResourceProvider(equivalence_tags=equivalence),
                    CourtesyResourceAdmissionPolicy(
                        maximum_utilization_percent=float(
                            resource_configuration.get(
                                "maximum_utilization_percent",
                                100,
                            )
                        )
                    ),
                    lease_store=InMemoryResourceLeaseStore(),
                )
            )

            def request_factory(
                identity: ScientificCandidateIdentity,
            ) -> ResourceRequest:
                return ResourceRequest(
                    request_id=f"native-{identity.evaluation_identity}",
                    requirements=(
                        ResourceRequirement(
                            resource_type="gpu",
                            quantity=int(
                                resource_configuration.get(
                                    "quantity_per_candidate",
                                    1,
                                )
                            ),
                        ),
                    ),
                    maximum_wait_seconds=float(
                        resource_configuration.get(
                            "maximum_wait_seconds",
                            14_400,
                        )
                    ),
                    stable_idle_seconds=float(
                        resource_configuration.get("stable_idle_seconds", 0)
                    ),
                    equivalence_requirements=equivalence,
                )

        if native_openevolve_resource_broker is not None and request_factory is None:
            raise ValueError("native_openevolve_resource_configuration_required")
        if context.output_dir is None:
            raise ValueError("native_openevolve_output_directory_required")
        model_name = None
        if approved_bridge is not None:
            model_name = approved_bridge.contract.model_config_contract.model_id
        native_runtime = StandardNativeOpenEvolveRuntime(
            backend=openevolve_backend,
            component=openevolve_backend.component,
            metadata=metadata,
            contract=contract,
            run_id=context.run_id,
            output_root=(
                context.output_dir.expanduser().resolve()
                / "runs"
                / context.run_id
                / "openevolve-native"
            ),
            evaluator=task_evaluator,
            verifier=selected_verifier,
            provenance_store=provenance_store,
            runtime_context=context,
            dataset_manifest=manifest,
            verification_policy=policy,
            clock=clock,
            approved_models=native_openevolve_models,
            approved_bridge=approved_bridge,
            approved_bridge_model_name=model_name,
            resource_broker=native_openevolve_resource_broker,
            resource_request_factory=request_factory,
        )
    openevolve_available = openevolve_backend is not None
    direct_backend = DirectSearchBackend(
        metadata,
        task.normalise_configuration,
    )
    registry = SearchBackendRegistry()
    registry.register(
        SearchCapability(
            SearchType.DIRECT, True, "BACKEND_AVAILABLE", "DIRECT backend selected"
        ),
        direct_backend,
    )
    registry.register(
        SearchCapability(
            SearchType.OPTUNA,
            optuna_available,
            "BACKEND_AVAILABLE" if optuna_available else "BACKEND_UNAVAILABLE",
            (
                "OPTUNA ask/tell backend selected"
                if optuna_available
                else "OPTUNA requires an Optuna-capable task and the hpo extra"
            ),
        ),
        (
            OptunaAskTellBackend(optuna_storage_handle.storage)
            if optuna_available and optuna_storage_handle
            else None
        ),
    )
    registry.register(
        SearchCapability(
            SearchType.OPENEVOLVE,
            openevolve_available,
            "BACKEND_AVAILABLE" if openevolve_available else "BACKEND_UNAVAILABLE",
            (
                "OPENEVOLVE bounded program-search backend selected"
                if openevolve_available
                else "OPENEVOLVE requires a compatible task-owned evolvable component"
            ),
        ),
        openevolve_backend,
    )
    capabilities = registry.capabilities()
    retrieval_store = knowledge_retrieval_store or InMemoryKnowledgeRetrievalStore()
    template_registry = knowledge_template_registry or default_template_registry()
    provider_registry = KnowledgeProviderRegistry()
    if knowledge_provider is not None:
        provider_registry.register(
            knowledge_provider.provider_id,
            lambda: knowledge_provider,
        )
    if knowledge_configuration is not None:
        if (
            knowledge_provider is not None
            and knowledge_provider.provider_id != knowledge_configuration.provider_id
        ):
            raise ValueError("knowledge provider identity does not match configuration")
        requirement = contract.grounding
        if (
            knowledge_configuration.schema_version
            != requirement.knowledge_schema_version
        ):
            raise ValueError("runtime knowledge schema version does not match contract")
        if (
            knowledge_configuration.content_version
            != requirement.knowledge_content_version
        ):
            raise ValueError(
                "runtime knowledge content version does not match contract"
            )
        if knowledge_configuration.maximum_records > requirement.maximum_query_records:
            raise ValueError("runtime knowledge record limit weakens contract")
        if (
            knowledge_configuration.minimum_assertion_confidence is not None
            and knowledge_configuration.minimum_assertion_confidence
            < requirement.minimum_assertion_confidence
        ):
            raise ValueError("runtime knowledge confidence threshold weakens contract")
        if knowledge_configuration.allowed_trust_tiers is not None and not {
            item.value for item in knowledge_configuration.allowed_trust_tiers
        }.issubset(requirement.permitted_trust_tiers):
            raise ValueError("runtime knowledge trust tiers weaken contract")
        if (
            knowledge_configuration.query_timeout_seconds
            > requirement.maximum_retrieval_duration
        ):
            raise ValueError("runtime knowledge timeout weakens contract")
    budget_policy = agent_budget_policy or AgentBudgetPolicy()
    if not isinstance(task, AgentContextCapableTask):
        raise ValueError(
            f"task {task.task_id}@{task.task_version} does not provide safe agent context"
        )
    task_agent_context = task.create_agent_context(
        contract,
        context,
        capabilities,
    )
    raw_prior_results = context.task_options.get("campaign_prior_results", 5)
    if (
        isinstance(raw_prior_results, bool)
        or not isinstance(raw_prior_results, int)
        or not 1 <= raw_prior_results <= 30
    ):
        raise ValueError("campaign_prior_results_invalid")
    context_assembler = AgentContextAssembler(
        provenance_store,
        knowledge_retrieval_store=retrieval_store,
        clock=clock,
        limits=AgentContextLimits(
            maximum_prior_hypotheses=min(12, raw_prior_results),
            maximum_prior_results=raw_prior_results,
        ),
    )
    knowledge_coordinator = KnowledgeRetrievalCoordinator(
        store=retrieval_store,
        validator=KnowledgeBundleValidator(),
        runtime_context=context,
        clock=clock,
    )
    live_configured = any(
        item is not None
        for item in (
            model_client,
            planner_model_client,
            hypothesis_call_config,
            planner_call_config,
        )
    )
    if live_configured and not all(
        item is not None
        for item in (model_client, hypothesis_call_config, planner_call_config)
    ):
        raise ValueError(
            "live agents require a model client plus hypothesis and planner call configs"
        )
    if live_configured and (hypothesis_agent is not None or planner_agent is not None):
        raise ValueError(
            "live model configuration cannot be mixed with injected agents"
        )
    selected_hypothesis_agent = hypothesis_agent
    selected_planner_agent = planner_agent
    if live_configured:
        assert model_client and hypothesis_call_config and planner_call_config
        selected_hypothesis_agent = LiveHypothesisAgent(
            client=model_client,
            call_config=hypothesis_call_config,
            budget_policy=budget_policy,
            call_store=call_store,
            clock=clock,
        )
        selected_planner_agent = LivePlannerAgent(
            client=planner_model_client or model_client,
            call_config=planner_call_config,
            budget_policy=budget_policy,
            call_store=call_store,
            clock=clock,
            task=task,
            contract=contract,
        )
    return RuntimeDependencies(
        hypothesis_agent=selected_hypothesis_agent or MockHypothesisAgent(),
        planner_agent=selected_planner_agent
        or ConfiguredPlannerAgent(
            normalised,
            search_type=search_type,
            experiment_budget=(
                int(
                    experiment_configuration.get(
                        "trial_budget", contract.maximum_experiments
                    )
                )
                if search_type in {SearchType.OPTUNA, SearchType.OPENEVOLVE}
                else 1
            ),
        ),
        direct_search_backend=direct_backend,
        evaluator=task_evaluator,
        verifier=selected_verifier,
        provenance_store=provenance_store,
        agent_call_store=call_store,
        knowledge_retrieval_store=retrieval_store,
        knowledge_provider_registry=provider_registry,
        knowledge_provider=knowledge_provider,
        knowledge_configuration=knowledge_configuration,
        knowledge_template_registry=template_registry,
        knowledge_coordinator=knowledge_coordinator,
        agent_context_assembler=context_assembler,
        task_agent_context=task_agent_context,
        agent_budget_policy=budget_policy,
        checkpointer=checkpointer,
        clock=clock,
        id_generator=id_generator,
        task_descriptor=descriptor,
        dataset_manifest=manifest,
        artefact_policy=task.artefact_policy(),
        verification_policy=policy,
        task=task,
        runtime_context=context,
        experiment_metadata=metadata,
        search_capabilities=capabilities,
        search_backend_registry=registry,
        optuna_backend=(
            OptunaAskTellBackend(optuna_storage_handle.storage)
            if optuna_available and optuna_storage_handle
            else None
        ),
        optuna_storage_handle=optuna_storage_handle,
        openevolve_backend=openevolve_backend,
        native_openevolve_runtime=native_runtime,
    )


def task_memory_dependencies(
    task: ResearchTask,
    runtime_context: TaskRuntimeContext,
    contract,
    experiment_configuration: dict,
    *,
    hypothesis_agent: HypothesisAgent | None = None,
    planner_agent: PlannerAgent | None = None,
    evaluator: Evaluator | None = None,
    verifier: Verifier | None = None,
    provenance_store: ProvenanceStore | None = None,
    agent_call_store: AgentCallStore | None = None,
    model_client: StructuredModelClient | None = None,
    planner_model_client: StructuredModelClient | None = None,
    hypothesis_call_config: ModelCallConfig | None = None,
    planner_call_config: ModelCallConfig | None = None,
    agent_budget_policy: AgentBudgetPolicy | None = None,
    knowledge_provider: KnowledgeProvider | None = None,
    knowledge_configuration: KnowledgeProviderConfiguration | None = None,
    knowledge_retrieval_store: KnowledgeRetrievalStore | None = None,
    knowledge_template_registry: KnowledgeQueryTemplateRegistry | None = None,
    clock: Callable[[], datetime] = utc_now,
    id_generator: Callable[[str], str] = random_id,
    search_type: SearchType = SearchType.DIRECT,
    openevolve_mutation_operator: MutationOperator | None = None,
    openevolve_sandbox_runner: Any | None = None,
    openevolve_live_runtime: MetadataOnlyLiveOpenEvolveRuntime | None = None,
    native_openevolve_models: tuple[ApprovedModel, ...] = (),
    native_openevolve_resource_broker: ResourceBroker | None = None,
) -> RuntimeDependencies:
    if openevolve_live_runtime is not None:
        raise ValueError("live_mutation_durable_runtime_required")
    return _assemble_task_dependencies(
        task=task,
        runtime_context=runtime_context,
        contract=contract,
        experiment_configuration=experiment_configuration,
        provenance_store=provenance_store or SQLiteProvenanceStore(),
        checkpointer=memory_checkpointer(),
        hypothesis_agent=hypothesis_agent,
        planner_agent=planner_agent,
        evaluator=evaluator,
        verifier=verifier,
        agent_call_store=agent_call_store,
        model_client=model_client,
        planner_model_client=planner_model_client,
        hypothesis_call_config=hypothesis_call_config,
        planner_call_config=planner_call_config,
        agent_budget_policy=agent_budget_policy,
        knowledge_provider=knowledge_provider,
        knowledge_configuration=knowledge_configuration,
        knowledge_retrieval_store=knowledge_retrieval_store,
        knowledge_template_registry=knowledge_template_registry,
        clock=clock,
        id_generator=id_generator,
        search_type=search_type,
        openevolve_mutation_operator=openevolve_mutation_operator,
        openevolve_sandbox_runner=openevolve_sandbox_runner,
        openevolve_live_runtime=openevolve_live_runtime,
        native_openevolve_models=native_openevolve_models,
        native_openevolve_resource_broker=native_openevolve_resource_broker,
    )


@contextmanager
def task_sqlite_dependencies(
    task: ResearchTask,
    runtime_context: TaskRuntimeContext,
    contract,
    experiment_configuration: dict,
    checkpoint_path: str | Path,
    provenance_path: str | Path,
    optuna_path: str | Path | None = None,
    agent_calls_path: str | Path | None = None,
    knowledge_retrievals_path: str | Path | None = None,
    *,
    hypothesis_agent: HypothesisAgent | None = None,
    planner_agent: PlannerAgent | None = None,
    evaluator: Evaluator | None = None,
    verifier: Verifier | None = None,
    model_client: StructuredModelClient | None = None,
    planner_model_client: StructuredModelClient | None = None,
    hypothesis_call_config: ModelCallConfig | None = None,
    planner_call_config: ModelCallConfig | None = None,
    agent_budget_policy: AgentBudgetPolicy | None = None,
    knowledge_provider: KnowledgeProvider | None = None,
    knowledge_configuration: KnowledgeProviderConfiguration | None = None,
    knowledge_template_registry: KnowledgeQueryTemplateRegistry | None = None,
    clock: Callable[[], datetime] = utc_now,
    id_generator: Callable[[str], str] = random_id,
    search_type: SearchType = SearchType.DIRECT,
    openevolve_mutation_operator: MutationOperator | None = None,
    openevolve_sandbox_runner: Any | None = None,
    openevolve_live_runtime: MetadataOnlyLiveOpenEvolveRuntime | None = None,
    native_openevolve_models: tuple[ApprovedModel, ...] = (),
    native_openevolve_resource_broker: ResourceBroker | None = None,
) -> Iterator[RuntimeDependencies]:
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    provenance = Path(provenance_path).expanduser().resolve()
    optuna_file = (
        Path(optuna_path).expanduser().resolve() if optuna_path is not None else None
    )
    agent_calls_file = (
        Path(agent_calls_path).expanduser().resolve()
        if agent_calls_path is not None
        else None
    )
    knowledge_retrievals_file = (
        Path(knowledge_retrievals_path).expanduser().resolve()
        if knowledge_retrievals_path is not None
        else None
    )
    stores = [
        checkpoint,
        provenance,
        *(tuple([optuna_file]) if optuna_file else ()),
        *(tuple([agent_calls_file]) if agent_calls_file else ()),
        *(tuple([knowledge_retrievals_file]) if knowledge_retrievals_file else ()),
    ]
    if len(stores) != len(set(stores)):
        raise ValueError(
            "checkpoint, provenance, Optuna and agent-call stores must use "
            "separate files; the knowledge store must also be separate"
        )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    provenance.parent.mkdir(parents=True, exist_ok=True)
    if agent_calls_file is not None:
        agent_calls_file.parent.mkdir(parents=True, exist_ok=True)
    if knowledge_retrievals_file is not None:
        knowledge_retrievals_file.parent.mkdir(parents=True, exist_ok=True)
    saver, connection = sqlite_checkpointer(checkpoint)
    store = SQLiteProvenanceStore(provenance)
    call_store = (
        SQLiteAgentCallStore(agent_calls_file)
        if agent_calls_file is not None
        else InMemoryAgentCallStore()
    )
    retrieval_store = (
        SQLiteKnowledgeRetrievalStore(knowledge_retrievals_file)
        if knowledge_retrievals_file is not None
        else InMemoryKnowledgeRetrievalStore()
    )
    optuna_handle = (
        sqlite_storage(optuna_file)
        if (
            search_type == SearchType.OPTUNA
            and optuna_file is not None
            and importlib.util.find_spec("optuna") is not None
        )
        else None
    )
    try:
        yield _assemble_task_dependencies(
            task=task,
            runtime_context=runtime_context,
            contract=contract,
            experiment_configuration=experiment_configuration,
            provenance_store=store,
            checkpointer=saver,
            hypothesis_agent=hypothesis_agent,
            planner_agent=planner_agent,
            evaluator=evaluator,
            verifier=verifier,
            agent_call_store=call_store,
            model_client=model_client,
            planner_model_client=planner_model_client,
            hypothesis_call_config=hypothesis_call_config,
            planner_call_config=planner_call_config,
            agent_budget_policy=agent_budget_policy,
            knowledge_provider=knowledge_provider,
            knowledge_configuration=knowledge_configuration,
            knowledge_retrieval_store=retrieval_store,
            knowledge_template_registry=knowledge_template_registry,
            clock=clock,
            id_generator=id_generator,
            search_type=search_type,
            optuna_storage_handle=optuna_handle,
            openevolve_mutation_operator=openevolve_mutation_operator,
            openevolve_sandbox_runner=openevolve_sandbox_runner,
            openevolve_live_runtime=openevolve_live_runtime,
            native_openevolve_models=native_openevolve_models,
            native_openevolve_resource_broker=(native_openevolve_resource_broker),
        )
    finally:
        if optuna_handle is not None:
            optuna_handle.close()
        store.close()
        if isinstance(call_store, SQLiteAgentCallStore):
            call_store.close()
        if isinstance(retrieval_store, SQLiteKnowledgeRetrievalStore):
            retrieval_store.close()
        if knowledge_provider is not None:
            knowledge_provider.close()
        connection.close()


# PR 1 compatibility factories remain easy synthetic entry points.
def memory_dependencies(
    *,
    hypothesis_agent: HypothesisAgent | None = None,
    planner_agent: PlannerAgent | None = None,
    evaluator: Evaluator | None = None,
    verifier: Verifier | None = None,
    provenance_store: ProvenanceStore | None = None,
    agent_call_store: AgentCallStore | None = None,
    model_client: StructuredModelClient | None = None,
    planner_model_client: StructuredModelClient | None = None,
    hypothesis_call_config: ModelCallConfig | None = None,
    planner_call_config: ModelCallConfig | None = None,
    agent_budget_policy: AgentBudgetPolicy | None = None,
    clock: Callable[[], datetime] = utc_now,
    id_generator: Callable[[str], str] = random_id,
    search_type: SearchType = SearchType.DIRECT,
) -> RuntimeDependencies:
    from auto_researcher.tasks.synthetic import (
        SyntheticTask,
        default_synthetic_configuration,
        default_synthetic_contract,
        default_synthetic_openevolve_configuration,
    )

    return task_memory_dependencies(
        SyntheticTask(),
        TaskRuntimeContext(),
        default_synthetic_contract(
            search_types=frozenset({search_type}),
            maximum_experiments=(
                8
                if search_type == SearchType.OPTUNA
                else 4
                if search_type == SearchType.OPENEVOLVE
                else 1
            ),
        ),
        (
            {"trial_budget": 8}
            if search_type == SearchType.OPTUNA
            else default_synthetic_openevolve_configuration()
            if search_type == SearchType.OPENEVOLVE
            else default_synthetic_configuration()
        ),
        hypothesis_agent=hypothesis_agent,
        planner_agent=planner_agent,
        evaluator=evaluator,
        verifier=verifier,
        provenance_store=provenance_store,
        agent_call_store=agent_call_store,
        model_client=model_client,
        planner_model_client=planner_model_client,
        hypothesis_call_config=hypothesis_call_config,
        planner_call_config=planner_call_config,
        agent_budget_policy=agent_budget_policy,
        clock=clock,
        id_generator=id_generator,
        search_type=search_type,
    )


@contextmanager
def sqlite_dependencies(
    checkpoint_path: str | Path,
    provenance_path: str | Path,
    *,
    hypothesis_agent: HypothesisAgent | None = None,
    planner_agent: PlannerAgent | None = None,
    evaluator: Evaluator | None = None,
    verifier: Verifier | None = None,
    agent_calls_path: str | Path | None = None,
    clock: Callable[[], datetime] = utc_now,
    id_generator: Callable[[str], str] = random_id,
) -> Iterator[RuntimeDependencies]:
    from auto_researcher.tasks.synthetic import (
        SyntheticTask,
        default_synthetic_configuration,
        default_synthetic_contract,
    )

    with task_sqlite_dependencies(
        SyntheticTask(),
        TaskRuntimeContext(),
        default_synthetic_contract(),
        default_synthetic_configuration(),
        checkpoint_path,
        provenance_path,
        agent_calls_path=agent_calls_path,
        hypothesis_agent=hypothesis_agent,
        planner_agent=planner_agent,
        evaluator=evaluator,
        verifier=verifier,
        clock=clock,
        id_generator=id_generator,
    ) as dependencies:
        yield dependencies
