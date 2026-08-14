"""Standard fail-closed assembly for approved metadata-only live mutation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator
import yaml

from auto_researcher.agents.call_store import AgentCallStore, SQLiteAgentCallStore
from auto_researcher.providers.protocols import StructuredModelClient
from auto_researcher.providers.anthropic import ANTHROPIC_ENVIRONMENT_SECRET
from auto_researcher.runtime.identity import payload_hash
from auto_researcher.search.openevolve.identity import component_interface_identity
from auto_researcher.search.openevolve.live_boundary import (
    metadata_only_model_exposure_identity,
)
from auto_researcher.search.openevolve.live_models import (
    MetadataOnlyLiveMutationApproval,
    MetadataOnlyOpenEvolveModelCallContext,
    OpenEvolveModelBridgeContract,
    parse_live_mutation_approval,
)
from auto_researcher.search.openevolve.production_bridge import (
    DurableOpenEvolveModelBridge,
    ProviderFactory,
    load_mutation_prompt,
)
from auto_researcher.search.openevolve.protocols import EvolvableComponent
from auto_researcher.search.openevolve.upstream import (
    build_approved_live_upstream_runtime,
    default_adapter_contract,
)
from auto_researcher.search.openevolve.upstream_models import (
    ExecutorIsolationResult,
    HardenedExecutorPolicy,
)
from auto_researcher.search.openevolve.hardened_executor import HardenedDockerExecutor
from auto_researcher.secrets import (
    ResolvedSecret,
    SecretProvider,
    SecretReference,
    SecretResolutionError,
    SecretResolutionErrorCode,
    provider_for_reference,
)
from auto_researcher.tasks.protocols import (
    MetadataOnlyLiveMutationCapableTask,
    ResearchTask,
)


class LiveRuntimeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MetadataOnlyLiveOpenEvolveConfiguration(LiveRuntimeModel):
    """Value-free artifact and provider-secret references for live mutation."""

    protocol_version: Literal["metadata-only-live-openevolve-runtime-v1"] = (
        "metadata-only-live-openevolve-runtime-v1"
    )
    mode: Literal["metadata_only"] = "metadata_only"
    approval_file: Path
    bridge_contract_file: Path
    adapter_lock_file: Path
    prompt_file: Path
    executor_policy_file: Path
    isolation_evidence_file: Path
    credential: SecretReference = ANTHROPIC_ENVIRONMENT_SECRET

    @model_validator(mode="after")
    def credential_is_required(self) -> "MetadataOnlyLiveOpenEvolveConfiguration":
        if not self.credential.required:
            raise ValueError("live_mutation_credentials_must_be_required")
        return self

    @field_validator(
        "approval_file",
        "bridge_contract_file",
        "adapter_lock_file",
        "prompt_file",
        "executor_policy_file",
        "isolation_evidence_file",
    )
    @classmethod
    def artifact_is_an_absolute_file(cls, value: Path) -> Path:
        path = value.expanduser()
        if not path.is_absolute():
            raise ValueError("live_mutation_runtime_paths_must_be_absolute")
        path = path.resolve()
        if not path.is_file():
            raise ValueError("live_mutation_runtime_artifact_unavailable")
        return path


@dataclass(frozen=True)
class MetadataOnlyLiveOpenEvolveRuntime:
    configuration: MetadataOnlyLiveOpenEvolveConfiguration
    thread_id: str
    provider_factory: ProviderFactory | None = None
    secret_provider_factory: Callable[[SecretReference], SecretProvider] | None = None
    executor_validator: Callable[[HardenedDockerExecutor], None] | None = None

    def __post_init__(self) -> None:
        if not self.thread_id:
            raise ValueError("live_mutation_runtime_thread_required")


def _load_artifact_object(path: Path) -> dict:
    class UniqueKeySafeLoader(yaml.SafeLoader):
        pass

    def construct_unique_mapping(loader, node, deep=False):
        loader.flatten_mapping(node)
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in result:
                raise ValueError("duplicate runtime artifact mapping key")
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    UniqueKeySafeLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_unique_mapping,
    )
    try:
        value = yaml.load(
            path.read_text(encoding="utf-8"),
            Loader=UniqueKeySafeLoader,
        )
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise ValueError("live_mutation_runtime_artifact_invalid") from exc
    if not isinstance(value, dict):
        raise ValueError("live_mutation_runtime_artifact_invalid")
    return value


def default_live_mutation_provider_factory(
    contract: OpenEvolveModelBridgeContract,
    credential_reference: SecretReference = ANTHROPIC_ENVIRONMENT_SECRET,
    *,
    secret_provider_factory: Callable[[SecretReference], SecretProvider] | None = None,
) -> ProviderFactory | None:
    """Return the sole production provider path; fake production is injected."""

    config = contract.model_config_contract
    if config.provider == "fake-production":
        return None
    if config.provider != "anthropic":
        raise ValueError("live_mutation_provider_not_supported")

    resolved_credential: ResolvedSecret | None = None

    def create() -> StructuredModelClient:
        nonlocal resolved_credential
        from auto_researcher.providers.anthropic import create_anthropic_client

        if resolved_credential is None:
            resolver_factory = secret_provider_factory or provider_for_reference
            resolver = resolver_factory(credential_reference)
            resolved_credential = resolver.resolve(credential_reference)
            if resolved_credential is None:
                raise SecretResolutionError(
                    SecretResolutionErrorCode.MISSING,
                    credential_reference,
                ) from None
        return create_anthropic_client(config, credential=resolved_credential)

    return create


def assemble_metadata_only_live_openevolve(
    *,
    runtime: MetadataOnlyLiveOpenEvolveRuntime,
    task: ResearchTask,
    component: EvolvableComponent,
    research_contract,
    run_id: str,
    experiment_configuration: dict,
    call_store: AgentCallStore,
    workspace_root: Path,
    now: Callable,
):
    """Validate every attested identity and return the approved operator/runner."""

    if not isinstance(task, MetadataOnlyLiveMutationCapableTask):
        raise ValueError("metadata_only_live_mutation_boundary_unavailable")
    if not isinstance(call_store, SQLiteAgentCallStore):
        raise ValueError("live_mutation_durable_call_store_required")
    if not workspace_root.is_absolute():
        raise ValueError("live_mutation_workspace_must_be_absolute")

    files = runtime.configuration
    approval = parse_live_mutation_approval(_load_artifact_object(files.approval_file))
    if not isinstance(approval, MetadataOnlyLiveMutationApproval):
        raise ValueError("metadata_only_live_mutation_approval_required")
    bridge_contract = OpenEvolveModelBridgeContract.model_validate(
        _load_artifact_object(files.bridge_contract_file)
    )
    executor_policy = HardenedExecutorPolicy.model_validate(
        _load_artifact_object(files.executor_policy_file)
    )
    isolation = ExecutorIsolationResult.model_validate(
        _load_artifact_object(files.isolation_evidence_file)
    )
    adapter = default_adapter_contract(files.adapter_lock_file)
    prompt = load_mutation_prompt(files.prompt_file)
    spec = component.component_spec()
    boundary = task.live_mutation_boundary()
    openevolve = experiment_configuration.get("openevolve")
    if not isinstance(openevolve, dict):
        raise ValueError("openevolve_finite_configuration_required")
    maximum_model_calls = openevolve.get("maximum_model_calls")
    if not isinstance(maximum_model_calls, int) or maximum_model_calls <= 0:
        raise ValueError("live_mutation_finite_budget_required")
    context = MetadataOnlyOpenEvolveModelCallContext(
        run_id=run_id,
        thread_id=runtime.thread_id,
        contract_id=research_contract.contract_id,
        contract_hash=payload_hash(research_contract),
        task_id=task.task_id,
        task_version=task.task_version,
        # The adapter replaces this placeholder with the authoritative request ID
        # before each bridge completion. It is not an approval-granting identity.
        search_request_id="unbound-until-authoritative-reservation",
        generation=1,
        parent_candidate_id="seed-placeholder",
        component_id=spec.component_id,
        component_version=spec.component_version,
        component_interface_hash=component_interface_identity(spec),
        model_exposure_identity=metadata_only_model_exposure_identity(spec),
        underlying_dataset_class=boundary.underlying_dataset_class,
        adapter_id=adapter.adapter_id,
        adapter_version=adapter.adapter_version,
        adapter_identity_hash=payload_hash(adapter),
        executor_policy_hash=payload_hash(executor_policy),
        image_digest=executor_policy.image_digest,
        mutable_file=spec.mutable_file,
        model_budget_identity=payload_hash(
            {
                "domain": "metadata-only-live-openevolve-budget-v1",
                "run_id": run_id,
                "approval_hash": approval.approval_hash,
                "maximum_model_calls": maximum_model_calls,
                "maximum_model_cost": approval.maximum_total_cost,
            }
        ),
        maximum_model_calls=maximum_model_calls,
        maximum_model_cost=approval.maximum_total_cost,
    )
    bridge = DurableOpenEvolveModelBridge(
        contract=bridge_contract,
        context=context,
        approval=approval,
        store=call_store,
        provider_factory=(
            runtime.provider_factory
            if runtime.provider_factory is not None
            else default_live_mutation_provider_factory(
                bridge_contract,
                runtime.configuration.credential,
                secret_provider_factory=runtime.secret_provider_factory,
            )
        ),
        now=now,
        system_prompt=prompt,
        metadata_only_boundary=boundary,
    )
    # Fail before graph execution, while complete() repeats this immediately
    # before a provider dispatch.
    bridge.validate_runtime_approval()
    operator, runner = build_approved_live_upstream_runtime(
        adapter,
        bridge,
        executor_policy,
        isolation,
        task=task,
        component_spec=spec,
        workspace_root=workspace_root,
    )
    validator = runtime.executor_validator or (
        lambda executor: executor.validate_environment()
    )
    validator(runner)
    return operator, runner


__all__ = [
    "MetadataOnlyLiveOpenEvolveConfiguration",
    "MetadataOnlyLiveOpenEvolveRuntime",
    "assemble_metadata_only_live_openevolve",
    "default_live_mutation_provider_factory",
]
