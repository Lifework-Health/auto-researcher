"""Pinned upstream-adapter and hardened-executor contracts."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from auto_researcher.contracts.models import FrozenJsonDict

UPSTREAM_REPOSITORY: Final = (
    "https://github.com/algorithmicsuperintelligence/openevolve"
)
UPSTREAM_TAG: Final = "v0.3.2"
UPSTREAM_COMMIT: Final = "411fb59c886c18704caaffb611e17cf9e7d824d2"
UPSTREAM_PACKAGE_VERSION: Final = "0.3.2"
UPSTREAM_WHEEL_SHA256: Final = (
    "df998b0731d9c1a80883b4aae452cc43405a3e9c61b46d676d06235b4db49366"
)
UPSTREAM_INSTALLED_RECORD_HASH: Final = (
    "74b05bb9cbd19a7045c6a0984328b02842031c782f293ced695a545158212fe6"
)


class AdapterModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class UpstreamOpenEvolveAdapterContract(AdapterModel):
    protocol_version: Literal["upstream-openevolve-adapter-v1"] = (
        "upstream-openevolve-adapter-v1"
    )
    adapter_id: Literal["auto-researcher-upstream-openevolve"] = (
        "auto-researcher-upstream-openevolve"
    )
    adapter_version: Literal["1"] = "1"
    upstream_repository: Literal[
        "https://github.com/algorithmicsuperintelligence/openevolve"
    ] = UPSTREAM_REPOSITORY
    upstream_tag: Literal["v0.3.2"] = UPSTREAM_TAG
    upstream_commit: Literal["411fb59c886c18704caaffb611e17cf9e7d824d2"] = (
        UPSTREAM_COMMIT
    )
    upstream_package_version: Literal["0.3.2"] = UPSTREAM_PACKAGE_VERSION
    upstream_wheel_sha256: Literal[
        "df998b0731d9c1a80883b4aae452cc43405a3e9c61b46d676d06235b4db49366"
    ] = UPSTREAM_WHEEL_SHA256
    installed_record_hash: Literal[
        "74b05bb9cbd19a7045c6a0984328b02842031c782f293ced695a545158212fe6"
    ] = UPSTREAM_INSTALLED_RECORD_HASH
    dependency_lock_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    supported_search_contract: Literal["openevolve-search-v1"] = "openevolve-search-v1"
    supported_candidate_representation: Literal["python_source_v1"] = "python_source_v1"
    mutation_mode_mapping: Literal["full-source-replacement-v1"] = (
        "full-source-replacement-v1"
    )
    population_policy_mapping: Literal["upstream-suggestion-core-authority-v1"] = (
        "upstream-suggestion-core-authority-v1"
    )
    evaluator_owner: Literal["AUTO_RESEARCHER"] = "AUTO_RESEARCHER"
    model_client_owner: Literal["AUTO_RESEARCHER"] = "AUTO_RESEARCHER"
    persistence_owner: Literal["AUTO_RESEARCHER"] = "AUTO_RESEARCHER"
    unsupported_features: tuple[str, ...]
    compatibility_flags: FrozenJsonDict


class UpstreamOpenEvolveAdapterState(AdapterModel):
    protocol_version: Literal["upstream-openevolve-state-v1"] = (
        "upstream-openevolve-state-v1"
    )
    adapter_identity_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    proposal_count: int = Field(default=0, ge=0)
    cursor: int = Field(default=0, ge=0)
    upstream_program_ids: tuple[str, ...] = ()
    upstream_parent_recommendations: tuple[str, ...] = ()
    final_core_decisions: tuple[str, ...] = ()
    bounded_metadata: FrozenJsonDict = Field(default_factory=dict)


class UpstreamMutationEnvelope(AdapterModel):
    protocol_version: Literal["upstream-mutation-envelope-v1"] = (
        "upstream-mutation-envelope-v1"
    )
    mutable_file: str
    source: str
    description: str = Field(min_length=1, max_length=2_000)
    upstream_program_id: str | None = None
    dependency_requests: tuple[str, ...] = ()
    provider_configuration: FrozenJsonDict = Field(default_factory=dict)


class HardenedExecutorPolicy(AdapterModel):
    protocol_version: Literal["openevolve-executor-policy-v2"] = (
        "openevolve-executor-policy-v2"
    )
    executor_id: Literal["openevolve-hardened-executor-v2"] = (
        "openevolve-hardened-executor-v2"
    )
    runtime_type: Literal["docker"] = "docker"
    runtime_version: str = Field(min_length=1)
    image_reference: str = Field(min_length=1)
    image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    base_image_digest: Literal[
        "sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7"
    ] = "sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7"
    entrypoint_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_child_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    build_recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    worker_protocol_version: Literal["openevolve-hardened-worker-result-v2"] = (
        "openevolve-hardened-worker-result-v2"
    )
    workspace_policy_version: Literal["openevolve-workspace-policy-v1"] = (
        "openevolve-workspace-policy-v1"
    )
    network_mode: Literal["none"] = "none"
    read_only_root: Literal[True] = True
    non_root_user: Literal[True] = True
    capabilities_dropped: Literal[True] = True
    no_new_privileges: Literal[True] = True
    environment_inheritance: Literal[False] = False
    workspace_mount: Literal["/workspace"] = "/workspace"
    input_mount: Literal["/input"] = "/input"
    writable_host_output_mount: Literal[False] = False
    tmpfs_inode_overhead: Literal[1] = 1
    supervisor_pid_overhead: Literal[4] = 4
    tmpfs_noexec: Literal[True] = True
    tmpfs_nosuid: Literal[True] = True
    tmpfs_nodev: Literal[True] = True
    fixed_uid: Literal[65532] = 65532
    fixed_gid: Literal[65532] = 65532


class ExecutorIsolationResult(AdapterModel):
    protocol_version: Literal["executor-isolation-result-v1"] = (
        "executor-isolation-result-v1"
    )
    executor_policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    network_isolation_verified: bool
    mount_isolation_verified: bool
    environment_sanitisation_verified: bool
    safe_checks: FrozenJsonDict
    safe_error_code: str | None = None


class ModelBridgeReservation(AdapterModel):
    protocol_version: Literal["upstream-model-bridge-reservation-v1"] = (
        "upstream-model-bridge-reservation-v1"
    )
    reservation_id: str
    mutation_reservation_id: str
    provider: str
    model_id: str
    prompt_version: str
    maximum_output_bytes: int = Field(gt=0)
    completed: bool = False
    response_hash: str | None = None
