"""Narrow adapter over pinned upstream OpenEvolve data mechanics."""

from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path

from auto_researcher.runtime.identity import payload_hash
from auto_researcher.search.openevolve.identity import source_hash
from auto_researcher.search.openevolve.models import (
    EvolvableComponentSpec,
    MutationReservation,
    OpenEvolveCandidate,
)
from auto_researcher.search.openevolve.protocols import StructuredMutationClient
from auto_researcher.search.openevolve.upstream_models import (
    ExecutorIsolationResult,
    HardenedExecutorPolicy,
    ModelBridgeReservation,
    UPSTREAM_INSTALLED_RECORD_HASH,
    UPSTREAM_PACKAGE_VERSION,
    UpstreamMutationEnvelope,
    UpstreamOpenEvolveAdapterContract,
    UpstreamOpenEvolveAdapterState,
)

DISABLED_UPSTREAM_FEATURES = (
    "controller",
    "evaluator",
    "provider_clients",
    "embeddings",
    "network",
    "subprocess_execution",
    "package_installation",
    "filesystem_mutation",
    "persistence",
    "resume",
    "budgets",
    "stopping",
    "scientific_judgement",
    "parallel_execution",
    "telemetry_prompts",
)


def dependency_lock_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def installed_record_hash() -> str:
    try:
        distribution = importlib.metadata.distribution("openevolve")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ValueError("upstream_openevolve_dependency_unavailable") from exc
    rows = []
    for item in distribution.files or ():
        name = str(item)
        if item.hash and (
            name.startswith("openevolve/")
            or name.endswith(".dist-info/METADATA")
            or name.endswith(".dist-info/WHEEL")
        ):
            rows.append(f"{name}|{item.hash.mode}|{item.hash.value}")
    return hashlib.sha256(("\n".join(sorted(rows)) + "\n").encode()).hexdigest()


def validate_upstream_dependency(contract: UpstreamOpenEvolveAdapterContract) -> None:
    try:
        version = importlib.metadata.version("openevolve")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ValueError("upstream_openevolve_dependency_unavailable") from exc
    if (
        version != UPSTREAM_PACKAGE_VERSION
        or contract.upstream_package_version != version
    ):
        raise ValueError("upstream_openevolve_identity_mismatch")
    if installed_record_hash() != UPSTREAM_INSTALLED_RECORD_HASH:
        raise ValueError("upstream_openevolve_dependency_hash_mismatch")
    from openevolve.database import Program

    required = {"id", "code", "parent_id", "generation", "metrics", "metadata"}
    if not required.issubset(Program.__dataclass_fields__):
        raise ValueError("upstream_openevolve_api_incompatible")


class AutoResearcherOpenEvolveModelBridge:
    """Minimal structured bridge; upstream receives no client or credentials."""

    def __init__(
        self,
        client: StructuredMutationClient,
        *,
        provider: str = "fake",
        model_id: str = "fake-structured-v1",
        prompt_version: str = "upstream-mutation-v1",
        maximum_output_bytes: int = 64_000,
    ):
        self.client = client
        self.provider = provider
        self.model_id = model_id
        self.prompt_version = prompt_version
        self.maximum_output_bytes = maximum_output_bytes
        self._completed: dict[str, tuple[dict, ModelBridgeReservation]] = {}

    def complete(
        self, request: dict, mutation_reservation_id: str
    ) -> tuple[dict, ModelBridgeReservation]:
        reservation_id = f"upstream-model-{payload_hash({'request': request, 'mutation': mutation_reservation_id})[:24]}"
        if reservation_id in self._completed:
            return self._completed[reservation_id]
        response = self.client.propose_mutation(request)
        if not isinstance(response, dict):
            raise ValueError("upstream_mutation_response_invalid")
        encoded = str(response).encode()
        if len(encoded) > self.maximum_output_bytes:
            raise ValueError("upstream_mutation_output_oversize")
        record = ModelBridgeReservation(
            reservation_id=reservation_id,
            mutation_reservation_id=mutation_reservation_id,
            provider=self.provider,
            model_id=self.model_id,
            prompt_version=self.prompt_version,
            maximum_output_bytes=self.maximum_output_bytes,
            completed=True,
            response_hash=payload_hash(response),
        )
        self._completed[reservation_id] = (response, record)
        return response, record


class UpstreamOpenEvolveAdapter:
    operator_id = "pinned-upstream-openevolve"
    operator_version = "upstream-openevolve-adapter-v1"
    model_calls_per_mutation = 1
    provenance = "FAKE_MODEL"

    def __init__(
        self,
        contract: UpstreamOpenEvolveAdapterContract,
        bridge: AutoResearcherOpenEvolveModelBridge,
    ):
        validate_upstream_dependency(contract)
        self.contract = contract
        self.bridge = bridge
        self.state = UpstreamOpenEvolveAdapterState(
            adapter_identity_hash=payload_hash(contract)
        )
        self._completed_mutations: dict[str, tuple[str, str, str | None]] = {}

    def mutate(
        self,
        reservation: MutationReservation,
        parent: OpenEvolveCandidate,
        component: EvolvableComponentSpec,
    ) -> tuple[str, str, str | None]:
        if reservation.reservation_id in self._completed_mutations:
            return self._completed_mutations[reservation.reservation_id]
        from openevolve.database import Program

        upstream_parent = Program(
            id=f"upstream-{parent.candidate_id}",
            code=parent.source_payload,
            parent_id=None,
            generation=parent.generation,
            metrics={},
            metadata={"authoritative_candidate_id": parent.candidate_id},
        )
        request = {
            "protocol": "upstream-adapter-mutation-request-v1",
            "parent": {
                "id": upstream_parent.id,
                "code": upstream_parent.code,
                "generation": upstream_parent.generation,
            },
            "mutable_file": component.mutable_file,
            "interface_contract": component.immutable_interface_contract,
            "maximum_source_bytes": component.maximum_source_bytes,
        }
        response, call = self.bridge.complete(request, reservation.reservation_id)
        try:
            envelope = UpstreamMutationEnvelope.model_validate(response)
        except Exception as exc:
            raise ValueError("upstream_mutation_response_invalid") from exc
        if (
            envelope.mutable_file != component.mutable_file
            or envelope.dependency_requests
            or envelope.provider_configuration
        ):
            raise ValueError("upstream_candidate_reconciliation_failed")
        if len(envelope.source.encode("utf-8")) > component.maximum_source_bytes:
            raise ValueError("upstream_mutation_output_oversize")
        canonical = envelope.source.replace("\r\n", "\n").replace("\r", "\n")
        upstream_id = (
            envelope.upstream_program_id or f"upstream-{source_hash(canonical)[:20]}"
        )
        self.state = self.state.model_copy(
            update={
                "proposal_count": self.state.proposal_count + 1,
                "cursor": reservation.birth_index,
                "upstream_program_ids": (*self.state.upstream_program_ids, upstream_id),
                "upstream_parent_recommendations": (
                    *self.state.upstream_parent_recommendations,
                    parent.candidate_id,
                ),
                "bounded_metadata": {"last_model_reservation": call.reservation_id},
            }
        )
        result = canonical, envelope.description, call.reservation_id
        self._completed_mutations[reservation.reservation_id] = result
        return result

    def recommend_parent(
        self, candidates: tuple[OpenEvolveCandidate, ...], scores: dict[str, float]
    ) -> str:
        """Use upstream population bookkeeping as a suggestion, never authority."""
        from openevolve.config import DatabaseConfig
        from openevolve.database import Program, ProgramDatabase

        database = ProgramDatabase(
            DatabaseConfig(
                population_size=max(1, len(candidates)),
                archive_size=max(1, len(candidates)),
                num_islands=1,
                feature_dimensions=["complexity"],
                random_seed=0,
            )
        )
        for candidate in candidates:
            database.add(
                Program(
                    id=f"upstream-{candidate.candidate_id}",
                    code=candidate.source_payload,
                    parent_id=(
                        f"upstream-{candidate.parent_candidate_ids[0]}"
                        if candidate.parent_candidate_ids
                        else None
                    ),
                    generation=candidate.generation,
                    metrics={"combined_score": scores.get(candidate.candidate_id, 0.0)},
                    metadata={"authoritative_candidate_id": candidate.candidate_id},
                )
            )
        best = database.get_best_program(metric="combined_score")
        if best is None:
            raise ValueError("upstream_candidate_reconciliation_failed")
        recommendation = str(best.metadata["authoritative_candidate_id"])
        self.state = self.state.model_copy(
            update={
                "upstream_parent_recommendations": (
                    *self.state.upstream_parent_recommendations,
                    recommendation,
                ),
            }
        )
        return recommendation


def default_adapter_contract(lock_path: Path) -> UpstreamOpenEvolveAdapterContract:
    return UpstreamOpenEvolveAdapterContract(
        dependency_lock_hash=dependency_lock_hash(lock_path),
        unsupported_features=DISABLED_UPSTREAM_FEATURES,
        compatibility_flags={
            "program_dataclass": True,
            "database_mapping": True,
            "direct_provider_calls": False,
        },
    )


def assert_live_mutation_eligible(
    adapter: UpstreamOpenEvolveAdapterContract,
    executor: HardenedExecutorPolicy,
    isolation: ExecutorIsolationResult,
    *,
    approved_adapter_hash: str,
    approved_image_digest: str,
    contract_permits_live_mutation: bool,
    operator_approved: bool,
    maximum_model_calls: int,
    maximum_candidate_evaluations: int,
) -> None:
    if payload_hash(adapter) != approved_adapter_hash:
        raise ValueError("upstream_openevolve_identity_mismatch")
    if executor.image_digest != approved_image_digest:
        raise ValueError("hardened_executor_image_mismatch")
    if not (
        isolation.network_isolation_verified
        and isolation.mount_isolation_verified
        and isolation.environment_sanitisation_verified
    ):
        raise ValueError("hardened_executor_network_isolation_unverified")
    if not contract_permits_live_mutation or not operator_approved:
        raise ValueError("live_mutation_approval_required")
    if maximum_model_calls <= 0 or maximum_candidate_evaluations <= 0:
        raise ValueError("live_mutation_finite_budget_required")
