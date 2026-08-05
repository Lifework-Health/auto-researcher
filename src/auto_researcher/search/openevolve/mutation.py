"""Auditable deterministic and fake-client mutation operators."""

from __future__ import annotations

from auto_researcher.runtime.identity import payload_hash
from auto_researcher.search.openevolve.models import (
    EvolvableComponentSpec,
    MutationReservation,
    OpenEvolveCandidate,
)
from auto_researcher.search.openevolve.protocols import StructuredMutationClient


class DeterministicMutationOperator:
    operator_id = "deterministic-fixture-replacement"
    operator_version = "deterministic-fixture-replacement-v1"
    model_calls_per_mutation = 0
    provenance = "DETERMINISTIC_FIXTURE"

    def mutate(
        self,
        reservation: MutationReservation,
        parent: OpenEvolveCandidate,
        component: EvolvableComponentSpec,
    ) -> tuple[str, str, str | None]:
        sources = component.deterministic_mutation_sources
        source = (
            sources[(reservation.generation - 1) % len(sources)]
            if sources
            else parent.source_payload
        )
        return (
            source,
            f"Applied deterministic replacement {reservation.birth_index}.",
            None,
        )


class FakeModelMutationOperator:
    operator_id = "fake-model-structured-replacement"
    operator_version = "fake-model-structured-replacement-v1"
    model_calls_per_mutation = 1
    provenance = "FAKE_MODEL"

    def __init__(
        self, client: StructuredMutationClient, *, maximum_output_bytes: int = 64_000
    ) -> None:
        self.client = client
        self.maximum_output_bytes = maximum_output_bytes

    def mutate(
        self,
        reservation: MutationReservation,
        parent: OpenEvolveCandidate,
        component: EvolvableComponentSpec,
    ) -> tuple[str, str, str | None]:
        request = {
            "protocol": "openevolve-mutation-request-v1",
            "reservation_id": reservation.reservation_id,
            "candidate_id": parent.candidate_id,
            "component_id": component.component_id,
            "component_version": component.component_version,
            "allowed_files": list(component.allowed_files),
            "mutable_file": component.mutable_file,
            "interface_contract": component.immutable_interface_contract,
            "output_schema": {"source": "utf8", "description": "bounded"},
            "source": parent.source_payload,
        }
        response = self.client.propose_mutation(request)
        if set(response) != {"source", "description"}:
            raise ValueError("candidate_patch_invalid")
        source = response["source"]
        description = response["description"]
        if not isinstance(source, str) or not isinstance(description, str):
            raise ValueError("candidate_patch_invalid")
        if len(source.encode("utf-8")) > min(
            self.maximum_output_bytes, component.maximum_source_bytes
        ):
            raise ValueError("candidate_output_limit")
        if not description or len(description) > 2_000:
            raise ValueError("candidate_patch_invalid")
        call_id = f"fake-mutation-{payload_hash({'request': request, 'response': response})[:24]}"
        return source, description, call_id
