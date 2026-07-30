"""The single installed PR 1 search backend."""

from __future__ import annotations

import hashlib
from typing import cast

from auto_researcher.contracts.enums import ProvenanceKind, SearchType
from auto_researcher.contracts.models import (
    ExperimentSpec,
    JsonValue,
    ResearchContract,
    SearchRequest,
)


class DirectSearchBackend:
    """Select one deterministic configuration; evaluation remains a separate step."""

    code_version = "auto-researcher-v2.1-pr1"
    dataset_version = "offline-mock-landscape-v1"

    def create_experiment(
        self,
        request: SearchRequest,
        contract: ResearchContract,
        *,
        run_id: str,
    ) -> ExperimentSpec:
        if request.search_type != SearchType.DIRECT:
            raise ValueError("DirectSearchBackend accepts only DIRECT requests")
        configuration: dict[str, JsonValue] = {}
        for name in sorted(request.search_space):
            values = request.search_space[name]
            configuration[name] = cast(list[JsonValue], values)[0] if isinstance(values, list) else values
        digest = hashlib.sha256(f"{run_id}\x1f{request.request_id}".encode()).hexdigest()[:16]
        return ExperimentSpec(
            experiment_id=f"experiment-{digest}",
            hypothesis_id=request.hypothesis_id,
            search_request_id=request.request_id,
            configuration=configuration,
            evaluator_id=contract.evaluator_id,
            code_version=self.code_version,
            dataset_version=self.dataset_version,
            provenance=ProvenanceKind.MOCK,
        )
