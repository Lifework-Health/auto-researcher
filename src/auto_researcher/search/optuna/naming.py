"""Deterministic, non-sensitive Optuna study identity."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from auto_researcher.contracts.models import ResearchContract, SearchRequest
from auto_researcher.search.optuna.models import OptunaStudySpec
from auto_researcher.tasks.models import ExperimentMetadata


@dataclass(frozen=True)
class StudyIdentity:
    study_name: str
    search_space_hash: str
    attributes: dict[str, str | int]


def _canonical_json(value) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def search_space_hash(spec: OptunaStudySpec) -> str:
    payload = {
        "schema_version": spec.schema_version,
        "search_space_version": spec.search_space_version,
        "direction": spec.direction.value,
        "parameters": [
            parameter.model_dump(mode="json") for parameter in spec.parameters
        ],
        "fixed_configuration": spec.fixed_configuration,
        "sampler": spec.sampler,
        "n_startup_trials": spec.n_startup_trials,
        "objective_metric": spec.objective_metric,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _safe_prefix(value: str, maximum: int = 24) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-")
    return (cleaned or "study")[:maximum]


def build_study_identity(
    *,
    run_id: str,
    contract: ResearchContract,
    request: SearchRequest,
    metadata: ExperimentMetadata,
    spec: OptunaStudySpec,
) -> StudyIdentity:
    space_hash = search_space_hash(spec)
    attributes: dict[str, str | int] = {
        "identity_schema_version": "1.0",
        "run_id": run_id,
        "task_id": contract.task_id,
        "task_version": contract.task_version,
        "objective_version": contract.objective_version,
        "task_constraints_version": contract.task_constraints_version,
        "evaluator_id": metadata.evaluator_id,
        "dataset_version": metadata.dataset_version,
        "code_version": metadata.code_version,
        "request_id": request.request_id,
        "search_space_hash": space_hash,
        "direction": spec.direction.value,
        "seed": spec.seed,
        "trial_budget": spec.trial_budget,
    }
    suffix = hashlib.sha256(
        _canonical_json(attributes).encode("utf-8")
    ).hexdigest()[:16]
    prefix = "-".join(
        (
            _safe_prefix(run_id),
            _safe_prefix(contract.task_id),
            _safe_prefix(contract.task_version),
        )
    )
    return StudyIdentity(
        study_name=f"{prefix}-{suffix}",
        search_space_hash=space_hash,
        attributes=attributes,
    )
