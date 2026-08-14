"""Task-owned projection of verified evaluations into native Optuna vectors."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from auto_researcher.contracts.models import EvaluationResult, VerificationResult
from auto_researcher.search.optuna.models import OptunaStudySpec


def projection_identity(spec: OptunaStudySpec) -> str:
    payload = {
        "protocol": "optuna-scientific-projection-v1",
        "objectives": [item.model_dump(mode="json") for item in spec.objective_specs],
        "constraints": [item.model_dump(mode="json") for item in spec.constraints],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _metric(
    metric: str,
    spec: OptunaStudySpec,
    evaluation: EvaluationResult,
    verification: VerificationResult,
) -> float:
    if metric == spec.objective_metric:
        value: Any = evaluation.primary_score
    elif metric == "verification.constraint_compliant":
        value = 1.0 if verification.constraint_compliant else 0.0
    else:
        value = evaluation.metrics.get(metric)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"approved Optuna metric {metric!r} is missing or non-numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"approved Optuna metric {metric!r} is non-finite")
    return numeric


def project_scientific_result(
    spec: OptunaStudySpec,
    evaluation: EvaluationResult,
    verification: VerificationResult,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if not evaluation.success or not verification.verified:
        raise ValueError("invalid_evaluation_cannot_be_projected_to_optuna")
    objectives = tuple(
        _metric(item.metric, spec, evaluation, verification)
        for item in spec.objective_specs
    )
    constraints: list[float] = []
    for item in spec.constraints:
        observed = _metric(item.metric, spec, evaluation, verification)
        if item.relation == "LESS_THAN_OR_EQUAL":
            constraints.append(observed - item.threshold)
        else:
            constraints.append(item.threshold - observed)
    return objectives, tuple(constraints)
