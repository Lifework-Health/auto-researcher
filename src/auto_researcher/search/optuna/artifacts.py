"""Safe aggregate study artefacts; raw datasets and backend databases stay external."""

from __future__ import annotations

from typing import Any

from auto_researcher.search.optuna.models import (
    OptunaStudyResult,
    OptunaStudySpec,
    OptunaTrialOutcome,
)
from auto_researcher.tasks.artifacts import atomic_json_write, safe_segment
from auto_researcher.tasks.models import ArtefactPolicy, TaskRuntimeContext

STUDY_ARTEFACT_FILENAMES = (
    "study_spec.json",
    "study_summary.json",
    "trials_summary.json",
    "selected_trial.json",
)
STUDY_ARTEFACT_TYPES = frozenset(
    {"study_spec", "study_summary", "trials_summary", "selected_trial"}
)


def study_artefact_references(
    context: TaskRuntimeContext,
    study_name: str,
) -> tuple[str, ...]:
    if context.output_dir is None or not context.run_id:
        return ()
    run_id = safe_segment(context.run_id, "run_id")
    study = safe_segment(study_name, "study_name")
    prefix = f"runs/{run_id}/studies/{study}"
    return tuple(f"{prefix}/{name}" for name in STUDY_ARTEFACT_FILENAMES)


def write_study_artefacts(
    context: TaskRuntimeContext,
    policy: ArtefactPolicy,
    spec: OptunaStudySpec,
    result: OptunaStudyResult,
    outcomes: list[OptunaTrialOutcome],
    selected: dict[str, Any] | None,
) -> tuple[str, ...]:
    disallowed = STUDY_ARTEFACT_TYPES - policy.allowed_artefact_types
    prohibited = STUDY_ARTEFACT_TYPES & policy.prohibited_artefact_types
    if disallowed or prohibited:
        blocked = ", ".join(sorted(disallowed | prohibited))
        raise ValueError(f"task artefact policy does not permit: {blocked}")
    references = study_artefact_references(context, result.study_name)
    if not references:
        return ()
    safe_outcomes: list[dict[str, Any]] = []
    for outcome in outcomes:
        payload = outcome.model_dump(mode="json")
        if policy.contains_sensitive_data:
            payload.pop("parameters", None)
        safe_outcomes.append(payload)
    safe_selected = dict(selected or {"selected": False})
    if policy.contains_sensitive_data:
        safe_selected.pop("parameters", None)
    safe_spec: Any = spec
    if policy.contains_sensitive_data:
        safe_spec = spec.model_dump(mode="json")
        safe_spec["fixed_configuration"] = {"redacted": True}
    values = (
        safe_spec,
        result,
        safe_outcomes,
        safe_selected,
    )
    assert context.output_dir is not None
    for reference, value in zip(references, values, strict=True):
        atomic_json_write(context.output_dir / reference, value)
    return references
