"""Explicit LangGraph-owned Optuna ask/evaluate/verify/tell lifecycle."""

from __future__ import annotations

from datetime import datetime

from auto_researcher.contracts.enums import EventType
from auto_researcher.graph.state import ResearchState
from auto_researcher.runtime.dependencies import RuntimeDependencies
from auto_researcher.search.optuna.artifacts import (
    study_artefact_references,
    write_study_artefacts,
)
from auto_researcher.search.optuna.models import (
    OptunaStudyResult,
    OptunaStudySpec,
    OptunaTrialReference,
)
from auto_researcher.search.optuna.naming import build_study_identity
from auto_researcher.search.optuna.provenance import append_optuna_event
from auto_researcher.tasks.protocols import OptunaCapableTask


def _backend(dependencies: RuntimeDependencies):
    if dependencies.optuna_backend is None:
        raise RuntimeError("OPTUNA backend was routed without a configured backend")
    return dependencies.optuna_backend


def _identity(state: ResearchState, dependencies: RuntimeDependencies):
    request = state["search_request"]
    spec = state["optuna_study_spec"]
    assert request is not None and spec is not None
    return build_study_identity(
        run_id=state["run_id"],
        contract=state["contract"],
        request=request,
        metadata=dependencies.experiment_metadata,
        spec=spec,
    )


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


def optuna_prepare_study(
    state: ResearchState,
    dependencies: RuntimeDependencies,
) -> dict:
    request = state["search_request"]
    assert request is not None
    if not isinstance(dependencies.task, OptunaCapableTask):
        raise RuntimeError("selected task does not implement OptunaCapableTask")
    registered = dependencies.task.create_optuna_study_spec(
        state["contract"],
        request,
    )
    if (registered.task_id, registered.task_version) != (
        state["contract"].task_id,
        state["contract"].task_version,
    ):
        raise ValueError("task-owned Optuna specification identity mismatch")
    if registered.objective_metric != state["contract"].primary_metric:
        raise ValueError("Optuna objective metric does not match the contract")
    remaining = max(
        0,
        state["budget"].maximum_experiments
        - state["budget"].experiments_used,
    )
    effective_budget = min(
        registered.trial_budget,
        request.experiment_budget,
        remaining,
    )
    payload = registered.model_dump(mode="python")
    payload["trial_budget"] = effective_budget
    spec = OptunaStudySpec.model_validate(payload)
    identity = build_study_identity(
        run_id=state["run_id"],
        contract=state["contract"],
        request=request,
        metadata=dependencies.experiment_metadata,
        spec=spec,
    )
    summary = _backend(dependencies).prepare_or_load_study(
        identity,
        spec,
        started_at=dependencies.clock(),
        trial_budget=effective_budget,
    )
    if summary.trials_asked >= effective_budget and summary.current_trial is None:
        summary = summary.model_copy(
            update={
                "finished": True,
                "finish_reason": "effective_trial_budget_reached",
            }
        )
    attrs = _backend(dependencies).study_user_attrs(identity.study_name)
    timestamp = _parse_timestamp(str(attrs["started_at"]))
    hypothesis = state["active_hypothesis"]
    assert hypothesis is not None
    event_ids = [
        append_optuna_event(
            dependencies.provenance_store,
            run_id=state["run_id"],
            cycle=state["cycle"],
            study_name=identity.study_name,
            event_type=EventType.HYPOTHESIS_PROPOSED,
            actor="hypothesis_agent",
            inputs=(state["contract"].contract_id,),
            outputs=(hypothesis.hypothesis_id,),
            rationale=hypothesis.rationale,
            timestamp=timestamp,
            provenance=hypothesis.provenance,
        ),
        append_optuna_event(
            dependencies.provenance_store,
            run_id=state["run_id"],
            cycle=state["cycle"],
            study_name=identity.study_name,
            event_type=EventType.SEARCH_PLANNED,
            actor="planner_agent",
            inputs=(request.hypothesis_id,),
            outputs=(request.request_id,),
            rationale=request.rationale,
            timestamp=timestamp,
            provenance=hypothesis.provenance,
        ),
    ]
    if state.get("human_approval_granted") is not None:
        event_ids.append(
            append_optuna_event(
                dependencies.provenance_store,
                run_id=state["run_id"],
                cycle=state["cycle"],
                study_name=identity.study_name,
                event_type=EventType.HUMAN_DECISION,
                actor="human",
                inputs=(request.request_id,),
                outputs=(),
                rationale=(
                    "approved"
                    if state["human_approval_granted"]
                    else "rejected"
                ),
                timestamp=timestamp,
                provenance=dependencies.experiment_metadata.provenance,
            )
        )
    event_ids.append(
        append_optuna_event(
            dependencies.provenance_store,
            run_id=state["run_id"],
            cycle=state["cycle"],
            study_name=identity.study_name,
            event_type=EventType.OPTUNA_STUDY_STARTED,
            actor="optuna_prepare_study",
            inputs=(state["contract"].contract_id, request.request_id),
            outputs=(identity.study_name, identity.search_space_hash),
            rationale="Prepared or reconstructed the deterministic task-owned study.",
            timestamp=timestamp,
            provenance=dependencies.experiment_metadata.provenance,
        )
    )
    return {
        "optuna_study_spec": spec,
        "optuna_study_state": summary,
        "decision_event_ids": event_ids,
        "executed_nodes": ["optuna_prepare_study"],
    }


def optuna_ask_trial(
    state: ResearchState,
    dependencies: RuntimeDependencies,
) -> dict:
    summary = state["optuna_study_state"]
    spec = state["optuna_study_spec"]
    assert summary is not None and spec is not None
    identity = _identity(state, dependencies)
    reference, _ = _backend(dependencies).ask_or_recover_trial(
        identity,
        spec,
        slot_index=summary.trials_asked,
        asked_at=dependencies.clock(),
    )
    summary = _backend(dependencies).load_study_summary(
        identity,
        spec.direction,
        summary.trial_budget,
        current_trial=reference,
    )
    attrs = _backend(dependencies).trial_user_attrs(
        reference.study_name,
        reference.trial_number,
    )
    event_id = append_optuna_event(
        dependencies.provenance_store,
        run_id=state["run_id"],
        cycle=state["cycle"],
        study_name=reference.study_name,
        event_type=EventType.OPTUNA_TRIAL_PROPOSED,
        actor="optuna_ask_trial",
        inputs=(reference.study_name,),
        outputs=(f"trial:{reference.trial_number}",),
        rationale="Asked exactly one durable trial from the registered search space.",
        timestamp=_parse_timestamp(str(attrs["asked_at"])),
        provenance=dependencies.experiment_metadata.provenance,
        trial_number=reference.trial_number,
    )
    return {
        "optuna_study_state": summary,
        "decision_event_ids": [event_id],
        "executed_nodes": ["optuna_ask_trial"],
    }


def optuna_create_experiment(
    state: ResearchState,
    dependencies: RuntimeDependencies,
) -> dict:
    request = state["search_request"]
    spec = state["optuna_study_spec"]
    summary = state["optuna_study_state"]
    assert request is not None and spec is not None and summary is not None
    reference = summary.current_trial
    assert reference is not None
    experiment = _backend(dependencies).create_experiment_spec(
        task=dependencies.task,
        metadata=dependencies.experiment_metadata,
        spec=spec,
        request=request,
        reference=reference,
    )
    reference = reference.model_copy(update={"experiment_id": experiment.experiment_id})
    return {
        "optuna_study_state": summary.model_copy(
            update={"current_trial": reference}
        ),
        "experiment_spec": experiment,
        "evaluation_result": None,
        "verification_result": None,
        "executed_nodes": ["optuna_create_experiment"],
    }


def optuna_tell_trial(
    state: ResearchState,
    dependencies: RuntimeDependencies,
) -> dict:
    spec = state["optuna_study_spec"]
    summary = state["optuna_study_state"]
    experiment = state["experiment_spec"]
    evaluation = state["evaluation_result"]
    verification = state["verification_result"]
    assert (
        spec is not None
        and summary is not None
        and summary.current_trial is not None
        and experiment is not None
        and evaluation is not None
        and verification is not None
    )
    outcome = _backend(dependencies).tell_trial(
        spec=spec,
        reference=summary.current_trial,
        experiment=experiment,
        evaluation=evaluation,
        verification=verification,
        reported_at=dependencies.clock(),
    )
    identity = _identity(state, dependencies)
    summary = _backend(dependencies).load_study_summary(
        identity,
        spec.direction,
        summary.trial_budget,
    )
    return {
        "optuna_trial_outcome": outcome,
        "optuna_study_state": summary,
        "executed_nodes": ["optuna_tell_trial"],
    }


def optuna_record_trial(
    state: ResearchState,
    dependencies: RuntimeDependencies,
) -> dict:
    outcome = state["optuna_trial_outcome"]
    experiment = state["experiment_spec"]
    evaluation = state["evaluation_result"]
    verification = state["verification_result"]
    assert outcome and experiment and evaluation and verification
    attrs = _backend(dependencies).trial_user_attrs(
        state["optuna_study_state"].study_name,
        outcome.trial_number,
    )
    timestamp = _parse_timestamp(str(attrs["reported_at"]))
    common = {
        "store": dependencies.provenance_store,
        "run_id": state["run_id"],
        "cycle": state["cycle"],
        "study_name": state["optuna_study_state"].study_name,
        "timestamp": timestamp,
        "provenance": evaluation.provenance,
        "trial_number": outcome.trial_number,
    }
    event_ids = [
        append_optuna_event(
            **common,
            event_type=EventType.EXPERIMENT_PREPARED,
            actor="optuna_create_experiment",
            inputs=(experiment.search_request_id,),
            outputs=(experiment.experiment_id,),
            rationale="Prepared a task-normalised experiment from one sampled trial.",
        ),
        append_optuna_event(
            **common,
            event_type=EventType.EVALUATION_OBSERVED,
            actor="evaluator",
            inputs=(experiment.experiment_id,),
            outputs=(f"score:{evaluation.primary_score}",),
            rationale="Recorded the task evaluator result for this trial.",
        ),
        append_optuna_event(
            **common,
            event_type=EventType.EVIDENCE_VERIFIED,
            actor="verifier",
            inputs=(experiment.experiment_id,),
            outputs=(f"evidence:{verification.evidence_status.value}",),
            rationale="Applied mandatory structural and task-policy verification.",
        ),
        append_optuna_event(
            **common,
            event_type=EventType.OPTUNA_TRIAL_REPORTED,
            actor="optuna_tell_trial",
            inputs=(experiment.experiment_id,),
            outputs=(
                f"trial:{outcome.trial_number}",
                f"status:{outcome.status.value}",
                f"verified:{verification.verified}",
                f"feasible:{outcome.feasible}",
            ),
            rationale="Reported verified evidence; invalid evaluations were marked FAIL.",
        ),
    ]
    return {
        "decision_event_ids": event_ids,
        "executed_nodes": ["optuna_record_trial"],
    }


def optuna_decide_study(state: ResearchState) -> dict:
    summary = state["optuna_study_state"]
    assert summary is not None
    asked_enough = summary.trials_asked >= summary.trial_budget
    budget_exhausted = state["budget"].exhausted
    if asked_enough or budget_exhausted:
        reason = (
            state["budget"].exhaustion_reason
            if budget_exhausted
            else "effective_trial_budget_reached"
        )
        summary = summary.model_copy(
            update={"finished": True, "finish_reason": reason}
        )
    return {
        "optuna_study_state": summary,
        "optuna_trial_outcome": None,
        "executed_nodes": ["optuna_decide_study"],
    }


def optuna_finalise_study(
    state: ResearchState,
    dependencies: RuntimeDependencies,
) -> dict:
    summary = state["optuna_study_state"]
    spec = state["optuna_study_spec"]
    assert summary is not None and spec is not None and summary.finished
    backend = _backend(dependencies)
    feasible = summary.best_feasible_trial_number
    diagnostic = summary.best_overall_trial_number
    selected_number = feasible if feasible is not None else diagnostic
    selected_models = (
        backend.load_trial_models(summary.study_name, selected_number)
        if selected_number is not None
        else None
    )
    preliminary = OptunaStudyResult(
        study_name=summary.study_name,
        direction=summary.direction,
        trial_budget=summary.trial_budget,
        trials_asked=summary.trials_asked,
        trials_completed=summary.trials_completed,
        trials_failed=summary.trials_failed,
        best_feasible_trial_number=summary.best_feasible_trial_number,
        best_feasible_score=summary.best_feasible_score,
        best_overall_trial_number=summary.best_overall_trial_number,
        best_overall_score=summary.best_overall_score,
        feasible_trial_found=feasible is not None,
        finish_reason=summary.finish_reason or "study_finished",
    )
    references = study_artefact_references(
        dependencies.runtime_context,
        summary.study_name,
    )
    result = preliminary.model_copy(update={"artefact_references": references})
    selected_payload = None
    if selected_number is not None:
        selected_payload = {
            "selected": feasible is not None,
            "diagnostic_only": feasible is None,
            "trial_number": selected_number,
            "parameters": next(
                item.parameters
                for item in backend.trial_outcomes(summary.study_name)
                if item.trial_number == selected_number
            ),
        }
    write_study_artefacts(
        dependencies.runtime_context,
        dependencies.artefact_policy,
        spec,
        result,
        backend.trial_outcomes(summary.study_name),
        selected_payload,
    )
    completed_at = backend.set_study_completed_at(
        summary.study_name,
        dependencies.clock(),
    )
    event_id = append_optuna_event(
        dependencies.provenance_store,
        run_id=state["run_id"],
        cycle=state["cycle"],
        study_name=summary.study_name,
        event_type=EventType.OPTUNA_STUDY_COMPLETED,
        actor="optuna_finalise_study",
        inputs=(summary.study_name,),
        outputs=references,
        rationale=result.finish_reason,
        timestamp=_parse_timestamp(completed_at),
        provenance=dependencies.experiment_metadata.provenance,
    )
    update = {
        "optuna_study_result": result,
        "decision_event_ids": [event_id],
        "executed_nodes": ["optuna_finalise_study"],
    }
    if selected_models is not None:
        experiment, evaluation, verification = selected_models
        if feasible is not None:
            update.update(
                experiment_spec=experiment,
                evaluation_result=evaluation,
                verification_result=verification,
            )
        else:
            update.update(
                experiment_spec=None,
                evaluation_result=None,
                verification_result=None,
                diagnostic_experiment_spec=experiment,
                diagnostic_evaluation_result=evaluation,
                diagnostic_verification_result=verification,
            )
    return update
