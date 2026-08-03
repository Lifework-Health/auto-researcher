"""Evaluator invocation, durable reuse and budget accounting."""

from pathlib import Path

from auto_researcher.contracts.enums import SearchType
from auto_researcher.contracts.models import EvaluationResult
from auto_researcher.graph.state import ResearchState
from auto_researcher.provenance.reuse import EvaluationReuseRecord
from auto_researcher.runtime.dependencies import RuntimeDependencies
from auto_researcher.runtime.identity import payload_hash
from auto_researcher.tasks.artifacts import artefact_references, verify_artefact_bundle


def _evaluation_identity(
    state: ResearchState,
    dependencies: RuntimeDependencies,
) -> tuple[str, str]:
    experiment = state["experiment_spec"]
    assert experiment is not None
    experiment_hash = payload_hash(experiment)
    evaluator_version = getattr(
        dependencies.evaluator,
        "version",
        dependencies.evaluator.evaluator_id,
    )
    identity_hash = payload_hash(
        {
            "run_id": state["run_id"],
            "experiment_id": experiment.experiment_id,
            "evaluator_version": evaluator_version,
            "dataset_version": experiment.dataset_version,
            "code_version": experiment.code_version,
            "experiment_payload_hash": experiment_hash,
        }
    )
    return identity_hash, experiment_hash


def _verify_reused_evaluation(
    record: EvaluationReuseRecord,
    dependencies: RuntimeDependencies,
) -> EvaluationResult:
    result = record.result
    if payload_hash(result) != record.result_payload_hash:
        raise RuntimeError("stored_evaluation_payload_hash_mismatch")
    if not result.artefact_references:
        raise RuntimeError("completed_evaluation_artefact_bundle_missing")
    expected_references = artefact_references(
        dependencies.runtime_context,
        result.experiment_id,
    )
    if result.artefact_references != expected_references:
        raise RuntimeError("completed_evaluation_artefact_reference_conflict")
    integrity = verify_artefact_bundle(
        dependencies.runtime_context,
        result.experiment_id,
    )
    if not integrity.complete:
        raise RuntimeError("completed_evaluation_artefact_bundle_missing")
    if not integrity.untampered:
        raise RuntimeError("completed_evaluation_artefact_bundle_tampered")
    output_dir = dependencies.runtime_context.output_dir
    if output_dir is None:
        raise RuntimeError("completed_evaluation_artefact_bundle_missing")
    evaluation_reference = next(
        (
            reference
            for reference in result.artefact_references
            if Path(reference).name == "evaluation_result.json"
        ),
        None,
    )
    if evaluation_reference is None:
        raise RuntimeError("completed_evaluation_artefact_bundle_missing")
    published = EvaluationResult.model_validate_json(
        (output_dir / evaluation_reference).read_text(encoding="utf-8")
    )
    if published != result:
        raise RuntimeError("completed_evaluation_artefact_payload_conflict")
    return result


def evaluate_experiment(
    state: ResearchState,
    dependencies: RuntimeDependencies,
) -> dict:
    experiment = state["experiment_spec"]
    assert experiment is not None
    identity_hash, experiment_hash = _evaluation_identity(state, dependencies)
    evaluator_version = getattr(
        dependencies.evaluator,
        "version",
        dependencies.evaluator.evaluator_id,
    )
    existing = dependencies.provenance_store.get_evaluation_reuse(
        state["run_id"],
        experiment.experiment_id,
    )
    reused = existing is not None
    if existing is not None:
        if existing.scientific_identity_hash != identity_hash:
            raise RuntimeError("conflicting_completed_evaluation_identity")
        result = _verify_reused_evaluation(existing, dependencies)
    else:
        result = dependencies.evaluator.evaluate(experiment, state["contract"])
        dependencies.provenance_store.append_evaluation_reuse(
            EvaluationReuseRecord(
                run_id=state["run_id"],
                experiment_id=experiment.experiment_id,
                scientific_identity_hash=identity_hash,
                experiment_payload_hash=experiment_hash,
                result_payload_hash=payload_hash(result),
                evaluator_version=evaluator_version,
                dataset_version=experiment.dataset_version,
                code_version=experiment.code_version,
                result=result,
            )
        )
    cost = float(getattr(dependencies.evaluator, "cost_per_experiment", 0.0))
    budget = state["budget"].record_experiment(cost)
    request = state.get("search_request")
    is_optuna = request is not None and request.search_type == SearchType.OPTUNA
    errors = (
        [] if result.success or is_optuna else [result.error or "evaluation_failed"]
    )
    return {
        "evaluation_result": result,
        "budget": budget,
        "errors": errors,
        "executed_nodes": [
            "evaluate_experiment_reused" if reused else "evaluate_experiment"
        ],
    }
