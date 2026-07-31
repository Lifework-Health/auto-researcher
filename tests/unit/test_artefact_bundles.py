from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from auto_researcher.contracts.enums import ProvenanceKind
from auto_researcher.contracts.models import EvaluationResult, ExperimentSpec
from auto_researcher.tasks.artifacts import (
    ARTEFACT_FILENAMES,
    ArtefactBundleConflictError,
    artefact_references,
    verify_artefact_bundle,
    write_artefact_bundle,
)
from auto_researcher.tasks.models import DatasetManifest, TaskRuntimeContext


class InjectedFailure(RuntimeError):
    pass


@pytest.fixture
def bundle_values(tmp_path):
    context = TaskRuntimeContext(
        run_id="bundle-run",
        output_dir=tmp_path / "outputs",
    )
    experiment = ExperimentSpec(
        experiment_id="experiment-1",
        hypothesis_id="hypothesis-1",
        search_request_id="request-1",
        configuration={"value": 1},
        evaluator_id="test-evaluator",
        code_version="code-v1+scientific-json-v1+experiment-bundle-v2",
        dataset_version="dataset-v1",
        provenance=ProvenanceKind.REAL,
    )
    evaluation = EvaluationResult(
        experiment_id=experiment.experiment_id,
        success=True,
        primary_score=0.75,
        metrics={"objective": 0.75},
        constraint_results={"gate": True},
        artefact_references=artefact_references(context, experiment.experiment_id),
        evaluator_version="test-v1",
        provenance=ProvenanceKind.REAL,
    )
    dataset_manifest = DatasetManifest(
        task_id="test-task",
        dataset_version="dataset-v1",
        files=("aggregate",),
        hashes={"aggregate": "abc"},
        loader_version="loader-v1",
        created_at=datetime(2026, 7, 31, tzinfo=UTC),
    )
    evaluator_manifest = {
        "evaluator_id": "test-evaluator",
        "adapter_version": "test-v1",
    }
    return context, experiment, evaluation, dataset_manifest, evaluator_manifest


def _final_directory(context, experiment):
    return context.output_dir / "runs" / context.run_id / experiment.experiment_id


@pytest.mark.parametrize(
    "failure_stage",
    [
        "before_any_file_write",
        *(f"during_write:{name}" for name in ARTEFACT_FILENAMES),
        "before_directory_publication",
    ],
)
def test_faults_never_publish_a_partial_bundle(bundle_values, failure_stage):
    context, experiment, evaluation, dataset, evaluator = bundle_values

    def fail(stage):
        if stage == failure_stage:
            raise InjectedFailure(stage)

    with pytest.raises(InjectedFailure):
        write_artefact_bundle(
            context,
            experiment,
            evaluation,
            dataset,
            evaluator,
            fault_injector=fail,
        )

    assert not _final_directory(context, experiment).exists()
    parent = context.output_dir / "runs" / context.run_id
    assert not list(parent.glob(".*.tmp"))


def test_publication_failure_cleans_staging_directory(
    bundle_values,
    monkeypatch,
):
    context, experiment, evaluation, dataset, evaluator = bundle_values

    def fail_publication(source, destination):
        raise InjectedFailure("directory publication")

    monkeypatch.setattr(
        "auto_researcher.tasks.artifacts.os.replace",
        fail_publication,
    )
    with pytest.raises(InjectedFailure):
        write_artefact_bundle(
            context,
            experiment,
            evaluation,
            dataset,
            evaluator,
        )

    assert not _final_directory(context, experiment).exists()
    parent = context.output_dir / "runs" / context.run_id
    assert not list(parent.glob(".*.tmp"))


def test_successful_bundle_is_strict_complete_and_deterministic(bundle_values):
    context, experiment, evaluation, dataset, evaluator = bundle_values
    first = write_artefact_bundle(context, experiment, evaluation, dataset, evaluator)
    second = write_artefact_bundle(context, experiment, evaluation, dataset, evaluator)

    directory = _final_directory(context, experiment)
    assert {item.name for item in directory.iterdir()} == set(ARTEFACT_FILENAMES)
    assert all(
        (context.output_dir / reference).is_file() for reference in first.references
    )
    for path in directory.iterdir():
        payload = path.read_text(encoding="utf-8")
        assert "NaN" not in payload
        assert "Infinity" not in payload
        json.loads(payload, parse_constant=lambda token: pytest.fail(token))
    integrity = verify_artefact_bundle(context, experiment.experiment_id)
    assert integrity.complete
    assert integrity.untampered
    assert integrity.bundle_sha256 == first.bundle_sha256
    assert first.payload_sha256 == second.payload_sha256
    assert first.bundle_sha256 == second.bundle_sha256
    assert first.replayed is False
    assert second.replayed is True


def test_conflicting_replay_does_not_overwrite_completed_bundle(bundle_values):
    context, experiment, evaluation, dataset, evaluator = bundle_values
    receipt = write_artefact_bundle(context, experiment, evaluation, dataset, evaluator)
    original = {
        name: (_final_directory(context, experiment) / name).read_bytes()
        for name in ARTEFACT_FILENAMES
    }
    conflicting = evaluation.model_copy(
        update={"primary_score": 0.25, "metrics": {"objective": 0.25}}
    )

    with pytest.raises(ArtefactBundleConflictError):
        write_artefact_bundle(
            context,
            experiment,
            conflicting,
            dataset,
            evaluator,
        )

    assert {
        name: (_final_directory(context, experiment) / name).read_bytes()
        for name in ARTEFACT_FILENAMES
    } == original
    assert verify_artefact_bundle(context, experiment.experiment_id).bundle_sha256 == (
        receipt.bundle_sha256
    )


def test_bundle_verifier_detects_tampering(bundle_values):
    context, experiment, evaluation, dataset, evaluator = bundle_values
    write_artefact_bundle(context, experiment, evaluation, dataset, evaluator)
    path = _final_directory(context, experiment) / "dataset_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["dataset_version"] = "tampered"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    integrity = verify_artefact_bundle(context, experiment.experiment_id)
    assert integrity.complete
    assert not integrity.untampered
    assert integrity.reason_codes == ("bundle_hash_mismatch",)
