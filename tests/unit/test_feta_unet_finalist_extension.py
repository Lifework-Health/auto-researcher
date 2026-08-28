from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from auto_researcher.contracts.enums import ProvenanceKind, SearchType
from auto_researcher.contracts.models import (
    EvaluationResult,
    ExperimentSpec,
    ResearchContract,
)
from auto_researcher.tasks.artifacts import atomic_json_write
from auto_researcher.tasks.feta_unet_search.configuration import (
    FeTAUNetSearchConfiguration,
)
from auto_researcher.tasks.feta_unet_search.continuation import (
    CONTINUATION_VERSION,
    trajectory_identity,
)
from auto_researcher.tasks.feta_unet_search.finalist_extension import (
    EXTENSION_SCHEMA_VERSION,
    _validate_extension_root,
    build_extension_plan,
    execute_extension,
    write_extension_summary,
)
from auto_researcher.tasks.feta_unet_search.task import (
    _safe_initial_campaign_observations,
)
from auto_researcher.tasks.models import DatasetManifest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(
    root: Path,
    *,
    experiment_id: str,
    learning_rate: float,
    score: float,
    endpoint: float,
    duration: float,
) -> None:
    run_id = "v6-run"
    configuration = FeTAUNetSearchConfiguration(
        maximum_epochs=100,
        learning_rate=learning_rate,
    )
    result_root = root / "output" / "runs" / run_id / experiment_id
    checkpoint_root = (
        root
        / "workspace"
        / "campaign_namespace"
        / experiment_id
        / "checkpoints"
        / "fold-0"
    )
    result_root.mkdir(parents=True)
    checkpoint_root.mkdir(parents=True)
    experiment = ExperimentSpec(
        experiment_id=experiment_id,
        hypothesis_id="hypothesis-v6",
        search_request_id="search-v6",
        configuration=configuration.model_dump(mode="json"),
        evaluator_id="feta-basic-unet-search-evaluator",
        code_version="code-v6",
        dataset_version="dataset-v1",
        provenance=ProvenanceKind.REAL,
    )
    evaluation = EvaluationResult(
        experiment_id=experiment_id,
        success=True,
        primary_score=score,
        metrics={
            "fold_summaries": [
                {
                    "total_duration_seconds": duration,
                    "peak_gpu_memory_bytes": 40 * 1024**3,
                    "validation_history": [
                        {
                            "epoch": 100,
                            "validation_score": endpoint,
                            "best_epoch": 90,
                            "best_validation_score": score,
                        }
                    ],
                }
            ]
        },
        constraint_results={"verified": True},
        evaluator_version="evaluator-v6",
        provenance=ProvenanceKind.REAL,
    )
    manifest = DatasetManifest(
        task_id="feta_unet_search",
        dataset_version="dataset-v1",
        files=("development",),
        hashes={"development": "a" * 64},
        loader_version="loader-v1",
        created_at=datetime(2026, 8, 22, tzinfo=UTC),
        metadata={"manifest_hash": "b" * 64},
    )
    atomic_json_write(result_root / "experiment_spec.json", experiment)
    atomic_json_write(result_root / "evaluation_result.json", evaluation)
    atomic_json_write(result_root / "dataset_manifest.json", manifest)
    (checkpoint_root / "last.pt").write_bytes(b"last")
    (checkpoint_root / "best.pt").write_bytes(b"best")
    atomic_json_write(
        checkpoint_root / "continuation.json",
        {
            "schema_version": CONTINUATION_VERSION,
            "trajectory_identity": trajectory_identity(configuration),
            "completed_epoch": 100,
            "best_epoch": 90,
            "best_score": score,
            "last_checkpoint_sha256": _sha256(checkpoint_root / "last.pt"),
            "best_checkpoint_sha256": _sha256(checkpoint_root / "best.pt"),
        },
    )
    atomic_json_write(
        checkpoint_root / "validation-history.json",
        {
            "entries": [
                {
                    "epoch": 100,
                    "validation_score": endpoint,
                    "best_epoch": 90,
                    "best_validation_score": score,
                }
            ]
        },
    )


def _runtime(tmp_path: Path) -> tuple[Path, tuple[str, str]]:
    root = tmp_path / "runtime"
    identifiers = ("experiment-source-one", "experiment-source-two")
    _source(
        root,
        experiment_id=identifiers[0],
        learning_rate=0.0001,
        score=0.819,
        endpoint=0.81,
        duration=3_000.0,
    )
    _source(
        root,
        experiment_id=identifiers[1],
        learning_rate=0.0002,
        score=0.818,
        endpoint=0.817,
        duration=2_000.0,
    )
    return root, identifiers


def test_extension_plan_is_identity_bound_and_uses_measured_duration(tmp_path):
    root, identifiers = _runtime(tmp_path)

    plan = build_extension_plan(
        runtime_root=root,
        source_run_id="v6-run",
        extension_run_id="v6-extension",
        experiment_ids=identifiers,
    )

    assert plan["schema_version"] == EXTENSION_SCHEMA_VERSION
    assert plan["candidate_count"] == 2
    assert plan["estimated_execution_seconds"] == 5_000.0
    assert plan["estimated_execution_seconds_with_margin"] == 6_550.0
    assert {item["source_experiment_id"] for item in plan["candidates"]} == set(
        identifiers
    )
    assert {item["configuration"]["maximum_epochs"] for item in plan["candidates"]} == {
        150
    }
    assert len({item["trajectory_identity"] for item in plan["candidates"]}) == 2


def test_extension_root_must_be_a_dedicated_runtime_child(tmp_path):
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()

    _validate_extension_root(runtime_root, runtime_root / "finalist-extension")

    with pytest.raises(ValueError, match="feta_unet_finalist_extension_root_invalid"):
        _validate_extension_root(runtime_root, runtime_root / "output")
    with pytest.raises(ValueError, match="feta_unet_finalist_extension_root_invalid"):
        _validate_extension_root(runtime_root, tmp_path / "outside" / "extension")


def test_extension_summary_exports_safe_ranked_v7_seed_evidence(tmp_path):
    root, identifiers = _runtime(tmp_path)
    extension_root = root / "extension"
    plan = build_extension_plan(
        runtime_root=root,
        source_run_id="v6-run",
        extension_run_id="v6-extension",
        experiment_ids=identifiers,
    )
    scores = (0.821, 0.823)
    for item, score in zip(plan["candidates"], scores):
        result_root = (
            extension_root
            / "output"
            / "runs"
            / "v6-extension"
            / item["extension_experiment_id"]
        )
        result_root.mkdir(parents=True)
        evaluation = EvaluationResult(
            experiment_id=item["extension_experiment_id"],
            success=True,
            primary_score=score,
            metrics={
                "fold_summaries": [
                    {
                        "resumed_from_epoch": 100,
                        "total_duration_seconds": 2_500.0,
                        "peak_gpu_memory_bytes": 42 * 1024**3,
                        "validation_history": [
                            {
                                "epoch": 150,
                                "validation_score": score - 0.001,
                                "best_epoch": 145,
                                "best_validation_score": score,
                            }
                        ],
                    }
                ]
            },
            constraint_results={"verified": True},
            evaluator_version="evaluator-v6",
            provenance=ProvenanceKind.REAL,
        )
        atomic_json_write(result_root / "evaluation_result.json", evaluation)

    summary = write_extension_summary(
        runtime_root=root,
        source_run_id="v6-run",
        extension_root=extension_root,
        extension_run_id="v6-extension",
        experiment_ids=identifiers,
    )

    assert summary["completed_count"] == 2
    evidence = json.loads(
        (extension_root / "v7-seed-evidence.json").read_text(encoding="utf-8")
    )
    assert evidence["ready"] is True
    assert evidence["parent_candidates"][0]["best_score"] == 0.823
    assert evidence["initial_incumbent_configuration"]["maximum_epochs"] == 150
    assert {
        item["maximum_epochs"] for item in evidence["direct_root_configurations"]
    } == {25}
    assert _safe_initial_campaign_observations(
        evidence["initial_campaign_observations"]
    ) == tuple(evidence["initial_campaign_observations"])


def test_extension_execution_uses_standard_evaluator_boundary_and_finishes(
    tmp_path, monkeypatch
):
    root, identifiers = _runtime(tmp_path)
    extension_root = root / "finalist-extension"
    task_config_path = root / "config" / "campaign.yaml"
    contract_path = root / "config" / "contract.yaml"
    task_config_path.parent.mkdir(parents=True)
    task_config_path.write_text(
        yaml.safe_dump(
            {
                "runtime": {
                    "data_dir": str(root / "data"),
                    "workspace_dir": str(root / "workspace"),
                    "environment": {"CUDA_VISIBLE_DEVICES": "1"},
                    "options": {
                        "workspace_namespace": "campaign_namespace",
                        "shared_preprocessing_cache": True,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    contract = ResearchContract(
        contract_id="contract-v6",
        schema_version="1.0",
        task_id="feta_unet_search",
        task_version="1.0",
        objective_version="objective-v1",
        primary_metric="mean_subject_macro_dice",
        task_constraints_version="constraints-v1",
        question="Improve the development score.",
        objective="Maximise development Dice.",
        allowed_search_types=frozenset({SearchType.DIRECT}),
        evaluator_id="feta-basic-unet-search-evaluator",
        verifier_id="verifier-v1",
        maximum_cycles=1,
        maximum_experiments=2,
        maximum_cost=0.0,
        provenance=ProvenanceKind.REAL,
    )
    contract_path.write_text(
        yaml.safe_dump(contract.model_dump(mode="json")), encoding="utf-8"
    )

    class FakeEvaluator:
        def __init__(self, context, metadata, manifest):
            self.context = context

        def evaluate(self, experiment, received_contract):
            assert received_contract.contract_id == "contract-v6"
            assert experiment.configuration["maximum_epochs"] == 150
            evaluation = EvaluationResult(
                experiment_id=experiment.experiment_id,
                success=True,
                primary_score=0.825,
                metrics={
                    "fold_summaries": [
                        {
                            "resumed_from_epoch": 100,
                            "total_duration_seconds": 2_000.0,
                            "peak_gpu_memory_bytes": 42 * 1024**3,
                            "validation_history": [
                                {
                                    "epoch": 150,
                                    "validation_score": 0.824,
                                    "best_epoch": 145,
                                    "best_validation_score": 0.825,
                                }
                            ],
                        }
                    ]
                },
                constraint_results={"verified": True},
                evaluator_version="evaluator-v6",
                provenance=ProvenanceKind.REAL,
            )
            destination = (
                self.context.output_dir
                / "runs"
                / self.context.run_id
                / experiment.experiment_id
                / "evaluation_result.json"
            )
            atomic_json_write(destination, evaluation)
            return evaluation

    monkeypatch.setattr(
        "auto_researcher.tasks.feta_unet_search.finalist_extension.FeTAUNetSearchEvaluator",
        FakeEvaluator,
    )

    summary = execute_extension(
        runtime_root=root,
        source_run_id="v6-run",
        extension_root=extension_root,
        extension_run_id="v6-extension",
        task_config_path=task_config_path,
        contract_path=contract_path,
        experiment_ids=identifiers,
        maximum_wall_time_seconds=10_000.0,
    )

    assert summary["completed_count"] == 2
    assert (extension_root / "extension-plan.json").is_file()
    assert (
        json.loads(
            (extension_root / "v7-seed-evidence.json").read_text(encoding="utf-8")
        )["ready"]
        is True
    )
