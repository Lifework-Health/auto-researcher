"""Targeted 100-to-150 epoch continuation for verified V6 finalists."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from auto_researcher.contracts.models import (
    EvaluationResult,
    ExperimentSpec,
    ResearchContract,
)
from auto_researcher.runtime.identity import payload_hash
from auto_researcher.tasks.artifacts import atomic_json_write, safe_segment
from auto_researcher.tasks.feta_unet_search.configuration import (
    FeTAUNetSearchConfiguration,
)
from auto_researcher.tasks.feta_unet_search.continuation import (
    CONTINUATION_VERSION,
    trajectory_identity,
)
from auto_researcher.tasks.feta_unet_search.evaluator import (
    FeTAUNetSearchEvaluator,
)
from auto_researcher.tasks.models import (
    DatasetManifest,
    ExperimentMetadata,
    TaskRuntimeContext,
)

EXTENSION_SCHEMA_VERSION = "feta-unet-v6-finalist-extension-v1"
V7_SEED_SCHEMA_VERSION = "feta-unet-v7-seed-evidence-v1"
SOURCE_FIDELITY = 100
TARGET_FIDELITY = 150
DEFAULT_MAXIMUM_WALL_TIME_SECONDS = 4 * 60 * 60
FINALISATION_RESERVE_SECONDS = 5 * 60


@dataclass(frozen=True)
class SourceCandidate:
    experiment: ExperimentSpec
    evaluation: EvaluationResult
    dataset_manifest: DatasetManifest
    configuration: FeTAUNetSearchConfiguration
    candidate_root: Path
    workspace_namespace: str
    source_best_epoch: int
    source_best_score: float
    source_endpoint_score: float
    estimated_extension_seconds: float


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("feta_unet_finalist_extension_artefact_invalid") from exc


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError("feta_unet_finalist_extension_configuration_invalid") from exc
    if not isinstance(value, dict):
        raise ValueError("feta_unet_finalist_extension_configuration_invalid")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_extension_root(runtime_root: Path, extension_root: Path) -> None:
    runtime = runtime_root.expanduser().resolve()
    extension = extension_root.expanduser().resolve()
    if extension.parent != runtime or extension.name in {
        "config",
        "control",
        "logs",
        "output",
        "workspace",
    }:
        raise ValueError("feta_unet_finalist_extension_root_invalid")


def _finite_number(value: object, reason: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(reason)
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(reason)
    return converted


def _fold_summary(evaluation: EvaluationResult) -> dict[str, Any]:
    raw = evaluation.metrics.get("fold_summaries")
    if not isinstance(raw, list) or len(raw) != 1 or not isinstance(raw[0], dict):
        raise ValueError("feta_unet_finalist_extension_metrics_invalid")
    return raw[0]


def _history_endpoint(history: object, fidelity: int) -> tuple[int, float]:
    if not isinstance(history, list):
        raise ValueError("feta_unet_finalist_extension_history_invalid")
    matching = [
        item
        for item in history
        if isinstance(item, dict) and item.get("epoch") == fidelity
    ]
    if len(matching) != 1:
        raise ValueError("feta_unet_finalist_extension_history_invalid")
    best_epoch = matching[0].get("best_epoch")
    if isinstance(best_epoch, bool) or not isinstance(best_epoch, int):
        raise ValueError("feta_unet_finalist_extension_history_invalid")
    return best_epoch, _finite_number(
        matching[0].get("validation_score"),
        "feta_unet_finalist_extension_history_invalid",
    )


def _candidate_workspace_root(runtime_root: Path, experiment_id: str) -> Path:
    matches = tuple(
        path
        for path in (runtime_root / "workspace").glob(f"*/{experiment_id}")
        if path.is_dir()
    )
    if len(matches) != 1:
        raise ValueError("feta_unet_finalist_extension_workspace_invalid")
    return matches[0]


def _validate_continuation_source(
    candidate_root: Path,
    configuration: FeTAUNetSearchConfiguration,
) -> tuple[int, float]:
    checkpoint_root = candidate_root / "checkpoints" / "fold-0"
    manifest_path = checkpoint_root / "continuation.json"
    history_path = checkpoint_root / "validation-history.json"
    last_path = checkpoint_root / "last.pt"
    best_path = checkpoint_root / "best.pt"
    if not all(
        path.is_file() for path in (manifest_path, history_path, last_path, best_path)
    ):
        raise ValueError("feta_unet_finalist_extension_checkpoint_missing")
    manifest = _read_json(manifest_path)
    history_payload = _read_json(history_path)
    entries = (
        history_payload.get("entries") if isinstance(history_payload, dict) else None
    )
    if not isinstance(manifest, dict) or not isinstance(entries, list):
        raise ValueError("feta_unet_finalist_extension_checkpoint_invalid")
    if (
        manifest.get("schema_version") != CONTINUATION_VERSION
        or manifest.get("completed_epoch") != SOURCE_FIDELITY
        or manifest.get("trajectory_identity") != trajectory_identity(configuration)
        or manifest.get("last_checkpoint_sha256") != _sha256(last_path)
        or manifest.get("best_checkpoint_sha256") != _sha256(best_path)
    ):
        raise ValueError("feta_unet_finalist_extension_checkpoint_invalid")
    best_epoch = manifest.get("best_epoch")
    best_score = _finite_number(
        manifest.get("best_score"),
        "feta_unet_finalist_extension_checkpoint_invalid",
    )
    if (
        isinstance(best_epoch, bool)
        or not isinstance(best_epoch, int)
        or not 1 <= best_epoch <= SOURCE_FIDELITY
    ):
        raise ValueError("feta_unet_finalist_extension_checkpoint_invalid")
    _history_endpoint(entries, SOURCE_FIDELITY)
    return best_epoch, best_score


def load_source_candidate(
    *,
    runtime_root: Path,
    source_run_id: str,
    experiment_id: str,
) -> SourceCandidate:
    source_run_id = safe_segment(source_run_id, "source_run_id")
    experiment_id = safe_segment(experiment_id, "experiment_id")
    result_root = runtime_root / "output" / "runs" / source_run_id / experiment_id
    experiment = ExperimentSpec.model_validate(
        _read_json(result_root / "experiment_spec.json")
    )
    evaluation = EvaluationResult.model_validate(
        _read_json(result_root / "evaluation_result.json")
    )
    dataset_manifest = DatasetManifest.model_validate(
        _read_json(result_root / "dataset_manifest.json")
    )
    if (
        experiment.experiment_id != experiment_id
        or evaluation.experiment_id != experiment_id
        or not evaluation.success
        or evaluation.primary_score is None
    ):
        raise ValueError("feta_unet_finalist_extension_source_invalid")
    configuration = FeTAUNetSearchConfiguration.model_validate(experiment.configuration)
    if configuration.maximum_epochs != SOURCE_FIDELITY:
        raise ValueError("feta_unet_finalist_extension_source_fidelity_invalid")
    candidate_root = _candidate_workspace_root(runtime_root, experiment_id)
    best_epoch, best_score = _validate_continuation_source(
        candidate_root, configuration
    )
    if not math.isclose(float(evaluation.primary_score), best_score, abs_tol=1e-12):
        raise ValueError("feta_unet_finalist_extension_source_score_invalid")
    fold = _fold_summary(evaluation)
    history = fold.get("validation_history")
    _, endpoint = _history_endpoint(history, SOURCE_FIDELITY)
    duration = _finite_number(
        fold.get("total_duration_seconds"),
        "feta_unet_finalist_extension_duration_invalid",
    )
    if duration <= 0:
        raise ValueError("feta_unet_finalist_extension_duration_invalid")
    return SourceCandidate(
        experiment=experiment,
        evaluation=evaluation,
        dataset_manifest=dataset_manifest,
        configuration=configuration,
        candidate_root=candidate_root,
        workspace_namespace=candidate_root.parent.name,
        source_best_epoch=best_epoch,
        source_best_score=best_score,
        source_endpoint_score=endpoint,
        # The source 100-epoch result is itself a 50-to-100 continuation, so its
        # measured total duration is the best estimate for another 50 epochs.
        estimated_extension_seconds=duration,
    )


def _extension_experiment_id(source: SourceCandidate) -> str:
    identity = payload_hash(
        {
            "schema_version": EXTENSION_SCHEMA_VERSION,
            "source_experiment_id": source.experiment.experiment_id,
            "trajectory_identity": trajectory_identity(source.configuration),
            "target_fidelity": TARGET_FIDELITY,
        }
    )
    return f"experiment-{identity[:16]}"


def build_extension_plan(
    *,
    runtime_root: Path,
    source_run_id: str,
    extension_run_id: str,
    experiment_ids: tuple[str, ...],
) -> dict[str, Any]:
    extension_run_id = safe_segment(extension_run_id, "extension_run_id")
    if len(experiment_ids) != 2 or len(set(experiment_ids)) != 2:
        raise ValueError("feta_unet_finalist_extension_requires_two_sources")
    sources = tuple(
        load_source_candidate(
            runtime_root=runtime_root,
            source_run_id=source_run_id,
            experiment_id=experiment_id,
        )
        for experiment_id in experiment_ids
    )
    trajectories = {trajectory_identity(source.configuration) for source in sources}
    if len(trajectories) != 2:
        raise ValueError("feta_unet_finalist_extension_trajectories_not_distinct")
    namespaces = {source.workspace_namespace for source in sources}
    manifests = {
        payload_hash(source.dataset_manifest.model_dump(mode="json"))
        for source in sources
    }
    metadata = {
        (
            source.experiment.evaluator_id,
            source.experiment.code_version,
            source.experiment.dataset_version,
            source.experiment.provenance.value,
        )
        for source in sources
    }
    if len(namespaces) != 1 or len(manifests) != 1 or len(metadata) != 1:
        raise ValueError("feta_unet_finalist_extension_source_identity_mismatch")
    rows = []
    for source in sources:
        configuration = source.configuration.model_copy(
            update={"maximum_epochs": TARGET_FIDELITY}
        )
        fold = _fold_summary(source.evaluation)
        rows.append(
            {
                "source_experiment_id": source.experiment.experiment_id,
                "extension_experiment_id": _extension_experiment_id(source),
                "trajectory_identity": trajectory_identity(source.configuration),
                "workspace_namespace": source.workspace_namespace,
                "source_fidelity": SOURCE_FIDELITY,
                "target_fidelity": TARGET_FIDELITY,
                "source_best_epoch": source.source_best_epoch,
                "source_best_score": source.source_best_score,
                "source_endpoint_score": source.source_endpoint_score,
                "source_peak_gpu_memory_bytes": fold.get("peak_gpu_memory_bytes"),
                "estimated_extension_seconds": source.estimated_extension_seconds,
                "configuration": configuration.model_dump(mode="json"),
            }
        )
    estimated = sum(row["estimated_extension_seconds"] for row in rows)
    return {
        "schema_version": EXTENSION_SCHEMA_VERSION,
        "source_run_id": source_run_id,
        "extension_run_id": extension_run_id,
        "source_fidelity": SOURCE_FIDELITY,
        "target_fidelity": TARGET_FIDELITY,
        "candidate_count": len(rows),
        "estimated_execution_seconds": estimated,
        "estimated_execution_seconds_with_margin": (
            estimated * 1.25 + FINALISATION_RESERVE_SECONDS
        ),
        "candidates": rows,
    }


def _extension_spec(source: SourceCandidate) -> ExperimentSpec:
    return source.experiment.model_copy(
        update={
            "experiment_id": _extension_experiment_id(source),
            "hypothesis_id": "hypothesis-v6-finalist-extension",
            "search_request_id": (
                f"search-v6-finalist-extension-{source.experiment.experiment_id}"
            ),
            "configuration": source.configuration.model_copy(
                update={"maximum_epochs": TARGET_FIDELITY}
            ).model_dump(mode="json"),
        }
    )


def _runtime_context(
    *,
    task_config_path: Path,
    runtime_root: Path,
    extension_root: Path,
    extension_run_id: str,
    workspace_namespace: str,
) -> TaskRuntimeContext:
    task_config = _read_yaml(task_config_path)
    runtime = task_config.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("feta_unet_finalist_extension_configuration_invalid")
    data_dir = runtime.get("data_dir")
    workspace_dir = runtime.get("workspace_dir")
    environment = runtime.get("environment", {})
    options = runtime.get("options", {})
    if (
        not isinstance(data_dir, str)
        or not isinstance(workspace_dir, str)
        or not isinstance(environment, dict)
        or not isinstance(options, dict)
        or Path(workspace_dir).expanduser().resolve()
        != (runtime_root / "workspace").resolve()
        or options.get("workspace_namespace") != workspace_namespace
    ):
        raise ValueError("feta_unet_finalist_extension_runtime_identity_mismatch")
    return TaskRuntimeContext(
        run_id=extension_run_id,
        data_dir=Path(data_dir).expanduser().resolve(),
        workspace_dir=(runtime_root / "workspace").resolve(),
        output_dir=(extension_root / "output").resolve(),
        environment=environment,
        task_options={
            **options,
            "workspace_namespace": workspace_namespace,
            "shared_preprocessing_cache": True,
        },
    )


def _completed_result(
    extension_root: Path, extension_run_id: str, experiment_id: str
) -> EvaluationResult | None:
    path = (
        extension_root
        / "output"
        / "runs"
        / extension_run_id
        / experiment_id
        / "evaluation_result.json"
    )
    if not path.exists():
        return None
    return EvaluationResult.model_validate(_read_json(path))


def execute_extension(
    *,
    runtime_root: Path,
    source_run_id: str,
    extension_root: Path,
    extension_run_id: str,
    task_config_path: Path,
    contract_path: Path,
    experiment_ids: tuple[str, ...],
    maximum_wall_time_seconds: float,
) -> dict[str, Any]:
    _validate_extension_root(runtime_root, extension_root)
    plan = build_extension_plan(
        runtime_root=runtime_root,
        source_run_id=source_run_id,
        extension_run_id=extension_run_id,
        experiment_ids=experiment_ids,
    )
    if (
        maximum_wall_time_seconds <= 0
        or plan["estimated_execution_seconds_with_margin"] > maximum_wall_time_seconds
    ):
        raise ValueError("feta_unet_finalist_extension_wall_time_insufficient")
    extension_root.mkdir(parents=True, exist_ok=True)
    atomic_json_write(extension_root / "extension-plan.json", plan)
    contract = ResearchContract.model_validate(_read_yaml(contract_path))
    started = time.monotonic()
    for experiment_id in experiment_ids:
        source = load_source_candidate(
            runtime_root=runtime_root,
            source_run_id=source_run_id,
            experiment_id=experiment_id,
        )
        extension_id = _extension_experiment_id(source)
        existing = _completed_result(extension_root, extension_run_id, extension_id)
        if existing is not None:
            if not existing.success:
                raise ValueError("feta_unet_finalist_extension_existing_failure")
            continue
        remaining = maximum_wall_time_seconds - (time.monotonic() - started)
        required = source.estimated_extension_seconds * 1.25
        if remaining < required + FINALISATION_RESERVE_SECONDS:
            break
        context = _runtime_context(
            task_config_path=task_config_path,
            runtime_root=runtime_root,
            extension_root=extension_root,
            extension_run_id=extension_run_id,
            workspace_namespace=source.workspace_namespace,
        )
        metadata = ExperimentMetadata(
            evaluator_id=source.experiment.evaluator_id,
            code_version=source.experiment.code_version,
            dataset_version=source.experiment.dataset_version,
            provenance=source.experiment.provenance,
        )
        evaluator = FeTAUNetSearchEvaluator(
            context,
            metadata,
            source.dataset_manifest,
        )
        result = evaluator.evaluate(_extension_spec(source), contract)
        if not result.success:
            raise RuntimeError(result.error or "feta_unet_finalist_extension_failed")
        write_extension_summary(
            runtime_root=runtime_root,
            source_run_id=source_run_id,
            extension_root=extension_root,
            extension_run_id=extension_run_id,
            experiment_ids=experiment_ids,
        )
    summary = write_extension_summary(
        runtime_root=runtime_root,
        source_run_id=source_run_id,
        extension_root=extension_root,
        extension_run_id=extension_run_id,
        experiment_ids=experiment_ids,
    )
    if summary["completed_count"] != len(experiment_ids):
        raise RuntimeError("feta_unet_finalist_extension_incomplete")
    return summary


def _result_history(evaluation: EvaluationResult) -> list[dict[str, Any]]:
    history = _fold_summary(evaluation).get("validation_history")
    if not isinstance(history, list) or any(
        not isinstance(item, dict) for item in history
    ):
        raise ValueError("feta_unet_finalist_extension_history_invalid")
    return history


def write_extension_summary(
    *,
    runtime_root: Path,
    source_run_id: str,
    extension_root: Path,
    extension_run_id: str,
    experiment_ids: tuple[str, ...],
) -> dict[str, Any]:
    rows = []
    for experiment_id in experiment_ids:
        source = load_source_candidate(
            runtime_root=runtime_root,
            source_run_id=source_run_id,
            experiment_id=experiment_id,
        )
        extension_id = _extension_experiment_id(source)
        evaluation = _completed_result(extension_root, extension_run_id, extension_id)
        if evaluation is None:
            rows.append(
                {
                    "source_experiment_id": experiment_id,
                    "extension_experiment_id": extension_id,
                    "trajectory_identity": trajectory_identity(source.configuration),
                    "status": "PENDING",
                }
            )
            continue
        history = _result_history(evaluation)
        endpoint_rows = [
            item for item in history if item.get("epoch") == TARGET_FIDELITY
        ]
        if len(endpoint_rows) != 1:
            raise ValueError("feta_unet_finalist_extension_history_invalid")
        fold = _fold_summary(evaluation)
        configuration = source.configuration.model_copy(
            update={"maximum_epochs": TARGET_FIDELITY}
        )
        rows.append(
            {
                "source_experiment_id": experiment_id,
                "extension_experiment_id": extension_id,
                "trajectory_identity": trajectory_identity(source.configuration),
                "status": "COMPLETED" if evaluation.success else "FAILED",
                "best_score": evaluation.primary_score,
                "best_epoch": endpoint_rows[0].get("best_epoch"),
                "endpoint_score": endpoint_rows[0].get("validation_score"),
                "resumed_from_epoch": fold.get("resumed_from_epoch"),
                "total_duration_seconds": fold.get("total_duration_seconds"),
                "peak_gpu_memory_bytes": fold.get("peak_gpu_memory_bytes"),
                "configuration": configuration.model_dump(mode="json"),
            }
        )
    completed = [row for row in rows if row["status"] == "COMPLETED"]
    summary = {
        "schema_version": EXTENSION_SCHEMA_VERSION,
        "source_run_id": source_run_id,
        "extension_run_id": extension_run_id,
        "target_fidelity": TARGET_FIDELITY,
        "candidate_count": len(rows),
        "completed_count": len(completed),
        "candidates": rows,
    }
    atomic_json_write(extension_root / "extension-summary.json", summary)
    ranked = sorted(
        completed,
        key=lambda item: (float(item["best_score"]), item["trajectory_identity"]),
        reverse=True,
    )
    observations = [
        (
            "Verified V6 finalist trajectory "
            f"{item['trajectory_identity'][:12]} reached mean macro Dice "
            f"{float(item['best_score']):.12f} at best epoch "
            f"{item['best_epoch']} and {float(item['endpoint_score']):.12f} "
            "at 150 epochs."
        )
        for item in ranked
    ]
    v7_evidence = {
        "schema_version": V7_SEED_SCHEMA_VERSION,
        "source_extension_run_id": extension_run_id,
        "ready": len(ranked) == len(experiment_ids),
        "initial_campaign_observations": observations,
        "initial_incumbent_configuration": (
            ranked[0]["configuration"] if ranked else None
        ),
        "direct_root_configurations": [
            {**item["configuration"], "maximum_epochs": 25} for item in ranked
        ],
        "parent_candidates": ranked,
    }
    atomic_json_write(extension_root / "v7-seed-evidence.json", v7_evidence)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "run", "report"), required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--extension-root", type=Path, required=True)
    parser.add_argument("--extension-run-id", required=True)
    parser.add_argument("--experiment-id", action="append", required=True)
    parser.add_argument("--task-config", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument(
        "--maximum-wall-time-seconds",
        type=float,
        default=DEFAULT_MAXIMUM_WALL_TIME_SECONDS,
    )
    args = parser.parse_args(argv)
    experiment_ids = tuple(args.experiment_id)
    _validate_extension_root(args.runtime_root, args.extension_root)
    if args.mode == "preflight":
        value = build_extension_plan(
            runtime_root=args.runtime_root,
            source_run_id=args.source_run_id,
            extension_run_id=args.extension_run_id,
            experiment_ids=experiment_ids,
        )
    elif args.mode == "run":
        if args.task_config is None or args.contract is None:
            parser.error("--task-config and --contract are required for run mode")
        value = execute_extension(
            runtime_root=args.runtime_root,
            source_run_id=args.source_run_id,
            extension_root=args.extension_root,
            extension_run_id=args.extension_run_id,
            task_config_path=args.task_config,
            contract_path=args.contract,
            experiment_ids=experiment_ids,
            maximum_wall_time_seconds=args.maximum_wall_time_seconds,
        )
    else:
        value = write_extension_summary(
            runtime_root=args.runtime_root,
            source_run_id=args.source_run_id,
            extension_root=args.extension_root,
            extension_run_id=args.extension_run_id,
            experiment_ids=experiment_ids,
        )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
