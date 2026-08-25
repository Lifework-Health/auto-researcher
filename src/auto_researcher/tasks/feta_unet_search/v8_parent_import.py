"""Transactional cross-run import for verified V8 OpenEvolve parents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from auto_researcher.contracts.models import EvaluationResult, ExperimentSpec
from auto_researcher.graph.nodes.evaluate import validate_reused_evaluation
from auto_researcher.provenance.reuse import EvaluationReuseRecord
from auto_researcher.provenance.sqlite_store import SQLiteProvenanceStore
from auto_researcher.runtime.identity import payload_hash
from auto_researcher.tasks.artifacts import (
    ARTEFACT_BUNDLE_METADATA_KEY,
    artefact_bundle_identity,
    artefact_references,
    write_artefact_bundle,
)
from auto_researcher.tasks.models import DatasetManifest, TaskRuntimeContext
from auto_researcher.tasks.feta_unet_search.configuration import (
    FeTAUNetSearchConfiguration,
)


def _load_json(path: Path, model_type):
    try:
        return model_type.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("v8_parent_import_source_bundle_invalid") from exc


def import_verified_evaluation(
    *,
    source_output_dir: Path,
    source_provenance_db: Path,
    source_run_id: str,
    target_output_dir: Path,
    target_provenance_db: Path,
    target_run_id: str,
    experiment_id: str,
) -> dict[str, object]:
    """Republish one verified bundle and bind it to target-run reuse identity."""

    source_context = TaskRuntimeContext(
        run_id=source_run_id, output_dir=source_output_dir
    )
    target_context = TaskRuntimeContext(
        run_id=target_run_id, output_dir=target_output_dir
    )
    source_store = SQLiteProvenanceStore(source_provenance_db)
    target_store = SQLiteProvenanceStore(target_provenance_db)
    try:
        source = source_store.get_evaluation_reuse(source_run_id, experiment_id)
        if source is None:
            raise ValueError("v8_parent_import_source_reuse_missing")
        validate_reused_evaluation(
            source, SimpleNamespace(runtime_context=source_context)
        )
        existing = target_store.get_evaluation_reuse(target_run_id, experiment_id)
        if existing is not None:
            validate_reused_evaluation(
                existing, SimpleNamespace(runtime_context=target_context)
            )
            return {
                "experiment_id": experiment_id,
                "source_run_id": source_run_id,
                "target_run_id": target_run_id,
                "record_sha256": payload_hash(existing),
                "replayed": True,
            }

        source_directory = source_output_dir / "runs" / source_run_id / experiment_id
        experiment = _load_json(
            source_directory / "experiment_spec.json", ExperimentSpec
        )
        evaluation = _load_json(
            source_directory / "evaluation_result.json", EvaluationResult
        )
        dataset_manifest = _load_json(
            source_directory / "dataset_manifest.json", DatasetManifest
        )
        try:
            evaluator_manifest = json.loads(
                (source_directory / "evaluator_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            evaluator_manifest.pop(ARTEFACT_BUNDLE_METADATA_KEY)
        except (KeyError, OSError, TypeError, ValueError) as exc:
            raise ValueError("v8_parent_import_source_bundle_invalid") from exc
        if experiment.experiment_id != experiment_id or not evaluation.success:
            raise ValueError("v8_parent_import_source_identity_invalid")

        references = artefact_references(target_context, experiment_id)
        imported_evaluation = evaluation.model_copy(
            update={"artefact_references": references}
        )
        write_artefact_bundle(
            target_context,
            experiment,
            imported_evaluation,
            dataset_manifest,
            evaluator_manifest,
        )
        bundle = artefact_bundle_identity(target_context, experiment_id)
        experiment_hash = payload_hash(experiment)
        scientific_identity = payload_hash(
            {
                "run_id": target_run_id,
                "experiment_id": experiment_id,
                "evaluator_version": source.evaluator_version,
                "dataset_version": experiment.dataset_version,
                "code_version": experiment.code_version,
                "experiment_payload_hash": experiment_hash,
            }
        )
        imported = EvaluationReuseRecord(
            run_id=target_run_id,
            experiment_id=experiment_id,
            scientific_identity_hash=scientific_identity,
            experiment_payload_hash=experiment_hash,
            result_payload_hash=payload_hash(imported_evaluation),
            evaluator_version=source.evaluator_version,
            dataset_version=experiment.dataset_version,
            code_version=experiment.code_version,
            artefact_bundle_hash=bundle.bundle_sha256,
            artefact_bundle_schema_version=bundle.schema_version,
            result_encoding_version=bundle.result_encoding_version,
            expected_artefact_references=bundle.references,
            evaluator_manifest_payload_hash=bundle.evaluator_manifest_payload_hash,
            completed_at=source.completed_at,
            result=imported_evaluation,
        )
        target_store.append_evaluation_reuse(imported)
        validate_reused_evaluation(
            imported, SimpleNamespace(runtime_context=target_context)
        )
        return {
            "experiment_id": experiment_id,
            "source_run_id": source_run_id,
            "target_run_id": target_run_id,
            "source_record_sha256": payload_hash(source),
            "record_sha256": payload_hash(imported),
            "bundle_sha256": bundle.bundle_sha256,
            "replayed": False,
        }
    finally:
        source_store.close()
        target_store.close()


def _validated_parent_ids(
    task_config: Path, source_output_dir: Path, source_run_id: str
) -> tuple[str, ...]:
    try:
        raw = yaml.safe_load(task_config.read_text(encoding="utf-8"))
        selected = raw["runtime"]["options"]["campaign_portfolio"]["parent_selection"][
            "selected_parents"
        ]
    except (KeyError, OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ValueError("v8_parent_import_configuration_invalid") from exc
    parents = tuple(
        item for item in selected if item.get("selection_role") == "mandatory"
    )
    if len(parents) != 2:
        raise ValueError("v8_parent_import_configuration_invalid")
    for parent in parents:
        try:
            experiment_id = str(parent["experiment_id"])
            planned = FeTAUNetSearchConfiguration.model_validate(
                parent["configuration"]
            )
            source = _load_json(
                source_output_dir
                / "runs"
                / source_run_id
                / experiment_id
                / "experiment_spec.json",
                ExperimentSpec,
            )
            published = FeTAUNetSearchConfiguration.model_validate(source.configuration)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("v8_parent_import_source_identity_invalid") from exc
        if source.experiment_id != experiment_id or published != planned:
            raise ValueError("v8_parent_import_source_identity_invalid")
    return tuple(str(item["experiment_id"]) for item in parents)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--source-output-dir", type=Path, required=True)
    parser.add_argument("--source-provenance-db", type=Path, required=True)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--target-output-dir", type=Path, required=True)
    parser.add_argument("--target-provenance-db", type=Path, required=True)
    parser.add_argument("--target-run-id", required=True)
    parser.add_argument("--experiment-id", action="append", required=True)
    args = parser.parse_args(argv)
    expected_ids = _validated_parent_ids(
        args.task_config, args.source_output_dir, args.source_run_id
    )
    if tuple(args.experiment_id) != expected_ids:
        raise ValueError("v8_parent_import_experiment_set_invalid")
    rows = [
        import_verified_evaluation(
            source_output_dir=args.source_output_dir,
            source_provenance_db=args.source_provenance_db,
            source_run_id=args.source_run_id,
            target_output_dir=args.target_output_dir,
            target_provenance_db=args.target_provenance_db,
            target_run_id=args.target_run_id,
            experiment_id=experiment_id,
        )
        for experiment_id in args.experiment_id
    ]
    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
