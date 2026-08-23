from __future__ import annotations

from auto_researcher.graph.builder import build_graph
from auto_researcher.provenance.sqlite_store import SQLiteProvenanceStore
from auto_researcher.runtime.dependencies import task_memory_dependencies
from auto_researcher.runtime.execution import start_run
from auto_researcher.tasks import TaskRuntimeContext
from auto_researcher.tasks.feta_unet_search.v8_parent_import import (
    import_verified_evaluation,
)
from auto_researcher.tasks.synthetic import (
    SyntheticTask,
    default_synthetic_configuration,
    default_synthetic_contract,
)


def test_verified_parent_import_is_transactional_and_idempotent(tmp_path):
    source_output = tmp_path / "source-output"
    source_run = "source-run"
    contract = default_synthetic_contract()
    dependencies = task_memory_dependencies(
        SyntheticTask(),
        TaskRuntimeContext(run_id=source_run, output_dir=source_output),
        contract,
        default_synthetic_configuration(),
    )
    final = start_run(
        build_graph(dependencies),
        {"run_id": source_run, "thread_id": "source-thread", "contract": contract},
        {"configurable": {"thread_id": "source-thread"}},
    )
    experiment_id = final["experiment_spec"].experiment_id
    source_record = dependencies.provenance_store.get_evaluation_reuse(
        source_run, experiment_id
    )
    assert source_record is not None
    source_db = tmp_path / "source.sqlite"
    source_store = SQLiteProvenanceStore(source_db)
    source_store.append_evaluation_reuse(source_record)
    source_store.close()

    target_output = tmp_path / "target-output"
    target_db = tmp_path / "target.sqlite"
    arguments = {
        "source_output_dir": source_output,
        "source_provenance_db": source_db,
        "source_run_id": source_run,
        "target_output_dir": target_output,
        "target_provenance_db": target_db,
        "target_run_id": "target-run",
        "experiment_id": experiment_id,
    }
    first = import_verified_evaluation(**arguments)
    second = import_verified_evaluation(**arguments)

    assert first["replayed"] is False
    assert second["replayed"] is True
    assert first["record_sha256"] == second["record_sha256"]
    target_store = SQLiteProvenanceStore(target_db)
    imported = target_store.get_evaluation_reuse("target-run", experiment_id)
    target_store.close()
    assert imported is not None
    assert imported.result.primary_score == source_record.result.primary_score
    assert imported.expected_artefact_references == imported.result.artefact_references
    assert all(
        reference.startswith(f"runs/target-run/{experiment_id}/")
        for reference in imported.expected_artefact_references
    )
