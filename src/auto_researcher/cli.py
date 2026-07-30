"""Generic task-oriented command line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
import yaml

from auto_researcher.contracts.enums import RunStatus
from auto_researcher.contracts.models import ResearchContract
from auto_researcher.graph.builder import build_graph
from auto_researcher.provenance.sqlite_store import SQLiteProvenanceStore
from auto_researcher.runtime.dependencies import task_sqlite_dependencies
from auto_researcher.tasks import TaskRuntimeContext, default_task_registry
from auto_researcher.tasks.models import TaskPluginError
from auto_researcher.tasks.synthetic import (
    default_synthetic_configuration,
    default_synthetic_contract,
)

app = typer.Typer(no_args_is_help=True, help="Run task plugins on Auto Researcher.")
DEFAULT_DATA_DIR = Path(".auto-researcher")


def _mock_contract(max_cycles: int) -> ResearchContract:
    """PR 1 compatibility alias for the default synthetic contract."""
    return default_synthetic_contract(max_cycles)


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def _load_task_configuration(
    path: Path | None,
    task_id: str,
    task_version: str,
) -> tuple[dict, dict]:
    if path is None:
        if task_id != "synthetic":
            raise ValueError("--task-config is required for non-synthetic tasks")
        return default_synthetic_configuration(), {}
    payload = _load_yaml(path)
    task_identity = payload.get("task", {})
    if task_identity.get("id") != task_id:
        raise ValueError(
            f"task config identifies {task_identity.get('id')!r}, not {task_id!r}"
        )
    if str(task_identity.get("version")) != task_version:
        raise ValueError(
            "task config identifies version "
            f"{task_identity.get('version')!r}, not {task_version!r}"
        )
    experiment = payload.get("experiment")
    runtime = payload.get("runtime", {})
    if not isinstance(experiment, dict) or not isinstance(runtime, dict):
        raise ValueError("task config requires mapping sections: experiment and runtime")
    return experiment, runtime


@app.command()
def run(
    task_id: str = typer.Option("synthetic", "--task"),
    contract_path: Path | None = typer.Option(None, "--contract"),
    task_config: Path | None = typer.Option(None, "--task-config"),
    mock: bool = typer.Option(False, "--mock", help="Alias for --task synthetic."),
    run_id: str = typer.Option(..., "--run-id"),
    thread_id: str = typer.Option(..., "--thread-id"),
    max_cycles: int = typer.Option(1, "--max-cycles", min=1),
    checkpoint_db: Path = typer.Option(
        DEFAULT_DATA_DIR / "checkpoints.sqlite",
        "--checkpoint-db",
    ),
    provenance_db: Path = typer.Option(
        DEFAULT_DATA_DIR / "provenance.sqlite",
        "--provenance-db",
    ),
) -> None:
    """Run a registered task through the unchanged LangGraph control plane."""
    if mock:
        task_id = "synthetic"
    registry = default_task_registry()
    try:
        contract = (
            ResearchContract.model_validate(_load_yaml(contract_path))
            if contract_path
            else default_synthetic_contract(max_cycles)
        )
        if contract.task_id != task_id:
            raise ValueError(
                f"contract targets task {contract.task_id!r}, not {task_id!r}"
            )
        task = registry.get(task_id, contract.task_version)
        experiment, runtime = _load_task_configuration(
            task_config,
            task_id,
            contract.task_version,
        )
        runtime_context = TaskRuntimeContext(
            run_id=run_id,
            data_dir=runtime.get("data_dir"),
            workspace_dir=runtime.get("workspace_dir"),
            output_dir=runtime.get("output_dir", DEFAULT_DATA_DIR),
            environment=runtime.get("environment", {}),
            task_options=runtime.get("options", {}),
        )
        config = {"configurable": {"thread_id": thread_id}}
        with task_sqlite_dependencies(
            task,
            runtime_context,
            contract,
            experiment,
            checkpoint_db,
            provenance_db,
        ) as dependencies:
            graph = build_graph(dependencies)
            final = graph.invoke(
                {"run_id": run_id, "thread_id": thread_id, "contract": contract},
                config,
            )
    except (TaskPluginError, ValueError, FileNotFoundError) as exc:
        typer.echo(f"Task setup failed: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    verification = final.get("verification_result")
    evaluation = final.get("evaluation_result")
    typer.echo(f"Task: {contract.task_id}@{contract.task_version}")
    typer.echo(f"Run: {run_id}")
    typer.echo(f"Status: {final['status'].value}")
    typer.echo(f"Cycles: {final['budget'].cycles_used}/{final['budget'].maximum_cycles}")
    typer.echo(f"Primary score: {evaluation.primary_score if evaluation else 'n/a'}")
    typer.echo(
        "Evidence: "
        f"{verification.evidence_status.value if verification else 'n/a'} "
        f"({verification.provenance.value if verification else 'n/a'})"
    )
    typer.echo(f"Stop reason: {final.get('stop_reason') or 'none'}")
    typer.echo(f"Event IDs: {', '.join(final['decision_event_ids']) or 'none'}")
    if final["status"] == RunStatus.FAILED:
        raise typer.Exit(code=1)


@app.command("tasks")
def list_tasks() -> None:
    """List registered task descriptors and readiness without importing v2 eagerly."""
    registry = default_task_registry()
    for task in registry.list_tasks():
        descriptor = task.descriptor()
        readiness = task.readiness(TaskRuntimeContext())
        searches = ",".join(sorted(item.value for item in descriptor.supported_search_types))
        typer.echo(
            f"{descriptor.task_id}\t{descriptor.task_version}\t"
            f"{descriptor.display_name}\tready={str(readiness.ready).lower()}\t"
            f"search={searches}"
        )


@app.command()
def provenance(
    run_id: str = typer.Option(..., "--run-id"),
    provenance_db: Path = typer.Option(
        DEFAULT_DATA_DIR / "provenance.sqlite",
        "--provenance-db",
    ),
) -> None:
    """Print ordered append-only provenance events for a run."""
    store = SQLiteProvenanceStore(provenance_db)
    try:
        events = store.list_events(run_id)
    finally:
        store.close()
    if not events:
        typer.echo(f"No provenance events found for run {run_id!r}.")
        raise typer.Exit(code=1)
    for event in events:
        typer.echo(
            f"{event.timestamp.isoformat()} "
            f"{event.event_id} {event.event_type.value} "
            f"actor={event.actor} provenance={event.provenance.value}"
        )


if __name__ == "__main__":
    app()
