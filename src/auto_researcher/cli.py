"""Generic task-oriented command line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
import yaml

from auto_researcher.contracts.enums import RunStatus, SearchType
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
    search = payload.get("search")
    runtime = payload.get("runtime", {})
    if (experiment is None) == (search is None):
        raise ValueError("task config requires exactly one of experiment or search")
    if not isinstance(runtime, dict):
        raise ValueError("task config runtime section must be a mapping")
    selected = experiment if experiment is not None else search
    if not isinstance(selected, dict):
        raise ValueError("task config experiment/search section must be a mapping")
    selected = dict(selected)
    if search is not None:
        kind = selected.pop("type", None)
        if kind != SearchType.OPTUNA.value:
            raise ValueError("PR 3 search section type must be OPTUNA")
    return selected, runtime


def _configured_search_type(path: Path | None) -> SearchType:
    if path is None:
        return SearchType.DIRECT
    payload = _load_yaml(path)
    return SearchType.OPTUNA if "search" in payload else SearchType.DIRECT


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
    optuna_db: Path = typer.Option(
        DEFAULT_DATA_DIR / "optuna.sqlite",
        "--optuna-db",
    ),
) -> None:
    """Run a registered task through the unchanged LangGraph control plane."""
    if mock:
        task_id = "synthetic"
    registry = default_task_registry()
    try:
        search_type = _configured_search_type(task_config)
        raw_config = _load_yaml(task_config) if task_config else {}
        requested_budget = int(
            raw_config.get("search", {}).get("trial_budget", max_cycles)
        )
        contract = (
            ResearchContract.model_validate(_load_yaml(contract_path))
            if contract_path
            else default_synthetic_contract(
                max_cycles,
                search_types=frozenset({search_type}),
                maximum_experiments=requested_budget,
            )
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
            optuna_db if search_type == SearchType.OPTUNA else None,
            search_type=search_type,
        ) as dependencies:
            graph = build_graph(dependencies)
            final = graph.invoke(
                {"run_id": run_id, "thread_id": thread_id, "contract": contract},
                config,
            )
    except (TaskPluginError, ValueError, FileNotFoundError, RuntimeError) as exc:
        typer.echo(f"Task setup failed: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    verification = final.get("verification_result")
    evaluation = final.get("evaluation_result")
    typer.echo(f"Task: {contract.task_id}@{contract.task_version}")
    typer.echo(f"Search type: {search_type.value}")
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
    study = final.get("optuna_study_result")
    if study is not None:
        typer.echo(f"Study: {study.study_name}")
        typer.echo(f"Direction: {study.direction.value}")
        typer.echo(f"Trial budget: {study.trial_budget}")
        typer.echo(f"Trials asked: {study.trials_asked}")
        typer.echo(f"Trials completed: {study.trials_completed}")
        typer.echo(f"Trials failed: {study.trials_failed}")
        typer.echo(f"Best feasible score: {study.best_feasible_score}")
        typer.echo(f"Best feasible trial: {study.best_feasible_trial_number}")
        typer.echo(f"Best overall diagnostic score: {study.best_overall_score}")
        typer.echo(f"Study finish reason: {study.finish_reason}")
        typer.echo(
            "Study artefacts: "
            f"{', '.join(study.artefact_references) or 'none'}"
        )
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
