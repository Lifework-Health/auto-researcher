"""Offline command-line entry points for PR 1."""

from __future__ import annotations

from pathlib import Path

import typer

from auto_researcher.contracts.enums import ProvenanceKind, RunStatus, SearchType
from auto_researcher.contracts.models import ResearchContract
from auto_researcher.graph.builder import build_graph
from auto_researcher.provenance.sqlite_store import SQLiteProvenanceStore
from auto_researcher.runtime.dependencies import sqlite_dependencies

app = typer.Typer(no_args_is_help=True, help="Run the Auto Researcher v2.1 control plane.")
DEFAULT_DATA_DIR = Path(".auto-researcher")


def _mock_contract(max_cycles: int) -> ResearchContract:
    return ResearchContract(
        contract_id="offline-demo-contract",
        schema_version="1.0",
        objective_version="1",
        question="Can a bounded direct configuration improve the mock objective?",
        objective="the deterministic offline primary score",
        constraints={
            "model_depth": {"minimum": 1, "maximum": 8},
            "learning_rate": {"exclusive_minimum": 0.0, "maximum": 1.0},
            "regularization": {"minimum": 0.0},
        },
        allowed_search_types=frozenset({SearchType.DIRECT}),
        evaluator_id="mock-evaluator",
        verifier_id="deterministic-verifier",
        maximum_cycles=max_cycles,
        maximum_experiments=max_cycles,
        maximum_cost=1.0,
        requires_approval_for=frozenset(),
        provenance=ProvenanceKind.MOCK,
    )


@app.command()
def run(
    mock: bool = typer.Option(False, "--mock", help="Use deterministic offline components."),
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
    """Execute one offline research run and print its verified final state."""
    if not mock:
        raise typer.BadParameter("PR 1 supports only --mock; live agents are not implemented")
    contract = _mock_contract(max_cycles)
    config = {"configurable": {"thread_id": thread_id}}
    with sqlite_dependencies(checkpoint_db, provenance_db) as dependencies:
        graph = build_graph(dependencies)
        final = graph.invoke(
            {"run_id": run_id, "thread_id": thread_id, "contract": contract},
            config,
        )
        verification = final.get("verification_result")
        evaluation = final.get("evaluation_result")
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
