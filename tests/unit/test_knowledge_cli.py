from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from auto_researcher.cli import app
from auto_researcher.contracts.enums import KnowledgeRetrievalStatus
from auto_researcher.knowledge.store import SQLiteKnowledgeRetrievalStore
from tests.unit.test_knowledge_replay import _runtime


def test_knowledge_cli_lists_readiness_shows_and_retries_safely(tmp_path):
    repository = Path(__file__).resolve().parents[2]
    runner = CliRunner()
    providers = runner.invoke(app, ["knowledge", "providers"])
    assert providers.exit_code == 0
    assert "static" in providers.stdout
    assert "neo4j" in providers.stdout

    readiness = runner.invoke(
        app,
        [
            "knowledge",
            "readiness",
            "--task",
            "synthetic",
            "--contract",
            str(repository / "examples/knowledge/synthetic-contract.yaml"),
            "--task-config",
            str(repository / "examples/knowledge/synthetic-static.yaml"),
        ],
    )
    assert readiness.exit_code == 0
    assert "Ready: true" in readiness.stdout

    path = tmp_path / "knowledge.sqlite"
    store = SQLiteKnowledgeRetrievalStore(path)
    coordinator, _, provider, configuration, request, policy = _runtime(
        tmp_path,
        store=store,
    )
    coordinator.run(request, provider, configuration, policy)
    ambiguous_request = request.model_copy(
        update={"retrieval_id": "ambiguous-retrieval"}
    )
    completed = store.latest(request.retrieval_id)
    assert completed is not None
    reserved = completed.model_copy(
        update={
            "record_id": "ambiguous:1:reserved",
            "retrieval_id": ambiguous_request.retrieval_id,
            "status": KnowledgeRetrievalStatus.RESERVED,
            "request": ambiguous_request,
            "bundle": None,
            "errors": (),
            "provider_request_started": True,
        }
    )
    store.append(reserved)
    store.append(
        reserved.model_copy(
            update={
                "record_id": "ambiguous:2:indeterminate",
                "status": KnowledgeRetrievalStatus.INDETERMINATE,
            }
        )
    )
    store.close()

    listed = runner.invoke(
        app,
        [
            "knowledge",
            "retrievals",
            "list",
            "--run-id",
            request.run_id,
            "--knowledge-retrievals-db",
            str(path),
        ],
    )
    assert listed.exit_code == 0
    assert "COMPLETED" in listed.stdout
    assert "INDETERMINATE" in listed.stdout
    shown = runner.invoke(
        app,
        [
            "knowledge",
            "retrievals",
            "show",
            "--retrieval-id",
            request.retrieval_id,
            "--knowledge-retrievals-db",
            str(path),
        ],
    )
    assert shown.exit_code == 0
    assert "query_plan_hash" in shown.stdout
    assert "reference_ids" in shown.stdout
    assert "password" not in shown.stdout.casefold()
    assert "bolt://" not in shown.stdout
    assert "MATCH (" not in shown.stdout
    retried = runner.invoke(
        app,
        [
            "knowledge",
            "retrievals",
            "retry",
            "--retrieval-id",
            ambiguous_request.retrieval_id,
            "--knowledge-retrievals-db",
            str(path),
        ],
    )
    assert retried.exit_code == 0
    assert "linked to ambiguous-retrieval" in retried.stdout
