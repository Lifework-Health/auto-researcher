from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import yaml
from typer.testing import CliRunner

from auto_researcher.cli import app
from auto_researcher.contracts.enums import KnowledgeRetrievalStatus
from auto_researcher.knowledge.store import SQLiteKnowledgeRetrievalStore
from tests.helpers_read_safety import operator_configuration
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


def test_attestation_cli_reports_only_safe_identity_and_risk(tmp_path):
    attestation = operator_configuration().read_safety_attestation
    assert attestation is not None
    path = tmp_path / "attestation.yaml"
    path.write_text(
        yaml.safe_dump(attestation.model_dump(mode="json"), sort_keys=True),
        encoding="utf-8",
    )
    runner = CliRunner()
    validated = runner.invoke(
        app,
        ["knowledge", "attestation", "validate", "--file", str(path)],
    )
    assert validated.exit_code == 0, validated.stdout
    assert "Valid: true" in validated.stdout
    assert "PROFESSIONAL" in validated.stdout
    assert "DATABASE_CREDENTIAL_NOT_ENFORCED_READ_ONLY" in validated.stdout
    assert "canonical-json-sha256-v1" in validated.stdout
    assert "password" not in validated.stdout.casefold()
    assert "neo4j+s://" not in validated.stdout
    assert "@" not in validated.stdout.split("Templates:", 1)[0]

    tampered = attestation.model_copy(update={"attestation_hash": "1" * 64})
    path.write_text(
        yaml.safe_dump(tampered.model_dump(mode="json"), sort_keys=True),
        encoding="utf-8",
    )
    rejected = runner.invoke(
        app,
        ["knowledge", "attestation", "inspect", "--file", str(path)],
    )
    assert rejected.exit_code == 1
    assert "ATTESTATION_HASH_MISMATCH" in rejected.stdout


def test_attestation_cli_is_cross_process_and_yaml_order_stable(tmp_path):
    attestation = operator_configuration().read_safety_attestation
    assert attestation is not None
    payload = attestation.model_dump(mode="json")
    payload = dict(reversed(tuple(payload.items())))
    for field in (
        "evidence_references",
        "permitted_query_template_ids",
        "prohibited_capabilities",
    ):
        payload[field] = list(reversed(payload[field]))
    path = tmp_path / "reordered-attestation.yaml"
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True).rstrip(),
        encoding="utf-8",
    )
    executable = Path(sys.executable).parent / "auto-researcher"
    outputs = []
    for seed, action in (("2", "validate"), ("999", "inspect")):
        completed = subprocess.run(
            [
                str(executable),
                "knowledge",
                "attestation",
                action,
                "--file",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        outputs.append(completed.stdout)

    for output in outputs:
        assert "Valid: true" in output
        assert f"Attestation hash: {attestation.attestation_hash}" in output
        assert f"Configuration hash: {attestation.configuration_hash}" in output


def test_attestation_cli_rejects_legacy_duplicate_and_ambiguous_yaml(tmp_path):
    attestation = operator_configuration().read_safety_attestation
    assert attestation is not None
    runner = CliRunner()

    legacy = attestation.model_dump(mode="json")
    legacy.pop("attestation_hash_algorithm")
    legacy.pop("configuration_hash_algorithm")
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump(legacy), encoding="utf-8")
    result = runner.invoke(
        app,
        ["knowledge", "attestation", "validate", "--file", str(path)],
    )
    assert result.exit_code == 1
    assert "LEGACY_ATTESTATION_REGENERATION_REQUIRED" in result.stdout

    duplicate = attestation.model_dump(mode="json")
    duplicate["permitted_query_template_ids"] = [
        "generic.schema_preflight@1.0.0",
        "generic.schema_preflight@1.0.0",
    ]
    path.write_text(yaml.safe_dump(duplicate), encoding="utf-8")
    result = runner.invoke(
        app,
        ["knowledge", "attestation", "validate", "--file", str(path)],
    )
    assert result.exit_code == 1
    assert "ATTESTATION_DUPLICATE_UNORDERED_VALUE" in result.stdout

    valid_yaml = yaml.safe_dump(attestation.model_dump(mode="json"))
    path.write_text(
        valid_yaml + "\nattestation_id: duplicate-id\n",
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        ["knowledge", "attestation", "validate", "--file", str(path)],
    )
    assert result.exit_code == 1
    assert "ATTESTATION_CANONICALIZATION_FAILED" in result.stdout
