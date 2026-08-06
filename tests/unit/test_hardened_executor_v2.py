from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from auto_researcher.runtime.checkpoints import (
    checkpoint_serializer,
    sqlite_checkpointer,
)
from auto_researcher.runtime.execution import (
    RunExecutionError,
    inspect_terminal_run,
    resume_run,
    start_run,
)
from auto_researcher.runtime.identity import payload_hash
from auto_researcher.search.openevolve.hardened_executor import (
    HardenedDockerExecutor,
    docker_policy,
)
from auto_researcher.search.openevolve.models import (
    CandidateExecutionStatus,
    CandidatePreparationResult,
    CandidateValidationStatus,
)
from auto_researcher.search.openevolve.upstream_models import ExecutorIsolationResult
from tests.unit.test_checkpoint_serialization import (
    _terminal_fixture,
    _write_checkpoint,
)
from tests.unit.test_openevolve_contracts import (
    _backend,
    _candidate,
    _contract,
    _request,
)

ROOT = Path(__file__).parents[2]
DIGEST = "sha256:" + "a" * 64


def _policy():
    return docker_policy(
        "executor:test",
        DIGEST,
        ROOT / "docker/openevolve-executor/Dockerfile",
        ROOT / "docker/openevolve-executor/worker.py",
        "test-runtime",
    )


class IsolatedExecutor(HardenedDockerExecutor):
    def verify_isolation(self):
        return ExecutorIsolationResult(
            executor_policy_hash=payload_hash(self.policy),
            network_isolation_verified=True,
            mount_isolation_verified=True,
            environment_sanitisation_verified=True,
            safe_checks={},
        )


def _sandbox_policy():
    backend = _backend()
    return backend.create_search_contract(_request(), _contract()).sandbox_policy


@pytest.mark.parametrize("stdout", [b"{}\n", b"{}\n{}\n", b"not-json\n"])
def test_malformed_or_duplicate_supervisor_envelope_fails_closed(
    monkeypatch, tmp_path, stdout
):
    executor = IsolatedExecutor(_policy(), tmp_path)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=[], returncode=0, stdout=stdout, stderr=b""
        ),
    )
    backend = _backend()
    result = executor.prepare(
        _candidate(backend.component_spec.seed_source),
        backend.component_spec,
        _sandbox_policy(),
        backend.component.seed_configuration(),
    )
    assert result.execution_status == CandidateExecutionStatus.FAILED
    assert result.safe_error_code == "hardened_executor_worker_protocol_invalid"
    assert list(tmp_path.iterdir()) == []


def test_unsupported_inode_limit_fails_before_candidate_container(tmp_path):
    class UnsupportedExecutor(HardenedDockerExecutor):
        def verify_isolation(self):
            return ExecutorIsolationResult(
                executor_policy_hash=payload_hash(self.policy),
                network_isolation_verified=True,
                mount_isolation_verified=False,
                environment_sanitisation_verified=True,
                safe_checks={},
                safe_error_code="hardened_executor_file_count_limit_unsupported",
            )

    backend = _backend()
    result = UnsupportedExecutor(_policy(), tmp_path).prepare(
        _candidate(backend.component_spec.seed_source),
        backend.component_spec,
        _sandbox_policy(),
        backend.component.seed_configuration(),
    )
    assert result.execution_status == CandidateExecutionStatus.FAILED
    assert result.safe_error_code == "hardened_executor_file_count_limit_unsupported"
    assert list(tmp_path.iterdir()) == []


def test_unavailable_runtime_fails_without_fallback(monkeypatch):
    executor = HardenedDockerExecutor(_policy())
    monkeypatch.setattr(
        "auto_researcher.search.openevolve.hardened_executor.shutil.which",
        lambda name: None,
    )
    with pytest.raises(ValueError, match="hardened_executor_unavailable"):
        executor._inspect()


def test_runtime_version_drift_fails_closed(monkeypatch):
    executor = HardenedDockerExecutor(_policy())
    monkeypatch.setattr(
        "auto_researcher.search.openevolve.hardened_executor.shutil.which",
        lambda name: "/safe/docker",
    )
    results = iter(
        (
            subprocess.CompletedProcess([], 0, stdout="different-runtime\n", stderr=""),
            subprocess.CompletedProcess(
                [],
                0,
                stdout=(
                    '{"Id":"sha256:'
                    + "a" * 64
                    + '","Architecture":"arm64","Os":"linux",'
                    '"Config":{"Entrypoint":["python","/opt/runner/worker.py"],'
                    '"User":"65532:65532"}}\n'
                ),
                stderr="",
            ),
        )
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: next(results))
    with pytest.raises(ValueError, match="hardened_executor_runtime_mismatch"):
        executor._inspect()


def test_executor_v1_preparation_is_not_reused_as_v2():
    executor = HardenedDockerExecutor(_policy())
    v1 = CandidatePreparationResult(
        candidate_id="candidate-test",
        validation_status=CandidateValidationStatus.VALID,
        execution_status=CandidateExecutionStatus.COMPLETED,
        cleanup_complete=True,
    )
    assert executor.accepts_preparation(v1, _sandbox_policy()) is False


def test_executor_v2_preparation_evidence_reconstructs_and_tamper_fails():
    executor = HardenedDockerExecutor(_policy())
    sandbox = _sandbox_policy()
    preparation = CandidatePreparationResult(
        protocol_version="candidate-preparation-v2",
        candidate_id="candidate-test",
        validation_status=CandidateValidationStatus.VALID,
        execution_status=CandidateExecutionStatus.COMPLETED,
        cleanup_complete=True,
        executor_id=executor.runner_id,
        executor_policy_identity=payload_hash(executor.policy),
        execution_request_identity="1" * 64,
        workspace_policy_identity="2" * 64,
        worker_protocol_version="openevolve-hardened-worker-result-v2",
        supervisor_identity=executor.policy.entrypoint_hash,
        image_digest=executor.policy.image_digest,
        declared_file_count_limit=sandbox.file_count_limit,
        derived_inode_limit=sandbox.file_count_limit + 1,
        workspace_bytes_limit=sandbox.workspace_bytes,
        file_size_bytes_limit=sandbox.file_size_bytes,
    )
    serializer = checkpoint_serializer()
    restored = serializer.loads_typed(serializer.dumps_typed(preparation))
    assert type(restored) is CandidatePreparationResult
    assert executor.accepts_preparation(restored, sandbox) is True
    assert (
        executor.accepts_preparation(
            restored.model_copy(update={"declared_file_count_limit": 9}), sandbox
        )
        is False
    )
    assert (
        executor.accepts_preparation(
            restored.model_copy(update={"worker_protocol_version": None}), sandbox
        )
        is False
    )


def test_terminal_v2_preparation_inspection_is_read_only_and_never_constructs_docker(
    monkeypatch, tmp_path
):
    executor = HardenedDockerExecutor(_policy())
    sandbox = _sandbox_policy()
    preparation = CandidatePreparationResult(
        protocol_version="candidate-preparation-v2",
        candidate_id="candidate-terminal-v2",
        validation_status=CandidateValidationStatus.VALID,
        execution_status=CandidateExecutionStatus.COMPLETED,
        cleanup_complete=True,
        executor_id=executor.runner_id,
        executor_policy_identity=payload_hash(executor.policy),
        execution_request_identity="1" * 64,
        workspace_policy_identity="2" * 64,
        worker_protocol_version="openevolve-hardened-worker-result-v2",
        supervisor_identity=executor.policy.entrypoint_hash,
        image_digest=executor.policy.image_digest,
        declared_file_count_limit=sandbox.file_count_limit,
        derived_inode_limit=sandbox.file_count_limit + 1,
        workspace_bytes_limit=sandbox.workspace_bytes,
        file_size_bytes_limit=sandbox.file_size_bytes,
    )
    state, initial, config = _terminal_fixture()
    state["openevolve_preparation_result"] = preparation
    checkpoint = tmp_path / "terminal-v2.sqlite"
    _write_checkpoint(checkpoint, state)
    before_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    before_mtime = checkpoint.stat().st_mtime_ns
    saver, connection = sqlite_checkpointer(checkpoint)

    class View:
        def get_state(self, runtime_config):
            item = saver.get_tuple(runtime_config)
            values = item.checkpoint["channel_values"] if item else {}
            return SimpleNamespace(values=values)

        def invoke(self, *args, **kwargs):
            raise AssertionError("terminal guard allowed graph execution")

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("INSPECT attempted Docker or another subprocess")
        ),
    )
    view = View()
    first = inspect_terminal_run(view, config)
    second = inspect_terminal_run(view, config)
    assert payload_hash(first) == payload_hash(second) == payload_hash(state)
    assert type(first["openevolve_preparation_result"]) is CandidatePreparationResult
    assert first["openevolve_preparation_result"].protocol_version == (
        "candidate-preparation-v2"
    )
    with pytest.raises(
        RunExecutionError, match="thread_already_exists_use_resume_or_inspect"
    ):
        start_run(view, initial, config)
    with pytest.raises(RunExecutionError, match="thread_is_terminal_use_inspect"):
        resume_run(view, config)
    connection.close()
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == before_hash
    assert checkpoint.stat().st_mtime_ns == before_mtime


def test_executor_policy_identity_is_cross_process_deterministic():
    script = """
from pathlib import Path
from auto_researcher.runtime.identity import payload_hash
from auto_researcher.search.openevolve.hardened_executor import docker_policy
root=Path.cwd()
policy=docker_policy('executor:test','sha256:'+'a'*64,root/'docker/openevolve-executor/Dockerfile',root/'docker/openevolve-executor/worker.py','test-runtime')
print(payload_hash(policy))
"""
    outputs = set()
    for seed in ("1", "777", "98765"):
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.add(completed.stdout.strip())
    assert len(outputs) == 1
