from __future__ import annotations

import hashlib
import os
import shutil
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
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
    WORKSPACE_ROOT_UNAVAILABLE,
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


def _remove_operation(path: Path) -> None:
    shutil.rmtree(path)


def test_workspace_root_none_uses_system_temporary_directory():
    executor = HardenedDockerExecutor(_policy())
    operation = executor._operation_directory("openevolve-none-")
    try:
        assert operation.is_dir()
        assert executor.workspace_root_identity is None
    finally:
        _remove_operation(operation)


def test_existing_and_missing_workspace_roots_are_stable_and_retained(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    missing = tmp_path / "missing" / "nested"
    for root, created_by_executor in ((existing, False), (missing, True)):
        executor = HardenedDockerExecutor(_policy(), root)
        identity = executor.workspace_root_identity
        first = executor._operation_directory("openevolve-first-")
        second = executor._operation_directory("openevolve-second-")
        assert first.parent == root
        assert second.parent == root
        assert first != second
        assert executor.workspace_root_identity == identity
        _remove_operation(first)
        _remove_operation(second)
        assert root.is_dir()
        assert list(root.iterdir()) == []
        if created_by_executor:
            assert stat.S_IMODE(root.stat().st_mode) & 0o077 == 0


@pytest.mark.parametrize("kind", ["file", "symlink"])
def test_workspace_root_rejects_non_directory_final_component(tmp_path, kind):
    root = tmp_path / "invalid-root"
    if kind == "file":
        root.write_text("not a directory", encoding="utf-8")
    else:
        target = tmp_path / "target"
        target.mkdir()
        root.symlink_to(target, target_is_directory=True)
    executor = HardenedDockerExecutor(_policy(), root)
    with pytest.raises(ValueError, match=WORKSPACE_ROOT_UNAVAILABLE):
        executor._operation_directory("openevolve-invalid-")


@pytest.mark.parametrize("kind", ["file", "symlink"])
def test_prepare_returns_safe_failure_for_invalid_workspace_root(tmp_path, kind):
    root = tmp_path / "invalid-prepare-root"
    if kind == "file":
        root.write_text("not a directory", encoding="utf-8")
    else:
        target = tmp_path / "target"
        target.mkdir()
        root.symlink_to(target, target_is_directory=True)
    backend = _backend()
    result = IsolatedExecutor(_policy(), root).prepare(
        _candidate(backend.component_spec.seed_source),
        backend.component_spec,
        _sandbox_policy(),
        backend.component.seed_configuration(),
    )
    assert result.execution_status == CandidateExecutionStatus.FAILED
    assert result.safe_error_code == WORKSPACE_ROOT_UNAVAILABLE
    assert result.cleanup_complete is True


def test_workspace_root_rejects_insufficient_access(monkeypatch, tmp_path):
    root = tmp_path / "restricted"
    root.mkdir()
    original = os.access
    monkeypatch.setattr(
        os,
        "access",
        lambda path, mode: False if Path(path) == root else original(path, mode),
    )
    with pytest.raises(ValueError, match=WORKSPACE_ROOT_UNAVAILABLE):
        HardenedDockerExecutor(_policy(), root)._operation_directory(
            "openevolve-restricted-"
        )


def test_workspace_root_removed_before_child_creation_fails_safely(
    monkeypatch, tmp_path
):
    root = tmp_path / "removed"
    executor = HardenedDockerExecutor(_policy(), root)
    ensure = executor._ensure_workspace_root

    def remove_after_validation():
        value = ensure()
        assert value is not None
        value.rmdir()
        return value

    monkeypatch.setattr(executor, "_ensure_workspace_root", remove_after_validation)
    with pytest.raises(ValueError, match=WORKSPACE_ROOT_UNAVAILABLE):
        executor._operation_directory("openevolve-removed-")


def test_prepare_maps_operation_child_failure_to_workspace_code(monkeypatch, tmp_path):
    root = tmp_path / "operation-child-failure"
    backend = _backend()
    executor = IsolatedExecutor(_policy(), root)
    monkeypatch.setattr(
        "auto_researcher.search.openevolve.hardened_executor.tempfile.mkdtemp",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError()),
    )
    result = executor.prepare(
        _candidate(backend.component_spec.seed_source),
        backend.component_spec,
        _sandbox_policy(),
        backend.component.seed_configuration(),
    )
    assert result.execution_status == CandidateExecutionStatus.FAILED
    assert result.safe_error_code == WORKSPACE_ROOT_UNAVAILABLE
    assert result.cleanup_complete is True
    assert root.is_dir()
    assert list(root.iterdir()) == []


def test_workspace_root_supports_concurrent_operations(tmp_path):
    root = tmp_path / "shared" / "executor"
    executor = HardenedDockerExecutor(_policy(), root)
    with ThreadPoolExecutor(max_workers=4) as pool:
        operations = tuple(
            pool.map(
                lambda _: executor._operation_directory("openevolve-shared-"), range(8)
            )
        )
    assert len(set(operations)) == 8
    assert all(path.parent == root and path.is_dir() for path in operations)
    for operation in operations:
        _remove_operation(operation)
    assert root.is_dir()
    assert list(root.iterdir()) == []


def test_workspace_root_identity_survives_empty_parent_recreation(tmp_path):
    root = tmp_path / "resume" / "executor"
    first = HardenedDockerExecutor(_policy(), root)
    operation = first._operation_directory("openevolve-before-resume-")
    _remove_operation(operation)
    identity = first.workspace_root_identity
    root.rmdir()

    reconstructed = HardenedDockerExecutor(_policy(), root)
    resumed_operation = reconstructed._operation_directory("openevolve-resumed-")
    assert reconstructed.workspace_root_identity == identity
    assert resumed_operation.parent == root
    _remove_operation(resumed_operation)
    assert root.is_dir()
    assert list(root.iterdir()) == []


def test_verify_isolation_materialises_missing_root_and_cleans_child(
    monkeypatch, tmp_path
):
    root = tmp_path / "missing" / "isolation"
    executor = HardenedDockerExecutor(_policy(), root)
    monkeypatch.setattr(executor, "_inspect", lambda: None)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=[], returncode=0, stdout=b"{}\n", stderr=b""
        ),
    )
    result = executor.verify_isolation()
    assert result.safe_error_code == "hardened_executor_file_count_limit_unsupported"
    assert root.is_dir()
    assert list(root.iterdir()) == []


def test_prepare_maps_failed_input_directory_creation_to_workspace_code(
    monkeypatch, tmp_path
):
    root = tmp_path / "missing" / "prepare"
    backend = _backend()
    executor = IsolatedExecutor(_policy(), root)
    original = Path.mkdir

    def fail_input(path, *args, **kwargs):
        if path.name == "input" and path.parent.parent == root:
            raise PermissionError
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fail_input)
    result = executor.prepare(
        _candidate(backend.component_spec.seed_source),
        backend.component_spec,
        _sandbox_policy(),
        backend.component.seed_configuration(),
    )
    assert result.execution_status == CandidateExecutionStatus.FAILED
    assert result.safe_error_code == WORKSPACE_ROOT_UNAVAILABLE
    assert result.cleanup_complete is True
    assert root.is_dir()
    assert list(root.iterdir()) == []


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
