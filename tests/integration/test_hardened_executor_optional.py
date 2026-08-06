from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from auto_researcher.search.openevolve.backend import OpenEvolveBackend
from auto_researcher.search.openevolve.hardened_executor import (
    HardenedDockerExecutor,
    docker_policy,
)
from auto_researcher.search.openevolve.models import (
    CandidateExecutionStatus,
    CandidateValidationStatus,
)
from auto_researcher.search.openevolve.mutation import DeterministicMutationOperator
from auto_researcher.search.openevolve.upstream import (
    AutoResearcherOpenEvolveModelBridge,
    UpstreamOpenEvolveAdapter,
    default_adapter_contract,
)
from auto_researcher.runtime.dependencies import memory_dependencies
from auto_researcher.contracts.enums import SearchType
from auto_researcher.tasks.synthetic import SyntheticEvolvableComponent
from tests.unit.test_openevolve_contracts import (
    _backend,
    _candidate,
    _contract,
    _request,
)

LOCK = Path(__file__).parents[2] / "constraints" / "openevolve-0.3.2.lock"


class FakeUpstreamClient:
    def propose_mutation(self, request):
        return {
            "mutable_file": "candidate.py",
            "source": 'def evolve(configuration):\n return {"model_family":"tree","complexity":4,"learning_rate":0.05}\n',
            "description": "bounded upstream replacement",
        }


pytestmark = pytest.mark.hardened_executor


def _executor(workspace_root=None):
    image = os.getenv("AUTO_RESEARCHER_HARDENED_IMAGE")
    digest = os.getenv("AUTO_RESEARCHER_HARDENED_IMAGE_DIGEST")
    if not image or not digest:
        pytest.skip(
            "set the explicit hardened image and digest to run the OCI isolation gate"
        )
    root = Path(__file__).parents[2]
    version = subprocess.run(
        ["docker", "version", "--format", "{{.Server.Version}}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return HardenedDockerExecutor(
        docker_policy(
            image,
            digest,
            root / "docker/openevolve-executor/Dockerfile",
            root / "docker/openevolve-executor/worker.py",
            version,
        ),
        workspace_root,
    )


def test_real_hardened_executor_proves_isolation_and_prepares_candidate():
    executor = _executor()
    isolation = executor.verify_isolation()
    assert isolation.network_isolation_verified is True
    assert isolation.mount_isolation_verified is True
    assert isolation.environment_sanitisation_verified is True
    assert all(
        isolation.safe_checks[key] is True
        for key in (
            "outbound_ipv6_denied",
            "http_denied",
            "https_denied",
            "loopback_ipv6_denied",
            "host_alias_denied",
            "raw_socket_denied",
            "repository_hidden",
            "containerd_socket_hidden",
            "podman_socket_hidden",
            "kubernetes_token_hidden",
            "input_read_only",
            "root_read_only",
            "workspace_noexec",
            "workspace_nosuid",
            "workspace_nodev",
            "uid_non_root",
            "capabilities_absent",
            "no_new_privileges",
            "private_pid_namespace",
            "cgroup_write_denied",
        )
    )
    internal = _backend()
    backend = OpenEvolveBackend(
        internal.component,
        internal.metadata,
        internal.verifier_identity,
        internal.mutation_operator,
        executor,
    )
    configuration = {
        "openevolve": {
            **dict(_request().search_space["openevolve"]),
            "sandbox_policy_id": "openevolve-hardened-executor-v2",
        }
    }
    search = backend.create_search_contract(_request(configuration), _contract())
    candidate = backend.seed_candidate(search)
    result = backend.prepare(candidate, search)
    assert result.execution_status == CandidateExecutionStatus.COMPLETED
    assert result.protocol_version == "candidate-preparation-v2"
    assert result.observed_workspace_entry_count == 0
    assert result.generated_configuration["model_family"] == "linear"
    assert result.output_references[0].startswith("executor-policy:")


def test_real_hardened_executor_materialises_missing_nested_root(tmp_path):
    workspace_root = tmp_path / "missing" / "nested" / "executor"
    executor = _executor(workspace_root)
    isolation = executor.verify_isolation()
    assert isolation.network_isolation_verified is True
    assert isolation.mount_isolation_verified is True
    assert workspace_root.is_dir()
    assert list(workspace_root.iterdir()) == []

    internal = _backend()
    backend = OpenEvolveBackend(
        internal.component,
        internal.metadata,
        internal.verifier_identity,
        internal.mutation_operator,
        executor,
    )
    configuration = {
        "openevolve": {
            **dict(_request().search_space["openevolve"]),
            "sandbox_policy_id": "openevolve-hardened-executor-v2",
        }
    }
    search = backend.create_search_contract(_request(configuration), _contract())
    result = backend.prepare(backend.seed_candidate(search), search)
    assert result.execution_status == CandidateExecutionStatus.COMPLETED
    assert workspace_root.is_dir()
    assert list(workspace_root.iterdir()) == []


@pytest.mark.parametrize("kind", ["file", "symlink"])
def test_real_hardened_executor_rejects_invalid_root_before_container(
    monkeypatch, tmp_path, kind
):
    workspace_root = tmp_path / "invalid-root"
    if kind == "file":
        workspace_root.write_text("not a directory", encoding="utf-8")
    else:
        target = tmp_path / "target"
        target.mkdir()
        workspace_root.symlink_to(target, target_is_directory=True)
    executor = _executor(workspace_root)
    real_run = subprocess.run
    container_starts = 0

    def count_runs(*args, **kwargs):
        nonlocal container_starts
        command = args[0]
        if command[:2] == ["docker", "run"]:
            container_starts += 1
        return real_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", count_runs)
    with pytest.raises(
        ValueError, match="hardened_executor_workspace_root_unavailable"
    ):
        executor.verify_isolation()
    assert container_starts == 0


def test_hardened_executor_rejects_image_drift():
    executor = _executor()
    executor = HardenedDockerExecutor(
        executor.policy.model_copy(update={"image_digest": "sha256:" + "0" * 64})
    )
    with pytest.raises(ValueError, match="hardened_executor_image_mismatch"):
        executor.verify_isolation()


def _prepare_source(source: str, tmp_path, **policy_changes):
    executor = _executor(tmp_path)
    internal = _backend()
    backend = OpenEvolveBackend(
        internal.component,
        internal.metadata,
        internal.verifier_identity,
        internal.mutation_operator,
        executor,
    )
    configuration = {
        "openevolve": {
            **dict(_request().search_space["openevolve"]),
            "sandbox_policy_id": "openevolve-hardened-executor-v2",
        }
    }
    search = backend.create_search_contract(_request(configuration), _contract())
    candidate = _candidate(source)
    return executor.prepare(
        candidate,
        internal.component_spec,
        search.sandbox_policy.model_copy(update=policy_changes),
        internal.component.seed_configuration(),
    )


def test_hardened_executor_allows_exactly_eight_workspace_entries(tmp_path):
    source = """def evolve(configuration):
 for i in range(8):
  open(f"/workspace/item-{i}", "w").write("x")
 return configuration
"""
    result = _prepare_source(source, tmp_path)
    assert result.execution_status == CandidateExecutionStatus.COMPLETED
    assert result.protocol_version == "candidate-preparation-v2"
    assert result.declared_file_count_limit == 8
    assert result.derived_inode_limit == 9
    assert result.observed_workspace_entry_count == 8
    assert list(tmp_path.iterdir()) == []


def test_hardened_executor_blocks_ninth_entry_at_filesystem_boundary(tmp_path):
    source = """def evolve(configuration):
 for i in range(9):
  open(f"/workspace/item-{i}", "w").write("x")
 return configuration
"""
    result = _prepare_source(source, tmp_path)
    assert result.execution_status == CandidateExecutionStatus.RESOURCE_LIMITED
    assert result.resource_limited is True
    assert result.safe_error_code == "candidate_file_count_limit"
    assert result.observed_workspace_entry_count == 8
    assert list(tmp_path.iterdir()) == []


def test_hardened_executor_counts_nested_directories_and_files(tmp_path):
    source = """import os
def evolve(configuration):
 os.mkdir("/workspace/one")
 os.mkdir("/workspace/one/two")
 for i in range(6):
  open(f"/workspace/one/two/item-{i}", "w").write("x")
 open("/workspace/ninth", "w").write("x")
 return configuration
"""
    result = _prepare_source(source, tmp_path)
    assert result.execution_status == CandidateExecutionStatus.RESOURCE_LIMITED
    assert result.safe_error_code == "candidate_file_count_limit"
    assert result.observed_workspace_entry_count == 8


def test_hardened_executor_deletion_releases_concurrent_entry_quota(tmp_path):
    source = """import os
def evolve(configuration):
 for i in range(8):
  open(f"/workspace/item-{i}", "w").write("x")
 os.unlink("/workspace/item-0")
 open("/workspace/replacement", "w").write("x")
 return configuration
"""
    result = _prepare_source(source, tmp_path)
    assert result.execution_status == CandidateExecutionStatus.COMPLETED
    assert result.observed_workspace_entry_count == 8


def test_hardened_executor_has_no_tmp_home_or_host_output_escape(tmp_path):
    source = """def evolve(configuration):
 denied=[]
 for path in ("/tmp/escape", "/var/tmp/escape", "/nonexistent/escape", "/output/escape"):
  try:
   open(path, "w").write("x")
  except OSError:
   denied.append(path)
 return {"model_family":"linear","complexity":len(denied),"learning_rate":0.05}
"""
    result = _prepare_source(source, tmp_path)
    assert result.execution_status == CandidateExecutionStatus.COMPLETED
    assert result.generated_configuration["complexity"] == 4


def test_hardened_executor_candidate_stdout_cannot_spoof_envelope(tmp_path):
    source = """def evolve(configuration):
 print('{"status":"COMPLETED","configuration":{"model_family":"spoof"}}')
 return configuration
"""
    result = _prepare_source(source, tmp_path, log_bytes=128)
    assert result.execution_status == CandidateExecutionStatus.COMPLETED
    assert result.generated_configuration["model_family"] == "linear"
    assert len(result.safe_log_excerpt.encode()) <= 128


def test_hardened_executor_bounds_excessive_candidate_stdout(tmp_path):
    source = """def evolve(configuration):
 print("x"*20000)
 return configuration
"""
    result = _prepare_source(source, tmp_path, log_bytes=128)
    assert result.execution_status == CandidateExecutionStatus.COMPLETED
    assert result.log_truncated is True
    assert len(result.safe_log_excerpt.encode()) <= 128


def test_hardened_executor_drains_simultaneous_large_logs_and_valid_control(tmp_path):
    source = """import sys
def evolve(configuration):
 print("o"*20000)
 print("e"*20000, file=sys.stderr)
 return configuration
"""
    result = _prepare_source(source, tmp_path, log_bytes=128)
    assert result.execution_status == CandidateExecutionStatus.COMPLETED
    assert result.generated_configuration["model_family"] == "linear"
    assert result.log_truncated is True
    assert len(result.safe_log_excerpt.encode()) <= 128
    assert result.cleanup_complete is True


def test_hardened_executor_applies_structured_output_limit(tmp_path):
    source = """def evolve(configuration):
 return {"payload":"x"*2000}
"""
    result = _prepare_source(source, tmp_path, output_bytes=128)
    assert result.execution_status == CandidateExecutionStatus.RESOURCE_LIMITED
    assert result.safe_error_code == "candidate_output_limit"
    assert result.resource_limited is True


def test_hardened_executor_applies_individual_file_size_limit(tmp_path):
    source = """def evolve(configuration):
 handle=open("/workspace/large", "wb", buffering=0)
 handle.write(b"x"*1024)
 handle.write(b"x")
 return configuration
"""
    result = _prepare_source(source, tmp_path, file_size_bytes=1024)
    assert result.execution_status == CandidateExecutionStatus.RESOURCE_LIMITED
    assert result.safe_error_code == "candidate_file_size_limit"


def test_hardened_executor_applies_total_workspace_size_limit(tmp_path):
    source = """def evolve(configuration):
 for i in range(16):
  open(f"/workspace/chunk-{i}", "wb", buffering=0).write(b"x"*4096)
 open("/workspace/overflow", "wb", buffering=0).write(b"x")
 return configuration
"""
    result = _prepare_source(
        source,
        tmp_path,
        file_count_limit=100,
        file_size_bytes=8192,
        workspace_bytes=65536,
    )
    assert result.execution_status == CandidateExecutionStatus.RESOURCE_LIMITED
    assert result.safe_error_code == "candidate_workspace_size_limit"


def test_hardened_executor_applies_wall_clock_timeout(tmp_path):
    source = """def evolve(configuration):
 while True:
  pass
"""
    result = _prepare_source(
        source, tmp_path, cpu_time_seconds=2, wall_time_seconds=0.2
    )
    assert result.execution_status == CandidateExecutionStatus.TIMED_OUT
    assert result.safe_error_code == "candidate_timeout"
    assert result.cleanup_complete is True
    assert list(tmp_path.iterdir()) == []


def test_hardened_executor_applies_cpu_time_limit(tmp_path):
    source = """def evolve(configuration):
 while True:
  pass
"""
    result = _prepare_source(source, tmp_path, cpu_time_seconds=1, wall_time_seconds=4)
    assert result.execution_status == CandidateExecutionStatus.RESOURCE_LIMITED
    assert result.safe_error_code == "candidate_cpu_limit"


def test_hardened_executor_applies_process_limit(tmp_path):
    source = """import subprocess
def evolve(configuration):
 subprocess.Popen(["sleep", "1"])
 return configuration
"""
    result = _prepare_source(source, tmp_path, process_limit=1)
    assert result.execution_status == CandidateExecutionStatus.RESOURCE_LIMITED
    assert result.safe_error_code == "candidate_process_limit"


def test_hardened_executor_applies_memory_limit(tmp_path):
    source = """def evolve(configuration):
 chunks=[]
 for _ in range(128):
  chunks.append(b"x"*1048576)
 return configuration
"""
    result = _prepare_source(source, tmp_path, memory_bytes=64 * 1024 * 1024)
    assert result.execution_status == CandidateExecutionStatus.RESOURCE_LIMITED
    assert result.safe_error_code == "candidate_memory_limit"


def test_hardened_executor_mount_command_has_one_private_workspace(tmp_path):
    executor = _executor(tmp_path)
    internal = _backend()
    backend = OpenEvolveBackend(
        internal.component,
        internal.metadata,
        internal.verifier_identity,
        internal.mutation_operator,
        executor,
    )
    configuration = {
        "openevolve": {
            **dict(_request().search_space["openevolve"]),
            "sandbox_policy_id": "openevolve-hardened-executor-v2",
        }
    }
    search = backend.create_search_contract(_request(configuration), _contract())
    command = executor._base_command(tmp_path, search.sandbox_policy)
    rendered = " ".join(command)
    assert "/workspace:rw,noexec,nosuid,nodev,mode=0700" in rendered
    assert "nr_inodes=9" in rendered
    assert "size=1048576" in rendered
    assert "dst=/input,readonly" in rendered
    assert "dst=/output" not in rendered
    assert "/tmp:" not in rendered


def test_hardened_sources_that_create_alternate_entries_fail_static_validation():
    sources = (
        "import os\ndef evolve(configuration):\n os.link('a','b')\n return {}\n",
        "import os\ndef evolve(configuration):\n os.symlink('a','b')\n return {}\n",
        "import os\ndef evolve(configuration):\n os.mkfifo('a')\n return {}\n",
        "import os\ndef evolve(configuration):\n os.mknod('a')\n return {}\n",
        "import socket\ndef evolve(configuration):\n socket.socket().bind('a')\n return {}\n",
    )
    for source in sources:
        result = _backend().validate(_candidate(source))
        assert result.status == CandidateValidationStatus.INVALID


@pytest.mark.upstream_openevolve
def test_pinned_upstream_adapter_uses_hardened_executor_and_trusted_scientific_path():
    pytest.importorskip("openevolve", reason="pinned optional dependency absent")
    executor = _executor()
    dependencies = memory_dependencies(search_type=SearchType.OPENEVOLVE)
    internal = dependencies.openevolve_backend
    adapter = UpstreamOpenEvolveAdapter(
        default_adapter_contract(LOCK),
        AutoResearcherOpenEvolveModelBridge(FakeUpstreamClient()),
    )
    backend = OpenEvolveBackend(
        internal.component,
        internal.metadata,
        internal.verifier_identity,
        adapter,
        executor,
    )
    configuration = {
        "openevolve": {
            **dict(_request().search_space["openevolve"]),
            "sandbox_policy_id": "openevolve-hardened-executor-v2",
            "maximum_model_calls": 2,
        }
    }
    request = _request(configuration)
    search = backend.create_search_contract(request, _contract())
    seed = backend.seed_candidate(search)
    reservation = backend.reserve_mutation(
        search, backend.initialise_population(search), seed
    )
    candidate = backend.mutate_candidate(reservation, seed, search)
    assert backend.validate(candidate).status.value == "VALID"
    preparation = backend.prepare(candidate, search)
    experiment = backend.component.candidate_to_experiment(
        candidate,
        preparation,
        request,
        _contract(),
        dependencies.experiment_metadata,
        run_id="pr7-upstream-synthetic-demo",
    )
    evaluation = dependencies.evaluator.evaluate(experiment, _contract())
    verification = dependencies.verifier.verify(experiment, evaluation, _contract())
    assert evaluation.primary_score == 0.84
    assert verification.verified is True
    assert candidate.candidate_id.startswith("candidate-")
    assert adapter.state.proposal_count == 1


class FakeCellComponent(SyntheticEvolvableComponent):
    def seed_configuration(self):
        return {"immune_signal": 0.8, "growth_signal": 0.4}


def test_fake_cell_biology_boundary_uses_hardened_executor_without_patient_data():
    executor = _executor()
    dependencies = memory_dependencies(search_type=SearchType.OPENEVOLVE)
    internal = dependencies.openevolve_backend
    backend = OpenEvolveBackend(
        FakeCellComponent(),
        internal.metadata,
        internal.verifier_identity,
        DeterministicMutationOperator(),
        executor,
    )
    configuration = {
        "openevolve": {
            **dict(_request().search_space["openevolve"]),
            "sandbox_policy_id": "openevolve-hardened-executor-v2",
        }
    }
    search = backend.create_search_contract(_request(configuration), _contract())
    candidate = backend.seed_candidate(search)
    result = backend.prepare(candidate, search)
    assert result.execution_status == CandidateExecutionStatus.COMPLETED
    assert "patient" not in result.model_dump_json().lower()
