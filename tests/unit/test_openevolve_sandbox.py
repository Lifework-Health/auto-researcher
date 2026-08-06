from __future__ import annotations

from auto_researcher.search.openevolve.models import CandidateExecutionStatus
from auto_researcher.search.openevolve.sandbox import LocalSandboxRunner
from auto_researcher.tasks.synthetic import SyntheticEvolvableComponent
from tests.unit.test_openevolve_contracts import (
    _backend,
    _candidate,
    _contract,
    _request,
)


def _prepare_local_source(source, tmp_path, **updates):
    backend = _backend()
    search = backend.create_search_contract(_request(), _contract())
    return LocalSandboxRunner(tmp_path).prepare(
        _candidate(source),
        backend.component_spec,
        search.sandbox_policy.model_copy(update=updates),
        backend.component.seed_configuration(),
    )


def test_sandbox_prepares_valid_source_without_inheriting_credentials(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("OPENEVOLVE_TEST_SECRET", "must-not-be-inherited")
    backend = _backend()
    search = backend.create_search_contract(_request(), _contract())
    seed = backend.seed_candidate(search)
    result = LocalSandboxRunner(tmp_path).prepare(
        seed,
        backend.component_spec,
        search.sandbox_policy,
        backend.component.seed_configuration(),
    )
    assert result.execution_status == CandidateExecutionStatus.COMPLETED
    assert result.generated_configuration["model_family"] == "linear"
    assert "must-not-be-inherited" not in result.safe_log_excerpt
    assert result.cleanup_complete is True
    assert list(tmp_path.iterdir()) == []


def test_sandbox_bounds_stdout_and_sanitises_traceback(tmp_path):
    source = "def evolve(configuration):\n print('x'*20000)\n raise RuntimeError('private')\n"
    candidate = _candidate(source)
    backend = _backend()
    search = backend.create_search_contract(_request(), _contract())
    result = LocalSandboxRunner(tmp_path).prepare(
        candidate,
        SyntheticEvolvableComponent().component_spec(),
        search.sandbox_policy.model_copy(update={"log_bytes": 128}),
        {},
    )
    assert result.execution_status == CandidateExecutionStatus.FAILED
    assert len(result.safe_log_excerpt.encode()) <= 128
    assert "Traceback" not in result.safe_log_excerpt


def test_sandbox_timeout_is_safe_and_cleanup_is_complete(tmp_path):
    source = "def evolve(configuration):\n for _ in iter(int, 1):\n  pass\n return {}\n"
    candidate = _candidate(source)
    backend = _backend()
    search = backend.create_search_contract(_request(), _contract())
    result = LocalSandboxRunner(tmp_path).prepare(
        candidate,
        backend.component_spec,
        search.sandbox_policy.model_copy(update={"wall_time_seconds": 0.05}),
        {},
    )
    assert result.execution_status == CandidateExecutionStatus.TIMED_OUT
    assert result.safe_error_code == "candidate_timeout"
    assert list(tmp_path.iterdir()) == []


def test_local_fixture_counts_only_candidate_writable_workspace(tmp_path):
    source = """def evolve(configuration):
 for i in range(8):
  open(f"entry-{i}", "w").write("x")
 return configuration
"""
    result = _prepare_local_source(source, tmp_path)
    assert result.execution_status == CandidateExecutionStatus.COMPLETED
    assert list(tmp_path.iterdir()) == []


def test_local_fixture_rejects_ninth_concurrent_entry(tmp_path):
    source = """def evolve(configuration):
 for i in range(9):
  open(f"entry-{i}", "w").write("x")
 return configuration
"""
    result = _prepare_local_source(source, tmp_path)
    assert result.execution_status == CandidateExecutionStatus.RESOURCE_LIMITED
    assert result.safe_error_code == "candidate_file_count_limit"
    assert list(tmp_path.iterdir()) == []
