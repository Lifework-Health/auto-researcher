from __future__ import annotations

import importlib.metadata
from pathlib import Path

import pytest
from pydantic import ValidationError

from auto_researcher.runtime.checkpoints import checkpoint_serializer
from auto_researcher.runtime.identity import payload_hash
from auto_researcher.search.openevolve.hardened_executor import docker_policy
from auto_researcher.search.openevolve.integration_artifacts import (
    publish_integration_bundle,
)
from auto_researcher.search.openevolve.upstream import (
    AutoResearcherOpenEvolveModelBridge,
    UpstreamOpenEvolveAdapter,
    assert_live_mutation_eligible,
    default_adapter_contract,
    validate_upstream_dependency,
)
from auto_researcher.search.openevolve.upstream_models import (
    ExecutorIsolationResult,
    UpstreamOpenEvolveAdapterState,
)
from tests.unit.test_openevolve_contracts import _backend, _contract, _request

LOCK = Path(__file__).parents[2] / "constraints" / "openevolve-0.3.2.lock"
pytestmark = pytest.mark.upstream_openevolve
pytest.importorskip("openevolve", reason="pinned optional OpenEvolve dependency absent")


class FakeUpstreamClient:
    calls = 0

    def propose_mutation(self, request):
        self.calls += 1
        assert "api_key" not in str(request).lower()
        return {
            "mutable_file": "candidate.py",
            "source": 'def evolve(configuration):\n return {"model_family":"tree","complexity":4,"learning_rate":0.05}\n',
            "description": "upstream-orchestrated full replacement",
            "upstream_program_id": "upstream-program-1",
        }


def test_pinned_dependency_and_adapter_contract_are_valid():
    contract = default_adapter_contract(LOCK)
    validate_upstream_dependency(contract)
    assert contract.upstream_commit == "411fb59c886c18704caaffb611e17cf9e7d824d2"
    assert contract.evaluator_owner == contract.model_client_owner == "AUTO_RESEARCHER"
    assert "provider_clients" in contract.unsupported_features


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"upstream_commit": "0" * 40}, "validation"),
        ({"upstream_package_version": "0.3.1"}, "validation"),
        ({"dependency_lock_hash": "not-a-hash"}, "validation"),
    ],
)
def test_upstream_identity_drift_fails_closed(change, code):
    values = default_adapter_contract(LOCK).model_dump(mode="python")
    values.update(change)
    if code == "validation":
        with pytest.raises(ValidationError):
            type(default_adapter_contract(LOCK)).model_validate(values)
    else:
        with pytest.raises(ValueError, match=code):
            validate_upstream_dependency(
                type(default_adapter_contract(LOCK)).model_validate(values)
            )


def test_changed_installed_distribution_hash_is_rejected(monkeypatch):
    monkeypatch.setattr(
        "auto_researcher.search.openevolve.upstream.installed_record_hash",
        lambda: "0" * 64,
    )
    with pytest.raises(
        ValueError, match="upstream_openevolve_dependency_hash_mismatch"
    ):
        validate_upstream_dependency(default_adapter_contract(LOCK))


def test_missing_dependency_has_stable_safe_code(monkeypatch):
    original = importlib.metadata.distribution

    def missing(name):
        if name == "openevolve":
            raise importlib.metadata.PackageNotFoundError(name)
        return original(name)

    monkeypatch.setattr(importlib.metadata, "distribution", missing)
    with pytest.raises(ValueError, match="upstream_openevolve_dependency_unavailable"):
        validate_upstream_dependency(default_adapter_contract(LOCK))


def test_bridge_is_exactly_once_and_maps_upstream_program_metadata():
    client = FakeUpstreamClient()
    adapter = UpstreamOpenEvolveAdapter(
        default_adapter_contract(LOCK), AutoResearcherOpenEvolveModelBridge(client)
    )
    backend = _backend()
    search = backend.create_search_contract(_request(), _contract())
    seed = backend.seed_candidate(search)
    population = backend.initialise_population(search)
    reservation = backend.reserve_mutation(search, population, seed)
    first = adapter.mutate(reservation, seed, backend.component_spec)
    second = adapter.mutate(reservation, seed, backend.component_spec)
    assert first == second
    assert client.calls == 1
    assert adapter.state.proposal_count == 1
    assert (
        UpstreamOpenEvolveAdapterState.model_validate_json(
            adapter.state.model_dump_json()
        )
        == adapter.state
    )


@pytest.mark.parametrize(
    "response",
    [
        {"mutable_file": "../evaluator.py", "source": "x", "description": "escape"},
        {
            "mutable_file": "candidate.py",
            "source": "x",
            "description": "deps",
            "dependency_requests": ("requests",),
        },
        {
            "mutable_file": "candidate.py",
            "source": "x",
            "description": "provider",
            "provider_configuration": {"api_key": "forbidden"},
        },
        {"files": {"candidate.py": "x", "verifier.py": "x"}, "description": "many"},
    ],
)
def test_hostile_upstream_envelopes_are_rejected(response):
    class Client:
        def propose_mutation(self, request):
            return response

    adapter = UpstreamOpenEvolveAdapter(
        default_adapter_contract(LOCK), AutoResearcherOpenEvolveModelBridge(Client())
    )
    backend = _backend()
    search = backend.create_search_contract(_request(), _contract())
    seed = backend.seed_candidate(search)
    reservation = backend.reserve_mutation(
        search, backend.initialise_population(search), seed
    )
    with pytest.raises(ValueError, match="upstream_"):
        adapter.mutate(reservation, seed, backend.component_spec)


def test_upstream_population_recommendation_is_metadata_not_identity():
    adapter = UpstreamOpenEvolveAdapter(
        default_adapter_contract(LOCK),
        AutoResearcherOpenEvolveModelBridge(FakeUpstreamClient()),
    )
    backend = _backend()
    search = backend.create_search_contract(_request(), _contract())
    seed = backend.seed_candidate(search)
    assert (
        adapter.recommend_parent((seed,), {seed.candidate_id: 0.78})
        == seed.candidate_id
    )


def test_adapter_state_is_checkpoint_allowlisted():
    state = UpstreamOpenEvolveAdapterState(
        adapter_identity_hash="1" * 64,
        proposal_count=2,
        cursor=2,
    )
    encoded = checkpoint_serializer().dumps_typed(state)
    assert checkpoint_serializer().loads_typed(encoded) == state


def test_adapter_and_executor_evidence_publish_transactionally(tmp_path):
    root = Path(__file__).parents[2]
    policy = docker_policy(
        "fixture-image",
        "sha256:" + "2" * 64,
        root / "docker/openevolve-executor/Dockerfile",
        root / "docker/openevolve-executor/worker.py",
        "fixture-runtime",
    )
    isolation = ExecutorIsolationResult(
        executor_policy_hash="3" * 64,
        network_isolation_verified=True,
        mount_isolation_verified=True,
        environment_sanitisation_verified=True,
        safe_checks={"fixture": True},
    )
    contract = default_adapter_contract(LOCK)
    state = UpstreamOpenEvolveAdapterState(adapter_identity_hash="4" * 64)
    first = publish_integration_bundle(
        tmp_path, "run-pr7", contract, state, policy, isolation
    )
    files = tuple((tmp_path / "runs" / "run-pr7").rglob("*.json"))
    mtimes = {path: path.stat().st_mtime_ns for path in files}
    second = publish_integration_bundle(
        tmp_path, "run-pr7", contract, state, policy, isolation
    )
    assert first == second
    assert mtimes == {path: path.stat().st_mtime_ns for path in files}


def test_live_eligibility_fails_closed_without_real_isolation_and_approval():
    root = Path(__file__).parents[2]
    contract = default_adapter_contract(LOCK)
    policy = docker_policy(
        "fixture-image",
        "sha256:" + "2" * 64,
        root / "docker/openevolve-executor/Dockerfile",
        root / "docker/openevolve-executor/worker.py",
        "fixture-runtime",
    )
    isolation = ExecutorIsolationResult(
        executor_policy_hash="3" * 64,
        network_isolation_verified=False,
        mount_isolation_verified=True,
        environment_sanitisation_verified=True,
        safe_checks={},
    )
    with pytest.raises(ValueError, match="network_isolation_unverified"):
        assert_live_mutation_eligible(
            contract,
            policy,
            isolation,
            approved_adapter_hash=payload_hash(contract),
            approved_image_digest=policy.image_digest,
            contract_permits_live_mutation=True,
            operator_approved=True,
            maximum_model_calls=1,
            maximum_candidate_evaluations=1,
        )
