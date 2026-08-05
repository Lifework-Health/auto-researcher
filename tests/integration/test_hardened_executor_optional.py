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
from auto_researcher.search.openevolve.models import CandidateExecutionStatus
from auto_researcher.search.openevolve.mutation import DeterministicMutationOperator
from auto_researcher.search.openevolve.upstream import (
    AutoResearcherOpenEvolveModelBridge,
    UpstreamOpenEvolveAdapter,
    default_adapter_contract,
)
from auto_researcher.runtime.dependencies import memory_dependencies
from auto_researcher.contracts.enums import SearchType
from auto_researcher.tasks.synthetic import SyntheticEvolvableComponent
from tests.unit.test_openevolve_contracts import _backend, _contract, _request

LOCK = Path(__file__).parents[2] / "constraints" / "openevolve-0.3.2.lock"


class FakeUpstreamClient:
    def propose_mutation(self, request):
        return {
            "mutable_file": "candidate.py",
            "source": 'def evolve(configuration):\n return {"model_family":"tree","complexity":4,"learning_rate":0.05}\n',
            "description": "bounded upstream replacement",
        }


pytestmark = pytest.mark.hardened_executor


def _executor():
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
        )
    )


def test_real_hardened_executor_proves_isolation_and_prepares_candidate():
    executor = _executor()
    isolation = executor.verify_isolation()
    assert isolation.network_isolation_verified is True
    assert isolation.mount_isolation_verified is True
    assert isolation.environment_sanitisation_verified is True
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
            "sandbox_policy_id": "openevolve-hardened-executor-v1",
        }
    }
    search = backend.create_search_contract(_request(configuration), _contract())
    candidate = backend.seed_candidate(search)
    result = backend.prepare(candidate, search)
    assert result.execution_status == CandidateExecutionStatus.COMPLETED
    assert result.generated_configuration["model_family"] == "linear"
    assert result.output_references[0].startswith("executor-policy:")


def test_hardened_executor_rejects_image_drift():
    executor = _executor()
    executor = HardenedDockerExecutor(
        executor.policy.model_copy(update={"image_digest": "sha256:" + "0" * 64})
    )
    with pytest.raises(ValueError, match="hardened_executor_image_mismatch"):
        executor.verify_isolation()


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
            "sandbox_policy_id": "openevolve-hardened-executor-v1",
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
            "sandbox_policy_id": "openevolve-hardened-executor-v1",
        }
    }
    search = backend.create_search_contract(_request(configuration), _contract())
    candidate = backend.seed_candidate(search)
    result = backend.prepare(candidate, search)
    assert result.execution_status == CandidateExecutionStatus.COMPLETED
    assert "patient" not in result.model_dump_json().lower()
