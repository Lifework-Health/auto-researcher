from __future__ import annotations

import os
import subprocess
import sys

import pytest
from pydantic import ValidationError

from auto_researcher.contracts.enums import SearchType
from auto_researcher.contracts.models import SearchRequest
from auto_researcher.runtime.dependencies import memory_dependencies
from auto_researcher.runtime.dependencies import task_memory_dependencies
from auto_researcher.search.openevolve.backend import OpenEvolveBackend
from auto_researcher.search.openevolve.identity import (
    candidate_id,
    openevolve_hash,
    source_hash,
)
from auto_researcher.search.openevolve.models import (
    CandidateOutcome,
    CandidateStatus,
    EvolvableComponentSpec,
    OpenEvolveCandidate,
)
from auto_researcher.search.protocols import SearchCapability
from auto_researcher.search.registry import SearchBackendRegistry
from auto_researcher.search.openevolve.mutation import FakeModelMutationOperator
from auto_researcher.search.openevolve.validation import validate_candidate
from auto_researcher.tasks.synthetic import (
    SyntheticEvolvableComponent,
    SyntheticTask,
    default_synthetic_contract,
    default_synthetic_openevolve_configuration,
)
from auto_researcher.tasks.models import TaskRuntimeContext


def _backend() -> OpenEvolveBackend:
    dependencies = memory_dependencies(search_type=SearchType.OPENEVOLVE)
    assert dependencies.openevolve_backend is not None
    return dependencies.openevolve_backend


def _request(configuration=None, budget=4) -> SearchRequest:
    return SearchRequest(
        request_id="search-openevolve-test",
        hypothesis_id="hypothesis-test",
        search_type=SearchType.OPENEVOLVE,
        target="bounded source evolution",
        search_space=configuration or default_synthetic_openevolve_configuration(),
        experiment_budget=budget,
        rationale="offline fixture",
    )


def _contract(maximum_experiments=4):
    return default_synthetic_contract(
        search_types=frozenset({SearchType.OPENEVOLVE}),
        maximum_experiments=maximum_experiments,
    )


def _candidate(source: str, component: EvolvableComponentSpec | None = None):
    component = component or SyntheticEvolvableComponent().component_spec()
    interface_hash = openevolve_hash(
        "openevolve-component-interface-v1",
        {
            "component_id": component.component_id,
            "component_version": component.component_version,
            "contract": component.immutable_interface_contract,
            "entry_point": component.entry_point,
            "parameter_schema": component.parameter_schema,
            "output_schema": component.output_schema,
            "allowed_files": component.allowed_files,
        },
    )
    digest = source_hash(source)
    return OpenEvolveCandidate(
        candidate_id=candidate_id(
            search_request_id="search-openevolve-test",
            component_interface_hash=interface_hash,
            source_sha256=digest,
        ),
        search_request_id="search-openevolve-test",
        parent_candidate_ids=(),
        generation=0,
        birth_index=0,
        mutation_operator="test",
        mutation_description="test source",
        mutable_file=component.mutable_file,
        source_payload=source,
        source_hash=digest,
        component_interface_hash=interface_hash,
        dependency_manifest_hash="0" * 64,
        sandbox_policy_id="openevolve-sandbox-v1",
        status=CandidateStatus.PROPOSED,
        creation_provenance="DETERMINISTIC_FIXTURE",
    )


def test_finite_search_contract_binds_task_component_evaluator_verifier_and_sandbox():
    search = _backend().create_search_contract(_request(), _contract())
    assert search.protocol_version == "openevolve-search-v1"
    assert search.maximum_candidate_evaluations == 4
    assert search.maximum_generations == 3
    assert search.evaluator_identity.startswith("synthetic-evaluator@")
    assert search.verifier_identity == "deterministic-verifier-v1@synthetic-policy-v1"
    assert search.sandbox_policy.network_access is False
    assert search.sandbox_policy.inherit_environment is False


@pytest.mark.parametrize(
    ("change", "code"),
    [
        (
            {"maximum_wall_time_seconds": None},
            "openevolve_finite_configuration_required",
        ),
        ({"sandbox_policy_id": "unknown"}, "openevolve_sandbox_policy_unavailable"),
        ({"evaluator_identity": "wrong"}, "openevolve_evaluator_identity_mismatch"),
        ({"verifier_identity": "wrong"}, "openevolve_verifier_identity_mismatch"),
    ],
)
def test_invalid_open_evolve_contract_is_rejected(change, code):
    configuration = default_synthetic_openevolve_configuration()
    values = dict(configuration["openevolve"])
    if change.get("maximum_wall_time_seconds", "present") is None:
        values.pop("maximum_wall_time_seconds")
    else:
        values.update(change)
    with pytest.raises((ValueError, ValidationError), match=code):
        _backend().create_search_contract(
            _request({"openevolve": values}),
            _contract(),
        )


def test_candidate_budget_cannot_exceed_research_contract():
    with pytest.raises(ValueError, match="candidate_budget_exceeds_contract"):
        _backend().create_search_contract(_request(budget=4), _contract(3))


def test_mutation_enabled_search_requires_seed_plus_evolved_evaluation():
    configuration = default_synthetic_openevolve_configuration()
    configuration["openevolve"]["maximum_generations"] = 1
    with pytest.raises(
        ValueError, match="openevolve_mutation_evaluation_budget_too_small"
    ):
        _backend().create_search_contract(
            _request(configuration, budget=1),
            _contract(maximum_experiments=1),
        )


def test_two_evaluations_fund_seed_and_one_evolved_candidate():
    configuration = default_synthetic_openevolve_configuration()
    configuration["openevolve"].update(
        {"maximum_generations": 1, "maximum_model_calls": 0}
    )
    search = _backend().create_search_contract(
        _request(configuration, budget=2),
        _contract(maximum_experiments=2),
    )
    assert search.maximum_candidate_evaluations == 2
    assert search.maximum_generations == 1


def test_model_call_budget_must_fund_one_enabled_mutation():
    internal = _backend()
    backend = OpenEvolveBackend(
        internal.component,
        internal.metadata,
        internal.verifier_identity,
        FakeModelMutationOperator(_FakeMutationClient()),
        internal.sandbox_runner,
    )
    configuration = default_synthetic_openevolve_configuration()
    configuration["openevolve"].update(
        {"maximum_generations": 1, "maximum_model_calls": 0}
    )
    with pytest.raises(ValueError, match="openevolve_model_call_budget_too_small"):
        backend.create_search_contract(
            _request(configuration, budget=2),
            _contract(maximum_experiments=2),
        )


def test_runtime_dependency_preflight_rejects_unreachable_mutation_without_effects():
    class Mutation:
        operator_id = "preflight-mutation"
        operator_version = "preflight-v1"
        model_calls_per_mutation = 1
        provenance = "FAKE_MODEL"
        calls = 0

        def mutate(self, *args, **kwargs):
            self.calls += 1
            raise AssertionError("budget preflight allowed mutation")

    class Runner:
        runner_id = "openevolve-sandbox-v1"
        calls = 0

        def prepare(self, *args, **kwargs):
            self.calls += 1
            raise AssertionError("budget preflight allowed preparation")

    mutation = Mutation()
    runner = Runner()
    configuration = default_synthetic_openevolve_configuration()
    configuration["openevolve"].update(
        {"maximum_generations": 1, "maximum_model_calls": 1}
    )
    with pytest.raises(
        ValueError, match="openevolve_mutation_evaluation_budget_too_small"
    ):
        task_memory_dependencies(
            SyntheticTask(),
            TaskRuntimeContext(),
            _contract(maximum_experiments=1),
            configuration,
            search_type=SearchType.OPENEVOLVE,
            openevolve_mutation_operator=mutation,
            openevolve_sandbox_runner=runner,
        )
    assert mutation.calls == 0
    assert runner.calls == 0


def test_component_surface_is_exactly_one_safe_file():
    values = SyntheticEvolvableComponent().component_spec().model_dump(mode="python")
    values["allowed_files"] = ("candidate.py", "../evaluator.py")
    with pytest.raises(ValidationError):
        EvolvableComponentSpec.model_validate(values)


HOSTILE_SOURCES = [
    "import os\ndef evolve(configuration):\n return os.system('id')\n",
    "import subprocess\ndef evolve(configuration):\n return {}\n",
    "import socket\ndef evolve(configuration):\n return {}\n",
    "import requests\ndef evolve(configuration):\n return {}\n",
    "import urllib.request\ndef evolve(configuration):\n return {}\n",
    "def evolve(configuration):\n return __import__('os')\n",
    "def evolve(configuration):\n return eval('1')\n",
    "def evolve(configuration):\n exec('x=1')\n return {}\n",
    "def evolve(configuration):\n return open('/etc/passwd').read()\n",
    "def evolve(configuration):\n return open('../secret').read()\n",
    "def evolve(configuration):\n return os.environ\n",
    "from pathlib import Path\ndef evolve(configuration):\n return list(Path.home().iterdir())\n",
    "def evolve(configuration):\n return globals()\n",
    "import multiprocessing\ndef evolve(configuration):\n return {}\n",
    "import auto_researcher\ndef evolve(configuration):\n return {}\n",
    "def evolve(configuration):\n configuration['evaluator_id']='changed'\n return configuration\n",
    "def evolve(configuration):\n return evolve(configuration)\n",
    "def evolve(configuration):\n while True:\n  pass\n",
    "class Escape:\n pass\ndef evolve(configuration):\n return {}\n",
    "def evolve(configuration):\n return (1).__class__.__mro__\n",
]


@pytest.mark.parametrize("source", HOSTILE_SOURCES)
def test_hostile_candidates_fail_static_validation(source):
    result = validate_candidate(
        _candidate(source),
        SyntheticEvolvableComponent().component_spec(),
    )
    assert result.status.value == "INVALID"
    assert result.safe_error_code in {
        "candidate_forbidden_import",
        "candidate_forbidden_operation",
    }


def test_syntax_and_binary_payloads_fail_closed():
    for source in (
        "def evolve(:\n pass\n",
        "def evolve(configuration):\n return {}\x00\n",
    ):
        result = validate_candidate(
            _candidate(source),
            SyntheticEvolvableComponent().component_spec(),
        )
        assert result.status.value == "INVALID"


class _FakeMutationClient:
    def propose_mutation(self, request):
        assert request["allowed_files"] == ["candidate.py"]
        assert "prompt" not in request
        return {
            "source": 'def evolve(configuration):\n return {"model_family":"tree","complexity":4,"learning_rate":0.05}\n',
            "description": "bounded fake replacement",
        }


def test_fake_model_mutation_is_structured_and_identity_bound():
    backend = _backend()
    search = backend.create_search_contract(_request(), _contract())
    seed = backend.seed_candidate(search)
    population = backend.initialise_population(search).model_copy(
        update={"active_population_candidate_ids": (seed.candidate_id,)}
    )
    reservation = backend.reserve_mutation(search, population, seed)
    operator = FakeModelMutationOperator(_FakeMutationClient())
    source, description, call_id = operator.mutate(
        reservation,
        seed,
        backend.component_spec,
    )
    assert source.startswith("def evolve")
    assert description == "bounded fake replacement"
    assert call_id.startswith("fake-mutation-")


def test_candidate_and_interface_hashes_ignore_python_hash_seed():
    script = """
from auto_researcher.search.openevolve.identity import candidate_id, openevolve_hash, source_hash
from auto_researcher.tasks.synthetic import SyntheticEvolvableComponent
s=SyntheticEvolvableComponent().component_spec()
i=openevolve_hash('interface', {'files': frozenset(s.allowed_files), 'schema': s.output_schema})
print(candidate_id(search_request_id='r',component_interface_hash=i,source_sha256=source_hash(s.seed_source)))
"""
    outputs = set()
    for seed in ("2", "999"):
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        outputs.add(completed.stdout.strip())
    assert len(outputs) == 1


def test_registry_exposes_direct_optuna_and_openevolve_without_duplicate_registration():
    dependencies = memory_dependencies(search_type=SearchType.OPENEVOLVE)
    assert set(dependencies.search_backend_registry.capabilities()) == set(SearchType)
    assert (
        dependencies.search_backend_registry.backend(SearchType.OPENEVOLVE)
        is dependencies.openevolve_backend
    )
    registry = SearchBackendRegistry()
    capability = SearchCapability(SearchType.DIRECT, True, "ok", "installed")
    registry.register(capability, object())
    with pytest.raises(ValueError, match="already registered"):
        registry.register(capability, object())


def test_constrained_selection_preserves_negative_result_over_infeasible_high_score():
    backend = _backend()
    search = backend.create_search_contract(_request(), _contract())
    first = _candidate(
        'def evolve(configuration):\n return {"model_family":"linear","complexity":4,"learning_rate":0.05}\n'
    )
    second = _candidate(
        'def evolve(configuration):\n return {"model_family":"tree","complexity":4,"learning_rate":0.05}\n'
    )
    population = backend.initialise_population(search)
    population = backend.update_population(
        population,
        search,
        first,
        CandidateOutcome(
            candidate_id=first.candidate_id,
            source_hash=first.source_hash,
            status=CandidateStatus.VERIFIED,
            objective_value=-1.0,
            constraint_compliant=True,
            verified=True,
            selection_outcome="ranked",
            replacement_outcome="active",
        ),
    )
    population = backend.update_population(
        population,
        search,
        second,
        CandidateOutcome(
            candidate_id=second.candidate_id,
            source_hash=second.source_hash,
            status=CandidateStatus.VERIFIED,
            objective_value=100.0,
            constraint_compliant=False,
            verified=True,
            selection_outcome="ranked",
            replacement_outcome="archive",
        ),
    )
    assert population.best_known_candidate_ids == (first.candidate_id,)
    assert set(population.archive_candidate_ids) == {
        first.candidate_id,
        second.candidate_id,
    }
