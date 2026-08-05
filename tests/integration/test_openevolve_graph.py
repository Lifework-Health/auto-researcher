from __future__ import annotations

import hashlib
import itertools
import os
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from auto_researcher.contracts.enums import RunStatus, SearchType
from auto_researcher.graph.builder import build_graph
from auto_researcher.runtime.dependencies import (
    task_memory_dependencies,
    task_sqlite_dependencies,
)
from auto_researcher.runtime.execution import (
    RunExecutionError,
    inspect_terminal_run,
    resume_run,
    start_run,
)
from auto_researcher.runtime.identity import payload_hash
from auto_researcher.search.openevolve.artifacts import verify_search_artefacts
from auto_researcher.search.openevolve.backend import OpenEvolveBackend
from auto_researcher.search.openevolve.models import (
    CandidateOutcome,
    CandidateStatus,
    EvolvableComponentSpec,
)
from auto_researcher.search.openevolve.mutation import DeterministicMutationOperator
from auto_researcher.search.openevolve.sandbox import LocalSandboxRunner
from auto_researcher.tasks.models import TaskRuntimeContext
from auto_researcher.tasks.synthetic import (
    SyntheticEvolvableComponent,
    SyntheticTask,
    default_synthetic_contract,
    default_synthetic_openevolve_configuration,
)

FIXED_TIME = datetime(2026, 8, 5, 12, tzinfo=UTC)


def _contract():
    return default_synthetic_contract(
        maximum_cycles=1,
        search_types=frozenset({SearchType.OPENEVOLVE}),
        maximum_experiments=4,
    )


def _identity_input(run_id, thread_id, contract):
    return {"run_id": run_id, "thread_id": thread_id, "contract": contract}


def _ids():
    counter = itertools.count()
    return lambda prefix: f"{prefix}-{next(counter):024d}"


class _ZeroRuntimeRunner(LocalSandboxRunner):
    def prepare(self, *args, **kwargs):
        return (
            super().prepare(*args, **kwargs).model_copy(update={"runtime_seconds": 0.0})
        )


def _deterministic_dependencies(dependencies, workspace):
    existing = dependencies.openevolve_backend
    assert existing is not None
    backend = OpenEvolveBackend(
        existing.component,
        existing.metadata,
        existing.verifier_identity,
        DeterministicMutationOperator(),
        _ZeroRuntimeRunner(workspace),
    )
    return replace(dependencies, openevolve_backend=backend)


def test_synthetic_openevolve_improves_over_multiple_generations_and_publishes(
    tmp_path,
):
    run_id = "openevolve-synthetic-demo"
    thread_id = "openevolve-synthetic-thread"
    context = TaskRuntimeContext(
        run_id=run_id,
        output_dir=tmp_path / "artefacts",
        workspace_dir=tmp_path / "workspace",
        manifest_created_at=FIXED_TIME,
    )
    dependencies = task_memory_dependencies(
        SyntheticTask(),
        context,
        _contract(),
        default_synthetic_openevolve_configuration(),
        search_type=SearchType.OPENEVOLVE,
        clock=lambda: FIXED_TIME,
        id_generator=_ids(),
    )
    final = start_run(
        build_graph(dependencies),
        _identity_input(run_id, thread_id, _contract()),
        {"configurable": {"thread_id": thread_id}},
    )
    result = final["openevolve_search_result"]
    population = final["openevolve_population_state"]
    assert final["status"] == RunStatus.COMPLETED
    assert result.stop_reason == "objective_reached"
    assert result.generations_completed == 2
    assert [item.objective_value for item in population.outcomes] == [0.78, 0.84, 0.88]
    assert final["evaluation_result"].primary_score == 0.88
    assert final["verification_result"].verified is True
    assert final["verification_result"].constraint_compliant is True
    assert len(population.lineage) == 3
    assert verify_search_artefacts(context, result.search_request_id)[0] is True
    for outcome in population.outcomes:
        assert outcome.evaluation is not None
        assert len(outcome.evaluation.artefact_references) == 4
    event_types = [
        item.event_type.value
        for item in dependencies.provenance_store.list_events(run_id)
    ]
    assert event_types.count("EXPERIMENT_PREPARED") == 3
    assert event_types.count("EVALUATION_OBSERVED") == 3
    assert event_types.count("EVIDENCE_VERIFIED") == 3
    assert event_types[-1] == "OPENEVOLVE_SEARCH_STOPPED"


def test_duplicate_source_is_archived_without_reevaluation():
    backend = task_memory_dependencies(
        SyntheticTask(),
        TaskRuntimeContext(),
        _contract(),
        default_synthetic_openevolve_configuration(),
        search_type=SearchType.OPENEVOLVE,
    ).openevolve_backend
    assert backend is not None
    request = backend.create_search_contract(
        __import__(
            "tests.unit.test_openevolve_contracts", fromlist=["_request"]
        )._request(),
        _contract(),
    )
    seed = backend.seed_candidate(request)
    population = backend.initialise_population(request)
    accepted = CandidateOutcome(
        candidate_id=seed.candidate_id,
        source_hash=seed.source_hash,
        status=CandidateStatus.VERIFIED,
        objective_value=-0.2,
        constraint_compliant=True,
        verified=True,
        selection_outcome="ranked",
        replacement_outcome="active",
    )
    population = backend.update_population(population, request, seed, accepted)
    reservation = backend.reserve_mutation(request, population, seed)
    duplicate = backend.mutate_candidate(
        reservation,
        seed,
        request,
    ).model_copy(
        update={
            "source_payload": seed.source_payload,
            "source_hash": seed.source_hash,
            "candidate_id": seed.candidate_id,
        }
    )
    rejected = CandidateOutcome(
        candidate_id=duplicate.candidate_id,
        source_hash=duplicate.source_hash,
        status=CandidateStatus.REJECTED,
        selection_outcome="rejected",
        rejection_reason="candidate_duplicate",
        replacement_outcome="archive_only",
    )
    updated = backend.update_population(population, request, duplicate, rejected)
    assert (
        updated.budget.candidate_evaluations == population.budget.candidate_evaluations
    )
    assert updated.diversity_metadata["duplicate_rejections"] == 1
    assert updated.active_population_candidate_ids == (seed.candidate_id,)
    assert updated.outcomes == population.outcomes


def _run_sqlite(root, *, interrupt_after=None):
    run_id = "openevolve-resume-equivalence"
    thread_id = "openevolve-resume-thread"
    contract = _contract()
    context = TaskRuntimeContext(
        run_id=run_id,
        output_dir=root / "artefacts",
        workspace_dir=root / "workspace",
        manifest_created_at=FIXED_TIME,
    )
    manager = task_sqlite_dependencies(
        SyntheticTask(),
        context,
        contract,
        default_synthetic_openevolve_configuration(),
        root / "checkpoints.sqlite",
        root / "provenance.sqlite",
        agent_calls_path=root / "agent-calls.sqlite",
        knowledge_retrievals_path=root / "knowledge.sqlite",
        search_type=SearchType.OPENEVOLVE,
        clock=lambda: FIXED_TIME,
        id_generator=_ids(),
    )
    return manager, context, contract, run_id, thread_id


def test_interrupted_resume_matches_uninterrupted_population_and_terminal_guards(
    tmp_path,
):
    uninterrupted_manager, _, contract, run_id, thread_id = _run_sqlite(
        tmp_path / "continuous"
    )
    with uninterrupted_manager as raw:
        dependencies = _deterministic_dependencies(raw, tmp_path / "continuous-sandbox")
        continuous = start_run(
            build_graph(dependencies),
            _identity_input(run_id, thread_id, contract),
            {"configurable": {"thread_id": thread_id}},
        )

    interrupted_manager, _, _, _, _ = _run_sqlite(tmp_path / "resumed")
    config = {"configurable": {"thread_id": thread_id}}
    with interrupted_manager as raw:
        dependencies = _deterministic_dependencies(raw, tmp_path / "resumed-sandbox")
        paused = start_run(
            build_graph(
                dependencies,
                interrupt_after=["prepare_openevolve_candidate"],
            ),
            _identity_input(run_id, thread_id, contract),
            config,
        )
        assert paused["status"] == RunStatus.RUNNING
        assert paused["evaluation_result"] is None

    resumed_manager, _, _, _, _ = _run_sqlite(tmp_path / "resumed")
    with resumed_manager as raw:
        dependencies = _deterministic_dependencies(raw, tmp_path / "resumed-sandbox")
        resumed = resume_run(build_graph(dependencies), config)
        inspected_once = inspect_terminal_run(build_graph(dependencies), config)
        inspected_twice = inspect_terminal_run(build_graph(dependencies), config)
        assert inspected_once == inspected_twice == resumed
        with pytest.raises(
            RunExecutionError,
            match="thread_already_exists_use_resume_or_inspect",
        ):
            start_run(
                build_graph(dependencies),
                _identity_input(run_id, thread_id, contract),
                config,
            )
        with pytest.raises(
            RunExecutionError,
            match="thread_is_terminal_use_inspect",
        ):
            resume_run(build_graph(dependencies), config)

    assert payload_hash(resumed["openevolve_population_state"]) == payload_hash(
        continuous["openevolve_population_state"]
    )
    assert resumed["openevolve_search_result"] == continuous["openevolve_search_result"]
    assert resumed["evaluation_result"] == continuous["evaluation_result"]
    assert resumed["verification_result"] == continuous["verification_result"]


class _CellBiologyComponent(SyntheticEvolvableComponent):
    SOURCE = """def evolve(configuration):
    pathway_score = configuration["immune_signal"] + configuration["growth_signal"]
    return {"model_family": "neural" if pathway_score >= 1 else "linear", "complexity": 4, "learning_rate": 0.05}
"""

    def component_spec(self):
        return EvolvableComponentSpec(
            component_id="fake-cell-pathway-aggregation",
            component_version="1.0",
            mutable_file="candidate.py",
            allowed_files=("candidate.py",),
            entry_point="evolve",
            immutable_interface_contract="evolve(synthetic pathway signals) -> immutable experiment configuration",
            parameter_schema={"immune_signal": "finite", "growth_signal": "finite"},
            output_schema={"model": "SyntheticConfiguration@1.0"},
            seed_source=self.SOURCE,
            deterministic_mutation_sources=(self.SOURCE,),
            maximum_source_bytes=4096,
            task_mutation_context={"data": "synthetic_non_patient"},
        )

    def seed_configuration(self):
        return {"immune_signal": 0.7, "growth_signal": 0.5}


def test_fake_cell_biology_component_uses_normal_evaluator_and_verifier(tmp_path):
    dependencies = task_memory_dependencies(
        SyntheticTask(),
        TaskRuntimeContext(
            run_id="fake-cell-demo",
            output_dir=tmp_path / "artefacts",
            workspace_dir=tmp_path / "workspace",
            manifest_created_at=FIXED_TIME,
        ),
        _contract(),
        default_synthetic_openevolve_configuration(),
        search_type=SearchType.OPENEVOLVE,
        clock=lambda: FIXED_TIME,
    )
    backend = OpenEvolveBackend(
        _CellBiologyComponent(),
        dependencies.experiment_metadata,
        "deterministic-verifier-v1@synthetic-policy-v1",
        DeterministicMutationOperator(),
        LocalSandboxRunner(tmp_path / "sandbox"),
    )
    from tests.unit.test_openevolve_contracts import _request

    request_model = _request()
    search = backend.create_search_contract(request_model, _contract())
    candidate = backend.seed_candidate(search)
    validation = backend.validate(candidate)
    assert validation.status.value == "VALID"
    preparation = backend.prepare(candidate, search)
    experiment = backend.component.candidate_to_experiment(
        candidate,
        preparation,
        request_model,
        _contract(),
        dependencies.experiment_metadata,
        run_id="fake-cell-demo",
    )
    evaluation = dependencies.evaluator.evaluate(experiment, _contract())
    verification = dependencies.verifier.verify(experiment, evaluation, _contract())
    assert evaluation.primary_score == 0.88
    assert verification.verified is True
    assert experiment.evaluator_id == "synthetic-evaluator"


def test_openevolve_checkpoint_reconstructs_cross_process_and_inspect_is_read_only(
    tmp_path,
):
    manager, _, contract, run_id, thread_id = _run_sqlite(tmp_path)
    config = {"configurable": {"thread_id": thread_id}}
    with manager as raw:
        dependencies = _deterministic_dependencies(raw, tmp_path / "sandbox")
        final = start_run(
            build_graph(dependencies),
            _identity_input(run_id, thread_id, contract),
            config,
        )
    checkpoint = tmp_path / "checkpoints.sqlite"
    before_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    before_mtime = checkpoint.stat().st_mtime_ns
    expected = payload_hash(final)
    script = """
import sys
from pathlib import Path
from types import SimpleNamespace
from auto_researcher.contracts.enums import SearchType
from auto_researcher.runtime.checkpoints import sqlite_checkpointer
from auto_researcher.runtime.execution import inspect_terminal_run
from auto_researcher.runtime.identity import payload_hash
from auto_researcher.search.openevolve.models import OpenEvolveCandidateCollection, OpenEvolvePopulationState
path=Path(sys.argv[1]); thread=sys.argv[2]
saver, connection=sqlite_checkpointer(path)
class View:
 def get_state(self, config):
  item=saver.get_tuple(config)
  return SimpleNamespace(values=item.checkpoint['channel_values'] if item else {})
state=inspect_terminal_run(View(), {'configurable': {'thread_id': thread}})
assert type(state['openevolve_population_state']) is OpenEvolvePopulationState
assert type(state['openevolve_candidates']) is OpenEvolveCandidateCollection
assert state['search_request'].search_type is SearchType.OPENEVOLVE
print(payload_hash(state))
connection.close()
"""
    hashes = set()
    for seed in ("17", "131"):
        completed = subprocess.run(
            [sys.executable, "-c", script, str(checkpoint), thread_id],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        hashes.add(completed.stdout.strip())
    assert hashes == {expected}
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == before_hash
    assert checkpoint.stat().st_mtime_ns == before_mtime


def test_tampered_openevolve_search_bundle_fails_closed(tmp_path):
    run_id = "openevolve-tamper"
    thread_id = "openevolve-tamper-thread"
    context = TaskRuntimeContext(
        run_id=run_id,
        output_dir=tmp_path / "artefacts",
        workspace_dir=tmp_path / "workspace",
        manifest_created_at=FIXED_TIME,
    )
    dependencies = task_memory_dependencies(
        SyntheticTask(),
        context,
        _contract(),
        default_synthetic_openevolve_configuration(),
        search_type=SearchType.OPENEVOLVE,
        clock=lambda: FIXED_TIME,
        id_generator=_ids(),
    )
    final = start_run(
        build_graph(dependencies),
        _identity_input(run_id, thread_id, _contract()),
        {"configurable": {"thread_id": thread_id}},
    )
    request_id = final["openevolve_search_result"].search_request_id
    candidate_index = (
        context.output_dir
        / "runs"
        / run_id
        / "openevolve"
        / request_id
        / "candidate_index.json"
    )
    candidate_index.write_text("{}\n", encoding="utf-8")
    assert verify_search_artefacts(context, request_id) == (False, None)
