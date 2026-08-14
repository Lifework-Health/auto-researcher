from __future__ import annotations

from datetime import UTC, datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import threading
import time

import optuna
import pytest

from auto_researcher.contracts.enums import EvidenceStatus, ProvenanceKind, SearchType
from auto_researcher.contracts.models import (
    EvaluationResult,
    SearchRequest,
    VerificationResult,
)
from auto_researcher.search.optuna.backend import OptunaAskTellBackend
from auto_researcher.search.optuna.components import (
    ApprovedOptunaComponentRegistry,
    NATIVE_PRUNERS,
    NATIVE_SAMPLERS,
    SamplerBuildContext,
    build_pruner,
    build_sampler,
)
from auto_researcher.search.optuna.models import (
    CategoricalParameterSpec,
    FloatParameterSpec,
    IntParameterSpec,
    OptimisationDirection,
    OptunaConditionSpec,
    OptunaConstraintSpec,
    OptunaDiagnosticsSpec,
    OptunaObjectiveSpec,
    OptunaPrunerSpec,
    OptunaSamplerSpec,
    OptunaStudySpec,
    OptunaTrialReference,
    OptunaTrialStatus,
)
from auto_researcher.search.optuna.naming import StudyIdentity, search_space_hash
from auto_researcher.search.optuna.operational import OptunaOperationalRecordStore
from auto_researcher.search.optuna.pruning import OptunaPruningAcknowledged
from auto_researcher.search.optuna.space import suggest_parameters
from auto_researcher.resources import (
    CourtesyResourceAdmissionPolicy,
    InMemoryResourceLeaseStore,
    ResourceBroker,
    ResourceCandidate,
    ResourceRequest,
    ResourceRequirement,
)
from auto_researcher.tasks.models import ExperimentMetadata


NOW = datetime(2026, 8, 14, 12, 0, tzinfo=UTC)


class EchoTask:
    def normalise_configuration(self, configuration):
        return dict(configuration)


def _spec(
    *,
    sampler: str = "random",
    multi: bool = False,
    constrained: bool = False,
    pruner: OptunaPrunerSpec | None = None,
    dynamic: bool = True,
) -> OptunaStudySpec:
    parameters = (
        CategoricalParameterSpec(name="optimizer", choices=("sgd", "adamw")),
        IntParameterSpec(name="depth", low=1, high=3),
        FloatParameterSpec(
            name="weight_decay",
            low=0.0,
            high=0.2,
            step=0.1,
            condition=(
                OptunaConditionSpec(parameter="optimizer", equals="adamw")
                if dynamic
                else None
            ),
        ),
    )
    objectives = (
        OptunaObjectiveSpec(
            name="score",
            direction=OptimisationDirection.MAXIMIZE,
            metric="score",
        ),
        *(
            (
                OptunaObjectiveSpec(
                    name="latency",
                    direction=OptimisationDirection.MINIMIZE,
                    metric="latency",
                ),
            )
            if multi
            else ()
        ),
    )
    return OptunaStudySpec(
        schema_version="2.0",
        task_id="echo",
        task_version="1.0",
        search_space_version="echo-space-v2",
        direction=OptimisationDirection.MAXIMIZE,
        parameters=parameters,
        fixed_configuration={},
        trial_budget=8,
        seed=17,
        sampler=OptunaSamplerSpec(type=sampler),
        n_startup_trials=1,
        objective_metric="score",
        objectives=objectives,
        constraints=(
            (
                OptunaConstraintSpec(
                    name="latency_limit",
                    metric="latency",
                    relation="LESS_THAN_OR_EQUAL",
                    threshold=2.0,
                ),
            )
            if constrained
            else ()
        ),
        pruner=pruner or OptunaPrunerSpec(type="none"),
        intermediate_reporting=(pruner is not None and pruner.type != "none"),
        diagnostics=OptunaDiagnosticsSpec(parameter_importance=True),
    )


def _identity(spec: OptunaStudySpec, name: str = "full-strength") -> StudyIdentity:
    return StudyIdentity(
        study_name=name,
        search_space_hash=search_space_hash(spec),
        attributes={
            "identity_schema_version": "2.0",
            "run_id": "run-v2",
            "request_id": "request-v2",
        },
    )


def _request() -> SearchRequest:
    return SearchRequest(
        request_id="request-v2",
        hypothesis_id="hypothesis-v2",
        search_type=SearchType.OPTUNA,
        target="native Optuna parity",
        search_space={},
        experiment_budget=8,
        rationale="test native public APIs",
    )


def _metadata() -> ExperimentMetadata:
    return ExperimentMetadata(
        evaluator_id="echo-evaluator",
        code_version="echo-code-v1",
        dataset_version="echo-data-v1",
        provenance=ProvenanceKind.SIMULATED,
    )


def _terminal_models(experiment, *, score: float, latency: float, compliant=True):
    evaluation = EvaluationResult(
        experiment_id=experiment.experiment_id,
        success=True,
        primary_score=score,
        metrics={"score": score, "latency": latency},
        constraint_results={"latency": compliant},
        evaluator_version="echo-evaluator-v1",
        provenance=ProvenanceKind.SIMULATED,
    )
    verification = VerificationResult(
        experiment_id=experiment.experiment_id,
        verified=True,
        claimed_score=score,
        measured_score=score,
        constraint_compliant=compliant,
        evidence_status=EvidenceStatus.INCONCLUSIVE,
        reasons=("deterministic",),
        provenance=ProvenanceKind.SIMULATED,
    )
    return evaluation, verification


def _prepare(backend: OptunaAskTellBackend, spec: OptunaStudySpec, name: str):
    identity = _identity(spec, name)
    backend.prepare_or_load_study(
        identity,
        spec,
        started_at=NOW,
        trial_budget=spec.trial_budget,
    )
    return identity


def test_native_sampler_registry_covers_exact_public_material_set() -> None:
    assert NATIVE_SAMPLERS == {
        "native_default",
        "tpe",
        "random",
        "cmaes",
        "gp",
        "nsgaii",
        "nsgaiii",
        "qmc",
        "grid",
        "brute_force",
    }
    storage = optuna.storages.InMemoryStorage()
    store = OptunaOperationalRecordStore(storage)
    registry = ApprovedOptunaComponentRegistry()
    for name in sorted(NATIVE_SAMPLERS - {"cmaes"}):
        spec = _spec(sampler=name, dynamic=name != "grid")
        component = build_sampler(
            spec.sampler_spec,
            SamplerBuildContext(spec, spec.seed, False, store),
            registry,
        )
        if name == "native_default":
            assert component is None
        else:
            assert isinstance(component, optuna.samplers.BaseSampler)
    if importlib.util.find_spec("cmaes") is None:
        spec = _spec(sampler="cmaes")
        with pytest.raises(RuntimeError, match="hpo-cmaes"):
            build_sampler(
                spec.sampler_spec,
                SamplerBuildContext(spec, spec.seed, False, store),
                registry,
            )


def test_approved_custom_sampler_and_pruner_registry_has_no_import_paths() -> None:
    registry = ApprovedOptunaComponentRegistry()
    registry.register_sampler(
        "approved-random",
        lambda _configuration, context: optuna.samplers.RandomSampler(
            seed=context.seed
        ),
    )
    registry.register_pruner(
        "approved-nop",
        lambda _configuration: optuna.pruners.NopPruner(),
    )
    spec = _spec(dynamic=False).model_copy(update={"schema_version": "1.0"})
    context = SamplerBuildContext(
        spec,
        spec.seed,
        False,
        OptunaOperationalRecordStore(optuna.storages.InMemoryStorage()),
    )
    assert isinstance(
        build_sampler(OptunaSamplerSpec(type="approved-random"), context, registry),
        optuna.samplers.RandomSampler,
    )
    assert isinstance(
        build_pruner(OptunaPrunerSpec(type="approved-nop"), registry),
        optuna.pruners.NopPruner,
    )


def test_custom_sampler_capabilities_default_to_fail_closed() -> None:
    registry = ApprovedOptunaComponentRegistry()
    registry.register_sampler(
        "conservative-random",
        lambda _configuration, context: optuna.samplers.RandomSampler(
            seed=context.seed
        ),
    )
    storage = optuna.storages.InMemoryStorage()
    store = OptunaOperationalRecordStore(storage)
    simple = _spec(dynamic=False).model_copy(update={"schema_version": "1.0"})
    assert isinstance(
        build_sampler(
            OptunaSamplerSpec(type="conservative-random"),
            SamplerBuildContext(simple, simple.seed, False, store),
            registry,
        ),
        optuna.samplers.RandomSampler,
    )
    incompatible = (
        (_spec(constrained=True, dynamic=False), False, "constraints"),
        (_spec(multi=True, dynamic=False), False, "multi_objective"),
        (_spec(dynamic=False), True, "shared_worker"),
        (_spec(dynamic=True), False, "dynamic_space"),
    )
    for study_spec, shared_workers, reason in incompatible:
        with pytest.raises(ValueError, match=reason):
            build_sampler(
                OptunaSamplerSpec(type="conservative-random"),
                SamplerBuildContext(
                    study_spec,
                    study_spec.seed,
                    shared_workers,
                    store,
                ),
                registry,
            )


def test_constraint_aware_custom_sampler_binds_and_invokes_durable_callback() -> None:
    calls: list[int] = []
    registry = ApprovedOptunaComponentRegistry()

    def factory(_configuration, context):
        callback = context.constraints_callback
        assert callback is not None

        def observed_callback(trial):
            calls.append(trial.number)
            return callback(trial)

        return optuna.samplers.TPESampler(
            seed=context.seed,
            n_startup_trials=0,
            constraints_func=observed_callback,
        )

    registry.register_sampler(
        "constraint-aware",
        factory,
        supports_constraints=True,
        supports_dynamic_space=True,
    )
    spec = _spec(constrained=True).model_copy(
        update={"sampler": OptunaSamplerSpec(type="constraint-aware")}
    )
    backend = OptunaAskTellBackend(
        optuna.storages.InMemoryStorage(),
        component_registry=registry,
    )
    identity = _prepare(backend, spec, "custom-constraint-callback")
    reference, _ = backend.ask_or_recover_trial(
        identity, spec, slot_index=0, asked_at=NOW
    )
    experiment = backend.create_experiment_spec(
        task=EchoTask(),
        metadata=_metadata(),
        spec=spec,
        request=_request(),
        reference=reference,
    )
    evaluation, verification = _terminal_models(
        experiment, score=0.7, latency=3.0, compliant=False
    )
    outcome = backend.tell_trial(
        spec=spec,
        reference=reference,
        experiment=experiment,
        evaluation=evaluation,
        verification=verification,
        reported_at=NOW,
    )
    assert outcome.status == OptunaTrialStatus.COMPLETE
    assert calls == [reference.trial_number]


def test_explicit_shared_worker_safe_custom_sampler_is_accepted() -> None:
    registry = ApprovedOptunaComponentRegistry()
    registry.register_sampler(
        "shared-random",
        lambda _configuration, context: optuna.samplers.RandomSampler(
            seed=context.seed
        ),
        shared_worker_safe=True,
        supports_dynamic_space=True,
    )
    spec = _spec()
    sampler = build_sampler(
        OptunaSamplerSpec(type="shared-random"),
        SamplerBuildContext(
            spec,
            spec.seed,
            True,
            OptunaOperationalRecordStore(optuna.storages.InMemoryStorage()),
        ),
        registry,
    )
    assert isinstance(sampler, optuna.samplers.RandomSampler)


def test_false_custom_sampler_combination_fails_before_study_ask() -> None:
    registry = ApprovedOptunaComponentRegistry()
    registry.register_sampler(
        "misdeclared-random",
        lambda _configuration, context: optuna.samplers.RandomSampler(
            seed=context.seed
        ),
    )
    spec = _spec(constrained=True).model_copy(
        update={"sampler": OptunaSamplerSpec(type="misdeclared-random")}
    )
    storage = optuna.storages.InMemoryStorage()
    backend = OptunaAskTellBackend(storage, component_registry=registry)
    with pytest.raises(ValueError, match="constraints_unsupported"):
        _prepare(backend, spec, "custom-fails-before-ask")
    assert optuna.study.get_all_study_names(storage=storage) == []


def test_multi_objective_trials_use_three_equivalent_resources_without_identity_leak() -> (
    None
):
    class ThreeGPUProvider:
        def candidates(self, _request):
            return tuple(
                ResourceCandidate(
                    resource_id=f"gpu:{index}",
                    resource_type="gpu",
                    equivalence_tags=frozenset({"equivalent-simulated-gpu"}),
                )
                for index in range(3)
            )

    spec = _spec(sampler="grid", multi=True, dynamic=False).model_copy(
        update={
            "parameters": (IntParameterSpec(name="depth", low=0, high=2),),
            "diagnostics": OptunaDiagnosticsSpec(),
        }
    )
    backend = OptunaAskTellBackend(optuna.storages.InMemoryStorage())
    identity = _prepare(backend, spec, "three-resource-pareto")
    # The standard local runtime deliberately permits one RUNNING trial. This
    # focused placement probe asks three native trials before concurrent work,
    # matching the explicit coordinated-worker execution shape.
    study = optuna.load_study(
        study_name=identity.study_name,
        storage=backend.storage,
        sampler=optuna.samplers.GridSampler({"depth": [0, 1, 2]}, seed=spec.seed),
    )
    references_list = []
    for index in range(3):
        trial = study.ask()
        parameters = suggest_parameters(trial, spec)
        trial.set_user_attr("slot_index", index)
        trial.set_user_attr("study_name", identity.study_name)
        references_list.append(
            OptunaTrialReference(
                study_name=identity.study_name,
                trial_number=trial.number,
                slot_index=index,
                parameters=parameters,
                status=OptunaTrialStatus.RUNNING,
            )
        )
    references = tuple(references_list)
    experiments = tuple(
        backend.create_experiment_spec(
            task=EchoTask(),
            metadata=_metadata(),
            spec=spec,
            request=_request(),
            reference=reference,
        )
        for reference in references
    )
    broker = ResourceBroker(
        ThreeGPUProvider(),
        CourtesyResourceAdmissionPolicy(),
        lease_store=InMemoryResourceLeaseStore(),
        poll_seconds=0.001,
    )
    barrier = threading.Barrier(3)
    active: set[str] = set()
    observed: set[str] = set()
    maximum_parallel = 0
    lock = threading.Lock()

    def evaluate(index: int):
        nonlocal maximum_parallel
        experiment = experiments[index]
        request = ResourceRequest(
            request_id=f"placement-{experiment.experiment_id}",
            requirements=(ResourceRequirement(resource_type="gpu"),),
            maximum_wait_seconds=2,
            equivalence_requirements=frozenset({"equivalent-simulated-gpu"}),
        )
        admission = broker.acquire(
            request,
            worker_id=f"worker-{index}",
            lease_ttl=timedelta(seconds=2),
        )
        assert admission.lease is not None
        resource_id = admission.lease.resource_id
        with lock:
            active.add(resource_id)
            observed.add(resource_id)
            maximum_parallel = max(maximum_parallel, len(active))
        barrier.wait(timeout=2)
        time.sleep(0.02)
        with lock:
            active.remove(resource_id)
        broker.release_lease(admission.lease.lease_id, worker_id=f"worker-{index}")
        depth = float(references[index].parameters["depth"])
        return _terminal_models(experiment, score=depth, latency=depth)

    with ThreadPoolExecutor(max_workers=3) as executor:
        terminal = tuple(executor.map(evaluate, range(3)))

    for reference, experiment, (evaluation, verification) in zip(
        references, experiments, terminal, strict=True
    ):
        backend.tell_trial(
            spec=spec,
            reference=reference,
            experiment=experiment,
            evaluation=evaluation,
            verification=verification,
            reported_at=NOW,
        )
    summary = backend.load_study_summary(identity, spec, 3)

    assert observed == {"gpu:0", "gpu:1", "gpu:2"}
    assert maximum_parallel == 3
    assert summary.pareto_trial_numbers == (0, 1, 2)
    assert all("gpu" not in reference.parameters for reference in references)
    assert all("gpu" not in experiment.configuration for experiment in experiments)


def _exercise_finite_sampler_exhaustion(sampler: str, study_name: str) -> None:
    spec = _spec(sampler=sampler, dynamic=False).model_copy(
        update={
            "parameters": (
                CategoricalParameterSpec(name="choice", choices=("a", "b")),
            ),
            "trial_budget": 2,
            "diagnostics": OptunaDiagnosticsSpec(),
        }
    )
    backend = OptunaAskTellBackend(optuna.storages.InMemoryStorage())
    identity = _prepare(backend, spec, study_name)
    for slot, score in enumerate((0.25, 0.75)):
        reference, _ = backend.ask_or_recover_trial(
            identity, spec, slot_index=slot, asked_at=NOW
        )
        experiment = backend.create_experiment_spec(
            task=EchoTask(),
            metadata=_metadata(),
            spec=spec,
            request=_request(),
            reference=reference,
        )
        evaluation, verification = _terminal_models(
            experiment,
            score=score,
            latency=1.0,
        )
        outcome = backend.tell_trial(
            spec=spec,
            reference=reference,
            experiment=experiment,
            evaluation=evaluation,
            verification=verification,
            reported_at=NOW,
        )
        assert outcome.status == OptunaTrialStatus.COMPLETE
        assert outcome.objective_values == (score,)
    study = backend._load_study(identity.study_name, spec)
    assert len(study.trials) == 2
    assert [trial.state for trial in study.trials] == [
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.COMPLETE,
    ]
    assert [tuple(trial.values or ()) for trial in study.trials] == [
        (0.25,),
        (0.75,),
    ]
    summary = backend.load_study_summary(identity, spec, 2)
    assert summary.trials_asked == summary.trial_budget == 2
    assert summary.trials_completed == 2


def test_grid_ask_tell_exhaustion_commits_native_state_and_values() -> None:
    _exercise_finite_sampler_exhaustion("grid", "grid-exhaustion")


def test_bruteforce_ask_tell_exhaustion_commits_native_state_and_values() -> None:
    _exercise_finite_sampler_exhaustion("brute_force", "bruteforce-exhaustion")


def test_exhaustion_adapter_does_not_swallow_unrelated_runtime_error() -> None:
    class BrokenStudy:
        message = "unrelated sampler failure"

        def tell(self, *_args, **_kwargs):
            raise RuntimeError(self.message)

    study = BrokenStudy()
    for message in (
        "unrelated sampler failure",
        "`Study.stop` is supposed to be invoked inside an objective function or a callback. extra",
    ):
        study.message = message
        with pytest.raises(RuntimeError, match="unrelated|extra"):
            OptunaAskTellBackend(optuna.storages.InMemoryStorage())._tell_public(
                study,
                0,
                1.0,
                state=optuna.trial.TrialState.COMPLETE,
            )


def test_native_pruner_factory_covers_exact_public_pruners() -> None:
    assert NATIVE_PRUNERS == {
        "native_default",
        "none",
        "median",
        "patient",
        "percentile",
        "successive_halving",
        "hyperband",
        "threshold",
        "wilcoxon",
    }
    options = {
        "percentile": {"percentile": 50.0},
        "patient": {"patience": 2},
        "threshold": {"upper": 1.0},
    }
    registry = ApprovedOptunaComponentRegistry()
    for name in sorted(NATIVE_PRUNERS):
        component = build_pruner(
            OptunaPrunerSpec(type=name, options=options.get(name, {})),
            registry,
        )
        if name == "native_default":
            assert component is None
        else:
            assert isinstance(component, optuna.pruners.BasePruner)


def test_dynamic_conditional_space_uses_native_trial_suggestions() -> None:
    spec = _spec(sampler="random")
    backend = OptunaAskTellBackend(optuna.storages.InMemoryStorage())
    identity = _prepare(backend, spec, "conditional-native")
    seen = set()
    for slot in range(8):
        reference, _ = backend.ask_or_recover_trial(
            identity,
            spec,
            slot_index=slot,
            asked_at=NOW,
        )
        seen.add(reference.parameters["optimizer"])
        assert ("weight_decay" in reference.parameters) is (
            reference.parameters["optimizer"] == "adamw"
        )
        experiment = backend.create_experiment_spec(
            task=EchoTask(),
            metadata=_metadata(),
            spec=spec,
            request=_request(),
            reference=reference,
        )
        evaluation, verification = _terminal_models(
            experiment,
            score=0.5 + slot / 100,
            latency=1.0,
        )
        backend.tell_trial(
            spec=spec,
            reference=reference,
            experiment=experiment,
            evaluation=evaluation,
            verification=verification,
            reported_at=NOW,
        )
    assert seen == {"sgd", "adamw"}


def test_native_multi_objective_pareto_has_no_scalar_winner() -> None:
    spec = _spec(sampler="nsgaii", multi=True)
    backend = OptunaAskTellBackend(optuna.storages.InMemoryStorage())
    identity = _prepare(backend, spec, "native-pareto")
    for slot, (score, latency) in enumerate(((0.8, 3.0), (0.7, 1.0), (0.6, 4.0))):
        reference, _ = backend.ask_or_recover_trial(
            identity, spec, slot_index=slot, asked_at=NOW
        )
        experiment = backend.create_experiment_spec(
            task=EchoTask(),
            metadata=_metadata(),
            spec=spec,
            request=_request(),
            reference=reference,
        )
        evaluation, verification = _terminal_models(
            experiment, score=score, latency=latency
        )
        outcome = backend.tell_trial(
            spec=spec,
            reference=reference,
            experiment=experiment,
            evaluation=evaluation,
            verification=verification,
            reported_at=NOW,
        )
        assert outcome.objective_value is None
        assert outcome.objective_values == (score, latency)
    summary = backend.load_study_summary(identity, spec, 3)
    assert summary.best_overall_trial_number is None
    assert summary.pareto_trial_numbers == (0, 1)


def test_native_constraint_vector_keeps_infeasible_trial_complete() -> None:
    spec = _spec(sampler="tpe", constrained=True)
    backend = OptunaAskTellBackend(optuna.storages.InMemoryStorage())
    identity = _prepare(backend, spec, "native-constraints")
    reference, _ = backend.ask_or_recover_trial(
        identity, spec, slot_index=0, asked_at=NOW
    )
    experiment = backend.create_experiment_spec(
        task=EchoTask(),
        metadata=_metadata(),
        spec=spec,
        request=_request(),
        reference=reference,
    )
    evaluation, verification = _terminal_models(
        experiment, score=0.9, latency=3.5, compliant=True
    )
    outcome = backend.tell_trial(
        spec=spec,
        reference=reference,
        experiment=experiment,
        evaluation=evaluation,
        verification=verification,
        reported_at=NOW,
    )
    assert outcome.status == OptunaTrialStatus.COMPLETE
    assert outcome.feasible is False
    assert outcome.constraint_values == (1.5,)
    frozen = backend._load_study(identity.study_name, spec).trials[0]
    assert frozen.state == optuna.trial.TrialState.COMPLETE
    assert tuple(frozen.system_attrs["constraints"]) == (1.5,)


def test_missing_objective_is_native_fail_without_fabricated_value() -> None:
    spec = _spec(multi=True)
    backend = OptunaAskTellBackend(optuna.storages.InMemoryStorage())
    identity = _prepare(backend, spec, "missing-objective")
    reference, _ = backend.ask_or_recover_trial(
        identity, spec, slot_index=0, asked_at=NOW
    )
    experiment = backend.create_experiment_spec(
        task=EchoTask(),
        metadata=_metadata(),
        spec=spec,
        request=_request(),
        reference=reference,
    )
    evaluation, verification = _terminal_models(experiment, score=0.9, latency=1.0)
    evaluation = evaluation.model_copy(update={"metrics": {"score": 0.9}})
    outcome = backend.tell_trial(
        spec=spec,
        reference=reference,
        experiment=experiment,
        evaluation=evaluation,
        verification=verification,
        reported_at=NOW,
    )
    assert outcome.status == OptunaTrialStatus.FAIL
    assert outcome.objective_values == ()
    assert outcome.objective_value is None


def test_native_cooperative_pruning_is_durable_and_distinct_from_fail() -> None:
    spec = _spec(pruner=OptunaPrunerSpec(type="threshold", options={"upper": 0.5}))
    backend = OptunaAskTellBackend(optuna.storages.InMemoryStorage())
    identity = _prepare(backend, spec, "native-pruning")
    reference, _ = backend.ask_or_recover_trial(
        identity, spec, slot_index=0, asked_at=NOW
    )
    reporter = backend.intermediate_reporter(spec=spec, reference=reference)
    assert reporter.report(0.8, 3) is True
    with pytest.raises(OptunaPruningAcknowledged):
        reporter.acknowledge_pruning()
    outcome = backend.prune_trial(spec=spec, reference=reference, reported_at=NOW)
    assert outcome.status == OptunaTrialStatus.PRUNED
    assert outcome.objective_values == ()
    assert outcome.pruned_at_step == 3
    assert outcome.intermediate_values == {3: 0.8}


def _interrupted_pruning_case(
    name: str,
    *,
    value: float,
    acknowledge: bool,
):
    spec = _spec(pruner=OptunaPrunerSpec(type="threshold", options={"upper": 0.5}))
    storage = optuna.storages.InMemoryStorage()
    owner = OptunaAskTellBackend(storage)
    identity = _prepare(owner, spec, name)
    reference, _ = owner.ask_or_recover_trial(
        identity, spec, slot_index=0, asked_at=NOW
    )
    reporter = owner.intermediate_reporter(spec=spec, reference=reference)
    requested = reporter.report(value, 3)
    if acknowledge:
        assert requested is True
        with pytest.raises(OptunaPruningAcknowledged):
            reporter.acknowledge_pruning()
    restarted = OptunaAskTellBackend(storage)
    outcome = restarted.recover_interrupted_reporting_trial(
        spec=spec,
        reference=reference,
        reported_at=NOW,
    )
    return owner, restarted, reporter, spec, reference, outcome


def test_prune_request_crash_before_acknowledgement_recovers_fail() -> None:
    owner, _, reporter, spec, reference, outcome = _interrupted_pruning_case(
        "prune-request-only-crash",
        value=0.8,
        acknowledge=False,
    )
    assert outcome.status == OptunaTrialStatus.FAIL
    assert outcome.objective_values == ()
    with pytest.raises(OptunaPruningAcknowledged):
        reporter.acknowledge_pruning()
    with pytest.raises(RuntimeError, match="conflicting report"):
        owner.prune_trial(spec=spec, reference=reference, reported_at=NOW)
    frozen = owner._load_study(reference.study_name, spec).trials[0]
    assert frozen.state == optuna.trial.TrialState.FAIL
    assert frozen.values is None


def test_acknowledged_prune_crash_before_tell_recovers_pruned() -> None:
    _, _, _, _, _, outcome = _interrupted_pruning_case(
        "acknowledged-prune-crash",
        value=0.8,
        acknowledge=True,
    )
    assert outcome.status == OptunaTrialStatus.PRUNED
    assert outcome.pruned_at_step == 3
    assert outcome.objective_values == ()


def test_non_pruning_report_crash_recovers_fail() -> None:
    _, _, _, _, _, outcome = _interrupted_pruning_case(
        "non-pruning-report-crash",
        value=0.4,
        acknowledge=False,
    )
    assert outcome.status == OptunaTrialStatus.FAIL
    assert outcome.objective_values == ()


def test_acknowledged_prune_replay_is_idempotently_pruned() -> None:
    owner, restarted, reporter, spec, reference, first = _interrupted_pruning_case(
        "acknowledged-prune-replay",
        value=0.8,
        acknowledge=True,
    )
    with pytest.raises(OptunaPruningAcknowledged):
        reporter.acknowledge_pruning()
    second = restarted.recover_interrupted_reporting_trial(
        spec=spec,
        reference=reference,
        reported_at=NOW,
    )
    third = owner.prune_trial(spec=spec, reference=reference, reported_at=NOW)
    assert first == second == third
    assert first.status == OptunaTrialStatus.PRUNED
    assert first.objective_values == ()


def test_multi_objective_pruning_is_rejected_as_absent_upstream() -> None:
    with pytest.raises(ValueError, match="multi_objective_pruning_not_supported"):
        _spec(
            multi=True,
            pruner=OptunaPrunerSpec(type="median"),
        )


def test_v2_scientific_identity_deduplicates_trials_without_deleting_them() -> None:
    spec = _spec(dynamic=False)
    backend = OptunaAskTellBackend(optuna.storages.InMemoryStorage())
    first = backend.create_experiment_spec(
        task=EchoTask(),
        metadata=_metadata(),
        spec=spec,
        request=_request(),
        reference=optuna_reference(0),
    )
    second = backend.create_experiment_spec(
        task=EchoTask(),
        metadata=_metadata(),
        spec=spec,
        request=_request(),
        reference=optuna_reference(9),
    )
    assert first.experiment_id == second.experiment_id


def optuna_reference(number: int):
    from auto_researcher.search.optuna.models import OptunaTrialReference

    return OptunaTrialReference(
        study_name="semantic-study",
        trial_number=number,
        slot_index=number,
        parameters={"optimizer": "sgd", "depth": 2, "weight_decay": 0.1},
        status=OptunaTrialStatus.RUNNING,
    )


def test_diagnostics_are_explicitly_operational_not_scientific_evidence() -> None:
    spec = _spec(dynamic=False)
    backend = OptunaAskTellBackend(optuna.storages.InMemoryStorage())
    identity = _prepare(backend, spec, "diagnostics")
    for slot in range(3):
        reference, _ = backend.ask_or_recover_trial(
            identity, spec, slot_index=slot, asked_at=NOW
        )
        experiment = backend.create_experiment_spec(
            task=EchoTask(),
            metadata=_metadata(),
            spec=spec,
            request=_request(),
            reference=reference,
        )
        evaluation, verification = _terminal_models(
            experiment, score=0.5 + slot / 10, latency=1.0
        )
        backend.tell_trial(
            spec=spec,
            reference=reference,
            experiment=experiment,
            evaluation=evaluation,
            verification=verification,
            reported_at=NOW,
        )
    diagnostics = backend.study_diagnostics(identity.study_name, spec)
    assert diagnostics.completed_trials == 3
    assert diagnostics.best_trial_number is not None
    assert diagnostics.epistemic_status == "OPERATIONAL_SEARCH_DIAGNOSTIC"
