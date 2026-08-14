from __future__ import annotations

import multiprocessing
import os
import time
from datetime import datetime, timedelta, timezone
from queue import Empty
from uuid import uuid4

import pytest

pytestmark = [pytest.mark.hpo, pytest.mark.postgresql]


def _database_url() -> str:
    value = os.environ.get("AUTO_RESEARCHER_TEST_POSTGRESQL_URL")
    if not value:
        pytest.skip("set AUTO_RESEARCHER_TEST_POSTGRESQL_URL for the PostgreSQL gate")
    return value


def _spec():
    from auto_researcher.search.optuna.models import (
        FloatParameterSpec,
        OptimisationDirection,
        OptunaStudySpec,
    )

    return OptunaStudySpec(
        schema_version="1.0",
        task_id="postgres-concurrency",
        task_version="1",
        search_space_version="1",
        direction=OptimisationDirection.MAXIMIZE,
        parameters=(FloatParameterSpec(name="x", low=-10.0, high=10.0),),
        trial_budget=12,
        seed=404,
        n_startup_trials=6,
        objective_metric="score",
    )


def _identity(study_name: str):
    from auto_researcher.search.optuna.naming import StudyIdentity

    return StudyIdentity(
        study_name=study_name,
        search_space_hash="postgres-concurrency-space",
        attributes={
            "run_id": f"run-{study_name}",
            "request_id": f"request-{study_name}",
            "task_id": "postgres-concurrency",
        },
    )


def _trial_worker(
    study_name: str,
    worker_id: str,
    ready,
    release,
    results,
) -> None:
    import optuna
    from optuna.storages import RDBStorage
    from optuna.trial import TrialState
    from sqlalchemy import create_engine

    from auto_researcher.search.optuna.backend import OptunaAskTellBackend
    from auto_researcher.search.optuna.coordination import (
        PostgresOptunaCoordination,
        SharedTrialBudgetExhausted,
    )

    url = os.environ["AUTO_RESEARCHER_TEST_POSTGRESQL_URL"]
    engine = create_engine(url, pool_pre_ping=True)
    coordination = PostgresOptunaCoordination(engine)
    storage = RDBStorage(url)
    backend = OptunaAskTellBackend(
        storage,
        shared_workers=True,
        coordination=coordination,
        worker_id=worker_id,
    )
    identity = _identity(study_name)
    spec = _spec()
    first = True
    try:
        while True:
            try:
                reference, claim = backend.ask_and_claim_trial(
                    identity,
                    spec,
                    trial_budget=12,
                    claim_ttl=timedelta(seconds=30),
                )
            except SharedTrialBudgetExhausted:
                break
            if first:
                ready.put((worker_id, reference.trial_number))
                release.wait(timeout=20)
                first = False
            score = -abs(float(reference.parameters["x"]) - 1.25)
            digest = f"report-{reference.trial_number}-{score}"
            claim = coordination.record_report(
                claim,
                report_digest=digest,
            )
            study = optuna.load_study(study_name=study_name, storage=storage)
            coordination.run_owned_and_release(
                claim,
                lambda: study.tell(
                    reference.trial_number,
                    score,
                    state=TrialState.COMPLETE,
                    skip_if_finished=True,
                ),
            )
            results.put((worker_id, reference.trial_number, reference.parameters["x"]))
    finally:
        engine.dispose()
        storage.engine.dispose()


def _lost_worker(study_name: str, output) -> None:
    from optuna.storages import RDBStorage
    from sqlalchemy import create_engine

    from auto_researcher.search.optuna.backend import OptunaAskTellBackend
    from auto_researcher.search.optuna.coordination import PostgresOptunaCoordination

    url = os.environ["AUTO_RESEARCHER_TEST_POSTGRESQL_URL"]
    engine = create_engine(url)
    storage = RDBStorage(url)
    coordination = PostgresOptunaCoordination(engine)
    backend = OptunaAskTellBackend(
        storage,
        shared_workers=True,
        coordination=coordination,
        worker_id="lost-worker",
    )
    reference, claim = backend.ask_and_claim_trial(
        _identity(study_name),
        _spec(),
        trial_budget=12,
        claim_ttl=timedelta(seconds=1),
    )
    output.put((reference.trial_number, claim.model_dump(mode="json")))
    # Deliberately disappear without tell/release/engine cleanup.


def _lease_racer(request_id: str, worker_id: str, start, output) -> None:
    from sqlalchemy import create_engine

    from auto_researcher.resources import (
        AdmissionClass,
        PostgresResourceLeaseStore,
        ResourceCandidate,
        ResourceLeaseConflict,
        ResourceRequest,
        ResourceRequirement,
    )

    engine = create_engine(os.environ["AUTO_RESEARCHER_TEST_POSTGRESQL_URL"])
    store = PostgresResourceLeaseStore(engine)
    request = ResourceRequest(
        request_id=request_id,
        requirements=(ResourceRequirement(resource_type="gpu"),),
        admission_class=AdmissionClass.PRIMARY,
    )
    try:
        start.wait(timeout=20)
        lease = store.acquire(
            request,
            ResourceCandidate(resource_id="gpu:race", resource_type="gpu"),
            worker_id=worker_id,
            now=datetime.now(timezone.utc),
            ttl=timedelta(minutes=1),
        )
        output.put(("acquired", lease.model_dump(mode="json")))
    except ResourceLeaseConflict:
        output.put(("conflict", worker_id))
    finally:
        engine.dispose()


class _PassthroughTask:
    def normalise_configuration(self, configuration):
        return configuration


class _DelayedCPUProvider:
    def __init__(self, delay_seconds: float, resource_id: str) -> None:
        self.started = time.monotonic()
        self.delay_seconds = delay_seconds
        self.resource_id = resource_id

    def candidates(self, request):
        del request
        if time.monotonic() - self.started < self.delay_seconds:
            return ()
        from auto_researcher.resources import ResourceCandidate

        return (
            ResourceCandidate(
                resource_id=self.resource_id,
                resource_type="cpu",
            ),
        )


def _worker_domain_inputs(study_name: str):
    from auto_researcher.contracts.enums import ProvenanceKind, SearchType
    from auto_researcher.contracts.models import SearchRequest
    from auto_researcher.tasks.models import ExperimentMetadata

    return (
        _PassthroughTask(),
        ExperimentMetadata(
            evaluator_id="deterministic-cpu",
            code_version="integration-test",
            dataset_version="integration-test",
            provenance=ProvenanceKind.SIMULATED,
        ),
        SearchRequest(
            request_id=f"worker-request-{study_name}",
            hypothesis_id="worker-heartbeat",
            search_type=SearchType.OPTUNA,
            target="postgres-worker-seam",
            search_space={"trial_budget": 12, "seed": 404},
            experiment_budget=12,
            rationale="exercise coordinated worker lifecycle",
        ),
    )


def _evaluation_for(context, *, score: float = 1.0):
    from auto_researcher.contracts.enums import ProvenanceKind
    from auto_researcher.contracts.models import EvaluationResult

    return EvaluationResult(
        experiment_id=context.experiment.experiment_id,
        success=True,
        primary_score=score,
        metrics={"score": score},
        constraint_results={"deterministic": True},
        evaluator_version="integration-test",
        provenance=ProvenanceKind.SIMULATED,
    )


def _verify(experiment, evaluation):
    from auto_researcher.contracts.enums import EvidenceStatus, ProvenanceKind
    from auto_researcher.contracts.models import VerificationResult

    return VerificationResult(
        experiment_id=experiment.experiment_id,
        verified=True,
        claimed_score=evaluation.primary_score,
        measured_score=evaluation.primary_score,
        constraint_compliant=True,
        evidence_status=EvidenceStatus.INCONCLUSIVE,
        reasons=(),
        provenance=ProvenanceKind.SIMULATED,
    )


def _context():
    return multiprocessing.get_context("spawn")


def _prepare_study(study_name: str):
    from datetime import datetime, timezone

    from optuna.storages import RDBStorage
    from sqlalchemy import create_engine

    from auto_researcher.search.optuna.backend import OptunaAskTellBackend
    from auto_researcher.search.optuna.coordination import PostgresOptunaCoordination

    url = _database_url()
    storage = RDBStorage(url)
    engine = create_engine(url)
    coordination = PostgresOptunaCoordination(engine)
    backend = OptunaAskTellBackend(
        storage,
        shared_workers=True,
        coordination=coordination,
        worker_id="preparer",
    )
    backend.prepare_or_load_study(
        _identity(study_name),
        _spec(),
        started_at=datetime.now(timezone.utc),
        trial_budget=12,
    )
    return storage, engine, coordination


def test_typed_storage_factory_resolves_runtime_secret_without_exposing_it():
    from sqlalchemy.engine import make_url

    from auto_researcher.search.optuna.storage import (
        PostgreSQLStorageConfiguration,
        postgresql_storage,
    )
    from auto_researcher.secrets.models import SecretProviderKind, SecretReference
    from auto_researcher.secrets.providers import EnvironmentSecretProvider

    parsed = make_url(_database_url())
    secret_value = parsed.password or "ephemeral-trust-value"
    configuration = PostgreSQLStorageConfiguration(
        host=str(parsed.host),
        port=int(parsed.port or 5432),
        database=str(parsed.database),
        username=str(parsed.username),
        password=SecretReference(
            logical_name="test_postgresql_password",
            provider=SecretProviderKind.ENVIRONMENT,
            provider_identifier="AR_TEST_POSTGRES_PASSWORD",
        ),
        alias="integration-gate",
    )

    handle = postgresql_storage(
        configuration,
        secret_provider=EnvironmentSecretProvider(
            {"AR_TEST_POSTGRES_PASSWORD": secret_value}
        ),
    )
    try:
        assert handle.safe_reference == "postgresql:integration-gate"
        assert secret_value not in repr(handle)
        assert secret_value not in handle.safe_reference
    finally:
        handle.close()


def test_three_processes_share_native_study_budget_and_observe_parallel_running():
    import optuna
    import psycopg
    from optuna.trial import TrialState

    study_name = f"ar-postgres-{uuid4()}"
    storage, engine, _ = _prepare_study(study_name)
    context = _context()
    ready = context.Queue()
    results = context.Queue()
    release = context.Event()
    workers = [
        context.Process(
            target=_trial_worker,
            args=(study_name, f"worker-{index}", ready, release, results),
        )
        for index in range(3)
    ]
    try:
        for worker in workers:
            worker.start()
        initial = [ready.get(timeout=30) for _ in workers]
        study = optuna.load_study(study_name=study_name, storage=storage)
        running = [
            trial
            for trial in study.get_trials(deepcopy=True)
            if trial.state == TrialState.RUNNING
        ]
        assert len(running) >= 3
        assert len({trial_number for _, trial_number in initial}) == 3
        release.set()
        for worker in workers:
            worker.join(timeout=60)
            assert worker.exitcode == 0
        completed = []
        while True:
            try:
                completed.append(results.get_nowait())
            except Empty:
                break
        trials = study.get_trials(deepcopy=True)
        assert len(trials) == 12
        assert len({trial.number for trial in trials}) == 12
        assert all(trial.state == TrialState.COMPLETE for trial in trials)
        assert len(completed) == 12
        assert len({number for _, number, _ in completed}) == 12
        assert len({value for _, _, value in completed}) > 3
        assert psycopg.__version__
        with engine.connect() as connection:
            version = connection.exec_driver_sql("SHOW server_version").scalar_one()
        assert version
    finally:
        release.set()
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
        optuna.delete_study(study_name=study_name, storage=storage)
        storage.engine.dispose()
        engine.dispose()


def test_lost_worker_claim_expires_and_trial_is_publicly_reconciled():
    import optuna
    from optuna.trial import TrialState

    from auto_researcher.search.optuna.backend import OptunaAskTellBackend
    from auto_researcher.search.optuna.coordination import TrialClaim, WorkerClaimLost

    study_name = f"ar-loss-{uuid4()}"
    storage, engine, coordination = _prepare_study(study_name)
    context = _context()
    output = context.Queue()
    process = context.Process(target=_lost_worker, args=(study_name, output))
    try:
        process.start()
        trial_number, raw_claim = output.get(timeout=30)
        old_claim = TrialClaim.model_validate(raw_claim)
        process.join(timeout=30)
        assert process.exitcode == 0
        time.sleep(1.2)
        recovery_claim = coordination.take_over_stale(
            study_name=study_name,
            trial_number=trial_number,
            recovery_worker_id="recovery-worker",
            ttl=timedelta(seconds=30),
        )
        with pytest.raises(WorkerClaimLost):
            coordination.assert_owner(old_claim)
        backend = OptunaAskTellBackend(
            storage,
            shared_workers=True,
            coordination=coordination,
            worker_id="recovery-worker",
        )
        backend.fail_claimed_trial(claim=recovery_claim, spec=_spec())
        trial = optuna.load_study(study_name=study_name, storage=storage).get_trials()[
            trial_number
        ]
        assert trial.state == TrialState.FAIL

        replacement, replacement_claim = backend.ask_and_claim_trial(
            _identity(study_name),
            _spec(),
            trial_budget=12,
            claim_ttl=timedelta(seconds=30),
        )
        study = optuna.load_study(study_name=study_name, storage=storage)
        coordination.run_owned_and_release(
            replacement_claim,
            lambda: study.tell(
                replacement.trial_number,
                1.0,
                state=TrialState.COMPLETE,
                skip_if_finished=True,
            ),
        )
        assert study.get_trials()[replacement.trial_number].state == TrialState.COMPLETE
    finally:
        if process.is_alive():
            process.terminate()
        optuna.delete_study(study_name=study_name, storage=storage)
        storage.engine.dispose()
        engine.dispose()


def test_unclaimed_ask_crash_window_waits_for_grace_then_reconciles():
    import optuna
    from optuna.trial import TrialState

    from auto_researcher.search.optuna.backend import OptunaAskTellBackend
    from auto_researcher.search.optuna.distributions import fixed_distributions

    study_name = f"ar-orphan-{uuid4()}"
    storage, engine, coordination = _prepare_study(study_name)
    study = optuna.load_study(study_name=study_name, storage=storage)
    orphan = study.ask(fixed_distributions=fixed_distributions(_spec().parameters))
    backend = OptunaAskTellBackend(
        storage,
        shared_workers=True,
        coordination=coordination,
        worker_id="recovery-worker",
    )
    try:
        immediate = backend.reconcile_unclaimed_trials(
            identity=_identity(study_name),
            spec=_spec(),
            orphan_grace=timedelta(minutes=1),
            now=datetime.now(timezone.utc),
        )
        assert immediate == ()

        reconciled = backend.reconcile_unclaimed_trials(
            identity=_identity(study_name),
            spec=_spec(),
            orphan_grace=timedelta(minutes=1),
            now=datetime.now(timezone.utc) + timedelta(minutes=5),
        )
        assert reconciled == (orphan.number,)
        assert study.get_trials()[orphan.number].state == TrialState.FAIL
    finally:
        optuna.delete_study(study_name=study_name, storage=storage)
        storage.engine.dispose()
        engine.dispose()


def test_unclaimed_orphan_prefers_database_asked_at_attribute():
    import optuna
    from optuna.trial import TrialState

    from auto_researcher.search.optuna.backend import OptunaAskTellBackend
    from auto_researcher.search.optuna.distributions import fixed_distributions

    study_name = f"ar-asked-at-orphan-{uuid4()}"
    storage, engine, coordination = _prepare_study(study_name)
    study = optuna.load_study(study_name=study_name, storage=storage)
    orphan = study.ask(fixed_distributions=fixed_distributions(_spec().parameters))
    now = datetime.now(timezone.utc)
    orphan.set_user_attr("asked_at", (now - timedelta(minutes=5)).isoformat())
    backend = OptunaAskTellBackend(
        storage,
        shared_workers=True,
        coordination=coordination,
        worker_id="recovery-worker",
        worker_session_id="asked-at-recovery-session",
    )
    try:
        reconciled = backend.reconcile_unclaimed_trials(
            identity=_identity(study_name),
            spec=_spec(),
            orphan_grace=timedelta(minutes=1),
            now=now,
        )
        assert reconciled == (orphan.number,)
        assert study.get_trials()[orphan.number].state == TrialState.FAIL
    finally:
        optuna.delete_study(study_name=study_name, storage=storage)
        storage.engine.dispose()
        engine.dispose()


def test_coordinated_worker_heartbeats_long_evaluator_and_completes():
    import optuna
    from optuna.trial import TrialState

    from auto_researcher.search.optuna.backend import OptunaAskTellBackend
    from auto_researcher.search.optuna.coordination import WorkerClaimConflict
    from auto_researcher.search.optuna.worker import CoordinatedOptunaWorker

    study_name = f"ar-worker-heartbeat-{uuid4()}"
    storage, engine, coordination = _prepare_study(study_name)
    backend = OptunaAskTellBackend(
        storage,
        shared_workers=True,
        coordination=coordination,
        worker_id="heartbeat-worker",
        worker_session_id="heartbeat-worker-session",
    )
    task, metadata, request = _worker_domain_inputs(study_name)
    verifier_called = False

    def evaluator(context):
        time.sleep(1.3)
        with pytest.raises(WorkerClaimConflict, match="worker_claim_not_stale"):
            coordination.take_over_stale(
                study_name=study_name,
                trial_number=context.trial.trial_number,
                recovery_worker_id="takeover-worker",
                ttl=timedelta(seconds=1),
            )
        time.sleep(1.0)
        return _evaluation_for(context)

    def verifier(experiment, evaluation):
        nonlocal verifier_called
        verifier_called = True
        return _verify(experiment, evaluation)

    worker = CoordinatedOptunaWorker(
        backend=backend,
        identity=_identity(study_name),
        study_spec=_spec(),
        trial_budget=12,
        claim_ttl=timedelta(seconds=1),
        claim_heartbeat_interval=timedelta(milliseconds=200),
        task=task,
        metadata=metadata,
        search_request=request,
        evaluator=evaluator,
        verifier=verifier,
    )
    try:
        result = worker.run_one()
        frozen = optuna.load_study(
            study_name=study_name,
            storage=storage,
        ).get_trials()[result.reference.trial_number]
        durable_claim = coordination.claim_for_trial(
            study_name,
            result.reference.trial_number,
        )
        assert verifier_called is True
        assert frozen.state == TrialState.COMPLETE
        assert durable_claim is not None and durable_claim.released_at is not None
        assert result.claim.heartbeat_at > result.claim.claimed_at
        assert durable_claim.heartbeat_at == result.claim.heartbeat_at
    finally:
        optuna.delete_study(study_name=study_name, storage=storage)
        storage.engine.dispose()
        engine.dispose()


def test_coordinated_worker_surfaces_lost_heartbeat_without_verify_or_tell():
    import optuna
    from optuna.trial import TrialState

    from auto_researcher.search.optuna.backend import OptunaAskTellBackend
    from auto_researcher.search.optuna.coordination import WorkerClaimLost
    from auto_researcher.search.optuna.worker import CoordinatedOptunaWorker

    study_name = f"ar-worker-heartbeat-loss-{uuid4()}"
    storage, engine, coordination = _prepare_study(study_name)
    backend = OptunaAskTellBackend(
        storage,
        shared_workers=True,
        coordination=coordination,
        worker_id="losing-worker",
        worker_session_id="losing-worker-session",
    )
    task, metadata, request = _worker_domain_inputs(study_name)
    verifier_called = False
    original_heartbeat = coordination.heartbeat
    heartbeat_calls = 0

    def heartbeat_then_take_over(claim, *, ttl):
        nonlocal heartbeat_calls
        heartbeat_calls += 1
        if heartbeat_calls == 2:
            time.sleep(0.35)
            coordination.take_over_stale(
                study_name=claim.study_name,
                trial_number=claim.trial_number,
                recovery_worker_id="recovery-worker",
                ttl=timedelta(seconds=2),
            )
        return original_heartbeat(claim, ttl=ttl)

    coordination.heartbeat = heartbeat_then_take_over

    def evaluator(context):
        time.sleep(0.8)
        return _evaluation_for(context)

    def verifier(experiment, evaluation):
        nonlocal verifier_called
        verifier_called = True
        return _verify(experiment, evaluation)

    worker = CoordinatedOptunaWorker(
        backend=backend,
        identity=_identity(study_name),
        study_spec=_spec(),
        trial_budget=12,
        claim_ttl=timedelta(milliseconds=300),
        claim_heartbeat_interval=timedelta(milliseconds=50),
        task=task,
        metadata=metadata,
        search_request=request,
        evaluator=evaluator,
        verifier=verifier,
    )
    try:
        with pytest.raises(WorkerClaimLost, match="worker_claim_lost_or_stale"):
            worker.run_one()
        study = optuna.load_study(study_name=study_name, storage=storage)
        frozen = study.get_trials()[0]
        current_claim = coordination.claim_for_trial(study_name, frozen.number)
        assert verifier_called is False
        assert frozen.state == TrialState.RUNNING
        assert current_claim is not None
        assert current_claim.worker_id == "recovery-worker"
        assert current_claim.released_at is None

        recovery_backend = OptunaAskTellBackend(
            storage,
            shared_workers=True,
            coordination=coordination,
            worker_id="recovery-worker",
            worker_session_id="recovery-session",
        )
        recovery_backend.fail_claimed_trial(claim=current_claim, spec=_spec())
    finally:
        coordination.heartbeat = original_heartbeat
        optuna.delete_study(study_name=study_name, storage=storage)
        storage.engine.dispose()
        engine.dispose()


def test_coordinated_worker_heartbeats_while_waiting_for_resource():
    import optuna
    from optuna.trial import TrialState

    from auto_researcher.resources import (
        AdmissionClass,
        CourtesyResourceAdmissionPolicy,
        PostgresResourceLeaseStore,
        ResourceBroker,
        ResourceRequest,
        ResourceRequirement,
    )
    from auto_researcher.search.optuna.backend import OptunaAskTellBackend
    from auto_researcher.search.optuna.worker import CoordinatedOptunaWorker

    study_name = f"ar-resource-wait-heartbeat-{uuid4()}"
    request_id = f"resource-wait-{uuid4()}"
    storage, engine, coordination = _prepare_study(study_name)
    lease_store = PostgresResourceLeaseStore(engine)
    backend = OptunaAskTellBackend(
        storage,
        shared_workers=True,
        coordination=coordination,
        worker_id="resource-wait-worker",
        worker_session_id="resource-wait-session",
    )
    provider = _DelayedCPUProvider(1.4, f"cpu:delayed-{uuid4()}")
    broker = ResourceBroker(
        provider,
        CourtesyResourceAdmissionPolicy(),
        lease_store=lease_store,
        poll_seconds=0.05,
    )
    task, metadata, search_request = _worker_domain_inputs(study_name)
    resource_request = ResourceRequest(
        request_id=request_id,
        requirements=(ResourceRequirement(resource_type="cpu"),),
        admission_class=AdmissionClass.PRIMARY,
        maximum_wait_seconds=5,
    )
    worker = CoordinatedOptunaWorker(
        backend=backend,
        identity=_identity(study_name),
        study_spec=_spec(),
        trial_budget=12,
        claim_ttl=timedelta(seconds=1),
        claim_heartbeat_interval=timedelta(milliseconds=200),
        task=task,
        metadata=metadata,
        search_request=search_request,
        evaluator=_evaluation_for,
        verifier=_verify,
        resource_broker=broker,
        resource_request_factory=lambda reference, experiment: resource_request,
        resource_lease_ttl=timedelta(seconds=1),
        resource_heartbeat_interval=timedelta(milliseconds=200),
    )
    try:
        result = worker.run_one()
        frozen = optuna.load_study(
            study_name=study_name,
            storage=storage,
        ).get_trials()[result.reference.trial_number]
        assert time.monotonic() - provider.started >= 1.4
        assert result.claim.heartbeat_at > result.claim.claimed_at
        assert result.resource_id == provider.resource_id
        assert frozen.state == TrialState.COMPLETE
    finally:
        from sqlalchemy import text

        with engine.begin() as connection:
            connection.execute(
                text("DELETE FROM ar_resource_lease WHERE request_id = :request_id"),
                {"request_id": request_id},
            )
        optuna.delete_study(study_name=study_name, storage=storage)
        storage.engine.dispose()
        engine.dispose()


def test_coordinated_worker_renews_resource_lease_during_long_evaluator():
    import optuna
    from optuna.trial import TrialState
    from sqlalchemy import text

    from auto_researcher.resources import (
        AdmissionClass,
        CourtesyResourceAdmissionPolicy,
        PostgresResourceLeaseStore,
        ResourceBroker,
        ResourceCandidate,
        ResourceLeaseConflict,
        ResourceRequest,
        ResourceRequirement,
    )
    from auto_researcher.search.optuna.backend import OptunaAskTellBackend
    from auto_researcher.search.optuna.worker import CoordinatedOptunaWorker

    study_name = f"ar-resource-renew-heartbeat-{uuid4()}"
    request_id = f"resource-renew-{uuid4()}"
    contender_request_id = f"resource-contender-{uuid4()}"
    resource_id = f"cpu:renew-{uuid4()}"
    storage, engine, coordination = _prepare_study(study_name)
    lease_store = PostgresResourceLeaseStore(engine)
    backend = OptunaAskTellBackend(
        storage,
        shared_workers=True,
        coordination=coordination,
        worker_id="resource-renew-worker",
        worker_session_id="resource-renew-session",
    )
    broker = ResourceBroker(
        _DelayedCPUProvider(0, resource_id),
        CourtesyResourceAdmissionPolicy(),
        lease_store=lease_store,
        poll_seconds=0.05,
    )
    task, metadata, search_request = _worker_domain_inputs(study_name)
    resource_request = ResourceRequest(
        request_id=request_id,
        requirements=(ResourceRequirement(resource_type="cpu"),),
        admission_class=AdmissionClass.PRIMARY,
    )
    contender_request = ResourceRequest(
        request_id=contender_request_id,
        requirements=(ResourceRequirement(resource_type="cpu"),),
        admission_class=AdmissionClass.PRIMARY,
    )
    observed_lease_id = None

    def evaluator(context):
        nonlocal observed_lease_id
        assert context.resource_admission is not None
        assert context.resource_admission.lease is not None
        initial = context.resource_admission.lease
        observed_lease_id = initial.lease_id
        time.sleep(1.2)
        active = lease_store.active_for(resource_id, now=datetime.now(timezone.utc))
        assert active is not None
        assert active.lease_id == initial.lease_id
        assert active.heartbeat_at > initial.heartbeat_at
        with pytest.raises(ResourceLeaseConflict):
            lease_store.acquire(
                contender_request,
                ResourceCandidate(resource_id=resource_id, resource_type="cpu"),
                worker_id="resource-contender",
                now=datetime.now(timezone.utc),
                ttl=timedelta(seconds=1),
            )
        time.sleep(0.4)
        return _evaluation_for(context)

    worker = CoordinatedOptunaWorker(
        backend=backend,
        identity=_identity(study_name),
        study_spec=_spec(),
        trial_budget=12,
        claim_ttl=timedelta(seconds=1),
        claim_heartbeat_interval=timedelta(milliseconds=200),
        task=task,
        metadata=metadata,
        search_request=search_request,
        evaluator=evaluator,
        verifier=_verify,
        resource_broker=broker,
        resource_request_factory=lambda reference, experiment: resource_request,
        resource_lease_ttl=timedelta(milliseconds=600),
        resource_heartbeat_interval=timedelta(milliseconds=100),
    )
    try:
        result = worker.run_one()
        frozen = optuna.load_study(
            study_name=study_name,
            storage=storage,
        ).get_trials()[result.reference.trial_number]
        with engine.connect() as connection:
            row = connection.execute(
                text(
                    "SELECT lease_id::text, acquired_at, heartbeat_at, released_at "
                    "FROM ar_resource_lease WHERE request_id = :request_id"
                ),
                {"request_id": request_id},
            ).one()
        assert observed_lease_id == f"lease-{row.lease_id}"
        assert row.heartbeat_at > row.acquired_at
        assert row.released_at is not None
        assert frozen.state == TrialState.COMPLETE
    finally:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "DELETE FROM ar_resource_lease "
                    "WHERE request_id IN (:request_id, :contender_request_id)"
                ),
                {
                    "request_id": request_id,
                    "contender_request_id": contender_request_id,
                },
            )
        optuna.delete_study(study_name=study_name, storage=storage)
        storage.engine.dispose()
        engine.dispose()


def test_resource_lease_database_constraints_win_cross_process_race():
    from sqlalchemy import create_engine, text

    request_id = f"resource-race-{uuid4()}"
    context = _context()
    output = context.Queue()
    start = context.Event()
    racers = [
        context.Process(
            target=_lease_racer,
            args=(request_id, f"worker-{index}", start, output),
        )
        for index in range(2)
    ]
    try:
        for racer in racers:
            racer.start()
        start.set()
        for racer in racers:
            racer.join(timeout=30)
            assert racer.exitcode == 0
        outcomes = [output.get(timeout=5) for _ in racers]
        assert sorted(kind for kind, _ in outcomes) == ["acquired", "conflict"]
    finally:
        for racer in racers:
            if racer.is_alive():
                racer.terminate()
        engine = create_engine(_database_url())
        with engine.begin() as connection:
            connection.execute(
                text("DELETE FROM ar_resource_lease WHERE request_id = :request_id"),
                {"request_id": request_id},
            )
        engine.dispose()


def test_exact_resource_acquire_is_idempotent_across_process_race():
    from sqlalchemy import create_engine, text

    request_id = f"resource-exact-race-{uuid4()}"
    context = _context()
    output = context.Queue()
    start = context.Event()
    racers = [
        context.Process(
            target=_lease_racer,
            args=(request_id, "same-worker", start, output),
        )
        for _ in range(2)
    ]
    engine = create_engine(_database_url())
    try:
        for racer in racers:
            racer.start()
        start.set()
        for racer in racers:
            racer.join(timeout=30)
            assert racer.exitcode == 0
        outcomes = [output.get(timeout=5) for _ in racers]
        assert [kind for kind, _ in outcomes] == ["acquired", "acquired"]
        leases = [payload for _, payload in outcomes]
        assert len({lease["lease_id"] for lease in leases}) == 1
        assert len({lease["acquired_at"] for lease in leases}) == 1
        assert len({lease["expires_at"] for lease in leases}) == 1
        with engine.connect() as connection:
            active_rows = connection.execute(
                text(
                    "SELECT COUNT(*) FROM ar_resource_lease "
                    "WHERE request_id = :request_id AND released_at IS NULL"
                ),
                {"request_id": request_id},
            ).scalar_one()
        assert active_rows == 1
    finally:
        for racer in racers:
            if racer.is_alive():
                racer.terminate()
        with engine.begin() as connection:
            connection.execute(
                text("DELETE FROM ar_resource_lease WHERE request_id = :request_id"),
                {"request_id": request_id},
            )
        engine.dispose()
