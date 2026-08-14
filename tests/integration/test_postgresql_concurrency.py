from __future__ import annotations

import multiprocessing
import os
import time
from datetime import timedelta
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


def _lease_racer(request_id: str, worker_id: str, output) -> None:
    from datetime import datetime, timezone

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
        lease = store.acquire(
            request,
            ResourceCandidate(resource_id="gpu:race", resource_type="gpu"),
            worker_id=worker_id,
            now=datetime.now(timezone.utc),
            ttl=timedelta(minutes=1),
        )
        output.put(("acquired", lease.worker_id))
    except ResourceLeaseConflict:
        output.put(("conflict", worker_id))
    finally:
        engine.dispose()


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
    from datetime import datetime, timezone

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


def test_resource_lease_database_constraints_win_cross_process_race():
    from sqlalchemy import create_engine, text

    request_id = f"resource-race-{uuid4()}"
    context = _context()
    output = context.Queue()
    racers = [
        context.Process(
            target=_lease_racer,
            args=(request_id, f"worker-{index}", output),
        )
        for index in range(2)
    ]
    try:
        for racer in racers:
            racer.start()
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
