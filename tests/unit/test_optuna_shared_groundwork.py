from __future__ import annotations

import pytest
from pydantic import ValidationError

from auto_researcher.search.optuna.backend import OptunaAskTellBackend
from auto_researcher.search.optuna.models import (
    IntParameterSpec,
    OptimisationDirection,
    OptunaStudySpec,
)
from auto_researcher.search.optuna.naming import StudyIdentity
from auto_researcher.search.optuna.storage import (
    OptunaStorageConfiguration,
    PostgreSQLStorageConfiguration,
    in_memory_storage,
    sqlite_storage,
)
from auto_researcher.secrets.models import SecretProviderKind, SecretReference


def postgresql_configuration() -> PostgreSQLStorageConfiguration:
    return PostgreSQLStorageConfiguration(
        host="postgres.internal",
        database="research",
        username="auto_researcher",
        password=SecretReference(
            logical_name="optuna_database_password",
            provider=SecretProviderKind.ENVIRONMENT,
            provider_identifier="OPTUNA_DATABASE_PASSWORD",
        ),
        alias="shared-hpo",
    )


def test_storage_configuration_rejects_shared_sqlite(tmp_path) -> None:
    with pytest.raises(ValidationError, match="sqlite_shared_workers_not_supported"):
        OptunaStorageConfiguration(
            backend="sqlite",
            shared_workers=True,
            sqlite_path=tmp_path / "optuna.db",
        )
    with pytest.raises(ValueError, match="sqlite_shared_workers_not_supported"):
        sqlite_storage(tmp_path / "optuna.db", shared_workers=True)


def test_postgresql_configuration_contains_reference_but_never_secret_value() -> None:
    configuration = OptunaStorageConfiguration(
        backend="postgresql",
        shared_workers=True,
        postgresql=postgresql_configuration(),
    )

    rendered = repr(configuration)
    assert "database-password-value" not in rendered
    assert configuration.safe_reference == "postgresql:shared-hpo"
    assert "OPTUNA_DATABASE_PASSWORD" not in configuration.safe_reference


@pytest.mark.hpo
def test_distributed_workers_use_native_distinct_seeded_tpe_streams() -> None:
    spec = OptunaStudySpec(
        task_id="test",
        schema_version="1.0",
        task_version="1",
        search_space_version="1",
        direction=OptimisationDirection.MAXIMIZE,
        parameters=(IntParameterSpec(name="x", low=0, high=10_000),),
        trial_budget=12,
        seed=1729,
        n_startup_trials=12,
        objective_metric="score",
    )
    # The sampler remains Optuna's public TPESampler; worker identity only derives
    # distinct native RNG seeds from the configured study seed.
    first = OptunaAskTellBackend(
        in_memory_storage().storage,
        shared_workers=True,
        coordination=object(),  # not exercised by this sampler-only test
        worker_id="worker-a",
    )._sampler(spec)
    second = OptunaAskTellBackend(
        in_memory_storage().storage,
        shared_workers=True,
        coordination=object(),
        worker_id="worker-b",
    )._sampler(spec)

    import optuna
    from optuna.distributions import IntDistribution

    study_a = optuna.create_study(sampler=first)
    study_b = optuna.create_study(sampler=second)
    suggestions_a = [
        study_a.ask(fixed_distributions={"x": IntDistribution(0, 10_000)}).params["x"]
        for _ in range(6)
    ]
    suggestions_b = [
        study_b.ask(fixed_distributions={"x": IntDistribution(0, 10_000)}).params["x"]
        for _ in range(6)
    ]

    assert type(first).__name__ == "TPESampler"
    assert suggestions_a != suggestions_b


def test_shared_validation_allows_multiple_valid_running_trials() -> None:
    backend = OptunaAskTellBackend(
        in_memory_storage().storage,
        shared_workers=True,
        coordination=object(),
        worker_id="worker-a",
    )
    identity = StudyIdentity(
        study_name="study",
        search_space_hash="hash",
        attributes={"run_id": "run", "request_id": "request"},
    )

    class Trial:
        def __init__(self, slot: int) -> None:
            self.user_attrs = {
                "run_id": "run",
                "request_id": "request",
                "slot_index": slot,
            }

    backend._validate_running_trials([Trial(0), Trial(1), Trial(2)], identity)
    with pytest.raises(Exception, match="different run or request"):
        foreign = Trial(3)
        foreign.user_attrs["run_id"] = "other"
        backend._validate_running_trials([foreign], identity)
