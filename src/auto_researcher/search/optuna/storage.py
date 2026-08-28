"""Typed Optuna storage factories for local and shared-worker execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from auto_researcher.secrets.models import SecretReference
from auto_researcher.secrets.providers import SecretProvider, provider_for_reference


class OptunaStorageBackend(StrEnum):
    MEMORY = "memory"
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"


class PostgreSQLStorageConfiguration(BaseModel):
    """Non-sensitive PostgreSQL location plus a managed-secret reference."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    host: str = Field(min_length=1, max_length=253)
    port: int = Field(default=5432, ge=1, le=65535)
    database: str = Field(min_length=1, max_length=63)
    username: str = Field(min_length=1, max_length=63)
    password: SecretReference
    alias: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$",
    )
    connect_timeout_seconds: int = Field(default=10, ge=1, le=120)

    @property
    def safe_reference(self) -> str:
        return f"postgresql:{self.alias or self.database}"


class OptunaStorageConfiguration(BaseModel):
    """Task-agnostic operational storage selection.

    This model is intentionally unsuitable for a resolved password or a
    credential-bearing DSN. Only a managed-secret reference can be configured.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)

    backend: OptunaStorageBackend = OptunaStorageBackend.MEMORY
    shared_workers: bool = False
    sqlite_path: Path | None = None
    postgresql: PostgreSQLStorageConfiguration | None = None

    @model_validator(mode="after")
    def backend_configuration_is_coherent(self) -> "OptunaStorageConfiguration":
        if self.backend is OptunaStorageBackend.MEMORY:
            if self.sqlite_path is not None or self.postgresql is not None:
                raise ValueError("memory storage does not accept database settings")
            if self.shared_workers:
                raise ValueError("memory storage cannot coordinate shared workers")
        elif self.backend is OptunaStorageBackend.SQLITE:
            if self.sqlite_path is None or self.postgresql is not None:
                raise ValueError("sqlite storage requires only sqlite_path")
            if self.shared_workers:
                raise ValueError("sqlite_shared_workers_not_supported")
        elif self.backend is OptunaStorageBackend.POSTGRESQL:
            if self.postgresql is None or self.sqlite_path is not None:
                raise ValueError("postgresql storage requires only postgresql settings")
            if not self.shared_workers:
                raise ValueError("postgresql storage is reserved for shared workers")
        return self

    @property
    def safe_reference(self) -> str:
        if self.backend is OptunaStorageBackend.MEMORY:
            return "memory"
        if self.backend is OptunaStorageBackend.SQLITE:
            assert self.sqlite_path is not None
            return f"sqlite:{self.sqlite_path.name}"
        assert self.postgresql is not None
        return self.postgresql.safe_reference


class OptunaStorageError(RuntimeError):
    """Bounded storage failure which never includes connection details."""


class OptunaStorageUnavailableError(OptunaStorageError):
    pass


class OptunaStorageAuthenticationError(OptunaStorageError):
    pass


@dataclass(frozen=True)
class OptunaStorageHandle:
    storage: Any = field(repr=False)
    safe_reference: str
    coordination_engine: Any | None = field(default=None, repr=False)
    shared_workers: bool = False

    def close(self) -> None:
        engine = getattr(self.storage, "engine", None)
        if engine is not None:
            engine.dispose()
        if (
            self.coordination_engine is not None
            and self.coordination_engine is not engine
        ):
            self.coordination_engine.dispose()


def _hpo_imports() -> tuple[Any, Any, Any]:
    try:
        from optuna.storages import InMemoryStorage, RDBStorage
        from sqlalchemy import create_engine
    except ImportError as exc:
        raise RuntimeError(
            "OPTUNA search requires the HPO dependency. "
            "Install with `pip install -e '.[hpo]'`."
        ) from exc
    return InMemoryStorage, RDBStorage, create_engine


def in_memory_storage() -> OptunaStorageHandle:
    InMemoryStorage, _, _ = _hpo_imports()
    return OptunaStorageHandle(storage=InMemoryStorage(), safe_reference="memory")


def sqlite_storage(
    path: str | Path, *, shared_workers: bool = False
) -> OptunaStorageHandle:
    if shared_workers:
        raise ValueError("sqlite_shared_workers_not_supported")
    _, RDBStorage, _ = _hpo_imports()
    try:
        from sqlalchemy.engine import URL
    except ImportError as exc:  # pragma: no cover - covered by _hpo_imports
        raise RuntimeError("OPTUNA HPO dependency unavailable") from exc
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return OptunaStorageHandle(
        storage=RDBStorage(
            url=URL.create("sqlite", database=str(resolved)).render_as_string(
                hide_password=False
            )
        ),
        safe_reference=f"sqlite:{resolved.name}",
    )


def _postgres_error(exc: Exception) -> OptunaStorageError:
    name = type(exc).__name__.casefold()
    text = str(getattr(exc, "orig", "")).casefold()
    if "authentication" in text or "password" in text or "invalidpassword" in name:
        return OptunaStorageAuthenticationError("optuna_storage_authentication_failed")
    return OptunaStorageUnavailableError("optuna_shared_storage_unavailable")


def postgresql_storage(
    configuration: PostgreSQLStorageConfiguration,
    *,
    secret_provider: SecretProvider | None = None,
) -> OptunaStorageHandle:
    """Construct native Optuna RDBStorage without retaining the resolved password."""

    _, RDBStorage, create_engine = _hpo_imports()
    try:
        from sqlalchemy.engine import URL
    except ImportError as exc:  # pragma: no cover - covered by _hpo_imports
        raise RuntimeError("OPTUNA HPO dependency unavailable") from exc
    provider = secret_provider or provider_for_reference(configuration.password)
    resolved = provider.resolve(configuration.password)
    if resolved is None:
        raise OptunaStorageAuthenticationError("optuna_storage_authentication_failed")
    # URL.create quotes credentials safely. The clear value exists only in this
    # narrow runtime construction scope and is never retained in configuration,
    # safe_reference, repr, study identity, or scientific state.
    url = URL.create(
        "postgresql+psycopg",
        username=configuration.username,
        password=resolved.reveal(),
        host=configuration.host,
        port=configuration.port,
        database=configuration.database,
        query={"connect_timeout": str(configuration.connect_timeout_seconds)},
    )
    runtime_url = url.render_as_string(hide_password=False)
    storage = None
    coordination_engine = None
    try:
        storage = RDBStorage(url=runtime_url)
        coordination_engine = create_engine(runtime_url, pool_pre_ping=True)
        with coordination_engine.connect() as connection:
            connection.exec_driver_sql("SELECT 1")
    except Exception as exc:
        if coordination_engine is not None:
            coordination_engine.dispose()
        if storage is not None:
            storage_engine = getattr(storage, "engine", None)
            if storage_engine is not None:
                storage_engine.dispose()
        raise _postgres_error(exc) from None
    finally:
        runtime_url = "<redacted>"
        resolved = None
        del url
    assert storage is not None and coordination_engine is not None
    return OptunaStorageHandle(
        storage=storage,
        coordination_engine=coordination_engine,
        safe_reference=configuration.safe_reference,
        shared_workers=True,
    )


def configured_storage(
    configuration: OptunaStorageConfiguration,
    *,
    secret_provider: SecretProvider | None = None,
) -> OptunaStorageHandle:
    if configuration.backend is OptunaStorageBackend.MEMORY:
        return in_memory_storage()
    if configuration.backend is OptunaStorageBackend.SQLITE:
        assert configuration.sqlite_path is not None
        return sqlite_storage(
            configuration.sqlite_path,
            shared_workers=configuration.shared_workers,
        )
    assert configuration.postgresql is not None
    return postgresql_storage(
        configuration.postgresql,
        secret_provider=secret_provider,
    )
