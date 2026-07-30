"""Optuna storage factories for sequential local execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OptunaStorageHandle:
    storage: Any
    safe_reference: str

    def close(self) -> None:
        engine = getattr(self.storage, "engine", None)
        if engine is not None:
            engine.dispose()


def in_memory_storage() -> OptunaStorageHandle:
    try:
        from optuna.storages import InMemoryStorage
    except ImportError as exc:
        raise RuntimeError(
            "OPTUNA search requires the HPO dependency. "
            "Install with `pip install -e '.[hpo]'`."
        ) from exc
    return OptunaStorageHandle(
        storage=InMemoryStorage(),
        safe_reference="memory",
    )


def sqlite_storage(path: str | Path) -> OptunaStorageHandle:
    try:
        from optuna.storages import RDBStorage
        from sqlalchemy.engine import URL
    except ImportError as exc:
        raise RuntimeError(
            "OPTUNA search requires the HPO dependency. "
            "Install with `pip install -e '.[hpo]'`."
        ) from exc
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return OptunaStorageHandle(
        storage=RDBStorage(
            url=URL.create("sqlite", database=str(resolved)).render_as_string(
                hide_password=False
            )
        ),
        safe_reference=resolved.name,
    )
