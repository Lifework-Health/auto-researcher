"""Deterministic, instance-scoped research task registry."""

from __future__ import annotations

from collections.abc import Callable

from auto_researcher.tasks.models import DuplicateTaskError, UnknownTaskError
from auto_researcher.tasks.protocols import ResearchTask

TaskFactory = Callable[[], ResearchTask]


class TaskRegistry:
    def __init__(self) -> None:
        self._factories: dict[tuple[str, str], TaskFactory] = {}

    def register(self, task_factory: TaskFactory) -> None:
        task = task_factory()
        key = (task.task_id, task.task_version)
        if key in self._factories:
            raise DuplicateTaskError(
                f"task {task.task_id!r} version {task.task_version!r} is already registered"
            )
        self._factories[key] = task_factory

    def get(self, task_id: str, task_version: str | None = None) -> ResearchTask:
        matches = sorted(key for key in self._factories if key[0] == task_id)
        if not matches:
            raise UnknownTaskError(
                f"unknown research task {task_id!r}; available: "
                f"{', '.join(sorted({key[0] for key in self._factories})) or 'none'}"
            )
        if task_version is None:
            key = matches[-1]
        else:
            key = (task_id, task_version)
            if key not in self._factories:
                versions = ", ".join(version for _, version in matches)
                raise UnknownTaskError(
                    f"task {task_id!r} version {task_version!r} is unavailable; "
                    f"registered versions: {versions}"
                )
        return self._factories[key]()

    def list_tasks(self) -> list[ResearchTask]:
        return [self._factories[key]() for key in sorted(self._factories)]

    def contains(self, task_id: str) -> bool:
        return any(registered_id == task_id for registered_id, _ in self._factories)


def default_task_registry() -> TaskRegistry:
    """Built-ins remain importable even when optional iCCA dependencies are absent."""
    from auto_researcher.tasks.icca_nbs import ICCANBSTask
    from auto_researcher.tasks.iris_knn import IrisKNNTask
    from auto_researcher.tasks.feta_seg import FeTASegTask
    from auto_researcher.tasks.synthetic import SyntheticTask

    registry = TaskRegistry()
    registry.register(SyntheticTask)
    registry.register(IrisKNNTask)
    registry.register(FeTASegTask)
    registry.register(ICCANBSTask)
    return registry
