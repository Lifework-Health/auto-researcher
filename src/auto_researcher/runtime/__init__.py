"""Lazy runtime exports; importing identity helpers must not assemble providers."""

__all__ = [
    "EXECUTION_ERROR_VOCABULARY_VERSION",
    "ExecutionMode",
    "RuntimeDependencies",
    "RunExecutionError",
    "inspect_terminal_run",
    "memory_dependencies",
    "resume_run",
    "start_run",
    "task_memory_dependencies",
    "task_sqlite_dependencies",
    "validate_start_run",
]


def __getattr__(name: str):
    dependency_names = {
        "RuntimeDependencies",
        "memory_dependencies",
        "task_memory_dependencies",
        "task_sqlite_dependencies",
    }
    execution_names = {
        "EXECUTION_ERROR_VOCABULARY_VERSION",
        "ExecutionMode",
        "RunExecutionError",
        "inspect_terminal_run",
        "resume_run",
        "start_run",
        "validate_start_run",
    }
    if name in dependency_names:
        from auto_researcher.runtime import dependencies

        return getattr(dependencies, name)
    if name in execution_names:
        from auto_researcher.runtime import execution

        return getattr(execution, name)
    raise AttributeError(name)
