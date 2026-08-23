"""Explicit, identity-safe START, RESUME and REPLAY_INSPECT operations."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Mapping

from langgraph.types import Command

from auto_researcher.contracts.enums import ReadSafetyMode, RunStatus
from auto_researcher.contracts.models import ResearchContract, RunExecutionIdentity
from auto_researcher.runtime.identity import payload_hash

EXECUTION_PROTOCOL_VERSION = "run-execution-v2"
EXECUTION_ERROR_VOCABULARY_VERSION = "run-execution-errors-v1"
GRAPH_SCHEMA_VERSION = "auto-researcher-graph-v1"
TERMINAL_STATUSES = frozenset(
    {RunStatus.COMPLETED, RunStatus.STOPPED, RunStatus.FAILED}
)


class ExecutionMode(StrEnum):
    START = "START"
    RESUME = "RESUME"
    REPLAY_INSPECT = "REPLAY_INSPECT"


class RunExecutionError(RuntimeError):
    """Safe runtime rejection with no graph-side effects."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _thread_id(config: Mapping[str, Any]) -> str:
    configurable = config.get("configurable", {})
    thread_id = (
        configurable.get("thread_id") if isinstance(configurable, dict) else None
    )
    if not isinstance(thread_id, str) or not thread_id:
        raise RunExecutionError("thread_id_is_required")
    return thread_id


def _snapshot_values(graph: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = graph.get_state(dict(config))
    values = getattr(snapshot, "values", None)
    return dict(values) if values else {}


def execution_identity(
    initial_input: Mapping[str, Any],
    config: Mapping[str, Any],
) -> RunExecutionIdentity:
    thread_id = _thread_id(config)
    supplied_thread = initial_input.get("thread_id")
    if supplied_thread != thread_id:
        raise RunExecutionError("conflicting_thread_id")
    run_id = initial_input.get("run_id")
    contract = initial_input.get("contract")
    if not isinstance(run_id, str) or not run_id:
        raise RunExecutionError("run_id_is_required")
    if not isinstance(contract, ResearchContract):
        try:
            contract = ResearchContract.model_validate(contract)
        except Exception as exc:
            raise RunExecutionError("research_contract_is_required") from exc
    canonical_initial = {
        key: value
        for key, value in initial_input.items()
        if key != "execution_identity"
    }
    return RunExecutionIdentity(
        execution_protocol=EXECUTION_PROTOCOL_VERSION,
        graph_schema_version=GRAPH_SCHEMA_VERSION,
        thread_id=thread_id,
        run_id=run_id,
        contract_id=contract.contract_id,
        task_id=contract.task_id,
        task_version=contract.task_version,
        contract_hash=payload_hash(contract),
        initial_input_hash=payload_hash(canonical_initial),
    )


def _stored_identity(values: Mapping[str, Any]) -> RunExecutionIdentity:
    raw = values.get("execution_identity")
    if raw is None:
        raise RunExecutionError("checkpoint_execution_identity_missing")
    if type(raw) is not RunExecutionIdentity:
        raise RunExecutionError("checkpoint_execution_identity_invalid")
    identity = raw
    contract = values.get("contract")
    if type(contract) is not ResearchContract or any(
        type(mode) is not ReadSafetyMode
        for mode in contract.grounding.permitted_read_safety_modes
    ):
        raise RunExecutionError("checkpoint_execution_identity_invalid")
    if (
        values.get("thread_id") != identity.thread_id
        or values.get("run_id") != identity.run_id
        or contract.contract_id != identity.contract_id
        or contract.task_id != identity.task_id
        or contract.task_version != identity.task_version
        or payload_hash(contract) != identity.contract_hash
    ):
        raise RunExecutionError("checkpoint_execution_identity_conflict")
    return identity


def _compare_identity(
    stored: RunExecutionIdentity,
    requested: RunExecutionIdentity,
) -> None:
    if stored.thread_id != requested.thread_id:
        raise RunExecutionError("conflicting_thread_id")
    if stored.run_id != requested.run_id:
        raise RunExecutionError("conflicting_run_identity")
    if (stored.task_id, stored.task_version) != (
        requested.task_id,
        requested.task_version,
    ):
        raise RunExecutionError("conflicting_task_identity")
    if (
        stored.contract_id != requested.contract_id
        or stored.contract_hash != requested.contract_hash
    ):
        raise RunExecutionError("conflicting_contract_identity")
    if stored.initial_input_hash != requested.initial_input_hash:
        raise RunExecutionError("conflicting_initial_input_identity")


def validate_start_run(
    graph: Any,
    initial_input: Mapping[str, Any],
    config: Mapping[str, Any],
) -> RunExecutionIdentity:
    """Validate START identity without invoking any graph node."""

    requested = execution_identity(initial_input, config)
    existing = _snapshot_values(graph, config)
    if existing:
        stored = _stored_identity(existing)
        _compare_identity(stored, requested)
        raise RunExecutionError("thread_already_exists_use_resume_or_inspect")
    return requested


def start_run(
    graph: Any,
    initial_input: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Start only a checkpoint thread that has never existed."""

    requested = validate_start_run(graph, initial_input, config)
    payload = dict(initial_input)
    payload["execution_identity"] = requested
    return graph.invoke(payload, dict(config))


def can_resume_recoverable_planner_failure(values: Mapping[str, Any]) -> bool:
    """Recognise the narrow legacy checkpoint shape fixed by this release."""

    try:
        status = RunStatus(values["status"])
    except (KeyError, ValueError):
        return False
    errors = tuple(values.get("errors", ()))
    stop_reason = values.get("stop_reason")
    legacy_agent_failure = (
        stop_reason in {"planner_agent_failed", "agent_context_too_large"}
        and errors
        and set(errors).issubset(
            {"planner_agent_failed", "agent_context_too_large"}
        )
    )
    directive_projection_failure = (
        stop_reason == "research_director_openevolve_context_invalid"
        and errors == ("research_director_openevolve_context_invalid",)
        and values.get("planner_failure_stage")
        == "research_directive_projection"
        and values.get("active_research_directive") is not None
    )
    projection_recovery_call_limit = (
        stop_reason == "maximum_agent_calls_per_cycle_reached"
        and set(errors)
        == {
            "research_director_openevolve_context_invalid",
            "maximum_agent_calls_per_cycle_reached",
        }
        and values.get("planner_failure_stage") == "model_call"
        and set(values.get("recovered_error_codes", ()))
        == {"research_director_openevolve_context_invalid"}
        and values.get("active_research_directive") is not None
    )
    v8_duplicate_portfolio_recovery = (
        stop_reason == "planner_agent_failed"
        and set(errors)
        == {
            "research_director_openevolve_context_invalid",
            "maximum_agent_calls_per_cycle_reached",
            "planner_agent_failed",
        }
        and values.get("planner_failure_stage") == "portfolio_policy"
        and values.get("active_research_directive") is not None
    )
    return (
        status == RunStatus.FAILED
        and (
            legacy_agent_failure
            or directive_projection_failure
            or projection_recovery_call_limit
            or v8_duplicate_portfolio_recovery
        )
        and values.get("active_hypothesis") is not None
        and values.get("search_request") is None
        and "plan_search" in values.get("executed_nodes", ())
    )


def resume_run(
    graph: Any,
    config: Mapping[str, Any],
    resume_value: Any = None,
    *,
    expected_identity: RunExecutionIdentity | None = None,
) -> dict[str, Any]:
    """Continue a checkpoint, including the exact recoverable planner boundary."""

    values = _snapshot_values(graph, config)
    if not values:
        raise RunExecutionError("thread_not_found")
    stored = _stored_identity(values)
    if expected_identity is not None:
        _compare_identity(stored, expected_identity)
    status = RunStatus(values["status"])
    if can_resume_recoverable_planner_failure(values):
        recovered = list(dict.fromkeys(values.get("errors", ())))
        graph.update_state(
            dict(config),
            {
                "status": RunStatus.RUNNING,
                "stop_reason": None,
                "planner_failure_code": None,
                "planner_failure_stage": None,
                "recovered_error_codes": recovered,
            },
            as_node="generate_hypothesis",
        )
        return graph.invoke(None, dict(config))
    if status in TERMINAL_STATUSES:
        raise RunExecutionError("thread_is_terminal_use_inspect")
    continuation: Any = None if resume_value is None else Command(resume=resume_value)
    return graph.invoke(continuation, dict(config))


def inspect_terminal_run(
    graph: Any,
    config: Mapping[str, Any],
    *,
    expected_identity: RunExecutionIdentity | None = None,
) -> dict[str, Any]:
    """Read a terminal state without invoking any LangGraph node."""

    values = _snapshot_values(graph, config)
    if not values:
        raise RunExecutionError("thread_not_found")
    stored = _stored_identity(values)
    if expected_identity is not None:
        _compare_identity(stored, expected_identity)
    if RunStatus(values["status"]) not in TERMINAL_STATUSES:
        raise RunExecutionError("thread_is_not_terminal_use_resume")
    return values
