"""Courteous task-owned GPU admission for FeTA search candidates."""

from __future__ import annotations

import math
import os
import pwd
import re
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from auto_researcher.tasks.models import TaskRuntimeContext

GPU_SCHEDULER_VERSION = "feta-search-courteous-gpu-admission-v1"
WAIT_LOG_INTERVAL_SECONDS = 300
REGISTERED_FIDELITIES = (25, 50, 100, 150, 300)
OPPORTUNISTIC_DEFAULT_FIDELITIES = (25, 50, 100)


class GPUSchedulerPolicy(BaseModel):
    """Strict runtime-only admission policy; disabled unless explicitly enabled."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["disabled", "primary", "opportunistic"] = "disabled"
    physical_gpu_index: int | None = Field(default=None, strict=True, ge=0)
    poll_seconds: int = Field(default=20, strict=True, gt=0)
    stable_idle_seconds: int = Field(default=0, strict=True, ge=0)
    minimum_free_memory_mib: int = Field(default=40000, strict=True, gt=0)
    maximum_utilization_percent: int = Field(
        default=10, strict=True, ge=0, le=100
    )
    allowed_fidelities: tuple[int, ...] = REGISTERED_FIDELITIES

    @model_validator(mode="before")
    @classmethod
    def apply_mode_defaults(cls, value: Any) -> Any:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("gpu scheduler configuration must be a mapping")
        configured = dict(value)
        mode = configured.get("mode", "disabled")
        if "stable_idle_seconds" not in configured:
            configured["stable_idle_seconds"] = (
                60 if mode == "opportunistic" else 0
            )
        if "allowed_fidelities" not in configured:
            configured["allowed_fidelities"] = (
                OPPORTUNISTIC_DEFAULT_FIDELITIES
                if mode == "opportunistic"
                else REGISTERED_FIDELITIES
            )
        return configured

    @field_validator("allowed_fidelities", mode="before")
    @classmethod
    def validate_fidelity_input(cls, value: Any) -> Any:
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError("allowed_fidelities must be a non-empty sequence")
        if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
            raise ValueError("allowed_fidelities must contain integers")
        return value

    @field_validator("allowed_fidelities")
    @classmethod
    def validate_fidelities(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if len(set(value)) != len(value) or any(
            fidelity not in REGISTERED_FIDELITIES for fidelity in value
        ):
            raise ValueError("allowed_fidelities are invalid")
        return value

    @model_validator(mode="after")
    def validate_mode_binding(self) -> "GPUSchedulerPolicy":
        if self.mode == "disabled" and self.physical_gpu_index is not None:
            raise ValueError("disabled scheduler cannot select a GPU")
        if self.mode != "disabled" and self.physical_gpu_index is None:
            raise ValueError("enabled scheduler requires physical_gpu_index")
        return self


@dataclass(frozen=True)
class GPUComputeProcess:
    pid: int
    username: str | None = None


@dataclass(frozen=True)
class GPUProbeResult:
    free_memory_mib: int
    utilization_percent: int
    compute_processes: tuple[GPUComputeProcess, ...] = ()


@dataclass(frozen=True)
class GPUAdmissionTelemetry:
    mode: Literal["primary", "opportunistic"]
    physical_gpu_index: int
    wait_seconds: float
    poll_count: int
    free_memory_mib: int
    utilization_percent: int
    foreign_process_count: int
    stable_idle_seconds: int

    def as_metrics(self) -> dict[str, Any]:
        return {
            "gpu_scheduler_version": GPU_SCHEDULER_VERSION,
            "gpu_scheduler_mode": self.mode,
            "physical_gpu_index": self.physical_gpu_index,
            "gpu_admission_wait_seconds": self.wait_seconds,
            "gpu_admission_poll_count": self.poll_count,
            "gpu_admission_free_memory_mib": self.free_memory_mib,
            "gpu_admission_utilization_percent": self.utilization_percent,
            "gpu_admission_foreign_process_count": self.foreign_process_count,
            "gpu_admission_stable_idle_seconds": self.stable_idle_seconds,
        }


class GPUProbe(Protocol):
    def probe(self, physical_gpu_index: int) -> GPUProbeResult: ...


def _username_for_pid(pid: int) -> str | None:
    try:
        uid = Path(f"/proc/{pid}").stat().st_uid
        return pwd.getpwuid(uid).pw_name
    except (KeyError, OSError):
        return None


class NvidiaSmiGPUProbe:
    """Read physical-card occupancy without creating a CUDA context."""

    def _query(self, arguments: list[str]) -> str:
        try:
            completed = subprocess.run(
                ["nvidia-smi", *arguments],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError("feta_search_gpu_probe_failed") from exc
        if completed.returncode != 0:
            raise RuntimeError("feta_search_gpu_probe_failed")
        return completed.stdout

    def probe(self, physical_gpu_index: int) -> GPUProbeResult:
        status = self._query(
            [
                f"--id={physical_gpu_index}",
                "--query-gpu=memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
        )
        rows = [row.strip() for row in status.splitlines() if row.strip()]
        if len(rows) != 1:
            raise ValueError("feta_search_gpu_probe_parse_failed")
        fields = [field.strip() for field in rows[0].split(",")]
        try:
            if len(fields) != 2:
                raise ValueError
            free_memory_mib, utilization_percent = (int(field) for field in fields)
            if free_memory_mib < 0 or not 0 <= utilization_percent <= 100:
                raise ValueError
        except ValueError as exc:
            raise ValueError("feta_search_gpu_probe_parse_failed") from exc

        process_output = self._query(
            [
                f"--id={physical_gpu_index}",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ]
        )
        process_rows = [row.strip() for row in process_output.splitlines() if row.strip()]
        if process_rows and all(
            row.lower().startswith("no running processes") for row in process_rows
        ):
            process_rows = []
        if any(re.fullmatch(r"[0-9]+", row) is None for row in process_rows):
            raise ValueError("feta_search_gpu_probe_parse_failed")
        processes = tuple(
            GPUComputeProcess(pid=int(row), username=_username_for_pid(int(row)))
            for row in process_rows
        )
        return GPUProbeResult(
            free_memory_mib=free_memory_mib,
            utilization_percent=utilization_percent,
            compute_processes=processes,
        )


def gpu_scheduler_policy(context: TaskRuntimeContext) -> GPUSchedulerPolicy:
    raw = context.task_options.get("gpu_scheduler")
    try:
        return GPUSchedulerPolicy.model_validate(raw if raw is not None else {})
    except (TypeError, ValueError, ValidationError) as exc:
        raise ValueError("feta_search_gpu_scheduler_configuration_invalid") from exc


def verify_physical_cuda_binding(
    policy: GPUSchedulerPolicy,
    *,
    environ: Mapping[str, str] | None = None,
) -> None:
    if policy.mode == "disabled":
        return
    visible = (environ if environ is not None else os.environ).get(
        "CUDA_VISIBLE_DEVICES"
    )
    if visible is None or re.fullmatch(r"[0-9]+", visible) is None:
        raise ValueError("feta_search_gpu_binding_mismatch")
    assert policy.physical_gpu_index is not None
    if int(visible) != policy.physical_gpu_index:
        raise ValueError("feta_search_gpu_binding_mismatch")


def _stdout_logger(record: str) -> None:
    print(record, flush=True)


def wait_for_gpu_admission(
    policy: GPUSchedulerPolicy,
    *,
    maximum_epochs: int,
    probe: GPUProbe | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    current_pid: int | None = None,
    environ: Mapping[str, str] | None = None,
    logger: Callable[[str], None] = _stdout_logger,
) -> GPUAdmissionTelemetry | None:
    """Wait indefinitely at a candidate boundary until the selected card is free."""

    if policy.mode == "disabled":
        return None
    verify_physical_cuda_binding(policy, environ=environ)
    if maximum_epochs not in policy.allowed_fidelities:
        raise ValueError("feta_search_gpu_fidelity_disallowed")

    physical_gpu_index = policy.physical_gpu_index
    assert physical_gpu_index is not None
    selected_probe = probe or NvidiaSmiGPUProbe()
    own_pid = os.getpid() if current_pid is None else current_pid
    started = clock()
    idle_started: float | None = None
    poll_count = 0
    last_wait_reason: str | None = None
    last_wait_log_time: float | None = None

    def log_waiting(reason: str, record: str, now: float) -> None:
        nonlocal last_wait_log_time, last_wait_reason
        if (
            reason != last_wait_reason
            or last_wait_log_time is None
            or now - last_wait_log_time >= WAIT_LOG_INTERVAL_SECONDS
        ):
            logger(record)
            last_wait_reason = reason
            last_wait_log_time = now

    while True:
        try:
            observed = selected_probe.probe(physical_gpu_index)
        except (RuntimeError, ValueError) as exc:
            if str(exc) in {
                "feta_search_gpu_probe_failed",
                "feta_search_gpu_probe_parse_failed",
            }:
                raise
            raise RuntimeError("feta_search_gpu_probe_failed") from exc
        except Exception as exc:
            raise RuntimeError("feta_search_gpu_probe_failed") from exc
        poll_count += 1
        now = clock()
        foreign = tuple(
            process for process in observed.compute_processes if process.pid != own_pid
        )

        reason: str | None = None
        if foreign:
            reason = "foreign_process"
        elif observed.free_memory_mib < policy.minimum_free_memory_mib:
            reason = "low_free_memory"
        elif observed.utilization_percent > policy.maximum_utilization_percent:
            reason = "high_utilization"

        if reason is not None:
            idle_started = None
            detail = ""
            if reason == "foreign_process":
                process = foreign[0]
                detail = f" pid={process.pid}"
                if process.username is not None:
                    detail += f" user={process.username}"
            elif reason == "low_free_memory":
                detail = f" free_memory_mib={observed.free_memory_mib}"
            else:
                detail = f" utilization_percent={observed.utilization_percent}"
            log_waiting(
                reason,
                "FETA_GPU_SCHEDULER "
                f"mode={policy.mode} gpu={physical_gpu_index} state=WAITING "
                f"reason={reason}{detail}",
                now,
            )
            sleeper(policy.poll_seconds)
            continue

        if idle_started is None:
            idle_started = now
        idle_seconds = max(0.0, now - idle_started)
        if idle_seconds < policy.stable_idle_seconds:
            log_waiting(
                "stability_window",
                "FETA_GPU_SCHEDULER "
                f"mode={policy.mode} gpu={physical_gpu_index} state=WAITING "
                f"reason=stability_window idle_seconds={int(idle_seconds)}",
                now,
            )
            sleeper(policy.poll_seconds)
            continue

        wait_seconds = max(0.0, now - started)
        logger(
            "FETA_GPU_SCHEDULER "
            f"mode={policy.mode} gpu={physical_gpu_index} state=ADMITTED "
            f"wait_seconds={int(wait_seconds)} "
            f"free_memory_mib={observed.free_memory_mib} "
            f"utilization_percent={observed.utilization_percent}"
        )
        return GPUAdmissionTelemetry(
            mode=policy.mode,
            physical_gpu_index=physical_gpu_index,
            wait_seconds=wait_seconds,
            poll_count=poll_count,
            free_memory_mib=observed.free_memory_mib,
            utilization_percent=observed.utilization_percent,
            foreign_process_count=len(foreign),
            stable_idle_seconds=policy.stable_idle_seconds,
        )


def scheduler_telemetry_is_valid(
    metrics: Mapping[str, Any], policy: GPUSchedulerPolicy
) -> bool:
    """Validate operational telemetry without contributing to the scientific score."""

    if policy.mode == "disabled":
        return True
    try:
        wait_seconds = metrics["gpu_admission_wait_seconds"]
        poll_count = metrics["gpu_admission_poll_count"]
        free_memory_mib = metrics["gpu_admission_free_memory_mib"]
        utilization_percent = metrics["gpu_admission_utilization_percent"]
        foreign_process_count = metrics["gpu_admission_foreign_process_count"]
        stable_idle_seconds = metrics["gpu_admission_stable_idle_seconds"]
        return (
            metrics.get("gpu_scheduler_version") == GPU_SCHEDULER_VERSION
            and metrics.get("gpu_scheduler_mode") == policy.mode
            and metrics.get("physical_gpu_index") == policy.physical_gpu_index
            and isinstance(wait_seconds, (int, float))
            and not isinstance(wait_seconds, bool)
            and math.isfinite(float(wait_seconds))
            and float(wait_seconds) >= 0
            and isinstance(poll_count, int)
            and not isinstance(poll_count, bool)
            and poll_count >= 1
            and isinstance(free_memory_mib, int)
            and not isinstance(free_memory_mib, bool)
            and free_memory_mib >= policy.minimum_free_memory_mib
            and isinstance(utilization_percent, int)
            and not isinstance(utilization_percent, bool)
            and 0 <= utilization_percent <= policy.maximum_utilization_percent
            and isinstance(foreign_process_count, int)
            and not isinstance(foreign_process_count, bool)
            and foreign_process_count == 0
            and isinstance(stable_idle_seconds, int)
            and not isinstance(stable_idle_seconds, bool)
            and stable_idle_seconds == policy.stable_idle_seconds
        )
    except (KeyError, TypeError, ValueError):
        return False
