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

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from auto_researcher.resources import (
    AdmissionClass,
    AdmissionOutcome,
    CourtesyResourceAdmissionPolicy,
    InvalidResourceState,
    ResourceBroker,
    ResourceCandidate,
    ResourceCapacity,
    ResourceInspectionError,
    ResourceOwner,
    ResourceRequest,
    ResourceRequirement,
)
from auto_researcher.tasks.feta_seg_search.configuration import FIDELITY_LEVELS
from auto_researcher.tasks.models import TaskRuntimeContext

GPU_SCHEDULER_VERSION = "feta-search-courteous-gpu-admission-v1"
WAIT_LOG_INTERVAL_SECONDS = 300
REGISTERED_FIDELITIES = FIDELITY_LEVELS
OPPORTUNISTIC_DEFAULT_FIDELITIES = (25, 50, 100)


class GPUSchedulerPolicy(BaseModel):
    """Strict runtime-only admission policy; disabled unless explicitly enabled."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["disabled", "primary", "opportunistic"] = "disabled"
    physical_gpu_index: int | None = Field(default=None, strict=True, ge=0)
    poll_seconds: int = Field(default=20, strict=True, gt=0)
    stable_idle_seconds: int = Field(default=0, strict=True, ge=0)
    minimum_free_memory_mib: int = Field(default=40000, strict=True, gt=0)
    maximum_utilization_percent: int = Field(default=10, strict=True, ge=0, le=100)
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
            configured["stable_idle_seconds"] = 60 if mode == "opportunistic" else 0
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
        process_rows = [
            row.strip() for row in process_output.splitlines() if row.strip()
        ]
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


class FeTAGPUResourceProvider:
    """Adapt the physical-card probe to task-neutral resource candidates."""

    def __init__(
        self,
        probe: GPUProbe,
        *,
        physical_gpu_index: int,
        current_pid: int,
    ) -> None:
        self.probe = probe
        self.physical_gpu_index = physical_gpu_index
        self.current_pid = current_pid

    def candidates(self, _request: ResourceRequest) -> tuple[ResourceCandidate, ...]:
        observed = self.probe.probe(self.physical_gpu_index)
        foreign = tuple(
            process
            for process in observed.compute_processes
            if process.pid != self.current_pid
        )
        owners = tuple(
            ResourceOwner(
                namespace="process_pid",
                owner_id=str(process.pid),
                display_name=process.username,
            )
            for process in foreign
        )
        return (
            ResourceCandidate(
                resource_id=f"gpu:{self.physical_gpu_index}",
                resource_type="gpu",
                quantity=1,
                capacities=(
                    ResourceCapacity(name="memory_mib", value=observed.free_memory_mib),
                ),
                utilization_percent=observed.utilization_percent,
                foreign_owners=owners,
                equivalence_tags=frozenset({"cuda-logical-device-0"}),
            ),
        )


def gpu_resource_request(policy: GPUSchedulerPolicy) -> ResourceRequest:
    """Translate FeTA runtime policy without admitting scientific configuration."""

    if policy.mode == "disabled":
        raise ValueError("disabled gpu scheduler has no resource request")
    return ResourceRequest(
        request_id=f"feta-gpu-{policy.physical_gpu_index}",
        requirements=(
            ResourceRequirement(
                resource_type="gpu",
                quantity=1,
                minimum_capacities=(
                    ResourceCapacity(
                        name="memory_mib", value=policy.minimum_free_memory_mib
                    ),
                ),
            ),
        ),
        admission_class=AdmissionClass(policy.mode),
        priority=100 if policy.mode == "primary" else 0,
        maximum_wait_seconds=None,
        stable_idle_seconds=policy.stable_idle_seconds,
        equivalence_requirements=frozenset({"cuda-logical-device-0"}),
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

    def observe(decision, candidate, now: float) -> None:
        if decision.outcome is AdmissionOutcome.ADMITTED:
            return
        reason = {
            "foreign_owner": "foreign_process",
            "low_memory_mib": "low_free_memory",
        }.get(decision.reason, decision.reason)
        detail = ""
        if reason == "foreign_process" and candidate.foreign_owners:
            process = candidate.foreign_owners[0]
            detail = f" pid={process.owner_id}"
            if process.display_name is not None:
                detail += f" user={process.display_name}"
        elif reason == "low_free_memory":
            detail = f" free_memory_mib={int(candidate.capacity('memory_mib') or 0)}"
        elif reason == "high_utilization":
            detail = f" utilization_percent={int(candidate.utilization_percent or 0)}"
        elif reason == "stability_window":
            detail = f" idle_seconds={int(decision.stable_idle_seconds)}"
        log_waiting(
            reason,
            "FETA_GPU_SCHEDULER "
            f"mode={policy.mode} gpu={physical_gpu_index} state=WAITING "
            f"reason={reason}{detail}",
            now,
        )

    provider = FeTAGPUResourceProvider(
        selected_probe,
        physical_gpu_index=physical_gpu_index,
        current_pid=own_pid,
    )
    broker = ResourceBroker(
        provider,
        CourtesyResourceAdmissionPolicy(
            maximum_utilization_percent=policy.maximum_utilization_percent
        ),
        poll_seconds=policy.poll_seconds,
        sleeper=sleeper,
        clock=clock,
        decision_observer=observe,
    )
    try:
        admission = broker.wait_for_admission(gpu_resource_request(policy))
    except ResourceInspectionError as exc:
        cause = exc.__cause__
        if isinstance(cause, (RuntimeError, ValueError)) and str(cause) in {
            "feta_search_gpu_probe_failed",
            "feta_search_gpu_probe_parse_failed",
        }:
            raise cause
        raise RuntimeError("feta_search_gpu_probe_failed") from exc
    except InvalidResourceState as exc:
        raise RuntimeError("feta_search_gpu_probe_failed") from exc

    telemetry = admission.telemetry
    free_memory_mib = next(
        int(capacity.value)
        for capacity in telemetry.observed_capacities
        if capacity.name == "memory_mib"
    )
    utilization_percent = int(telemetry.utilization_percent or 0)
    logger(
        "FETA_GPU_SCHEDULER "
        f"mode={policy.mode} gpu={physical_gpu_index} state=ADMITTED "
        f"wait_seconds={int(telemetry.wait_seconds)} "
        f"free_memory_mib={free_memory_mib} "
        f"utilization_percent={utilization_percent}"
    )
    return GPUAdmissionTelemetry(
        mode=policy.mode,
        physical_gpu_index=physical_gpu_index,
        wait_seconds=telemetry.wait_seconds,
        poll_count=telemetry.poll_count,
        free_memory_mib=free_memory_mib,
        utilization_percent=utilization_percent,
        foreign_process_count=telemetry.foreign_owner_count,
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
