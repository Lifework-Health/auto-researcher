"""Generic whole-GPU discovery and worker-local CUDA process binding."""

from __future__ import annotations

import os
import pwd
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from auto_researcher.resources.models import (
    ResourceCandidate,
    ResourceCapacity,
    ResourceLease,
    ResourceOwner,
    ResourceRequest,
)


class InvalidGPUProviderSnapshot(RuntimeError):
    pass


@dataclass(frozen=True)
class NvidiaComputeProcess:
    pid: int
    username: str | None = None


@dataclass(frozen=True)
class NvidiaGPUObservation:
    physical_index: int
    uuid: str
    free_memory_mib: int
    utilization_percent: int
    compute_processes: tuple[NvidiaComputeProcess, ...] = ()


class NvidiaGPUEnumerator(Protocol):
    def observations(self) -> tuple[NvidiaGPUObservation, ...]: ...


def _username_for_pid(pid: int) -> str | None:
    try:
        uid = Path(f"/proc/{pid}").stat().st_uid
        return pwd.getpwuid(uid).pw_name
    except (KeyError, OSError):
        return None


class NvidiaSmiGPUEnumerator:
    """Enumerate every visible physical GPU without importing CUDA libraries."""

    def _query(self, arguments: list[str]) -> str:
        try:
            completed = subprocess.run(
                ["nvidia-smi", *arguments],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            raise InvalidGPUProviderSnapshot("nvidia_gpu_inspection_failed") from None
        if completed.returncode != 0:
            raise InvalidGPUProviderSnapshot("nvidia_gpu_inspection_failed")
        return completed.stdout

    def observations(self) -> tuple[NvidiaGPUObservation, ...]:
        gpu_output = self._query(
            [
                "--query-gpu=index,uuid,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
        )
        process_output = self._query(
            [
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ]
        )
        processes: dict[str, list[NvidiaComputeProcess]] = {}
        process_rows = [
            row.strip() for row in process_output.splitlines() if row.strip()
        ]
        if process_rows and all(
            row.casefold().startswith("no running processes") for row in process_rows
        ):
            process_rows = []
        for row in process_rows:
            fields = [field.strip() for field in row.split(",")]
            if len(fields) != 2 or re.fullmatch(r"[0-9]+", fields[1]) is None:
                raise InvalidGPUProviderSnapshot("nvidia_gpu_snapshot_invalid")
            pid = int(fields[1])
            processes.setdefault(fields[0], []).append(
                NvidiaComputeProcess(pid=pid, username=_username_for_pid(pid))
            )

        observations: list[NvidiaGPUObservation] = []
        for row in (row.strip() for row in gpu_output.splitlines() if row.strip()):
            fields = [field.strip() for field in row.split(",")]
            try:
                if len(fields) != 4 or not fields[1]:
                    raise ValueError
                index = int(fields[0])
                free_memory = int(fields[2])
                utilization = int(fields[3])
                if index < 0 or free_memory < 0 or not 0 <= utilization <= 100:
                    raise ValueError
            except ValueError:
                raise InvalidGPUProviderSnapshot(
                    "nvidia_gpu_snapshot_invalid"
                ) from None
            observations.append(
                NvidiaGPUObservation(
                    physical_index=index,
                    uuid=fields[1],
                    free_memory_mib=free_memory,
                    utilization_percent=utilization,
                    compute_processes=tuple(processes.get(fields[1], ())),
                )
            )
        indexes = [item.physical_index for item in observations]
        if not observations or len(indexes) != len(set(indexes)):
            raise InvalidGPUProviderSnapshot("nvidia_gpu_snapshot_invalid")
        return tuple(sorted(observations, key=lambda item: item.physical_index))


class NvidiaGPUResourceProvider:
    """Expose each independently allocatable physical NVIDIA GPU as one candidate."""

    def __init__(
        self,
        enumerator: NvidiaGPUEnumerator | None = None,
        *,
        current_pid: int | None = None,
        eligible_indexes: frozenset[int] | None = None,
        equivalence_tags: frozenset[str] = frozenset(),
    ) -> None:
        self.enumerator = enumerator or NvidiaSmiGPUEnumerator()
        self.current_pid = os.getpid() if current_pid is None else current_pid
        self.eligible_indexes = eligible_indexes
        self.equivalence_tags = equivalence_tags

    def candidates(self, _request: ResourceRequest) -> tuple[ResourceCandidate, ...]:
        try:
            observations = self.enumerator.observations()
        except InvalidGPUProviderSnapshot:
            raise
        except Exception:
            raise InvalidGPUProviderSnapshot("nvidia_gpu_inspection_failed") from None
        candidates = []
        for observed in observations:
            if (
                self.eligible_indexes is not None
                and observed.physical_index not in self.eligible_indexes
            ):
                continue
            foreign = tuple(
                process
                for process in observed.compute_processes
                if process.pid != self.current_pid
            )
            candidates.append(
                ResourceCandidate(
                    resource_id=f"gpu:{observed.physical_index}",
                    resource_type="gpu",
                    quantity=1,
                    capacities=(
                        ResourceCapacity(
                            name="memory_mib", value=observed.free_memory_mib
                        ),
                    ),
                    utilization_percent=observed.utilization_percent,
                    foreign_owners=tuple(
                        ResourceOwner(
                            namespace="process_pid",
                            owner_id=str(process.pid),
                            display_name=process.username,
                        )
                        for process in foreign
                    ),
                    equivalence_tags=frozenset({"nvidia-cuda", "whole-physical-gpu"})
                    | self.equivalence_tags,
                )
            )
        return tuple(candidates)


def physical_gpu_index(resource_id: str) -> int:
    match = re.fullmatch(r"gpu:([0-9]+)", resource_id)
    if match is None:
        raise ValueError("leased_resource_is_not_a_physical_gpu")
    return int(match.group(1))


def cuda_environment_for_lease(
    lease: ResourceLease,
    *,
    base_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return a child-process environment; never mutate process-global state."""

    environment = dict(os.environ if base_environment is None else base_environment)
    environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_index(lease.resource_id))
    return environment
