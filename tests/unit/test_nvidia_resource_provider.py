from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

from auto_researcher.resources import (
    AdmissionClass,
    CourtesyResourceAdmissionPolicy,
    InMemoryResourceLeaseStore,
    NvidiaComputeProcess,
    NvidiaGPUObservation,
    NvidiaGPUResourceProvider,
    ResourceBroker,
    ResourceCapacity,
    ResourceRequest,
    ResourceRequirement,
    ResourceWaitTimeout,
    cuda_environment_for_lease,
)
from auto_researcher.runtime.identity import payload_hash
from auto_researcher.tasks.feta_seg_search.gpu_scheduler import (
    GPUSchedulerPolicy,
    gpu_resource_request,
)


class FakeEnumerator:
    def __init__(self, count: int) -> None:
        self.items = tuple(
            NvidiaGPUObservation(
                physical_index=index,
                uuid=f"GPU-{index}",
                free_memory_mib=80_000 - index,
                utilization_percent=index,
            )
            for index in range(count)
        )

    def observations(self) -> tuple[NvidiaGPUObservation, ...]:
        return self.items


def request(request_id: str, *, wait: float | None = None) -> ResourceRequest:
    return ResourceRequest(
        request_id=request_id,
        requirements=(
            ResourceRequirement(
                resource_type="gpu",
                quantity=1,
                minimum_capacities=(ResourceCapacity(name="memory_mib", value=40_000),),
            ),
        ),
        admission_class=AdmissionClass.PRIMARY,
        maximum_wait_seconds=wait,
        equivalence_requirements=frozenset({"nvidia-cuda", "whole-physical-gpu"}),
    )


def test_one_and_three_gpu_snapshots_expose_independent_whole_candidates() -> None:
    one = NvidiaGPUResourceProvider(FakeEnumerator(1), current_pid=99).candidates(
        request("one")
    )
    three = NvidiaGPUResourceProvider(FakeEnumerator(3), current_pid=99).candidates(
        request("three")
    )

    assert [candidate.resource_id for candidate in one] == ["gpu:0"]
    assert [candidate.resource_id for candidate in three] == [
        "gpu:0",
        "gpu:1",
        "gpu:2",
    ]
    assert all(candidate.quantity == 1 for candidate in three)


def test_three_requests_lease_three_gpus_and_fourth_waits() -> None:
    monotonic = [0.0]

    def sleep(seconds: float) -> None:
        monotonic[0] += seconds

    store = InMemoryResourceLeaseStore()
    broker = ResourceBroker(
        NvidiaGPUResourceProvider(FakeEnumerator(3), current_pid=99),
        CourtesyResourceAdmissionPolicy(maximum_utilization_percent=10),
        lease_store=store,
        poll_seconds=1,
        sleeper=sleep,
        clock=lambda: monotonic[0],
        wall_clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc)
        + timedelta(seconds=monotonic[0]),
    )
    admissions = [
        broker.acquire(
            request(f"trial-{index}"),
            worker_id=f"worker-{index}",
            lease_ttl=timedelta(minutes=5),
        )
        for index in range(3)
    ]
    assert {item.lease.resource_id for item in admissions if item.lease} == {
        "gpu:0",
        "gpu:1",
        "gpu:2",
    }
    with pytest.raises(ResourceWaitTimeout):
        broker.acquire(
            request("trial-4", wait=1),
            worker_id="worker-4",
            lease_ttl=timedelta(minutes=5),
        )


def test_same_request_restart_recovers_original_gpu_without_renewal() -> None:
    now = datetime(2026, 8, 14, tzinfo=timezone.utc)
    store = InMemoryResourceLeaseStore()
    broker = ResourceBroker(
        NvidiaGPUResourceProvider(FakeEnumerator(3), current_pid=99),
        CourtesyResourceAdmissionPolicy(),
        lease_store=store,
        poll_seconds=1,
        wall_clock=lambda: now,
    )
    first = broker.acquire(
        request("stable-trial"),
        worker_id="worker",
        lease_ttl=timedelta(minutes=5),
    )
    recovered = broker.acquire(
        request("stable-trial"),
        worker_id="worker",
        lease_ttl=timedelta(hours=1),
    )

    assert recovered.lease == first.lease
    assert recovered.lease.expires_at == first.lease.expires_at


def test_cuda_binding_uses_lease_without_mutating_process_environment() -> None:
    now = datetime(2026, 8, 14, tzinfo=timezone.utc)
    candidates = NvidiaGPUResourceProvider(
        FakeEnumerator(2), current_pid=99
    ).candidates(request("trial-placement"))
    store = InMemoryResourceLeaseStore()
    first_placement = store.acquire(
        request("trial-placement-0"),
        candidates[0],
        worker_id="worker-0",
        now=now,
        ttl=timedelta(minutes=5),
    )
    lease = store.acquire(
        request("trial-placement"),
        candidates[1],
        worker_id="worker",
        now=now,
        ttl=timedelta(minutes=5),
    )
    scientific_payload = {
        "configuration": {"learning_rate": 0.001, "maximum_epochs": 25},
        "dataset_version": "feta-2.1",
    }
    identity_before_placement = payload_hash(scientific_payload)
    original = os.environ.get("CUDA_VISIBLE_DEVICES")
    child = cuda_environment_for_lease(
        lease,
        base_environment={"SCIENTIFIC_IDENTITY": "unchanged"},
    )
    other_child = cuda_environment_for_lease(
        first_placement,
        base_environment={"SCIENTIFIC_IDENTITY": "unchanged"},
    )

    assert child == {
        "SCIENTIFIC_IDENTITY": "unchanged",
        "CUDA_VISIBLE_DEVICES": "1",
    }
    assert other_child["CUDA_VISIBLE_DEVICES"] == "0"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == original
    assert payload_hash(scientific_payload) == identity_before_placement


def test_compute_process_owners_are_classified_and_current_pid_is_not_foreign() -> None:
    class Enumerator:
        def observations(self):
            return (
                NvidiaGPUObservation(
                    physical_index=0,
                    uuid="GPU-0",
                    free_memory_mib=80_000,
                    utilization_percent=0,
                    compute_processes=(
                        NvidiaComputeProcess(pid=10, username="self"),
                        NvidiaComputeProcess(pid=11, username="other"),
                    ),
                ),
            )

    candidate = NvidiaGPUResourceProvider(Enumerator(), current_pid=10).candidates(
        request("owners")
    )[0]
    assert [
        (owner.owner_id, owner.display_name) for owner in candidate.foreign_owners
    ] == [("11", "other")]


def test_equivalent_pool_requires_logical_work_identity_not_physical_gpu() -> None:
    policy = GPUSchedulerPolicy(
        mode="primary",
        gpu_selection="equivalent_pool",
        physical_gpu_index=None,
    )
    with pytest.raises(ValueError, match="logical_request_id"):
        gpu_resource_request(policy)

    mapped = gpu_resource_request(policy, request_id="run-study-trial-7")
    assert mapped.request_id == "run-study-trial-7"
    assert "gpu:0" not in mapped.request_id
    assert mapped.equivalence_requirements == frozenset(
        {"nvidia-cuda", "whole-physical-gpu"}
    )
