from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from auto_researcher.resources import (
    AdmissionClass,
    CourtesyResourceAdmissionPolicy,
    InMemoryResourceLeaseStore,
    InvalidResourceRequest,
    InvalidResourceState,
    PreemptionPolicy,
    ResourceBroker,
    ResourceCandidate,
    ResourceCapacity,
    ResourceInspectionError,
    ResourceLeaseConflict,
    ResourceOwner,
    ResourceProvider,
    ResourceRequest,
    ResourceRequirement,
    ResourceWaitTimeout,
)
from auto_researcher.runtime.identity import payload_hash
from auto_researcher.tasks.feta_seg_search.gpu_scheduler import (
    GPUSchedulerPolicy,
    gpu_resource_request,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.wall_now = datetime(2026, 8, 13, tzinfo=timezone.utc)
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def wall(self) -> datetime:
        return self.wall_now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds
        self.wall_now += timedelta(seconds=seconds)


class SequenceProvider:
    def __init__(self, observations: list[tuple[ResourceCandidate, ...]]) -> None:
        self.observations = observations
        self.calls = 0

    def candidates(self, _request: ResourceRequest) -> tuple[ResourceCandidate, ...]:
        index = min(self.calls, len(self.observations) - 1)
        self.calls += 1
        return self.observations[index]


def cpu_candidate(
    resource_id: str = "cpu:0",
    *,
    available: bool = True,
    quantity: int = 8,
    foreign_owners: tuple[ResourceOwner, ...] = (),
    tags: frozenset[str] = frozenset({"x86_64", "avx2"}),
    state_valid: bool = True,
    state_reason: str | None = None,
) -> ResourceCandidate:
    return ResourceCandidate(
        resource_id=resource_id,
        resource_type="cpu",
        quantity=quantity,
        capacities=(ResourceCapacity(name="ram_mib", value=16384),),
        utilization_percent=0,
        available=available,
        foreign_owners=foreign_owners,
        equivalence_tags=tags,
        state_valid=state_valid,
        state_reason=state_reason,
    )


def request(
    *,
    request_id: str = "candidate-boundary-17",
    admission_class: AdmissionClass = AdmissionClass.PRIMARY,
    stable_idle_seconds: float = 0,
    maximum_wait_seconds: float = 300,
    priority: int = 10,
    preemption: PreemptionPolicy = PreemptionPolicy.NEVER,
) -> ResourceRequest:
    return ResourceRequest(
        request_id=request_id,
        requirements=(
            ResourceRequirement(
                resource_type="cpu",
                quantity=4,
                minimum_capacities=(ResourceCapacity(name="ram_mib", value=8192),),
            ),
        ),
        admission_class=admission_class,
        priority=priority,
        maximum_wait_seconds=maximum_wait_seconds,
        preemption=preemption,
        stable_idle_seconds=stable_idle_seconds,
        equivalence_requirements=frozenset({"x86_64"}),
    )


def broker_for(
    provider: ResourceProvider, clock: FakeClock, **kwargs
) -> ResourceBroker:
    return ResourceBroker(
        provider,
        CourtesyResourceAdmissionPolicy(maximum_utilization_percent=80),
        poll_seconds=10,
        sleeper=clock.sleep,
        clock=clock.monotonic,
        wall_clock=clock.wall,
        **kwargs,
    )


def test_primary_resource_is_admitted_immediately() -> None:
    provider = SequenceProvider([(cpu_candidate(),)])
    clock = FakeClock()

    admission = broker_for(provider, clock).wait_for_admission(request())

    assert admission.decision.reason == "eligible"
    assert admission.telemetry.wait_seconds == 0
    assert admission.telemetry.poll_count == 1
    assert clock.sleeps == []


def test_feta_policy_maps_to_generic_gpu_count_vram_and_admission_class() -> None:
    policy = GPUSchedulerPolicy(
        mode="opportunistic",
        physical_gpu_index=1,
        minimum_free_memory_mib=42000,
    )

    mapped = gpu_resource_request(policy)

    assert mapped.admission_class is AdmissionClass.OPPORTUNISTIC
    assert mapped.stable_idle_seconds == 60
    assert mapped.preemption.value == "never"
    assert mapped.requirements[0].resource_type == "gpu"
    assert mapped.requirements[0].quantity == 1
    assert mapped.requirements[0].minimum_capacities == (
        ResourceCapacity(name="memory_mib", value=42000),
    )


def test_opportunistic_admission_requires_continuously_idle_stability_window() -> None:
    provider = SequenceProvider([(cpu_candidate(),)])
    clock = FakeClock()

    admission = broker_for(provider, clock).wait_for_admission(
        request(
            admission_class=AdmissionClass.OPPORTUNISTIC,
            stable_idle_seconds=25,
        )
    )

    assert admission.telemetry.wait_seconds == 30
    assert admission.telemetry.observed_continuous_idle_seconds == 30
    assert admission.telemetry.required_stable_idle_seconds == 25
    assert provider.calls == 4
    assert clock.sleeps == [10, 10, 10]


def test_busy_waits_and_automatically_admits_without_scientific_failure() -> None:
    provider = SequenceProvider([(cpu_candidate(available=False),), (cpu_candidate(),)])
    clock = FakeClock()

    admission = broker_for(provider, clock).wait_for_admission(request())

    assert admission.decision.reason == "eligible"
    assert admission.telemetry.wait_seconds == 10
    assert provider.calls == 2
    assert clock.sleeps == [10]


def test_foreign_owner_prevents_opportunistic_admission_until_it_leaves() -> None:
    owner = ResourceOwner(namespace="scheduler_worker", owner_id="other-worker")
    provider = SequenceProvider(
        [
            (cpu_candidate(foreign_owners=(owner,)),),
            (cpu_candidate(),),
        ]
    )
    clock = FakeClock()

    admission = broker_for(provider, clock).wait_for_admission(
        request(admission_class=AdmissionClass.OPPORTUNISTIC)
    )

    assert admission.telemetry.wait_seconds == 10
    assert admission.telemetry.foreign_owner_count == 0


def test_disappearing_candidate_resets_continuously_observed_idle_window() -> None:
    provider = SequenceProvider(
        [
            (cpu_candidate(),),
            (),
            (cpu_candidate(),),
            (cpu_candidate(),),
            (cpu_candidate(),),
        ]
    )
    clock = FakeClock()

    admission = broker_for(provider, clock).wait_for_admission(
        request(
            admission_class=AdmissionClass.OPPORTUNISTIC,
            stable_idle_seconds=20,
        )
    )

    assert admission.telemetry.wait_seconds == 40
    assert admission.telemetry.observed_continuous_idle_seconds == 20
    assert provider.calls == 5


def test_stability_timer_resets_when_resource_becomes_busy() -> None:
    provider = SequenceProvider(
        [
            (cpu_candidate(),),
            (cpu_candidate(),),
            (cpu_candidate(available=False),),
            (cpu_candidate(),),
        ]
    )
    clock = FakeClock()

    admission = broker_for(provider, clock).wait_for_admission(
        request(
            admission_class=AdmissionClass.OPPORTUNISTIC,
            stable_idle_seconds=20,
        )
    )

    assert admission.telemetry.wait_seconds == 50
    assert clock.sleeps == [10, 10, 10, 10, 10]


def test_each_candidate_boundary_rechecks_current_resource_state() -> None:
    provider = SequenceProvider(
        [
            (cpu_candidate(),),
            (cpu_candidate(available=False),),
            (cpu_candidate(),),
        ]
    )
    clock = FakeClock()
    broker = broker_for(provider, clock)

    first = broker.wait_for_admission(request())
    second = broker.wait_for_admission(request(request_id="candidate-boundary-18"))

    assert first.telemetry.wait_seconds == 0
    assert second.telemetry.wait_seconds == 10
    assert provider.calls == 3


def test_broker_acquires_renews_and_releases_resource_lease() -> None:
    provider = SequenceProvider([(cpu_candidate(),)])
    clock = FakeClock()
    store = InMemoryResourceLeaseStore()
    broker = broker_for(provider, clock, lease_store=store)

    admission = broker.acquire(
        request(), worker_id="worker-a", lease_ttl=timedelta(seconds=30)
    )
    assert admission.lease is not None
    assert admission.lease.allocation_semantics == "whole_candidate"
    assert store.active_for("cpu:0", now=clock.wall()) == admission.lease

    clock.sleep(10)
    renewed = broker.renew_lease(
        admission.lease.lease_id,
        worker_id="worker-a",
        lease_ttl=timedelta(seconds=30),
    )
    assert renewed.heartbeat_at == clock.wall()

    released = broker.release_lease(admission.lease.lease_id, worker_id="worker-a")
    assert released.released_at == clock.wall()
    assert store.active_for("cpu:0", now=clock.wall()) is None


def test_stale_lease_is_detected_recovered_and_resource_can_be_reclaimed() -> None:
    clock = FakeClock()
    store = InMemoryResourceLeaseStore()
    candidate = cpu_candidate()
    first = store.acquire(
        request(),
        candidate,
        worker_id="worker-a",
        now=clock.wall(),
        ttl=timedelta(seconds=20),
    )
    clock.sleep(20)

    assert store.stale(now=clock.wall()) == (first,)
    recovered = store.recover_stale(now=clock.wall())
    assert recovered[0].released_at == clock.wall()
    second = store.acquire(
        request(),
        candidate,
        worker_id="worker-b",
        now=clock.wall(),
        ttl=timedelta(seconds=20),
    )
    assert second.worker_id == "worker-b"


def test_restart_reacquire_recovers_exact_same_live_lease() -> None:
    clock = FakeClock()
    store = InMemoryResourceLeaseStore()
    first_broker = broker_for(
        SequenceProvider([(cpu_candidate(),)]), clock, lease_store=store
    )
    first = first_broker.acquire(
        request(), worker_id="worker-a", lease_ttl=timedelta(seconds=30)
    )
    assert first.lease is not None

    restarted_broker = broker_for(
        SequenceProvider([(cpu_candidate(),)]), clock, lease_store=store
    )
    recovered = restarted_broker.acquire(
        request(), worker_id="worker-a", lease_ttl=timedelta(seconds=30)
    )

    assert recovered.decision.reason == "lease_recovered"
    assert recovered.lease is first.lease
    assert recovered.lease.lease_id == first.lease.lease_id
    assert recovered.lease.expires_at == first.lease.expires_at


def test_atomic_whole_candidate_lease_rejects_other_worker_or_request() -> None:
    clock = FakeClock()
    store = InMemoryResourceLeaseStore()
    candidate = cpu_candidate(quantity=8)
    original = store.acquire(
        request(),
        candidate,
        worker_id="worker-a",
        now=clock.wall(),
        ttl=timedelta(seconds=20),
    )
    assert original.allocation_semantics == "whole_candidate"

    with pytest.raises(ResourceLeaseConflict, match="resource_already_leased"):
        store.acquire(
            request(),
            candidate,
            worker_id="worker-b",
            now=clock.wall(),
            ttl=timedelta(seconds=20),
        )
    with pytest.raises(ResourceLeaseConflict, match="resource_already_leased"):
        store.acquire(
            request(request_id="different-request"),
            candidate,
            worker_id="worker-a",
            now=clock.wall(),
            ttl=timedelta(seconds=20),
        )


def test_whole_candidate_lease_does_not_partially_suballocate_quantity() -> None:
    clock = FakeClock()
    store = InMemoryResourceLeaseStore()
    candidate = cpu_candidate(quantity=8)
    store.acquire(
        request(request_id="four-of-eight-a"),
        candidate,
        worker_id="worker-a",
        now=clock.wall(),
        ttl=timedelta(seconds=20),
    )

    with pytest.raises(ResourceLeaseConflict, match="resource_already_leased"):
        store.acquire(
            request(request_id="four-of-eight-b"),
            candidate,
            worker_id="worker-b",
            now=clock.wall(),
            ttl=timedelta(seconds=20),
        )


def test_invalid_resource_state_fails_closed() -> None:
    provider = SequenceProvider(
        [
            (
                cpu_candidate(
                    state_valid=False,
                    state_reason="provider_state_unparseable",
                ),
            )
        ]
    )
    clock = FakeClock()

    with pytest.raises(InvalidResourceState, match="provider_state_unparseable"):
        broker_for(provider, clock).wait_for_admission(request())


def test_invalid_candidate_does_not_poison_valid_equivalent_candidate() -> None:
    provider = SequenceProvider(
        [
            (
                cpu_candidate(
                    "cpu:a",
                    state_valid=False,
                    state_reason="candidate_state_invalid",
                ),
                cpu_candidate("cpu:b"),
            )
        ]
    )

    admission = broker_for(provider, FakeClock()).wait_for_admission(request())

    assert admission.telemetry.resource_id == "cpu:b"


def test_provider_inspection_failure_fails_closed() -> None:
    class BrokenProvider:
        def candidates(self, _request: ResourceRequest):
            raise ValueError("snapshot unavailable")

    with pytest.raises(ResourceInspectionError, match="resource_inspection_failed"):
        broker_for(BrokenProvider(), FakeClock()).wait_for_admission(request())


@pytest.mark.parametrize(
    "invalid_request",
    [
        request(preemption=PreemptionPolicy.ALLOW),
        request().model_copy(
            update={
                "requirements": request().requirements
                + (ResourceRequirement(resource_type="ram", quantity=1),)
            }
        ),
    ],
)
def test_structurally_unsupported_request_fails_before_provider_inspection(
    invalid_request: ResourceRequest,
) -> None:
    provider = SequenceProvider([(cpu_candidate(),)])

    with pytest.raises(InvalidResourceRequest):
        broker_for(provider, FakeClock()).wait_for_admission(invalid_request)

    assert provider.calls == 0


def test_maximum_wait_shorter_than_poll_interval_consumes_exact_deadline() -> None:
    provider = SequenceProvider([(cpu_candidate(available=False),)])
    clock = FakeClock()

    with pytest.raises(ResourceWaitTimeout, match="maximum_wait"):
        broker_for(provider, clock).wait_for_admission(request(maximum_wait_seconds=3))

    assert clock.now == 3
    assert clock.sleeps == [3]
    assert provider.calls == 2


def test_non_divisible_maximum_wait_permits_final_deadline_observation() -> None:
    provider = SequenceProvider([(cpu_candidate(available=False),)])
    clock = FakeClock()

    with pytest.raises(ResourceWaitTimeout, match="maximum_wait"):
        broker_for(provider, clock).wait_for_admission(request(maximum_wait_seconds=25))

    assert clock.now == 25
    assert clock.sleeps == [10, 10, 5]
    assert provider.calls == 4


def test_resource_becoming_available_before_deadline_is_admitted() -> None:
    provider = SequenceProvider(
        [
            (cpu_candidate(available=False),),
            (cpu_candidate(available=False),),
            (cpu_candidate(),),
        ]
    )
    clock = FakeClock()

    admission = broker_for(provider, clock).wait_for_admission(
        request(maximum_wait_seconds=25)
    )

    assert admission.telemetry.wait_seconds == 20
    assert clock.sleeps == [10, 10]


def test_resource_unavailable_through_exact_deadline_times_out() -> None:
    provider = SequenceProvider(
        [
            (cpu_candidate(available=False),),
            (cpu_candidate(available=False),),
            (cpu_candidate(available=False),),
        ]
    )
    clock = FakeClock()

    with pytest.raises(ResourceWaitTimeout, match="maximum_wait"):
        broker_for(provider, clock).wait_for_admission(request(maximum_wait_seconds=20))

    assert clock.now == 20
    assert provider.calls == 3


def test_priority_is_coordination_intent_not_local_eligibility() -> None:
    admitted_resources = []
    for priority in (-100, 0, 100):
        admission = broker_for(
            SequenceProvider([(cpu_candidate(),)]), FakeClock()
        ).wait_for_admission(request(priority=priority))
        admitted_resources.append(admission.telemetry.resource_id)

    assert admitted_resources == ["cpu:0", "cpu:0", "cpu:0"]


def test_scientific_identity_is_invariant_across_equivalent_resources() -> None:
    scientific_configuration = {
        "dataset": "locked-split-v1",
        "seed": 42,
        "evaluator": "trusted-evaluator-v1",
        "learning_rate": 0.001,
    }
    identity = payload_hash(scientific_configuration)

    assignments = []
    for resource_id in ("cpu:east", "cpu:west"):
        provider = SequenceProvider([(cpu_candidate(resource_id),)])
        admission = broker_for(provider, FakeClock()).wait_for_admission(request())
        assignments.append(admission.telemetry.resource_id)
        assert payload_hash(scientific_configuration) == identity

    assert assignments == ["cpu:east", "cpu:west"]


def test_resource_telemetry_is_separate_from_scientific_metrics() -> None:
    scientific_metrics = {"objective_score": 0.91, "metric": "macro_dice"}
    provider = SequenceProvider([(cpu_candidate(),)])
    telemetry = (
        broker_for(provider, FakeClock()).wait_for_admission(request()).telemetry
    )

    operational = telemetry.as_operational_telemetry()

    assert scientific_metrics == {"objective_score": 0.91, "metric": "macro_dice"}
    assert set(scientific_metrics).isdisjoint(operational)
    assert "objective_score" not in operational
    assert "resource_id" in operational
