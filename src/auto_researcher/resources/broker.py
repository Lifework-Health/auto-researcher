"""Generic courteous admission policy and polling resource broker."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

from auto_researcher.resources.leases import ResourceLeaseConflict
from auto_researcher.resources.models import (
    AdmissionOutcome,
    PreemptionPolicy,
    ResourceAdmission,
    ResourceAdmissionDecision,
    ResourceAdmissionTelemetry,
    ResourceCandidate,
    ResourceLease,
    ResourceRequest,
)
from auto_researcher.resources.protocols import (
    ResourceAdmissionPolicy,
    ResourceLeaseStore,
    ResourceProvider,
)


class ResourceBrokerError(RuntimeError):
    pass


class ResourceInspectionError(ResourceBrokerError):
    pass


class InvalidResourceState(ResourceBrokerError):
    pass


class ResourceWaitTimeout(ResourceBrokerError):
    pass


class CourtesyResourceAdmissionPolicy:
    """Task-neutral busy/foreign-owner/capacity/stability admission semantics."""

    def __init__(self, *, maximum_utilization_percent: float | None = None) -> None:
        if maximum_utilization_percent is not None and not (
            0 <= maximum_utilization_percent <= 100
        ):
            raise ValueError("maximum utilization must be between 0 and 100")
        self.maximum_utilization_percent = maximum_utilization_percent

    def decide(
        self,
        request: ResourceRequest,
        candidate: ResourceCandidate,
        *,
        worker_id: str | None,
        stable_idle_seconds: float,
    ) -> ResourceAdmissionDecision:
        def decision(
            outcome: AdmissionOutcome,
            reason: str,
            *,
            continuously_idle: bool = False,
        ) -> ResourceAdmissionDecision:
            return ResourceAdmissionDecision(
                outcome=outcome,
                request_id=request.request_id,
                resource_id=candidate.resource_id,
                reason=reason,
                stable_idle_seconds=stable_idle_seconds,
                continuously_idle=continuously_idle,
            )

        if not candidate.state_valid:
            return decision(
                AdmissionOutcome.REJECTED, candidate.state_reason or "invalid_state"
            )
        if len(request.requirements) != 1:
            return decision(
                AdmissionOutcome.REJECTED, "resource_bundle_not_implemented"
            )
        matching = tuple(
            requirement
            for requirement in request.requirements
            if requirement.resource_type == candidate.resource_type
        )
        if len(matching) != 1:
            return decision(AdmissionOutcome.WAIT, "resource_type_mismatch")
        requirement = matching[0]
        if not request.equivalence_requirements.issubset(candidate.equivalence_tags):
            return decision(AdmissionOutcome.WAIT, "equivalence_mismatch")
        if candidate.quantity < requirement.quantity:
            return decision(AdmissionOutcome.WAIT, "insufficient_quantity")
        for minimum in requirement.minimum_capacities:
            observed = candidate.capacity(minimum.name)
            if observed is None:
                return decision(AdmissionOutcome.REJECTED, "capacity_state_missing")
            if observed < minimum.value:
                return decision(AdmissionOutcome.WAIT, f"low_{minimum.name}")
        foreign_owners = tuple(
            owner for owner in candidate.foreign_owners if owner != worker_id
        )
        if foreign_owners:
            return decision(AdmissionOutcome.WAIT, "foreign_owner")
        if not candidate.available:
            return decision(AdmissionOutcome.WAIT, "resource_busy")
        if (
            self.maximum_utilization_percent is not None
            and candidate.utilization_percent is None
        ):
            return decision(AdmissionOutcome.REJECTED, "utilization_state_missing")
        if (
            self.maximum_utilization_percent is not None
            and candidate.utilization_percent is not None
            and candidate.utilization_percent > self.maximum_utilization_percent
        ):
            return decision(AdmissionOutcome.WAIT, "high_utilization")
        if stable_idle_seconds < request.stable_idle_seconds:
            return decision(
                AdmissionOutcome.WAIT,
                "stability_window",
                continuously_idle=True,
            )
        return decision(AdmissionOutcome.ADMITTED, "eligible", continuously_idle=True)


DecisionObserver = Callable[[ResourceAdmissionDecision, ResourceCandidate, float], None]


class ResourceBroker:
    """Re-check a provider until policy admits, optionally claiming an atomic lease."""

    def __init__(
        self,
        provider: ResourceProvider,
        policy: ResourceAdmissionPolicy,
        *,
        lease_store: ResourceLeaseStore | None = None,
        poll_seconds: float = 20,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        decision_observer: DecisionObserver | None = None,
    ) -> None:
        if poll_seconds <= 0:
            raise ValueError("resource broker poll_seconds must be positive")
        self.provider = provider
        self.policy = policy
        self.lease_store = lease_store
        self.poll_seconds = poll_seconds
        self.sleeper = sleeper
        self.clock = clock
        self.wall_clock = wall_clock
        self.decision_observer = decision_observer

    def wait_for_admission(self, request: ResourceRequest) -> ResourceAdmission:
        return self._wait(request, worker_id=None, lease_ttl=None)

    def acquire(
        self,
        request: ResourceRequest,
        *,
        worker_id: str,
        lease_ttl: timedelta,
    ) -> ResourceAdmission:
        if self.lease_store is None:
            raise ResourceBrokerError("resource_lease_store_not_configured")
        if not worker_id or lease_ttl <= timedelta(0):
            raise ValueError("worker_id and a positive lease ttl are required")
        return self._wait(request, worker_id=worker_id, lease_ttl=lease_ttl)

    def renew_lease(
        self,
        lease_id: str,
        *,
        worker_id: str,
        lease_ttl: timedelta,
    ) -> ResourceLease:
        if self.lease_store is None:
            raise ResourceBrokerError("resource_lease_store_not_configured")
        return self.lease_store.renew(
            lease_id,
            worker_id=worker_id,
            now=self.wall_clock(),
            ttl=lease_ttl,
        )

    def release_lease(self, lease_id: str, *, worker_id: str) -> ResourceLease:
        if self.lease_store is None:
            raise ResourceBrokerError("resource_lease_store_not_configured")
        return self.lease_store.release(
            lease_id, worker_id=worker_id, now=self.wall_clock()
        )

    def recover_stale_leases(self) -> tuple[ResourceLease, ...]:
        if self.lease_store is None:
            raise ResourceBrokerError("resource_lease_store_not_configured")
        return self.lease_store.recover_stale(now=self.wall_clock())

    def _observe(
        self,
        decision: ResourceAdmissionDecision,
        candidate: ResourceCandidate,
        now: float,
    ) -> None:
        if self.decision_observer is not None:
            self.decision_observer(decision, candidate, now)

    def _wait(
        self,
        request: ResourceRequest,
        *,
        worker_id: str | None,
        lease_ttl: timedelta | None,
    ) -> ResourceAdmission:
        if request.preemption is not PreemptionPolicy.NEVER:
            raise ResourceBrokerError("resource_preemption_not_implemented")
        started = self.clock()
        idle_started: dict[str, float] = {}
        poll_count = 0
        while True:
            try:
                candidates = tuple(self.provider.candidates(request))
            except ResourceBrokerError:
                raise
            except Exception as exc:
                raise ResourceInspectionError("resource_inspection_failed") from exc
            poll_count += 1
            now = self.clock()
            if not candidates:
                raise InvalidResourceState("resource_provider_returned_no_candidates")

            for candidate in sorted(candidates, key=lambda item: item.resource_id):
                if not candidate.state_valid:
                    raise InvalidResourceState(
                        candidate.state_reason or "invalid_resource_state"
                    )
                active_lease = (
                    None
                    if self.lease_store is None
                    else self.lease_store.active_for(
                        candidate.resource_id, now=self.wall_clock()
                    )
                )
                if active_lease is not None:
                    idle_started.pop(candidate.resource_id, None)
                    decision = ResourceAdmissionDecision(
                        outcome=AdmissionOutcome.WAIT,
                        request_id=request.request_id,
                        resource_id=candidate.resource_id,
                        reason="resource_already_leased",
                    )
                    self._observe(decision, candidate, now)
                    continue

                initial = self.policy.decide(
                    request,
                    candidate,
                    worker_id=worker_id,
                    stable_idle_seconds=0,
                )
                if initial.outcome is AdmissionOutcome.REJECTED:
                    raise InvalidResourceState(initial.reason)
                if not initial.continuously_idle:
                    idle_started.pop(candidate.resource_id, None)
                    self._observe(initial, candidate, now)
                    continue
                candidate_idle_started = idle_started.setdefault(
                    candidate.resource_id, now
                )
                stable_idle_seconds = max(0.0, now - candidate_idle_started)
                decision = self.policy.decide(
                    request,
                    candidate,
                    worker_id=worker_id,
                    stable_idle_seconds=stable_idle_seconds,
                )
                self._observe(decision, candidate, now)
                if decision.outcome is not AdmissionOutcome.ADMITTED:
                    continue

                lease = None
                if worker_id is not None:
                    assert self.lease_store is not None and lease_ttl is not None
                    try:
                        lease = self.lease_store.acquire(
                            request,
                            candidate,
                            worker_id=worker_id,
                            now=self.wall_clock(),
                            ttl=lease_ttl,
                        )
                    except ResourceLeaseConflict:
                        idle_started.pop(candidate.resource_id, None)
                        conflict = decision.model_copy(
                            update={
                                "outcome": AdmissionOutcome.WAIT,
                                "reason": "resource_already_leased",
                            }
                        )
                        self._observe(conflict, candidate, now)
                        continue
                telemetry = ResourceAdmissionTelemetry(
                    request_id=request.request_id,
                    resource_id=candidate.resource_id,
                    resource_type=candidate.resource_type,
                    admission_class=request.admission_class,
                    wait_seconds=max(0.0, now - started),
                    poll_count=poll_count,
                    stable_idle_seconds=request.stable_idle_seconds,
                    observed_capacities=candidate.capacities,
                    utilization_percent=candidate.utilization_percent,
                    foreign_owner_count=len(
                        tuple(
                            owner
                            for owner in candidate.foreign_owners
                            if owner != worker_id
                        )
                    ),
                )
                return ResourceAdmission(
                    decision=decision, telemetry=telemetry, lease=lease
                )

            elapsed = max(0.0, now - started)
            if (
                request.maximum_wait_seconds is not None
                and elapsed + self.poll_seconds > request.maximum_wait_seconds
            ):
                raise ResourceWaitTimeout("resource_maximum_wait_exceeded")
            self.sleeper(self.poll_seconds)
