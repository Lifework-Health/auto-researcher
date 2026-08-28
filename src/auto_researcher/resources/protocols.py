"""Replaceable provider, policy, and lease boundaries for resource admission."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Protocol

from auto_researcher.resources.models import (
    ResourceAdmissionDecision,
    ResourceCandidate,
    ResourceLease,
    ResourceRequest,
)


class ResourceProvider(Protocol):
    """Return one authoritative snapshot or raise when it cannot be trusted.

    Foreign owners are already classified by the provider and carry an explicit
    identity namespace. Broker worker IDs are not compared to provider owner IDs.
    """

    def candidates(self, request: ResourceRequest) -> tuple[ResourceCandidate, ...]: ...


class ResourceAdmissionPolicy(Protocol):
    def decide(
        self,
        request: ResourceRequest,
        candidate: ResourceCandidate,
        *,
        worker_id: str | None,
        stable_idle_seconds: float,
    ) -> ResourceAdmissionDecision: ...


class ResourceLeaseStore(Protocol):
    """Authoritative atomic lease boundary.

    At most one active lease may exist for a logical request ID. Acquire is
    idempotent only for the same active resource, request ID, and worker ID;
    all other active claims on the request or resource conflict.
    """

    def active_for(
        self, resource_id: str, *, now: datetime
    ) -> ResourceLease | None: ...

    def active_for_request(
        self, request_id: str, *, now: datetime
    ) -> ResourceLease | None: ...

    def acquire(
        self,
        request: ResourceRequest,
        candidate: ResourceCandidate,
        *,
        worker_id: str,
        now: datetime,
        ttl: timedelta,
    ) -> ResourceLease: ...

    def renew(
        self,
        lease_id: str,
        *,
        worker_id: str,
        now: datetime,
        ttl: timedelta,
    ) -> ResourceLease: ...

    def release(
        self, lease_id: str, *, worker_id: str, now: datetime
    ) -> ResourceLease: ...

    def stale(self, *, now: datetime) -> tuple[ResourceLease, ...]: ...

    def recover_stale(self, *, now: datetime) -> tuple[ResourceLease, ...]: ...
