"""Small in-memory lease store with an interface suitable for durable replacement."""

from __future__ import annotations

from datetime import datetime, timedelta
from threading import RLock
from uuid import uuid4

from auto_researcher.resources.models import (
    ResourceCandidate,
    ResourceLease,
    ResourceRequest,
)


class ResourceLeaseError(RuntimeError):
    pass


class ResourceLeaseConflict(ResourceLeaseError):
    pass


class ResourceLeaseOwnershipError(ResourceLeaseError):
    pass


class ResourceLeaseNotFound(ResourceLeaseError):
    pass


class InMemoryResourceLeaseStore:
    """Process-local atomic leases; PostgreSQL/shared workers belong in PR 11.5."""

    def __init__(self) -> None:
        self._leases: dict[str, ResourceLease] = {}
        self._active_by_resource: dict[str, str] = {}
        self._lock = RLock()

    def _is_stale(self, lease: ResourceLease, now: datetime) -> bool:
        return lease.released_at is None and lease.expires_at <= now

    def _recover_stale_locked(self, now: datetime) -> tuple[ResourceLease, ...]:
        recovered: list[ResourceLease] = []
        for lease_id, lease in tuple(self._leases.items()):
            if not self._is_stale(lease, now):
                continue
            released = lease.model_copy(update={"released_at": now})
            self._leases[lease_id] = released
            self._active_by_resource.pop(lease.resource_id, None)
            recovered.append(released)
        return tuple(recovered)

    def active_for(self, resource_id: str, *, now: datetime) -> ResourceLease | None:
        with self._lock:
            self._recover_stale_locked(now)
            lease_id = self._active_by_resource.get(resource_id)
            return None if lease_id is None else self._leases[lease_id]

    def acquire(
        self,
        request: ResourceRequest,
        candidate: ResourceCandidate,
        *,
        worker_id: str,
        now: datetime,
        ttl: timedelta,
    ) -> ResourceLease:
        if not worker_id or ttl <= timedelta(0):
            raise ValueError("worker_id and a positive lease ttl are required")
        with self._lock:
            self._recover_stale_locked(now)
            active_id = self._active_by_resource.get(candidate.resource_id)
            if active_id is not None:
                active = self._leases[active_id]
                if (
                    active.worker_id == worker_id
                    and active.request_id == request.request_id
                ):
                    return active
                raise ResourceLeaseConflict("resource_already_leased")
            lease = ResourceLease(
                lease_id=f"lease-{uuid4()}",
                request_id=request.request_id,
                resource_id=candidate.resource_id,
                worker_id=worker_id,
                acquired_at=now,
                heartbeat_at=now,
                expires_at=now + ttl,
            )
            self._leases[lease.lease_id] = lease
            self._active_by_resource[candidate.resource_id] = lease.lease_id
            return lease

    def renew(
        self,
        lease_id: str,
        *,
        worker_id: str,
        now: datetime,
        ttl: timedelta,
    ) -> ResourceLease:
        if ttl <= timedelta(0):
            raise ValueError("a positive lease ttl is required")
        with self._lock:
            lease = self._leases.get(lease_id)
            if (
                lease is None
                or lease.released_at is not None
                or self._is_stale(lease, now)
            ):
                raise ResourceLeaseNotFound("active_resource_lease_not_found")
            if lease.worker_id != worker_id:
                raise ResourceLeaseOwnershipError("resource_lease_worker_mismatch")
            renewed = lease.model_copy(
                update={"heartbeat_at": now, "expires_at": now + ttl}
            )
            self._leases[lease_id] = renewed
            return renewed

    def release(self, lease_id: str, *, worker_id: str, now: datetime) -> ResourceLease:
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None or lease.released_at is not None:
                raise ResourceLeaseNotFound("active_resource_lease_not_found")
            if lease.worker_id != worker_id:
                raise ResourceLeaseOwnershipError("resource_lease_worker_mismatch")
            released = lease.model_copy(update={"released_at": now})
            self._leases[lease_id] = released
            self._active_by_resource.pop(lease.resource_id, None)
            return released

    def stale(self, *, now: datetime) -> tuple[ResourceLease, ...]:
        with self._lock:
            return tuple(
                lease for lease in self._leases.values() if self._is_stale(lease, now)
            )

    def recover_stale(self, *, now: datetime) -> tuple[ResourceLease, ...]:
        with self._lock:
            return self._recover_stale_locked(now)
