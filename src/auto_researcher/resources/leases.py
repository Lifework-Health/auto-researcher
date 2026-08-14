"""Small in-memory lease store with an interface suitable for durable replacement."""

from __future__ import annotations

import hashlib
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


class ResourceLeaseStoreUnavailable(ResourceLeaseError):
    pass


class InMemoryResourceLeaseStore:
    """Atomic whole-candidate and logical-request leases for one local process."""

    def __init__(self) -> None:
        self._leases: dict[str, ResourceLease] = {}
        self._active_by_resource: dict[str, str] = {}
        self._active_by_request: dict[str, str] = {}
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
            if self._active_by_resource.get(lease.resource_id) == lease_id:
                self._active_by_resource.pop(lease.resource_id, None)
            if self._active_by_request.get(lease.request_id) == lease_id:
                self._active_by_request.pop(lease.request_id, None)
            recovered.append(released)
        return tuple(recovered)

    def active_for(self, resource_id: str, *, now: datetime) -> ResourceLease | None:
        with self._lock:
            self._recover_stale_locked(now)
            lease_id = self._active_by_resource.get(resource_id)
            return None if lease_id is None else self._leases[lease_id]

    def active_for_request(
        self, request_id: str, *, now: datetime
    ) -> ResourceLease | None:
        with self._lock:
            self._recover_stale_locked(now)
            lease_id = self._active_by_request.get(request_id)
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
            request_lease_id = self._active_by_request.get(request.request_id)
            if request_lease_id is not None:
                request_lease = self._leases[request_lease_id]
                if (
                    request_lease.resource_id == candidate.resource_id
                    and request_lease.worker_id == worker_id
                ):
                    return request_lease
                raise ResourceLeaseConflict("request_already_leased")
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
            self._active_by_request[request.request_id] = lease.lease_id
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
            if self._active_by_resource.get(lease.resource_id) == lease_id:
                self._active_by_resource.pop(lease.resource_id, None)
            if self._active_by_request.get(lease.request_id) == lease_id:
                self._active_by_request.pop(lease.request_id, None)
            return released

    def stale(self, *, now: datetime) -> tuple[ResourceLease, ...]:
        with self._lock:
            return tuple(
                lease for lease in self._leases.values() if self._is_stale(lease, now)
            )

    def recover_stale(self, *, now: datetime) -> tuple[ResourceLease, ...]:
        with self._lock:
            return self._recover_stale_locked(now)


class PostgresResourceLeaseStore:
    """Database-time, whole-candidate leases shared across processes and nodes."""

    def __init__(self, engine, *, create_schema: bool = True) -> None:
        if engine.dialect.name != "postgresql":
            raise ValueError("resource_lease_store_requires_postgresql")
        self.engine = engine
        if create_schema:
            self.create_schema()

    @staticmethod
    def _text(statement: str):
        try:
            from sqlalchemy import text
        except ImportError as exc:  # pragma: no cover - shared extra provides it
            raise RuntimeError("PostgreSQL lease dependencies unavailable") from exc
        return text(statement)

    def create_schema(self) -> None:
        statements = (
            """
            CREATE TABLE IF NOT EXISTS ar_resource_lease (
                lease_id UUID PRIMARY KEY,
                request_id TEXT NOT NULL,
                resource_id TEXT NOT NULL,
                worker_id TEXT NOT NULL,
                acquired_at TIMESTAMPTZ NOT NULL,
                heartbeat_at TIMESTAMPTZ NOT NULL,
                expires_at TIMESTAMPTZ NOT NULL,
                released_at TIMESTAMPTZ NULL
            )
            """,
            """
            CREATE UNIQUE INDEX IF NOT EXISTS ar_active_resource_lease_uq
            ON ar_resource_lease (resource_id) WHERE released_at IS NULL
            """,
            """
            CREATE UNIQUE INDEX IF NOT EXISTS ar_active_request_lease_uq
            ON ar_resource_lease (request_id) WHERE released_at IS NULL
            """,
        )
        try:
            with self.engine.begin() as connection:
                schema_lock = int.from_bytes(
                    hashlib.sha256(b"auto-researcher-resource-lease-schema").digest()[
                        :8
                    ],
                    "big",
                    signed=True,
                )
                connection.execute(
                    self._text("SELECT pg_advisory_xact_lock(:key)"),
                    {"key": schema_lock},
                )
                for statement in statements:
                    connection.execute(self._text(statement))
        except Exception:
            raise ResourceLeaseStoreUnavailable(
                "resource_lease_store_unavailable"
            ) from None

    @staticmethod
    def _lease(row) -> ResourceLease:
        values = dict(row._mapping)
        values["lease_id"] = f"lease-{values['lease_id']}"
        return ResourceLease.model_validate(values)

    def _recover_stale(self, connection) -> tuple[ResourceLease, ...]:
        rows = connection.execute(
            self._text(
                """
                UPDATE ar_resource_lease
                SET released_at = CURRENT_TIMESTAMP
                WHERE released_at IS NULL AND expires_at <= CURRENT_TIMESTAMP
                RETURNING *
                """
            )
        ).all()
        return tuple(self._lease(row) for row in rows)

    def active_for(self, resource_id: str, *, now: datetime) -> ResourceLease | None:
        del now  # PostgreSQL time is authoritative across workers.
        try:
            with self.engine.begin() as connection:
                self._recover_stale(connection)
                row = connection.execute(
                    self._text(
                        """
                        SELECT * FROM ar_resource_lease
                        WHERE resource_id = :resource_id AND released_at IS NULL
                        """
                    ),
                    {"resource_id": resource_id},
                ).one_or_none()
        except Exception:
            raise ResourceLeaseStoreUnavailable(
                "resource_lease_store_unavailable"
            ) from None
        return None if row is None else self._lease(row)

    def active_for_request(
        self, request_id: str, *, now: datetime
    ) -> ResourceLease | None:
        del now
        try:
            with self.engine.begin() as connection:
                self._recover_stale(connection)
                row = connection.execute(
                    self._text(
                        """
                        SELECT * FROM ar_resource_lease
                        WHERE request_id = :request_id AND released_at IS NULL
                        """
                    ),
                    {"request_id": request_id},
                ).one_or_none()
        except Exception:
            raise ResourceLeaseStoreUnavailable(
                "resource_lease_store_unavailable"
            ) from None
        return None if row is None else self._lease(row)

    def acquire(
        self,
        request: ResourceRequest,
        candidate: ResourceCandidate,
        *,
        worker_id: str,
        now: datetime,
        ttl: timedelta,
    ) -> ResourceLease:
        del now
        if not worker_id or ttl <= timedelta(0):
            raise ValueError("worker_id and a positive lease ttl are required")
        lease_uuid = str(uuid4())
        try:
            with self.engine.begin() as connection:
                self._recover_stale(connection)
                request_row = connection.execute(
                    self._text(
                        """
                        SELECT * FROM ar_resource_lease
                        WHERE request_id = :request_id AND released_at IS NULL
                        FOR UPDATE
                        """
                    ),
                    {"request_id": request.request_id},
                ).one_or_none()
                if request_row is not None:
                    existing = self._lease(request_row)
                    if (
                        existing.worker_id == worker_id
                        and existing.resource_id == candidate.resource_id
                    ):
                        return existing
                    raise ResourceLeaseConflict("request_already_leased")
                resource_row = connection.execute(
                    self._text(
                        """
                        SELECT * FROM ar_resource_lease
                        WHERE resource_id = :resource_id AND released_at IS NULL
                        FOR UPDATE
                        """
                    ),
                    {"resource_id": candidate.resource_id},
                ).one_or_none()
                if resource_row is not None:
                    raise ResourceLeaseConflict("resource_already_leased")
                row = connection.execute(
                    self._text(
                        """
                        INSERT INTO ar_resource_lease (
                            lease_id, request_id, resource_id, worker_id,
                            acquired_at, heartbeat_at, expires_at
                        ) VALUES (
                            CAST(:lease_id AS UUID), :request_id, :resource_id, :worker_id,
                            CURRENT_TIMESTAMP, CURRENT_TIMESTAMP,
                            CURRENT_TIMESTAMP + (:ttl_seconds * INTERVAL '1 second')
                        )
                        RETURNING *
                        """
                    ),
                    {
                        "lease_id": lease_uuid,
                        "request_id": request.request_id,
                        "resource_id": candidate.resource_id,
                        "worker_id": worker_id,
                        "ttl_seconds": ttl.total_seconds(),
                    },
                ).one()
                return self._lease(row)
        except ResourceLeaseConflict:
            raise
        except Exception as exc:
            # A database-enforced partial unique index is the final arbiter for
            # races that passed both reads on separate connections.
            if type(exc).__name__ == "IntegrityError":
                raise ResourceLeaseConflict(
                    "resource_or_request_already_leased"
                ) from None
            raise ResourceLeaseStoreUnavailable(
                "resource_lease_store_unavailable"
            ) from None

    @staticmethod
    def _uuid(lease_id: str) -> str:
        if not lease_id.startswith("lease-"):
            raise ResourceLeaseNotFound("active_resource_lease_not_found")
        return lease_id.removeprefix("lease-")

    def renew(
        self,
        lease_id: str,
        *,
        worker_id: str,
        now: datetime,
        ttl: timedelta,
    ) -> ResourceLease:
        del now
        if ttl <= timedelta(0):
            raise ValueError("a positive lease ttl is required")
        try:
            with self.engine.begin() as connection:
                row = connection.execute(
                    self._text(
                        """
                        UPDATE ar_resource_lease
                        SET heartbeat_at = CURRENT_TIMESTAMP,
                            expires_at = CURRENT_TIMESTAMP
                                + (:ttl_seconds * INTERVAL '1 second')
                        WHERE lease_id = CAST(:lease_id AS UUID)
                          AND worker_id = :worker_id
                          AND released_at IS NULL
                          AND expires_at > CURRENT_TIMESTAMP
                        RETURNING *
                        """
                    ),
                    {
                        "lease_id": self._uuid(lease_id),
                        "worker_id": worker_id,
                        "ttl_seconds": ttl.total_seconds(),
                    },
                ).one_or_none()
                if row is not None:
                    return self._lease(row)
                owner = connection.execute(
                    self._text(
                        "SELECT worker_id FROM ar_resource_lease "
                        "WHERE lease_id = CAST(:lease_id AS UUID)"
                    ),
                    {"lease_id": self._uuid(lease_id)},
                ).scalar_one_or_none()
        except ResourceLeaseNotFound:
            raise
        except Exception:
            raise ResourceLeaseStoreUnavailable(
                "resource_lease_store_unavailable"
            ) from None
        if owner is not None and owner != worker_id:
            raise ResourceLeaseOwnershipError("resource_lease_worker_mismatch")
        raise ResourceLeaseNotFound("active_resource_lease_not_found")

    def release(self, lease_id: str, *, worker_id: str, now: datetime) -> ResourceLease:
        del now
        try:
            with self.engine.begin() as connection:
                row = connection.execute(
                    self._text(
                        """
                        UPDATE ar_resource_lease
                        SET released_at = CURRENT_TIMESTAMP
                        WHERE lease_id = CAST(:lease_id AS UUID)
                          AND worker_id = :worker_id
                          AND released_at IS NULL
                        RETURNING *
                        """
                    ),
                    {"lease_id": self._uuid(lease_id), "worker_id": worker_id},
                ).one_or_none()
                if row is not None:
                    return self._lease(row)
                owner = connection.execute(
                    self._text(
                        "SELECT worker_id FROM ar_resource_lease "
                        "WHERE lease_id = CAST(:lease_id AS UUID)"
                    ),
                    {"lease_id": self._uuid(lease_id)},
                ).scalar_one_or_none()
        except ResourceLeaseNotFound:
            raise
        except Exception:
            raise ResourceLeaseStoreUnavailable(
                "resource_lease_store_unavailable"
            ) from None
        if owner is not None and owner != worker_id:
            raise ResourceLeaseOwnershipError("resource_lease_worker_mismatch")
        raise ResourceLeaseNotFound("active_resource_lease_not_found")

    def stale(self, *, now: datetime) -> tuple[ResourceLease, ...]:
        del now
        try:
            with self.engine.connect() as connection:
                rows = connection.execute(
                    self._text(
                        """
                        SELECT * FROM ar_resource_lease
                        WHERE released_at IS NULL AND expires_at <= CURRENT_TIMESTAMP
                        ORDER BY acquired_at, lease_id
                        """
                    )
                ).all()
        except Exception:
            raise ResourceLeaseStoreUnavailable(
                "resource_lease_store_unavailable"
            ) from None
        return tuple(self._lease(row) for row in rows)

    def recover_stale(self, *, now: datetime) -> tuple[ResourceLease, ...]:
        del now
        try:
            with self.engine.begin() as connection:
                return self._recover_stale(connection)
        except Exception:
            raise ResourceLeaseStoreUnavailable(
                "resource_lease_store_unavailable"
            ) from None
