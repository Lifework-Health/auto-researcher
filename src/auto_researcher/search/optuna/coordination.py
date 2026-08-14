"""Small PostgreSQL coordination boundary around native Optuna ask/tell."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class OptunaCoordinationError(RuntimeError):
    """Stable failure that never includes database connection information."""


class SharedTrialBudgetExhausted(OptunaCoordinationError):
    pass


class WorkerClaimConflict(OptunaCoordinationError):
    pass


class WorkerClaimLost(OptunaCoordinationError):
    pass


class TellOwnershipMismatch(OptunaCoordinationError):
    pass


class TrialNoLongerRunning(OptunaCoordinationError):
    pass


class TrialClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    study_name: str = Field(min_length=1)
    trial_number: int = Field(ge=0)
    worker_id: str = Field(min_length=1)
    claim_id: str = Field(min_length=1)
    claimed_at: datetime
    heartbeat_at: datetime
    expires_at: datetime
    released_at: datetime | None = None
    report_digest: str | None = None


_T = TypeVar("_T")


def _lock_key(study_name: str) -> int:
    raw = hashlib.sha256(f"auto-researcher-optuna:{study_name}".encode()).digest()[:8]
    return int.from_bytes(raw, byteorder="big", signed=True)


class PostgresOptunaCoordination:
    """Own budget admission and fencing, never Optuna scientific trial state."""

    def __init__(self, engine: Any, *, create_schema: bool = True) -> None:
        if engine.dialect.name != "postgresql":
            raise ValueError("optuna_coordination_requires_postgresql")
        self.engine = engine
        if create_schema:
            self.create_schema()

    @staticmethod
    def _text(statement: str) -> Any:
        try:
            from sqlalchemy import text
        except ImportError as exc:  # pragma: no cover - shared extra provides it
            raise RuntimeError("OPTUNA HPO dependency unavailable") from exc
        return text(statement)

    def create_schema(self) -> None:
        statements = (
            """
            CREATE TABLE IF NOT EXISTS ar_optuna_study_budget (
                study_name TEXT PRIMARY KEY,
                trial_budget INTEGER NOT NULL CHECK (trial_budget > 0),
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS ar_optuna_trial_claim (
                study_name TEXT NOT NULL,
                trial_number INTEGER NOT NULL CHECK (trial_number >= 0),
                worker_id TEXT NOT NULL,
                claim_id UUID NOT NULL,
                claimed_at TIMESTAMPTZ NOT NULL,
                heartbeat_at TIMESTAMPTZ NOT NULL,
                expires_at TIMESTAMPTZ NOT NULL,
                released_at TIMESTAMPTZ NULL,
                report_digest TEXT NULL,
                PRIMARY KEY (study_name, trial_number)
            )
            """,
            """
            CREATE UNIQUE INDEX IF NOT EXISTS ar_optuna_active_claim_token_uq
            ON ar_optuna_trial_claim (study_name, trial_number, claim_id)
            WHERE released_at IS NULL
            """,
        )
        try:
            with self.engine.begin() as connection:
                connection.execute(
                    self._text("SELECT pg_advisory_xact_lock(:key)"),
                    {"key": _lock_key("__coordination_schema__")},
                )
                for statement in statements:
                    connection.execute(self._text(statement))
        except Exception:
            raise OptunaCoordinationError("study_coordination_unavailable") from None

    def run_study_locked(self, study_name: str, operation: Callable[[], _T]) -> _T:
        """Run a short recovery operation against the same lock used by ASK."""

        try:
            with self.engine.begin() as connection:
                connection.execute(
                    self._text("SELECT pg_advisory_xact_lock(:key)"),
                    {"key": _lock_key(study_name)},
                )
                return operation()
        except OptunaCoordinationError:
            raise
        except Exception:
            raise OptunaCoordinationError("study_coordination_failure") from None

    @staticmethod
    def _claim(row: Any) -> TrialClaim:
        values = {
            key: value
            for key, value in dict(row._mapping).items()
            if key in TrialClaim.model_fields
        }
        values["claim_id"] = str(values["claim_id"])
        return TrialClaim.model_validate(values)

    def admit_ask_and_claim(
        self,
        *,
        study_name: str,
        trial_budget: int,
        worker_id: str,
        ttl: timedelta,
        ask: Callable[[datetime, str], tuple[int, _T]],
    ) -> tuple[TrialClaim, _T]:
        """Serialize only budget check, native ask, and durable claim creation."""

        if not study_name or not worker_id or trial_budget <= 0 or ttl <= timedelta(0):
            raise ValueError("valid study, budget, worker, and claim ttl are required")
        claim_id = str(uuid4())
        try:
            with self.engine.begin() as connection:
                connection.execute(
                    self._text("SELECT pg_advisory_xact_lock(:key)"),
                    {"key": _lock_key(study_name)},
                )
                connection.execute(
                    self._text(
                        """
                        INSERT INTO ar_optuna_study_budget (study_name, trial_budget)
                        VALUES (:study_name, :trial_budget)
                        ON CONFLICT (study_name) DO NOTHING
                        """
                    ),
                    {"study_name": study_name, "trial_budget": trial_budget},
                )
                configured = connection.execute(
                    self._text(
                        """
                        SELECT trial_budget
                        FROM ar_optuna_study_budget
                        WHERE study_name = :study_name
                        FOR UPDATE
                        """
                    ),
                    {"study_name": study_name},
                ).scalar_one()
                if int(configured) != trial_budget:
                    raise OptunaCoordinationError("shared_trial_budget_mismatch")
                database_now = connection.execute(
                    self._text("SELECT CURRENT_TIMESTAMP")
                ).scalar_one()
                trial_number, value = ask(database_now, claim_id)
                row = connection.execute(
                    self._text(
                        """
                        INSERT INTO ar_optuna_trial_claim (
                            study_name, trial_number, worker_id, claim_id,
                            claimed_at, heartbeat_at, expires_at
                        ) VALUES (
                            :study_name, :trial_number, :worker_id, CAST(:claim_id AS UUID),
                            CURRENT_TIMESTAMP, CURRENT_TIMESTAMP,
                            CURRENT_TIMESTAMP + (:ttl_seconds * INTERVAL '1 second')
                        )
                        RETURNING *
                        """
                    ),
                    {
                        "study_name": study_name,
                        "trial_number": trial_number,
                        "worker_id": worker_id,
                        "claim_id": claim_id,
                        "ttl_seconds": ttl.total_seconds(),
                    },
                ).one()
                return self._claim(row), value
        except SharedTrialBudgetExhausted:
            raise
        except OptunaCoordinationError:
            raise
        except Exception:
            # A native ask may already have committed. Recovery deliberately treats
            # that RUNNING trial as unclaimed rather than guessing its owner.
            raise WorkerClaimConflict("worker_claim_conflict") from None

    def claim_for_trial(self, study_name: str, trial_number: int) -> TrialClaim | None:
        try:
            with self.engine.connect() as connection:
                row = connection.execute(
                    self._text(
                        """
                        SELECT * FROM ar_optuna_trial_claim
                        WHERE study_name = :study_name AND trial_number = :trial_number
                        """
                    ),
                    {"study_name": study_name, "trial_number": trial_number},
                ).one_or_none()
        except Exception:
            raise OptunaCoordinationError("study_coordination_unavailable") from None
        return None if row is None else self._claim(row)

    def assert_owner(self, claim: TrialClaim) -> TrialClaim:
        try:
            with self.engine.connect() as connection:
                row = connection.execute(
                    self._text(
                        """
                        SELECT * FROM ar_optuna_trial_claim
                        WHERE study_name = :study_name
                          AND trial_number = :trial_number
                          AND worker_id = :worker_id
                          AND claim_id = CAST(:claim_id AS UUID)
                          AND released_at IS NULL
                          AND expires_at > CURRENT_TIMESTAMP
                        """
                    ),
                    claim.model_dump(
                        include={"study_name", "trial_number", "worker_id", "claim_id"}
                    ),
                ).one_or_none()
        except Exception:
            raise OptunaCoordinationError("study_coordination_unavailable") from None
        if row is None:
            raise WorkerClaimLost("worker_claim_lost_or_stale")
        return self._claim(row)

    def heartbeat(self, claim: TrialClaim, *, ttl: timedelta) -> TrialClaim:
        if ttl <= timedelta(0):
            raise ValueError("a positive claim ttl is required")
        try:
            with self.engine.begin() as connection:
                row = connection.execute(
                    self._text(
                        """
                        UPDATE ar_optuna_trial_claim
                        SET heartbeat_at = CURRENT_TIMESTAMP,
                            expires_at = CURRENT_TIMESTAMP
                                + (:ttl_seconds * INTERVAL '1 second')
                        WHERE study_name = :study_name
                          AND trial_number = :trial_number
                          AND worker_id = :worker_id
                          AND claim_id = CAST(:claim_id AS UUID)
                          AND released_at IS NULL
                          AND expires_at > CURRENT_TIMESTAMP
                        RETURNING *
                        """
                    ),
                    {
                        **claim.model_dump(
                            include={
                                "study_name",
                                "trial_number",
                                "worker_id",
                                "claim_id",
                            }
                        ),
                        "ttl_seconds": ttl.total_seconds(),
                    },
                ).one_or_none()
        except Exception:
            raise OptunaCoordinationError("study_coordination_unavailable") from None
        if row is None:
            raise WorkerClaimLost("worker_claim_lost_or_stale")
        return self._claim(row)

    def record_report(self, claim: TrialClaim, *, report_digest: str) -> TrialClaim:
        if not report_digest:
            raise ValueError("report digest is required")
        try:
            with self.engine.begin() as connection:
                row = connection.execute(
                    self._text(
                        """
                        UPDATE ar_optuna_trial_claim
                        SET report_digest = COALESCE(report_digest, :report_digest)
                        WHERE study_name = :study_name
                          AND trial_number = :trial_number
                          AND worker_id = :worker_id
                          AND claim_id = CAST(:claim_id AS UUID)
                          AND released_at IS NULL
                          AND expires_at > CURRENT_TIMESTAMP
                          AND (report_digest IS NULL OR report_digest = :report_digest)
                        RETURNING *
                        """
                    ),
                    {
                        **claim.model_dump(
                            include={
                                "study_name",
                                "trial_number",
                                "worker_id",
                                "claim_id",
                            }
                        ),
                        "report_digest": report_digest,
                    },
                ).one_or_none()
        except Exception:
            raise OptunaCoordinationError("study_coordination_unavailable") from None
        if row is None:
            raise TellOwnershipMismatch("tell_ownership_or_report_mismatch")
        return self._claim(row)

    def run_owned_and_release(
        self,
        claim: TrialClaim,
        operation: Callable[[], _T],
    ) -> _T:
        """Fence a short tell/reconcile operation with a locked ownership row."""

        try:
            with self.engine.begin() as connection:
                row = connection.execute(
                    self._text(
                        """
                        SELECT * FROM ar_optuna_trial_claim
                        WHERE study_name = :study_name
                          AND trial_number = :trial_number
                        FOR UPDATE
                        """
                    ),
                    claim.model_dump(include={"study_name", "trial_number"}),
                ).one_or_none()
                if row is None:
                    raise WorkerClaimLost("worker_claim_lost_or_stale")
                current = self._claim(row)
                if (
                    current.worker_id != claim.worker_id
                    or current.claim_id != claim.claim_id
                    or current.released_at is not None
                ):
                    raise TellOwnershipMismatch("tell_ownership_mismatch")
                alive = connection.execute(
                    self._text("SELECT :expires_at > CURRENT_TIMESTAMP"),
                    {"expires_at": current.expires_at},
                ).scalar_one()
                if not alive:
                    raise WorkerClaimLost("worker_claim_lost_or_stale")
                result = operation()
                connection.execute(
                    self._text(
                        """
                        UPDATE ar_optuna_trial_claim
                        SET released_at = CURRENT_TIMESTAMP
                        WHERE study_name = :study_name
                          AND trial_number = :trial_number
                          AND claim_id = CAST(:claim_id AS UUID)
                        """
                    ),
                    claim.model_dump(
                        include={"study_name", "trial_number", "claim_id"}
                    ),
                )
                return result
        except OptunaCoordinationError:
            raise
        except Exception:
            raise OptunaCoordinationError("study_coordination_failure") from None

    def take_over_stale(
        self,
        *,
        study_name: str,
        trial_number: int,
        recovery_worker_id: str,
        ttl: timedelta,
    ) -> TrialClaim:
        """Issue a new fencing token only after database-time expiry."""

        if not study_name or not recovery_worker_id or ttl <= timedelta(0):
            raise ValueError("valid study, recovery worker, and ttl are required")
        new_claim_id = str(uuid4())
        try:
            with self.engine.begin() as connection:
                row = connection.execute(
                    self._text(
                        """
                        UPDATE ar_optuna_trial_claim
                        SET worker_id = :worker_id,
                            claim_id = CAST(:claim_id AS UUID),
                            claimed_at = CURRENT_TIMESTAMP,
                            heartbeat_at = CURRENT_TIMESTAMP,
                            expires_at = CURRENT_TIMESTAMP
                                + (:ttl_seconds * INTERVAL '1 second')
                        WHERE study_name = :study_name
                          AND trial_number = :trial_number
                          AND released_at IS NULL
                          AND expires_at <= CURRENT_TIMESTAMP
                        RETURNING *
                        """
                    ),
                    {
                        "study_name": study_name,
                        "trial_number": trial_number,
                        "worker_id": recovery_worker_id,
                        "claim_id": new_claim_id,
                        "ttl_seconds": ttl.total_seconds(),
                    },
                ).one_or_none()
        except Exception:
            raise OptunaCoordinationError("study_coordination_unavailable") from None
        if row is None:
            raise WorkerClaimConflict("worker_claim_not_stale")
        return self._claim(row)
