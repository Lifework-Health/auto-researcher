"""Task-agnostic resource admission and lease contracts."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ResourceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)


class AdmissionClass(str, Enum):
    PRIMARY = "primary"
    OPPORTUNISTIC = "opportunistic"


class PreemptionPolicy(str, Enum):
    NEVER = "never"
    ALLOW = "allow"


class AdmissionOutcome(str, Enum):
    ADMITTED = "admitted"
    WAIT = "wait"
    REJECTED = "rejected"


class ResourceCapacity(ResourceModel):
    """One generic scalar capacity such as memory_mib, bytes, or requests/minute."""

    name: str = Field(min_length=1)
    value: float = Field(ge=0)


class ResourceOwner(ResourceModel):
    """Provider-classified foreign ownership in an explicit identity namespace."""

    namespace: str = Field(min_length=1)
    owner_id: str = Field(min_length=1)
    display_name: str | None = None


class ResourceRequirement(ResourceModel):
    resource_type: str = Field(min_length=1)
    quantity: int = Field(default=1, ge=1)
    minimum_capacities: tuple[ResourceCapacity, ...] = ()

    @model_validator(mode="after")
    def capacity_names_are_unique(self) -> "ResourceRequirement":
        names = [capacity.name for capacity in self.minimum_capacities]
        if len(names) != len(set(names)):
            raise ValueError("resource capacity names must be unique")
        return self


class ResourceRequest(ResourceModel):
    """Operational intent only; priority is not local queue-order enforcement.

    Admission class may drive configured courtesy policy. Priority is retained for a
    future coordinator comparing multiple requests; the process-local v1 broker
    evaluates one request and does not order a shared queue.
    """

    request_id: str = Field(min_length=1)
    requirements: tuple[ResourceRequirement, ...] = Field(min_length=1)
    admission_class: AdmissionClass = AdmissionClass.PRIMARY
    priority: int = 0
    maximum_wait_seconds: float | None = Field(default=None, gt=0)
    preemption: PreemptionPolicy = PreemptionPolicy.NEVER
    stable_idle_seconds: float = Field(default=0, ge=0)
    equivalence_requirements: frozenset[str] = Field(default_factory=frozenset)

    @model_validator(mode="after")
    def resource_types_are_unique(self) -> "ResourceRequest":
        resource_types = [
            requirement.resource_type for requirement in self.requirements
        ]
        if len(resource_types) != len(set(resource_types)):
            raise ValueError("resource requirement types must be unique")
        return self


class ResourceCandidate(ResourceModel):
    """One indivisible allocation unit; a lease reserves this entire bundle.

    Quantity is matching capacity inside the bundle, not a partially leasable pool.
    Providers expose independently allocatable units as distinct candidate IDs.
    """

    resource_id: str = Field(min_length=1)
    resource_type: str = Field(min_length=1)
    quantity: int = Field(default=1, ge=0)
    capacities: tuple[ResourceCapacity, ...] = ()
    utilization_percent: float | None = Field(default=None, ge=0, le=100)
    available: bool = True
    foreign_owners: tuple[ResourceOwner, ...] = ()
    equivalence_tags: frozenset[str] = Field(default_factory=frozenset)
    state_valid: bool = True
    state_reason: str | None = None

    @model_validator(mode="after")
    def state_and_capacities_are_coherent(self) -> "ResourceCandidate":
        names = [capacity.name for capacity in self.capacities]
        if len(names) != len(set(names)):
            raise ValueError("resource capacity names must be unique")
        if not self.state_valid and not self.state_reason:
            raise ValueError("invalid resource state requires a reason")
        return self

    def capacity(self, name: str) -> float | None:
        return next(
            (capacity.value for capacity in self.capacities if capacity.name == name),
            None,
        )


class ResourceAdmissionDecision(ResourceModel):
    outcome: AdmissionOutcome
    request_id: str = Field(min_length=1)
    resource_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    stable_idle_seconds: float = Field(default=0, ge=0)
    continuously_idle: bool = False


class ResourceAdmissionTelemetry(ResourceModel):
    """Operational telemetry, deliberately separate from scientific metrics."""

    request_id: str = Field(min_length=1)
    resource_id: str = Field(min_length=1)
    resource_type: str = Field(min_length=1)
    admission_class: AdmissionClass
    wait_seconds: float = Field(ge=0)
    poll_count: int = Field(ge=1)
    observed_continuous_idle_seconds: float = Field(ge=0)
    required_stable_idle_seconds: float = Field(ge=0)
    observed_capacities: tuple[ResourceCapacity, ...] = ()
    utilization_percent: float | None = Field(default=None, ge=0, le=100)
    foreign_owner_count: int = Field(default=0, ge=0)

    def as_operational_telemetry(self) -> dict[str, object]:
        return self.model_dump(mode="json")


class ResourceLease(ResourceModel):
    """Whole-candidate ownership, idempotent by resource/request/worker identity."""

    lease_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    resource_id: str = Field(min_length=1)
    worker_id: str = Field(min_length=1)
    allocation_semantics: Literal["whole_candidate"] = "whole_candidate"
    acquired_at: datetime
    heartbeat_at: datetime
    expires_at: datetime
    released_at: datetime | None = None

    @model_validator(mode="after")
    def timestamps_are_ordered(self) -> "ResourceLease":
        timestamps = (self.acquired_at, self.heartbeat_at, self.expires_at)
        if any(timestamp.tzinfo is None for timestamp in timestamps):
            raise ValueError("resource lease timestamps must be timezone-aware")
        if self.heartbeat_at < self.acquired_at or self.expires_at <= self.heartbeat_at:
            raise ValueError("resource lease timestamps are invalid")
        if self.released_at is not None:
            if self.released_at.tzinfo is None or self.released_at < self.acquired_at:
                raise ValueError("resource lease release timestamp is invalid")
        return self


class ResourceAdmission(ResourceModel):
    decision: ResourceAdmissionDecision
    telemetry: ResourceAdmissionTelemetry
    lease: ResourceLease | None = None
