"""Generic resource admission, brokerage, telemetry, and leasing."""

from auto_researcher.resources.broker import (
    CourtesyResourceAdmissionPolicy,
    InvalidResourceRequest,
    InvalidResourceState,
    ResourceBroker,
    ResourceBrokerError,
    ResourceInspectionError,
    ResourceWaitTimeout,
)
from auto_researcher.resources.leases import (
    InMemoryResourceLeaseStore,
    ResourceLeaseConflict,
    ResourceLeaseError,
    ResourceLeaseNotFound,
    ResourceLeaseOwnershipError,
)
from auto_researcher.resources.models import (
    AdmissionClass,
    AdmissionOutcome,
    PreemptionPolicy,
    ResourceAdmission,
    ResourceAdmissionDecision,
    ResourceAdmissionTelemetry,
    ResourceCandidate,
    ResourceCapacity,
    ResourceLease,
    ResourceOwner,
    ResourceRequest,
    ResourceRequirement,
)
from auto_researcher.resources.protocols import (
    ResourceAdmissionPolicy,
    ResourceLeaseStore,
    ResourceProvider,
)

__all__ = [
    "AdmissionClass",
    "AdmissionOutcome",
    "CourtesyResourceAdmissionPolicy",
    "InMemoryResourceLeaseStore",
    "InvalidResourceRequest",
    "InvalidResourceState",
    "PreemptionPolicy",
    "ResourceAdmission",
    "ResourceAdmissionDecision",
    "ResourceAdmissionPolicy",
    "ResourceAdmissionTelemetry",
    "ResourceBroker",
    "ResourceBrokerError",
    "ResourceCandidate",
    "ResourceCapacity",
    "ResourceInspectionError",
    "ResourceLease",
    "ResourceLeaseConflict",
    "ResourceLeaseError",
    "ResourceLeaseNotFound",
    "ResourceLeaseOwnershipError",
    "ResourceLeaseStore",
    "ResourceOwner",
    "ResourceProvider",
    "ResourceRequest",
    "ResourceRequirement",
    "ResourceWaitTimeout",
]
