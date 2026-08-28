"""Task-agnostic one-trial worker seam around coordinated native ask/tell."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta, timezone
from threading import Event, Lock, Thread

from pydantic import BaseModel, ConfigDict

from auto_researcher.contracts.models import (
    EvaluationResult,
    ExperimentSpec,
    SearchRequest,
    VerificationResult,
)
from auto_researcher.resources import (
    ResourceAdmission,
    ResourceBroker,
    ResourceLease,
    ResourceLeaseError,
    ResourceRequest,
    cuda_environment_for_lease,
)
from auto_researcher.search.optuna.backend import OptunaAskTellBackend
from auto_researcher.search.optuna.coordination import (
    OptunaCoordinationError,
    TrialClaim,
)
from auto_researcher.search.optuna.models import (
    OptunaStudySpec,
    OptunaTrialOutcome,
    OptunaTrialReference,
)
from auto_researcher.search.optuna.naming import StudyIdentity
from auto_researcher.search.optuna.pruning import (
    OptunaIntermediateReporter,
    OptunaPruningAcknowledged,
)
from auto_researcher.tasks.models import ExperimentMetadata
from auto_researcher.tasks.protocols import ResearchTask


class CoordinatedWorkerError(RuntimeError):
    pass


class WorkerEvaluationFailed(CoordinatedWorkerError):
    pass


_PRESERVED_OPERATIONAL_ERRORS = (
    OptunaCoordinationError,
    ResourceLeaseError,
)


class _WorkerExecutionGuard:
    """Renew operational ownership and retain the first heartbeat failure."""

    def __init__(
        self,
        *,
        backend: OptunaAskTellBackend,
        claim: TrialClaim,
        claim_ttl: timedelta,
        claim_interval: timedelta,
        resource_broker: ResourceBroker | None,
        resource_lease_ttl: timedelta,
        resource_interval: timedelta,
    ) -> None:
        assert backend.coordination is not None
        self._coordination = backend.coordination
        self._claim = claim
        self._claim_ttl = claim_ttl
        self._claim_interval = claim_interval.total_seconds()
        self._resource_broker = resource_broker
        self._resource_lease_ttl = resource_lease_ttl
        self._resource_interval = resource_interval.total_seconds()
        self._resource_lease: ResourceLease | None = None
        self._failure: Exception | None = None
        self._lock = Lock()
        self._claim_stop = Event()
        self._resource_stop = Event()
        self._claim_thread: Thread | None = None
        self._resource_thread: Thread | None = None

    def _record_failure(self, failure: Exception) -> None:
        with self._lock:
            if self._failure is None:
                self._failure = failure

    def _renew_claim(self) -> None:
        with self._lock:
            claim = self._claim
        renewed = self._coordination.heartbeat(claim, ttl=self._claim_ttl)
        with self._lock:
            self._claim = renewed

    def _claim_loop(self) -> None:
        while not self._claim_stop.wait(self._claim_interval):
            try:
                self._renew_claim()
            except Exception as exc:
                self._record_failure(exc)
                return

    def start(self) -> None:
        # Renew synchronously before any experiment construction or resource wait.
        try:
            self._renew_claim()
        except Exception as exc:
            self._record_failure(exc)
            raise
        self._claim_thread = Thread(
            target=self._claim_loop,
            name="optuna-claim-heartbeat",
            daemon=True,
        )
        self._claim_thread.start()

    def _renew_resource(self) -> None:
        with self._lock:
            lease = self._resource_lease
        assert lease is not None and self._resource_broker is not None
        renewed = self._resource_broker.renew_lease(
            lease.lease_id,
            worker_id=lease.worker_id,
            lease_ttl=self._resource_lease_ttl,
        )
        with self._lock:
            self._resource_lease = renewed

    def _resource_loop(self) -> None:
        while not self._resource_stop.wait(self._resource_interval):
            try:
                self._renew_resource()
            except Exception as exc:
                self._record_failure(exc)
                return

    def start_resource_heartbeat(self, lease: ResourceLease) -> None:
        if self._resource_broker is None:
            raise RuntimeError("resource_broker_not_configured")
        with self._lock:
            self._resource_lease = lease
        try:
            self._renew_resource()
        except Exception as exc:
            self._record_failure(exc)
            raise
        self._resource_thread = Thread(
            target=self._resource_loop,
            name="resource-lease-heartbeat",
            daemon=True,
        )
        self._resource_thread.start()

    def raise_if_failed(self) -> None:
        with self._lock:
            failure = self._failure
        if failure is not None:
            raise failure

    def assert_owner(self) -> TrialClaim:
        self.raise_if_failed()
        with self._lock:
            claim = self._claim
        current = self._coordination.assert_owner(claim)
        with self._lock:
            self._claim = current
        self.raise_if_failed()
        return current

    def stop(self) -> None:
        # Signal both loops before joining either, so no heartbeat can race a
        # terminal claim release or resource release.
        self._claim_stop.set()
        self._resource_stop.set()
        if self._claim_thread is not None:
            self._claim_thread.join()
        if self._resource_thread is not None:
            self._resource_thread.join()

    @property
    def failure(self) -> Exception | None:
        with self._lock:
            return self._failure


class WorkerExecutionContext(BaseModel):
    """Operational placement passed to a host evaluator, never scientific identity."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    experiment: ExperimentSpec
    trial: OptunaTrialReference
    claim: TrialClaim
    resource_admission: ResourceAdmission | None = None
    process_environment: Mapping[str, str] | None = None
    intermediate_reporter: OptunaIntermediateReporter | None = None


class CoordinatedTrialResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reference: OptunaTrialReference
    claim: TrialClaim
    experiment: ExperimentSpec
    evaluation: EvaluationResult | None = None
    verification: VerificationResult | None = None
    outcome: OptunaTrialOutcome
    resource_id: str | None = None


Evaluator = Callable[[WorkerExecutionContext], EvaluationResult]
Verifier = Callable[[ExperimentSpec, EvaluationResult], VerificationResult]
ResourceRequestFactory = Callable[
    [OptunaTrialReference, ExperimentSpec], ResourceRequest | None
]


def optuna_trial_work_request_id(
    identity: StudyIdentity, reference: OptunaTrialReference
) -> str:
    """Stable operational work identity independent of the selected resource."""

    material = (
        f"{identity.attributes['run_id']}\x1f{identity.study_name}"
        f"\x1f{reference.trial_number}"
    ).encode("utf-8")
    return f"optuna-work-{hashlib.sha256(material).hexdigest()[:24]}"


class CoordinatedOptunaWorker:
    """Run one evaluator/verifier call without owning parameter suggestion.

    Heartbeats preserve fencing while injected work is running. Python cannot
    forcibly cancel an arbitrary evaluator already executing after a database or
    network failure; once that evaluator returns, the guard prevents verification
    and terminal Optuna writes when ownership loss has been detected.
    """

    def __init__(
        self,
        *,
        backend: OptunaAskTellBackend,
        identity: StudyIdentity,
        study_spec: OptunaStudySpec,
        trial_budget: int,
        claim_ttl: timedelta,
        claim_heartbeat_interval: timedelta | None = None,
        task: ResearchTask,
        metadata: ExperimentMetadata,
        search_request: SearchRequest,
        evaluator: Evaluator,
        verifier: Verifier,
        resource_broker: ResourceBroker | None = None,
        resource_request_factory: ResourceRequestFactory | None = None,
        resource_lease_ttl: timedelta = timedelta(hours=24),
        resource_heartbeat_interval: timedelta | None = None,
        base_process_environment: Mapping[str, str] | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        if trial_budget <= 0 or claim_ttl <= timedelta(0):
            raise ValueError("positive trial budget and claim ttl are required")
        if (resource_broker is None) != (resource_request_factory is None):
            raise ValueError("resource broker and request factory must be paired")
        claim_interval = claim_heartbeat_interval or claim_ttl / 4
        if claim_interval <= timedelta(0) or claim_interval > claim_ttl / 3:
            raise ValueError("claim heartbeat interval must be positive and <= ttl / 3")
        resource_interval = resource_heartbeat_interval or resource_lease_ttl / 4
        if resource_broker is not None and (
            resource_lease_ttl <= timedelta(0)
            or resource_interval <= timedelta(0)
            or resource_interval > resource_lease_ttl / 3
        ):
            raise ValueError(
                "resource heartbeat interval must be positive and <= ttl / 3"
            )
        self.backend = backend
        self.identity = identity
        self.study_spec = study_spec
        self.trial_budget = trial_budget
        self.claim_ttl = claim_ttl
        self.claim_heartbeat_interval = claim_interval
        self.task = task
        self.metadata = metadata
        self.search_request = search_request
        self.evaluator = evaluator
        self.verifier = verifier
        self.resource_broker = resource_broker
        self.resource_request_factory = resource_request_factory
        self.resource_lease_ttl = resource_lease_ttl
        self.resource_heartbeat_interval = resource_interval
        self.base_process_environment = base_process_environment
        self.clock = clock

    def run_one(self) -> CoordinatedTrialResult:
        reference, claim = self.backend.ask_and_claim_trial(
            self.identity,
            self.study_spec,
            trial_budget=self.trial_budget,
            claim_ttl=self.claim_ttl,
        )
        guard = _WorkerExecutionGuard(
            backend=self.backend,
            claim=claim,
            claim_ttl=self.claim_ttl,
            claim_interval=self.claim_heartbeat_interval,
            resource_broker=self.resource_broker,
            resource_lease_ttl=self.resource_lease_ttl,
            resource_interval=self.resource_heartbeat_interval,
        )
        admission: ResourceAdmission | None = None
        process_environment: Mapping[str, str] | None = None
        worker_id = claim.worker_id
        terminal_started = False
        try:
            guard.start()
            claim = guard.assert_owner()
            experiment = self.backend.create_experiment_spec(
                task=self.task,
                metadata=self.metadata,
                spec=self.study_spec,
                request=self.search_request,
                reference=reference,
            )
            if self.resource_request_factory is not None:
                assert self.resource_broker is not None
                resource_request = self.resource_request_factory(reference, experiment)
                if resource_request is not None:
                    claim = guard.assert_owner()
                    admission = self.resource_broker.acquire(
                        resource_request,
                        worker_id=worker_id,
                        lease_ttl=self.resource_lease_ttl,
                    )
                    claim = guard.assert_owner()
                    if admission.lease is not None:
                        guard.start_resource_heartbeat(admission.lease)
                    if (
                        admission.lease is not None
                        and admission.lease.resource_id.startswith("gpu:")
                    ):
                        process_environment = cuda_environment_for_lease(
                            admission.lease,
                            base_environment=self.base_process_environment,
                        )
            claim = guard.assert_owner()
            reporter = (
                self.backend.intermediate_reporter(
                    spec=self.study_spec,
                    reference=reference,
                )
                if self.study_spec.pruner.type != "none"
                else None
            )
            try:
                evaluation = self.evaluator(
                    WorkerExecutionContext(
                        experiment=experiment,
                        trial=reference,
                        claim=claim,
                        resource_admission=admission,
                        process_environment=process_environment,
                        intermediate_reporter=reporter,
                    )
                )
            except OptunaPruningAcknowledged:
                claim = guard.assert_owner()
                guard.stop()
                claim = guard.assert_owner()
                terminal_started = True
                outcome = self.backend.prune_claimed_trial(
                    claim=claim,
                    spec=self.study_spec,
                    reference=reference,
                    reported_at=self.clock(),
                )
                return CoordinatedTrialResult(
                    reference=reference,
                    claim=claim,
                    experiment=experiment,
                    outcome=outcome,
                    resource_id=(
                        admission.lease.resource_id
                        if admission is not None and admission.lease is not None
                        else None
                    ),
                )
            claim = guard.assert_owner()
            verification = self.verifier(experiment, evaluation)
            claim = guard.assert_owner()
            guard.stop()
            claim = guard.assert_owner()
            terminal_started = True
            outcome = self.backend.tell_claimed_trial(
                claim=claim,
                spec=self.study_spec,
                reference=reference,
                experiment=experiment,
                evaluation=evaluation,
                verification=verification,
                reported_at=self.clock(),
            )
            return CoordinatedTrialResult(
                reference=reference,
                claim=claim,
                experiment=experiment,
                evaluation=evaluation,
                verification=verification,
                outcome=outcome,
                resource_id=(
                    admission.lease.resource_id
                    if admission is not None and admission.lease is not None
                    else None
                ),
            )
        except Exception as exc:
            guard.stop()
            heartbeat_failure = guard.failure
            if heartbeat_failure is not None:
                if heartbeat_failure is exc:
                    raise
                raise heartbeat_failure from exc
            # Infrastructure/evaluator exceptions become native FAIL, never a
            # fabricated objective. A replacement, if policy admits one, is a new
            # Optuna trial.
            if not terminal_started:
                try:
                    current = guard.assert_owner()
                    self.backend.fail_claimed_trial(
                        claim=current,
                        spec=self.study_spec,
                    )
                except _PRESERVED_OPERATIONAL_ERRORS:
                    raise
            if isinstance(
                exc, (CoordinatedWorkerError, *_PRESERVED_OPERATIONAL_ERRORS)
            ):
                raise
            raise WorkerEvaluationFailed(
                "coordinated_worker_evaluation_failed"
            ) from exc
        finally:
            guard.stop()
            if (
                admission is not None
                and admission.lease is not None
                and self.resource_broker is not None
            ):
                try:
                    self.resource_broker.release_lease(
                        admission.lease.lease_id,
                        worker_id=worker_id,
                    )
                except Exception:
                    # Database expiry remains the durable recovery path.
                    pass
