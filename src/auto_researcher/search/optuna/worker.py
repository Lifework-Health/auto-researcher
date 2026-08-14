"""Task-agnostic one-trial worker seam around coordinated native ask/tell."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta, timezone
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
    ResourceRequest,
    cuda_environment_for_lease,
)
from auto_researcher.search.optuna.backend import OptunaAskTellBackend
from auto_researcher.search.optuna.coordination import TrialClaim
from auto_researcher.search.optuna.models import (
    OptunaStudySpec,
    OptunaTrialOutcome,
    OptunaTrialReference,
)
from auto_researcher.search.optuna.naming import StudyIdentity
from auto_researcher.tasks.models import ExperimentMetadata
from auto_researcher.tasks.protocols import ResearchTask


class CoordinatedWorkerError(RuntimeError):
    pass


class WorkerEvaluationFailed(CoordinatedWorkerError):
    pass


class WorkerExecutionContext(BaseModel):
    """Operational placement passed to a host evaluator, never scientific identity."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    experiment: ExperimentSpec
    trial: OptunaTrialReference
    claim: TrialClaim
    resource_admission: ResourceAdmission | None = None
    process_environment: Mapping[str, str] | None = None


class CoordinatedTrialResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reference: OptunaTrialReference
    claim: TrialClaim
    experiment: ExperimentSpec
    evaluation: EvaluationResult
    verification: VerificationResult
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
    """Run one evaluator/verifier call without owning parameter suggestion."""

    def __init__(
        self,
        *,
        backend: OptunaAskTellBackend,
        identity: StudyIdentity,
        study_spec: OptunaStudySpec,
        trial_budget: int,
        claim_ttl: timedelta,
        task: ResearchTask,
        metadata: ExperimentMetadata,
        search_request: SearchRequest,
        evaluator: Evaluator,
        verifier: Verifier,
        resource_broker: ResourceBroker | None = None,
        resource_request_factory: ResourceRequestFactory | None = None,
        resource_lease_ttl: timedelta = timedelta(hours=24),
        base_process_environment: Mapping[str, str] | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        if trial_budget <= 0 or claim_ttl <= timedelta(0):
            raise ValueError("positive trial budget and claim ttl are required")
        if (resource_broker is None) != (resource_request_factory is None):
            raise ValueError("resource broker and request factory must be paired")
        self.backend = backend
        self.identity = identity
        self.study_spec = study_spec
        self.trial_budget = trial_budget
        self.claim_ttl = claim_ttl
        self.task = task
        self.metadata = metadata
        self.search_request = search_request
        self.evaluator = evaluator
        self.verifier = verifier
        self.resource_broker = resource_broker
        self.resource_request_factory = resource_request_factory
        self.resource_lease_ttl = resource_lease_ttl
        self.base_process_environment = base_process_environment
        self.clock = clock

    def _assert_owner(self, claim: TrialClaim) -> TrialClaim:
        assert self.backend.coordination is not None
        return self.backend.coordination.assert_owner(claim)

    def run_one(self) -> CoordinatedTrialResult:
        reference, claim = self.backend.ask_and_claim_trial(
            self.identity,
            self.study_spec,
            trial_budget=self.trial_budget,
            claim_ttl=self.claim_ttl,
        )
        claim = self._assert_owner(claim)
        experiment = self.backend.create_experiment_spec(
            task=self.task,
            metadata=self.metadata,
            spec=self.study_spec,
            request=self.search_request,
            reference=reference,
        )
        admission: ResourceAdmission | None = None
        process_environment: Mapping[str, str] | None = None
        worker_id = claim.worker_id
        try:
            if self.resource_request_factory is not None:
                assert self.resource_broker is not None
                resource_request = self.resource_request_factory(reference, experiment)
                if resource_request is not None:
                    claim = self._assert_owner(claim)
                    admission = self.resource_broker.acquire(
                        resource_request,
                        worker_id=worker_id,
                        lease_ttl=self.resource_lease_ttl,
                    )
                    if (
                        admission.lease is not None
                        and admission.lease.resource_id.startswith("gpu:")
                    ):
                        process_environment = cuda_environment_for_lease(
                            admission.lease,
                            base_environment=self.base_process_environment,
                        )
            claim = self._assert_owner(claim)
            evaluation = self.evaluator(
                WorkerExecutionContext(
                    experiment=experiment,
                    trial=reference,
                    claim=claim,
                    resource_admission=admission,
                    process_environment=process_environment,
                )
            )
            claim = self._assert_owner(claim)
            verification = self.verifier(experiment, evaluation)
            claim = self._assert_owner(claim)
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
            # Infrastructure/evaluator exceptions become native FAIL, never a
            # fabricated objective. A replacement, if policy admits one, is a new
            # Optuna trial.
            try:
                current = self._assert_owner(claim)
                self.backend.fail_claimed_trial(claim=current, spec=self.study_spec)
            except Exception:
                pass
            if isinstance(exc, CoordinatedWorkerError):
                raise
            raise WorkerEvaluationFailed(
                "coordinated_worker_evaluation_failed"
            ) from exc
        finally:
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
