"""Replay-safe generic Optuna Study.ask/Study.tell backend."""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from auto_researcher.contracts.models import (
    EvaluationResult,
    ExperimentSpec,
    SearchRequest,
    VerificationResult,
)
from auto_researcher.search.optuna.distributions import fixed_distributions
from auto_researcher.search.optuna.coordination import (
    PostgresOptunaCoordination,
    SharedTrialBudgetExhausted,
    TrialNoLongerRunning,
    TrialClaim,
)
from auto_researcher.search.optuna.models import (
    OptimisationDirection,
    OptunaStudyState,
    OptunaTrialOutcome,
    OptunaTrialReference,
    OptunaTrialStatus,
    OptunaStudySpec,
)
from auto_researcher.search.optuna.naming import StudyIdentity
from auto_researcher.search.optuna.recovery import (
    AmbiguousRunningTrialError,
    ConflictingTrialReportError,
    StudyIdentityMismatchError,
)
from auto_researcher.search.optuna.selection import (
    SelectionCandidate,
    select_trials,
)
from auto_researcher.tasks.models import ExperimentMetadata
from auto_researcher.tasks.protocols import ResearchTask


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _timestamp(value: datetime) -> str:
    return value.isoformat()


def _report_key(trial_number: int) -> str:
    return f"auto_researcher_trial_report:{trial_number}"


class OptunaAskTellBackend:
    """Thin durable adapter; LangGraph, not this class, owns the trial loop."""

    def __init__(
        self,
        storage: Any,
        *,
        shared_workers: bool = False,
        coordination: PostgresOptunaCoordination | None = None,
        worker_id: str | None = None,
        worker_session_id: str | None = None,
    ) -> None:
        if shared_workers and (coordination is None or not worker_id):
            raise ValueError("shared Optuna requires coordination and worker_id")
        if worker_session_id is not None and not worker_session_id:
            raise ValueError("worker_session_id cannot be empty")
        self.storage = storage
        self.shared_workers = shared_workers
        self.coordination = coordination
        self.worker_id = worker_id
        # Operational runtime incarnation only. It never enters study, request,
        # experiment, or other scientific identity.
        self.worker_session_id = worker_session_id
        if shared_workers and self.worker_session_id is None:
            self.worker_session_id = str(uuid4())
        # Optuna persists trials, not sampler RNG state. Keep one sampler-bearing
        # Study object per prepared study for the lifetime of this runtime so
        # sequential ask calls advance the seeded sampler instead of resetting it.
        self._study_cache: dict[str, Any] = {}
        # Public Trial objects returned by Study.ask() are retained only for the
        # current runtime. Restart recovery uses public Study.tell(trial_number)
        # and never reconstructs a Trial from Optuna's private _trial_id.
        self._live_trials: dict[tuple[str, int], Any] = {}
        self._reported_digests: dict[tuple[str, int], str] = {}

    @staticmethod
    def _imports():
        try:
            import optuna
            from optuna.samplers import TPESampler
            from optuna.trial import TrialState
        except ImportError as exc:
            raise RuntimeError(
                "OPTUNA search requires the HPO dependency. "
                "Install with `pip install -e '.[hpo]'`."
            ) from exc
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        return optuna, TPESampler, TrialState

    def _sampler(self, spec: OptunaStudySpec):
        optuna, TPESampler, _ = self._imports()
        seed = spec.seed
        if self.shared_workers:
            assert self.worker_id is not None
            assert self.worker_session_id is not None
            # Optuna does not durably persist sampler RNG state. Include the
            # process incarnation so restarting one logical worker cannot replay
            # that worker's initial native TPE stream.
            material = (
                f"{spec.seed}\x1f{self.worker_id}\x1f{self.worker_session_id}"
            ).encode("utf-8")
            seed = int.from_bytes(hashlib.sha256(material).digest()[:4], "big")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=optuna.exceptions.ExperimentalWarning,
                message=r"Argument ``constant_liar``.*",
            )
            return TPESampler(
                seed=seed,
                n_startup_trials=spec.n_startup_trials,
                constant_liar=self.shared_workers,
            )

    def _load_study(self, study_name: str, spec: OptunaStudySpec):
        cached = self._study_cache.get(study_name)
        if cached is not None:
            return cached
        optuna, _, _ = self._imports()
        study = optuna.load_study(
            study_name=study_name,
            storage=self.storage,
            sampler=self._sampler(spec),
        )
        self._study_cache[study_name] = study
        return study

    def prepare_or_load_study(
        self,
        identity: StudyIdentity,
        spec: OptunaStudySpec,
        *,
        started_at: datetime,
        trial_budget: int,
    ) -> OptunaStudyState:
        optuna, _, TrialState = self._imports()
        existing_names = set(optuna.study.get_all_study_names(storage=self.storage))
        created = identity.study_name not in existing_names
        study = optuna.create_study(
            storage=self.storage,
            sampler=self._sampler(spec),
            study_name=identity.study_name,
            direction=spec.direction.value.lower(),
            load_if_exists=True,
        )
        if created:
            for key, value in identity.attributes.items():
                study.set_user_attr(key, value)
            study.set_user_attr("started_at", _timestamp(started_at))
        else:
            mismatches = {
                key: (study.user_attrs.get(key), expected)
                for key, expected in identity.attributes.items()
                if study.user_attrs.get(key) != expected
            }
            if mismatches:
                details = ", ".join(
                    f"{key}={actual!r} expected {expected!r}"
                    for key, (actual, expected) in sorted(mismatches.items())
                )
                raise StudyIdentityMismatchError(
                    f"Optuna study identity mismatch: {details}"
                )
            if study.direction.name != spec.direction.value:
                raise StudyIdentityMismatchError(
                    "Optuna study direction does not match the task specification"
                )
        self._study_cache[identity.study_name] = study

        running = [
            trial
            for trial in study.get_trials(deepcopy=True)
            if trial.state == TrialState.RUNNING
        ]
        self._validate_running_trials(running, identity)
        current = (
            self._reference_from_frozen(running[0])
            if running and not self.shared_workers
            else None
        )
        return self.load_study_summary(
            identity,
            spec.direction,
            trial_budget,
            current_trial=current,
        )

    def _validate_running_trials(self, running, identity: StudyIdentity) -> None:
        untagged = [trial for trial in running if "slot_index" not in trial.user_attrs]
        if untagged and not self.shared_workers:
            raise AmbiguousRunningTrialError(
                "found an untagged RUNNING Optuna trial; manual recovery is required"
            )
        if self.shared_workers and untagged:
            assert self.coordination is not None
            if any(
                self.coordination.claim_for_trial(identity.study_name, trial.number)
                is not None
                for trial in untagged
            ):
                raise AmbiguousRunningTrialError(
                    "claimed RUNNING trial is missing required ownership metadata"
                )
        tagged = [trial for trial in running if "slot_index" in trial.user_attrs]
        if any(
            trial.user_attrs.get("run_id") != identity.attributes["run_id"]
            or trial.user_attrs.get("request_id") != identity.attributes["request_id"]
            for trial in tagged
        ):
            raise AmbiguousRunningTrialError(
                "found a RUNNING trial owned by a different run or request"
            )
        slots = [int(trial.user_attrs["slot_index"]) for trial in tagged]
        if len(slots) != len(set(slots)):
            raise AmbiguousRunningTrialError(
                "duplicate RUNNING trial slot_index values require recovery"
            )
        if not self.shared_workers and len(running) > 1:
            raise AmbiguousRunningTrialError(
                "multiple RUNNING trials violate PR 3 sequential execution"
            )

    @staticmethod
    def _reference_from_frozen(trial) -> OptunaTrialReference:
        status = OptunaTrialStatus(trial.state.name)
        return OptunaTrialReference(
            study_name=trial.study_name
            if hasattr(trial, "study_name")
            else str(trial.user_attrs.get("study_name", "unknown")),
            trial_number=trial.number,
            slot_index=int(trial.user_attrs["slot_index"]),
            parameters=trial.params,
            experiment_id=trial.user_attrs.get("experiment_id"),
            status=status,
        )

    def ask_or_recover_trial(
        self,
        identity: StudyIdentity,
        spec: OptunaStudySpec,
        *,
        slot_index: int,
        asked_at: datetime,
    ) -> tuple[OptunaTrialReference, bool]:
        if self.shared_workers:
            raise RuntimeError("shared workers must use ask_and_claim_trial")
        _, _, TrialState = self._imports()
        study = self._load_study(identity.study_name, spec)
        running = [
            trial
            for trial in study.get_trials(deepcopy=True)
            if trial.state == TrialState.RUNNING
        ]
        self._validate_running_trials(running, identity)
        matching = [
            trial
            for trial in running
            if int(trial.user_attrs["slot_index"]) == slot_index
        ]
        if matching:
            trial = matching[0]
            return (
                OptunaTrialReference(
                    study_name=identity.study_name,
                    trial_number=trial.number,
                    slot_index=slot_index,
                    parameters=trial.params,
                    experiment_id=trial.user_attrs.get("experiment_id"),
                    status=OptunaTrialStatus.RUNNING,
                ),
                True,
            )
        if running:
            raise AmbiguousRunningTrialError(
                f"RUNNING trial belongs to slot "
                f"{running[0].user_attrs['slot_index']}, not {slot_index}"
            )
        trial = study.ask(fixed_distributions=fixed_distributions(spec.parameters))
        experiment_id = self._experiment_id(identity.study_name, trial.number)
        trial.set_user_attr("study_name", identity.study_name)
        trial.set_user_attr("run_id", identity.attributes["run_id"])
        trial.set_user_attr("request_id", identity.attributes["request_id"])
        trial.set_user_attr("slot_index", slot_index)
        trial.set_user_attr("asked_at", _timestamp(asked_at))
        trial.set_user_attr("experiment_id", experiment_id)
        self._live_trials[(identity.study_name, trial.number)] = trial
        return (
            OptunaTrialReference(
                study_name=identity.study_name,
                trial_number=trial.number,
                slot_index=slot_index,
                parameters=trial.params,
                experiment_id=experiment_id,
                status=OptunaTrialStatus.RUNNING,
            ),
            False,
        )

    @staticmethod
    def _experiment_id(study_name: str, trial_number: int) -> str:
        digest = hashlib.sha256(
            f"{study_name}\x1f{trial_number}".encode("utf-8")
        ).hexdigest()[:16]
        return f"experiment-{digest}"

    def ask_and_claim_trial(
        self,
        identity: StudyIdentity,
        spec: OptunaStudySpec,
        *,
        trial_budget: int,
        claim_ttl: timedelta,
    ) -> tuple[OptunaTrialReference, TrialClaim]:
        """Atomically admit native ask and establish durable worker fencing."""

        if (
            not self.shared_workers
            or self.coordination is None
            or self.worker_id is None
        ):
            raise RuntimeError("shared_optuna_coordination_not_configured")
        study = self._load_study(identity.study_name, spec)

        def native_ask(database_now: datetime, claim_id: str):
            trials = study.get_trials(deepcopy=True)
            if len(trials) >= trial_budget:
                raise SharedTrialBudgetExhausted("shared_trial_budget_exhausted")
            slot_index = len(trials)
            trial = study.ask(fixed_distributions=fixed_distributions(spec.parameters))
            experiment_id = self._experiment_id(identity.study_name, trial.number)
            for key, value in {
                "study_name": identity.study_name,
                "run_id": identity.attributes["run_id"],
                "request_id": identity.attributes["request_id"],
                "slot_index": slot_index,
                "asked_at": _timestamp(database_now),
                "worker_id": self.worker_id,
                "claim_id": claim_id,
                "experiment_id": experiment_id,
            }.items():
                trial.set_user_attr(key, value)
            self._live_trials[(identity.study_name, trial.number)] = trial
            reference = OptunaTrialReference(
                study_name=identity.study_name,
                trial_number=trial.number,
                slot_index=slot_index,
                parameters=trial.params,
                experiment_id=experiment_id,
                status=OptunaTrialStatus.RUNNING,
            )
            return trial.number, reference

        claim, reference = self.coordination.admit_ask_and_claim(
            study_name=identity.study_name,
            trial_budget=trial_budget,
            worker_id=self.worker_id,
            ttl=claim_ttl,
            ask=native_ask,
        )
        return reference, claim

    def create_experiment_spec(
        self,
        *,
        task: ResearchTask,
        metadata: ExperimentMetadata,
        spec: OptunaStudySpec,
        request: SearchRequest,
        reference: OptunaTrialReference,
    ) -> ExperimentSpec:
        configuration = dict(spec.fixed_configuration)
        configuration.update(reference.parameters)
        normalised = task.normalise_configuration(configuration)
        experiment_id = self._experiment_id(
            reference.study_name, reference.trial_number
        )
        experiment = ExperimentSpec(
            experiment_id=experiment_id,
            hypothesis_id=request.hypothesis_id,
            search_request_id=request.request_id,
            configuration=normalised,
            evaluator_id=metadata.evaluator_id,
            code_version=metadata.code_version,
            dataset_version=metadata.dataset_version,
            provenance=metadata.provenance,
        )
        return experiment

    def tell_trial(
        self,
        *,
        spec: OptunaStudySpec,
        reference: OptunaTrialReference,
        experiment: ExperimentSpec,
        evaluation: EvaluationResult,
        verification: VerificationResult,
        reported_at: datetime,
    ) -> OptunaTrialOutcome:
        _, _, TrialState = self._imports()
        score = evaluation.primary_score
        complete = (
            evaluation.success
            and score is not None
            and math.isfinite(score)
            and verification.verified
            and experiment.experiment_id == evaluation.experiment_id
            and experiment.experiment_id == verification.experiment_id
            and experiment.provenance == evaluation.provenance
            and evaluation.provenance == verification.provenance
            and verification.measured_score is not None
            and math.isclose(
                float(verification.measured_score),
                float(score),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        )
        status = OptunaTrialStatus.COMPLETE if complete else OptunaTrialStatus.FAIL
        feasible = bool(complete and verification.constraint_compliant)
        payload = {
            "status": status.value,
            "objective_value": float(score) if complete and score is not None else None,
            "feasible": feasible,
            "experiment": experiment.model_dump(mode="json"),
            "evaluation": evaluation.model_dump(mode="json"),
            "verification": verification.model_dump(mode="json"),
        }
        digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
        study = self._load_study(reference.study_name, spec)
        frozen = self._trial_by_number(study, reference.trial_number)
        durable_report = study.user_attrs.get(_report_key(reference.trial_number), {})
        if frozen.state != TrialState.RUNNING:
            existing_digest = (
                frozen.user_attrs.get("report_digest")
                or durable_report.get("report_digest")
                or self._reported_digests.get(
                    (reference.study_name, reference.trial_number)
                )
            )
            expected_state = (
                TrialState.COMPLETE
                if status == OptunaTrialStatus.COMPLETE
                else TrialState.FAIL
            )
            same_value = status == OptunaTrialStatus.FAIL or (
                frozen.value is not None
                and score is not None
                and float(frozen.value) == float(score)
            )
            if (
                frozen.state == expected_state
                and existing_digest == digest
                and same_value
            ):
                return self._outcome_from_frozen(
                    frozen, durable_report, study_name=reference.study_name
                )
            raise ConflictingTrialReportError(
                f"trial {reference.trial_number} already has a conflicting report"
            )

        live_trial = self._live_trials.get(
            (reference.study_name, reference.trial_number)
        )
        if live_trial is not None:
            live_trial.set_user_attr("report_digest", digest)
            live_trial.set_user_attr("reported_at", _timestamp(reported_at))
            live_trial.set_user_attr("feasible", feasible)
            live_trial.set_user_attr("experiment", payload["experiment"])
            live_trial.set_user_attr("evaluation", payload["evaluation"])
            live_trial.set_user_attr("verification", payload["verification"])
            live_trial.set_user_attr("report_status", status.value)
        # A unique public Study user-attribute key preserves the exact report
        # across process reconstruction without rebuilding Trial from _trial_id.
        # It complements Optuna's authoritative value/state; it never mirrors a
        # competing TrialState or parameter proposal.
        study.set_user_attr(
            _report_key(reference.trial_number),
            {
                "report_digest": digest,
                "reported_at": _timestamp(reported_at),
                "feasible": feasible,
                "experiment": payload["experiment"],
                "evaluation": payload["evaluation"],
                "verification": payload["verification"],
                "report_status": status.value,
            },
        )
        self._reported_digests[(reference.study_name, reference.trial_number)] = digest
        if status == OptunaTrialStatus.COMPLETE:
            assert score is not None
            study.tell(
                reference.trial_number,
                float(score),
                state=TrialState.COMPLETE,
                skip_if_finished=True,
            )
        else:
            study.tell(
                reference.trial_number,
                state=TrialState.FAIL,
                skip_if_finished=True,
            )
        return self._outcome_from_frozen(
            self._trial_by_number(study, reference.trial_number),
            study.user_attrs.get(_report_key(reference.trial_number), {}),
            study_name=reference.study_name,
        )

    def report_identity(
        self,
        *,
        experiment: ExperimentSpec,
        evaluation: EvaluationResult,
        verification: VerificationResult,
    ) -> tuple[str, str]:
        """Return the exact immutable report digest and resulting Optuna state."""

        score = evaluation.primary_score
        complete = (
            evaluation.success
            and score is not None
            and math.isfinite(score)
            and verification.verified
            and experiment.experiment_id == evaluation.experiment_id
            and experiment.experiment_id == verification.experiment_id
            and experiment.provenance == evaluation.provenance
            and evaluation.provenance == verification.provenance
            and verification.measured_score is not None
            and math.isclose(
                float(verification.measured_score),
                float(score),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        )
        status = OptunaTrialStatus.COMPLETE if complete else OptunaTrialStatus.FAIL
        payload = {
            "status": status.value,
            "objective_value": float(score) if complete and score is not None else None,
            "feasible": bool(complete and verification.constraint_compliant),
            "experiment": experiment.model_dump(mode="json"),
            "evaluation": evaluation.model_dump(mode="json"),
            "verification": verification.model_dump(mode="json"),
        }
        return (
            hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest(),
            status.value,
        )

    def tell_claimed_trial(
        self,
        *,
        claim: TrialClaim,
        spec: OptunaStudySpec,
        reference: OptunaTrialReference,
        experiment: ExperimentSpec,
        evaluation: EvaluationResult,
        verification: VerificationResult,
        reported_at: datetime,
    ) -> OptunaTrialOutcome:
        if self.coordination is None:
            raise RuntimeError("shared_optuna_coordination_not_configured")
        digest, status = self.report_identity(
            experiment=experiment,
            evaluation=evaluation,
            verification=verification,
        )
        current = self.coordination.claim_for_trial(
            claim.study_name, claim.trial_number
        )
        if current is not None and current.released_at is not None:
            _, _, TrialState = self._imports()
            frozen = self._trial_by_number(
                self._load_study(reference.study_name, spec),
                reference.trial_number,
            )
            expected_state = (
                TrialState.COMPLETE
                if status == OptunaTrialStatus.COMPLETE.value
                else TrialState.FAIL
            )
            if current.report_digest != digest or frozen.state != expected_state:
                raise ConflictingTrialReportError(
                    f"trial {reference.trial_number} already has a conflicting report"
                )
            if expected_state == TrialState.COMPLETE and (
                frozen.value is None
                or evaluation.primary_score is None
                or float(frozen.value) != float(evaluation.primary_score)
            ):
                raise ConflictingTrialReportError(
                    f"trial {reference.trial_number} already has a conflicting value"
                )
            return OptunaTrialOutcome(
                trial_number=reference.trial_number,
                status=OptunaTrialStatus(status),
                objective_value=(
                    float(evaluation.primary_score)
                    if status == OptunaTrialStatus.COMPLETE.value
                    and evaluation.primary_score is not None
                    else None
                ),
                feasible=bool(
                    status == OptunaTrialStatus.COMPLETE.value
                    and verification.constraint_compliant
                ),
                experiment_id=experiment.experiment_id,
                parameters=reference.parameters,
                evaluation_artefact_references=evaluation.artefact_references,
                verification_status=verification.evidence_status.value,
            )
        recorded = self.coordination.record_report(
            claim,
            report_digest=digest,
        )
        return self.coordination.run_owned_and_release(
            recorded,
            lambda: self.tell_trial(
                spec=spec,
                reference=reference,
                experiment=experiment,
                evaluation=evaluation,
                verification=verification,
                reported_at=reported_at,
            ),
        )

    def fail_claimed_trial(self, *, claim: TrialClaim, spec: OptunaStudySpec) -> None:
        """Reconcile stale infrastructure ownership through public Study.tell()."""

        if self.coordination is None:
            raise RuntimeError("shared_optuna_coordination_not_configured")
        _, _, TrialState = self._imports()
        study = self._load_study(claim.study_name, spec)

        def fail() -> None:
            frozen = self._trial_by_number(study, claim.trial_number)
            if frozen.state != TrialState.RUNNING:
                raise TrialNoLongerRunning("trial_no_longer_running")
            study.tell(
                claim.trial_number,
                state=TrialState.FAIL,
                skip_if_finished=True,
            )

        self.coordination.run_owned_and_release(claim, fail)

    def reconcile_unclaimed_trials(
        self,
        *,
        identity: StudyIdentity,
        spec: OptunaStudySpec,
        orphan_grace: timedelta,
        now: datetime,
    ) -> tuple[int, ...]:
        """Fail bounded old ASK crash orphans without ever guessing ownership.

        Optuna records ``datetime_start`` using the asking host's clock. The
        configured grace must therefore exceed the deployment's maximum allowed
        host clock skew. Durable claimed-trial expiry uses PostgreSQL time instead.
        """

        if (
            not self.shared_workers
            or self.coordination is None
            or orphan_grace <= timedelta(0)
            or now.tzinfo is None
        ):
            raise ValueError("shared coordination, aware now, and grace are required")
        _, _, TrialState = self._imports()
        study = self._load_study(identity.study_name, spec)
        coordination = self.coordination
        assert coordination is not None

        def reconcile() -> tuple[int, ...]:
            reconciled: list[int] = []
            for trial in study.get_trials(deepcopy=True):
                if trial.state != TrialState.RUNNING:
                    continue
                if (
                    coordination.claim_for_trial(identity.study_name, trial.number)
                    is not None
                ):
                    continue
                asked_at = trial.user_attrs.get("asked_at")
                started: datetime | None = None
                if isinstance(asked_at, str):
                    try:
                        parsed = datetime.fromisoformat(asked_at)
                    except ValueError:
                        parsed = None
                    if parsed is not None and parsed.tzinfo is not None:
                        # ask_and_claim_trial obtains this value from PostgreSQL.
                        started = parsed
                if started is None:
                    if trial.datetime_start is None:
                        continue
                    started = trial.datetime_start
                if started.tzinfo is None:
                    # Optuna 4.9 RDB datetime_start is host-local and naive. Python
                    # applies the recovery host's zone rules here; shared workers
                    # must therefore use one configured timezone (preferably UTC)
                    # and orphan_grace must also cover the permitted clock skew.
                    started = started.astimezone(now.tzinfo)
                if now - started < orphan_grace:
                    continue
                study.tell(
                    trial.number,
                    state=TrialState.FAIL,
                    skip_if_finished=True,
                )
                reconciled.append(trial.number)
            return tuple(reconciled)

        return coordination.run_study_locked(identity.study_name, reconcile)

    @staticmethod
    def _outcome_from_frozen(
        trial,
        durable_report: dict[str, Any] | None = None,
        *,
        study_name: str | None = None,
    ) -> OptunaTrialOutcome:
        report = durable_report or {}
        status = OptunaTrialStatus(trial.state.name)
        evaluation = trial.user_attrs.get("evaluation") or report.get("evaluation", {})
        verification = trial.user_attrs.get("verification") or report.get(
            "verification", {}
        )
        return OptunaTrialOutcome(
            trial_number=trial.number,
            status=status,
            objective_value=trial.value,
            feasible=bool(
                trial.user_attrs.get("feasible", report.get("feasible", False))
            ),
            experiment_id=str(
                trial.user_attrs.get("experiment_id")
                or report.get("experiment", {}).get("experiment_id")
                or (
                    OptunaAskTellBackend._experiment_id(study_name, trial.number)
                    if study_name is not None
                    else f"unclaimed-trial-{trial.number}"
                )
            ),
            parameters=trial.params,
            evaluation_artefact_references=tuple(
                evaluation.get("artefact_references", ())
            ),
            verification_status=str(
                verification.get("evidence_status", "INCONCLUSIVE")
            ),
        )

    def load_study_summary(
        self,
        identity: StudyIdentity,
        direction: OptimisationDirection,
        trial_budget: int,
        *,
        current_trial: OptunaTrialReference | None = None,
    ) -> OptunaStudyState:
        _, _, TrialState = self._imports()
        study = self._load_study_by_identity(identity)
        trials = study.get_trials(deepcopy=True)
        completed = [trial for trial in trials if trial.state == TrialState.COMPLETE]
        failed = [trial for trial in trials if trial.state == TrialState.FAIL]
        selection = select_trials(
            (
                SelectionCandidate(
                    trial_number=trial.number,
                    score=float(trial.value),
                    feasible=bool(
                        trial.user_attrs.get(
                            "feasible",
                            study.user_attrs.get(_report_key(trial.number), {}).get(
                                "feasible", False
                            ),
                        )
                    ),
                )
                for trial in completed
                if trial.value is not None
            ),
            direction,
        )
        return OptunaStudyState(
            study_name=identity.study_name,
            search_space_hash=identity.search_space_hash,
            direction=direction,
            trial_budget=trial_budget,
            trials_asked=len(trials),
            trials_completed=len(completed),
            trials_failed=len(failed),
            current_trial=current_trial,
            best_feasible_trial_number=(
                selection.best_feasible.trial_number
                if selection.best_feasible
                else None
            ),
            best_feasible_score=(
                selection.best_feasible.score if selection.best_feasible else None
            ),
            best_overall_trial_number=(
                selection.best_overall.trial_number if selection.best_overall else None
            ),
            best_overall_score=(
                selection.best_overall.score if selection.best_overall else None
            ),
        )

    def _load_study_by_identity(self, identity: StudyIdentity):
        cached = self._study_cache.get(identity.study_name)
        if cached is not None:
            return cached
        optuna, _, _ = self._imports()
        return optuna.load_study(
            study_name=identity.study_name,
            storage=self.storage,
        )

    @staticmethod
    def _trial_by_number(study, trial_number: int):
        matches = [
            trial
            for trial in study.get_trials(deepcopy=True)
            if trial.number == trial_number
        ]
        if len(matches) != 1:
            raise KeyError(
                f"study {study.study_name!r} has no unique trial {trial_number}"
            )
        return matches[0]

    def load_trial_models(
        self,
        study_name: str,
        trial_number: int,
    ) -> tuple[ExperimentSpec, EvaluationResult, VerificationResult]:
        optuna, _, _ = self._imports()
        study = optuna.load_study(study_name=study_name, storage=self.storage)
        trial = self._trial_by_number(study, trial_number)
        report = study.user_attrs.get(_report_key(trial_number), {})
        return (
            ExperimentSpec.model_validate(
                trial.user_attrs.get("experiment") or report["experiment"]
            ),
            EvaluationResult.model_validate(
                trial.user_attrs.get("evaluation") or report["evaluation"]
            ),
            VerificationResult.model_validate(
                trial.user_attrs.get("verification") or report["verification"]
            ),
        )

    def trial_outcomes(self, study_name: str) -> list[OptunaTrialOutcome]:
        optuna, _, TrialState = self._imports()
        study = optuna.load_study(study_name=study_name, storage=self.storage)
        outcomes = []
        for trial in study.get_trials(deepcopy=True):
            if trial.state in {TrialState.COMPLETE, TrialState.FAIL}:
                outcomes.append(
                    self._outcome_from_frozen(
                        trial,
                        study.user_attrs.get(_report_key(trial.number), {}),
                        study_name=study_name,
                    )
                )
        return outcomes

    def set_study_completed_at(
        self,
        study_name: str,
        completed_at: datetime,
    ) -> str:
        optuna, _, _ = self._imports()
        study = optuna.load_study(study_name=study_name, storage=self.storage)
        existing = study.user_attrs.get("completed_at")
        if existing is not None:
            return str(existing)
        value = _timestamp(completed_at)
        study.set_user_attr("completed_at", value)
        return value

    def study_user_attrs(self, study_name: str) -> dict[str, Any]:
        optuna, _, _ = self._imports()
        study = optuna.load_study(study_name=study_name, storage=self.storage)
        return dict(study.user_attrs)

    def trial_user_attrs(
        self,
        study_name: str,
        trial_number: int,
    ) -> dict[str, Any]:
        optuna, _, _ = self._imports()
        study = optuna.load_study(study_name=study_name, storage=self.storage)
        attrs = dict(self._trial_by_number(study, trial_number).user_attrs)
        attrs.update(study.user_attrs.get(_report_key(trial_number), {}))
        return attrs
