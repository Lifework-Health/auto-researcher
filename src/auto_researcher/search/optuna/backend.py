"""Replay-safe generic Optuna Study.ask/Study.tell backend."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from typing import Any

from auto_researcher.contracts.models import (
    EvaluationResult,
    ExperimentSpec,
    SearchRequest,
    VerificationResult,
)
from auto_researcher.search.optuna.distributions import fixed_distributions
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


class OptunaAskTellBackend:
    """Thin durable adapter; LangGraph, not this class, owns the trial loop."""

    def __init__(self, storage: Any) -> None:
        self.storage = storage
        # Optuna persists trials, not sampler RNG state. Keep one sampler-bearing
        # Study object per prepared study for the lifetime of this runtime so
        # sequential ask calls advance the seeded sampler instead of resetting it.
        self._study_cache: dict[str, Any] = {}

    @staticmethod
    def _imports():
        try:
            import optuna
            from optuna.samplers import TPESampler
            from optuna.trial import Trial, TrialState
        except ImportError as exc:
            raise RuntimeError(
                "OPTUNA search requires the HPO dependency. "
                "Install with `pip install -e '.[hpo]'`."
            ) from exc
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        return optuna, TPESampler, Trial, TrialState

    def _load_study(self, study_name: str, spec: OptunaStudySpec):
        cached = self._study_cache.get(study_name)
        if cached is not None:
            return cached
        optuna, TPESampler, _, _ = self._imports()
        study = optuna.load_study(
            study_name=study_name,
            storage=self.storage,
            sampler=TPESampler(
                seed=spec.seed,
                n_startup_trials=spec.n_startup_trials,
            ),
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
        optuna, TPESampler, _, TrialState = self._imports()
        existing_names = set(optuna.study.get_all_study_names(storage=self.storage))
        created = identity.study_name not in existing_names
        study = optuna.create_study(
            storage=self.storage,
            sampler=TPESampler(
                seed=spec.seed,
                n_startup_trials=spec.n_startup_trials,
            ),
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
        current = self._reference_from_frozen(running[0]) if running else None
        return self.load_study_summary(
            identity,
            spec.direction,
            trial_budget,
            current_trial=current,
        )

    @staticmethod
    def _validate_running_trials(running, identity: StudyIdentity) -> None:
        if any("slot_index" not in trial.user_attrs for trial in running):
            raise AmbiguousRunningTrialError(
                "found an untagged RUNNING Optuna trial; manual recovery is required"
            )
        if any(
            trial.user_attrs.get("run_id") != identity.attributes["run_id"]
            or trial.user_attrs.get("request_id") != identity.attributes["request_id"]
            for trial in running
        ):
            raise AmbiguousRunningTrialError(
                "found a RUNNING trial owned by a different run or request"
            )
        slots = [int(trial.user_attrs["slot_index"]) for trial in running]
        if len(slots) != len(set(slots)) or len(running) > 1:
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
        _, _, _, TrialState = self._imports()
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
        trial.set_user_attr("study_name", identity.study_name)
        trial.set_user_attr("run_id", identity.attributes["run_id"])
        trial.set_user_attr("request_id", identity.attributes["request_id"])
        trial.set_user_attr("slot_index", slot_index)
        trial.set_user_attr("asked_at", _timestamp(asked_at))
        return (
            OptunaTrialReference(
                study_name=identity.study_name,
                trial_number=trial.number,
                slot_index=slot_index,
                parameters=trial.params,
                status=OptunaTrialStatus.RUNNING,
            ),
            False,
        )

    def create_experiment_spec(
        self,
        *,
        task: ResearchTask,
        metadata: ExperimentMetadata,
        spec: OptunaStudySpec,
        request: SearchRequest,
        reference: OptunaTrialReference,
    ) -> ExperimentSpec:
        _, _, Trial, _ = self._imports()
        configuration = dict(spec.fixed_configuration)
        configuration.update(reference.parameters)
        normalised = task.normalise_configuration(configuration)
        digest = hashlib.sha256(
            f"{reference.study_name}\x1f{reference.trial_number}".encode("utf-8")
        ).hexdigest()[:16]
        experiment_id = f"experiment-{digest}"
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
        study = self._load_study(reference.study_name, spec)
        frozen = self._trial_by_number(study, reference.trial_number)
        Trial(study, frozen._trial_id).set_user_attr(
            "experiment_id",
            experiment_id,
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
        _, _, Trial, TrialState = self._imports()
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
        if frozen.state != TrialState.RUNNING:
            existing_digest = frozen.user_attrs.get("report_digest")
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
                return self._outcome_from_frozen(frozen)
            raise ConflictingTrialReportError(
                f"trial {reference.trial_number} already has a conflicting report"
            )

        live_trial = Trial(study, frozen._trial_id)
        live_trial.set_user_attr("report_digest", digest)
        live_trial.set_user_attr("reported_at", _timestamp(reported_at))
        live_trial.set_user_attr("feasible", feasible)
        live_trial.set_user_attr("experiment", payload["experiment"])
        live_trial.set_user_attr("evaluation", payload["evaluation"])
        live_trial.set_user_attr("verification", payload["verification"])
        live_trial.set_user_attr("report_status", status.value)
        if status == OptunaTrialStatus.COMPLETE:
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
            self._trial_by_number(study, reference.trial_number)
        )

    @staticmethod
    def _outcome_from_frozen(trial) -> OptunaTrialOutcome:
        status = OptunaTrialStatus(trial.state.name)
        evaluation = trial.user_attrs.get("evaluation", {})
        verification = trial.user_attrs.get("verification", {})
        return OptunaTrialOutcome(
            trial_number=trial.number,
            status=status,
            objective_value=trial.value,
            feasible=bool(trial.user_attrs.get("feasible", False)),
            experiment_id=str(trial.user_attrs.get("experiment_id", "")),
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
        _, _, _, TrialState = self._imports()
        study = self._load_study_by_identity(identity)
        trials = study.get_trials(deepcopy=True)
        completed = [trial for trial in trials if trial.state == TrialState.COMPLETE]
        failed = [trial for trial in trials if trial.state == TrialState.FAIL]
        selection = select_trials(
            (
                SelectionCandidate(
                    trial_number=trial.number,
                    score=float(trial.value),
                    feasible=bool(trial.user_attrs.get("feasible", False)),
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
        optuna, _, _, _ = self._imports()
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
        optuna, _, _, _ = self._imports()
        study = optuna.load_study(study_name=study_name, storage=self.storage)
        trial = self._trial_by_number(study, trial_number)
        return (
            ExperimentSpec.model_validate(trial.user_attrs["experiment"]),
            EvaluationResult.model_validate(trial.user_attrs["evaluation"]),
            VerificationResult.model_validate(trial.user_attrs["verification"]),
        )

    def trial_outcomes(self, study_name: str) -> list[OptunaTrialOutcome]:
        optuna, _, _, TrialState = self._imports()
        study = optuna.load_study(study_name=study_name, storage=self.storage)
        outcomes = []
        for trial in study.get_trials(deepcopy=True):
            if trial.state in {TrialState.COMPLETE, TrialState.FAIL}:
                outcomes.append(self._outcome_from_frozen(trial))
        return outcomes

    def set_study_completed_at(
        self,
        study_name: str,
        completed_at: datetime,
    ) -> str:
        optuna, _, _, _ = self._imports()
        study = optuna.load_study(study_name=study_name, storage=self.storage)
        existing = study.user_attrs.get("completed_at")
        if existing is not None:
            return str(existing)
        value = _timestamp(completed_at)
        study.set_user_attr("completed_at", value)
        return value

    def study_user_attrs(self, study_name: str) -> dict[str, Any]:
        optuna, _, _, _ = self._imports()
        study = optuna.load_study(study_name=study_name, storage=self.storage)
        return dict(study.user_attrs)

    def trial_user_attrs(
        self,
        study_name: str,
        trial_number: int,
    ) -> dict[str, Any]:
        optuna, _, _, _ = self._imports()
        study = optuna.load_study(study_name=study_name, storage=self.storage)
        return dict(self._trial_by_number(study, trial_number).user_attrs)
