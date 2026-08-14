"""Durable public-API operational records for native Optuna callbacks."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


CONSTRAINT_PROTOCOL = "optuna-native-constraints-v1"
INTERMEDIATE_PROTOCOL = "optuna-intermediate-report-v1"


def _key(kind: str, trial_number: int) -> str:
    return f"auto_researcher_{kind}:{trial_number}"


def _digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ConstraintRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol: str = CONSTRAINT_PROTOCOL
    study_name: str = Field(min_length=1)
    trial_number: int = Field(ge=0)
    names: tuple[str, ...]
    values: tuple[float, ...]
    projection_identity: str = Field(min_length=1)
    digest: str = Field(min_length=64, max_length=64)

    @field_validator("values")
    @classmethod
    def values_are_finite(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        if not all(math.isfinite(value) for value in values):
            raise ValueError("constraint vectors must be finite")
        return values


class IntermediateRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol: str = INTERMEDIATE_PROTOCOL
    study_name: str = Field(min_length=1)
    trial_number: int = Field(ge=0)
    values: dict[int, float]
    prune_requested: bool = False
    pruned_at_step: int | None = Field(default=None, ge=0)
    digest: str = Field(min_length=64, max_length=64)


class OptunaOperationalRecordStore:
    """Use unique public Study user attributes as the durable callback seam.

    Optuna remains authoritative for TrialState and intermediate values. These
    records persist task-owned constraint projections and acknowledged pruning
    decisions without touching Optuna's private schema.
    """

    def __init__(self, storage: Any) -> None:
        self.storage = storage

    @staticmethod
    def _optuna() -> Any:
        try:
            import optuna
        except ImportError as exc:
            raise RuntimeError("OPTUNA HPO dependency unavailable") from exc
        return optuna

    def _study(self, study_name: str) -> Any:
        return self._optuna().load_study(
            study_name=study_name,
            storage=self.storage,
        )

    def persist_constraints(
        self,
        *,
        study_name: str,
        trial_number: int,
        names: Sequence[str],
        values: Sequence[float],
        projection_identity: str,
    ) -> ConstraintRecord:
        body = {
            "protocol": CONSTRAINT_PROTOCOL,
            "study_name": study_name,
            "trial_number": trial_number,
            "names": tuple(names),
            "values": tuple(float(value) for value in values),
            "projection_identity": projection_identity,
        }
        record = ConstraintRecord.model_validate({**body, "digest": _digest(body)})
        study = self._study(study_name)
        key = _key("constraints", trial_number)
        existing = study.user_attrs.get(key)
        if existing is not None and ConstraintRecord.model_validate(existing) != record:
            raise RuntimeError("conflicting_optuna_constraint_record")
        study.set_user_attr(key, record.model_dump(mode="json"))
        return record

    def load_constraints(
        self,
        study_name: str,
        trial_number: int,
    ) -> ConstraintRecord | None:
        payload = self._study(study_name).user_attrs.get(
            _key("constraints", trial_number)
        )
        if payload is None:
            return None
        record = ConstraintRecord.model_validate(payload)
        body = record.model_dump(mode="json", exclude={"digest"})
        if record.digest != _digest(body):
            raise RuntimeError("tampered_optuna_constraint_record")
        return record

    def constraints_for_frozen_trial(self, trial: Any) -> tuple[float, ...]:
        study_name = trial.user_attrs.get("study_name")
        if not isinstance(study_name, str) or not study_name:
            raise RuntimeError("optuna_constraint_study_identity_missing")
        record = self.load_constraints(study_name, int(trial.number))
        if record is None:
            if getattr(trial.state, "name", "") in {"PRUNED", "FAIL"}:
                return ()
            raise RuntimeError("optuna_constraint_vector_missing")
        return record.values

    def persist_intermediate(
        self,
        *,
        study_name: str,
        trial_number: int,
        values: dict[int, float],
        prune_requested: bool,
        pruned_at_step: int | None,
    ) -> IntermediateRecord:
        if not values or not all(
            step >= 0 and math.isfinite(value) for step, value in values.items()
        ):
            raise ValueError("intermediate reports require finite values and steps")
        body = {
            "protocol": INTERMEDIATE_PROTOCOL,
            "study_name": study_name,
            "trial_number": trial_number,
            "values": {
                int(step): float(value) for step, value in sorted(values.items())
            },
            "prune_requested": prune_requested,
            "pruned_at_step": pruned_at_step,
        }
        record = IntermediateRecord.model_validate({**body, "digest": _digest(body)})
        study = self._study(study_name)
        key = _key("intermediate", trial_number)
        existing_payload = study.user_attrs.get(key)
        if existing_payload is not None:
            existing = IntermediateRecord.model_validate(existing_payload)
            existing_body = existing.model_dump(mode="json", exclude={"digest"})
            if existing.digest != _digest(existing_body):
                raise RuntimeError("tampered_optuna_intermediate_record")
            if set(existing.values) - set(record.values):
                raise RuntimeError("optuna_intermediate_report_regression")
            if existing.prune_requested and not record.prune_requested:
                raise RuntimeError("optuna_prune_decision_regression")
        study.set_user_attr(key, record.model_dump(mode="json"))
        return record

    def load_intermediate(
        self,
        study_name: str,
        trial_number: int,
    ) -> IntermediateRecord | None:
        payload = self._study(study_name).user_attrs.get(
            _key("intermediate", trial_number)
        )
        if payload is None:
            return None
        record = IntermediateRecord.model_validate(payload)
        body = record.model_dump(mode="json", exclude={"digest"})
        if record.digest != _digest(body):
            raise RuntimeError("tampered_optuna_intermediate_record")
        return record
