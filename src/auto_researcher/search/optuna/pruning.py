"""Cooperative native Trial.report/should_prune integration."""

from __future__ import annotations

import math
from typing import Any

from auto_researcher.search.optuna.operational import OptunaOperationalRecordStore


class OptunaPruningAcknowledged(RuntimeError):
    """Raised only when the evaluator cooperatively stops at a safe checkpoint."""


class OptunaIntermediateReporter:
    def __init__(
        self,
        *,
        trial: Any,
        study_name: str,
        operational_store: OptunaOperationalRecordStore,
    ) -> None:
        self._trial = trial
        self._study_name = study_name
        self._store = operational_store
        self._values: dict[int, float] = {}
        self._prune_requested = False
        self._prune_requested_at_step: int | None = None
        self._prune_acknowledged = False
        self._prune_acknowledged_at_step: int | None = None

    def report(self, value: float, step: int) -> bool:
        numeric = float(value)
        if step < 0 or not math.isfinite(numeric):
            raise ValueError("intermediate Optuna values require finite value and step")
        existing = self._values.get(step)
        if existing is not None and existing != numeric:
            raise RuntimeError("conflicting_optuna_intermediate_value")
        self._trial.report(numeric, step)
        self._values[step] = numeric
        requested = bool(self._trial.should_prune())
        self._prune_requested = self._prune_requested or requested
        if requested and self._prune_requested_at_step is None:
            self._prune_requested_at_step = step
        self._store.persist_intermediate(
            study_name=self._study_name,
            trial_number=self._trial.number,
            values=self._values,
            prune_requested=self._prune_requested,
            prune_requested_at_step=self._prune_requested_at_step,
            prune_acknowledged=self._prune_acknowledged,
            prune_acknowledged_at_step=self._prune_acknowledged_at_step,
        )
        return requested

    def acknowledge_pruning(self) -> None:
        if not self._prune_requested:
            raise RuntimeError("Optuna pruning was not requested")
        self._prune_acknowledged = True
        self._prune_acknowledged_at_step = self._prune_requested_at_step
        self._store.persist_intermediate(
            study_name=self._study_name,
            trial_number=self._trial.number,
            values=self._values,
            prune_requested=True,
            prune_requested_at_step=self._prune_requested_at_step,
            prune_acknowledged=True,
            prune_acknowledged_at_step=self._prune_acknowledged_at_step,
        )
        raise OptunaPruningAcknowledged("optuna_pruning_cooperatively_acknowledged")
