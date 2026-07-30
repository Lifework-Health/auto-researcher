"""Domain-neutral feasible and diagnostic Optuna winner selection."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from auto_researcher.search.optuna.models import OptimisationDirection


@dataclass(frozen=True)
class SelectionCandidate:
    trial_number: int
    score: float
    feasible: bool


@dataclass(frozen=True)
class StudySelection:
    best_feasible: SelectionCandidate | None
    best_overall: SelectionCandidate | None


def _better(
    candidate: SelectionCandidate,
    current: SelectionCandidate | None,
    direction: OptimisationDirection,
) -> bool:
    if current is None:
        return True
    if candidate.score == current.score:
        return candidate.trial_number < current.trial_number
    if direction == OptimisationDirection.MAXIMIZE:
        return candidate.score > current.score
    return candidate.score < current.score


def select_trials(
    candidates: Iterable[SelectionCandidate],
    direction: OptimisationDirection,
) -> StudySelection:
    feasible: SelectionCandidate | None = None
    overall: SelectionCandidate | None = None
    for candidate in candidates:
        if not math.isfinite(candidate.score):
            continue
        if _better(candidate, overall, direction):
            overall = candidate
        if candidate.feasible and _better(candidate, feasible, direction):
            feasible = candidate
    return StudySelection(best_feasible=feasible, best_overall=overall)
