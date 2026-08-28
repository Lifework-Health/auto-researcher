"""Native Optuna diagnostics kept separate from scientific evidence."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from auto_researcher.search.optuna.models import (
    OptunaStudyDiagnostics,
    OptunaStudySpec,
)


def build_study_diagnostics(
    study: Any, spec: OptunaStudySpec
) -> OptunaStudyDiagnostics:
    try:
        from optuna.importance import (
            FanovaImportanceEvaluator,
            MeanDecreaseImpurityImportanceEvaluator,
            PedAnovaImportanceEvaluator,
            get_param_importances,
        )
        from optuna.trial import TrialState
    except ImportError as exc:
        raise RuntimeError("OPTUNA HPO dependency unavailable") from exc

    trials = study.get_trials(deepcopy=True)
    completed = [trial for trial in trials if trial.state == TrialState.COMPLETE]
    pruned = [trial for trial in trials if trial.state == TrialState.PRUNED]
    failed = [trial for trial in trials if trial.state == TrialState.FAIL]
    multi_objective = len(spec.objective_specs) > 1
    pareto = tuple(trial.number for trial in study.best_trials) if completed else ()
    best = None
    if completed and not multi_objective:
        try:
            best = study.best_trial.number
        except ValueError:
            # Native constrained studies intentionally have no best trial until
            # at least one COMPLETE trial is feasible.
            best = None
    importances: dict[str, Any] = {}
    if spec.diagnostics.parameter_importance and len(completed) >= 2:
        factories: dict[str, Callable[[], Any | None]] = {
            "native_default": lambda: None,
            "fanova": FanovaImportanceEvaluator,
            "mdi": MeanDecreaseImpurityImportanceEvaluator,
            "ped_anova": PedAnovaImportanceEvaluator,
        }
        for evaluator_name in spec.diagnostics.importance_evaluators:
            evaluator = factories[evaluator_name]()
            per_objective: dict[str, dict[str, float]] = {}
            for index, objective in enumerate(spec.objective_specs):
                target = (
                    (lambda trial, position=index: float(trial.values[position]))
                    if multi_objective
                    else None
                )
                try:
                    per_objective[objective.name] = get_param_importances(
                        study,
                        evaluator=evaluator,
                        target=target,
                    )
                except (RuntimeError, ValueError, ZeroDivisionError):
                    per_objective[objective.name] = {}
            importances[evaluator_name] = per_objective
    return OptunaStudyDiagnostics(
        sampler=type(study.sampler).__name__,
        pruner=type(study.pruner).__name__,
        completed_trials=len(completed),
        pruned_trials=len(pruned),
        failed_trials=len(failed),
        best_trial_number=best,
        pareto_trial_numbers=pareto if multi_objective else (),
        parameter_importances=importances,
    )
