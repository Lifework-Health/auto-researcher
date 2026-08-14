from __future__ import annotations

from datetime import UTC, datetime

from auto_researcher.contracts.enums import SearchType
from auto_researcher.graph.builder import build_graph
from auto_researcher.runtime.dependencies import (
    task_memory_dependencies,
    task_sqlite_dependencies,
)
from auto_researcher.search.optuna.models import (
    CategoricalParameterSpec,
    FloatParameterSpec,
    IntParameterSpec,
    OptimisationDirection,
    OptunaConditionSpec,
    OptunaConstraintSpec,
    OptunaDiagnosticsSpec,
    OptunaObjectiveSpec,
    OptunaPrunerSpec,
    OptunaSamplerSpec,
    OptunaStudySpec,
)
from auto_researcher.tasks.models import TaskRuntimeContext
from auto_researcher.tasks.synthetic import SyntheticTask, default_synthetic_contract
from auto_researcher.tasks.synthetic.evaluator import SyntheticEvaluator


NOW = datetime(2026, 8, 14, 13, 0, tzinfo=UTC)


class ReportingSyntheticEvaluator(SyntheticEvaluator):
    def evaluate_with_intermediate_reporting(self, experiment, contract, reporter):
        complexity = int(experiment.configuration["complexity"])
        value = 0.4 if complexity == 2 else 0.8
        if reporter.report(value, 1):
            reporter.acknowledge_pruning()
        return self.evaluate(experiment, contract)


class FullStrengthSyntheticTask(SyntheticTask):
    def normalise_configuration(self, configuration):
        payload = dict(configuration)
        payload.setdefault("learning_rate", 0.05)
        return super().normalise_configuration(payload)

    def create_evaluator(self, runtime_context):
        return ReportingSyntheticEvaluator(
            runtime_context,
            self.experiment_metadata(runtime_context),
            self.dataset_manifest(runtime_context),
        )

    def create_optuna_study_spec(self, contract, request):
        self.validate_contract(contract)
        return OptunaStudySpec(
            schema_version="2.0",
            task_id=self.task_id,
            task_version=self.task_version,
            search_space_version="synthetic-optuna-full-strength-v2",
            direction=OptimisationDirection.MAXIMIZE,
            parameters=(
                CategoricalParameterSpec(
                    name="model_family",
                    choices=("linear", "neural"),
                ),
                IntParameterSpec(name="complexity", low=1, high=2),
                FloatParameterSpec(
                    name="learning_rate",
                    low=0.05,
                    high=0.1,
                    step=0.05,
                    condition=OptunaConditionSpec(
                        parameter="model_family",
                        equals="neural",
                    ),
                ),
            ),
            trial_budget=request.experiment_budget,
            seed=29,
            sampler=OptunaSamplerSpec(
                type="nsgaii",
                options={"population_size": 4},
            ),
            objective_metric=contract.primary_metric,
            objectives=(
                OptunaObjectiveSpec(
                    name="balanced_score",
                    direction=OptimisationDirection.MAXIMIZE,
                    metric=contract.primary_metric,
                ),
            ),
            constraints=(
                OptunaConstraintSpec(
                    name="runtime_acceptance",
                    metric="runtime",
                    relation="LESS_THAN_OR_EQUAL",
                    threshold=0.25,
                ),
            ),
            pruner=OptunaPrunerSpec(
                type="threshold",
                options={"lower": 0.6},
            ),
            intermediate_reporting=True,
            diagnostics=OptunaDiagnosticsSpec(
                parameter_importance=True,
                importance_evaluators=("native_default", "ped_anova"),
            ),
            study_metadata={"diagnostics_are_scientific_evidence": False},
        )


class MultiObjectiveSyntheticTask(FullStrengthSyntheticTask):
    def create_optuna_study_spec(self, contract, request):
        base = super().create_optuna_study_spec(contract, request)
        return base.model_copy(
            update={
                "search_space_version": "synthetic-optuna-pareto-v2",
                "sampler": OptunaSamplerSpec(
                    type="nsgaii",
                    options={"population_size": 4},
                ),
                "objectives": (
                    base.objectives[0],
                    OptunaObjectiveSpec(
                        name="runtime",
                        direction=OptimisationDirection.MINIMIZE,
                        metric="runtime",
                    ),
                ),
                "constraints": (),
                "pruner": OptunaPrunerSpec(type="none"),
                "intermediate_reporting": False,
            }
        )


class RandomSamplerSyntheticTask(FullStrengthSyntheticTask):
    def create_optuna_study_spec(self, contract, request):
        base = super().create_optuna_study_spec(contract, request)
        return base.model_copy(
            update={
                "search_space_version": "synthetic-optuna-random-v2",
                "sampler": OptunaSamplerSpec(type="random"),
                "constraints": (),
            }
        )


def _invoke(graph, contract, run_id, thread_id, value=None):
    payload = (
        {"run_id": run_id, "thread_id": thread_id, "contract": contract}
        if value is None
        else value
    )
    return graph.invoke(payload, {"configurable": {"thread_id": thread_id}})


def test_standard_runtime_nsgaii_conditional_pruning_constraints_reuse_and_diagnostics(
    tmp_path,
) -> None:
    contract = default_synthetic_contract(
        search_types=frozenset({SearchType.OPTUNA}),
        maximum_experiments=10,
    )
    context = TaskRuntimeContext(
        run_id="standard-optuna-v2",
        output_dir=tmp_path,
        manifest_created_at=NOW,
    )
    dependencies = task_memory_dependencies(
        FullStrengthSyntheticTask(),
        context,
        contract,
        {"trial_budget": 10},
        search_type=SearchType.OPTUNA,
        clock=lambda: NOW,
    )
    final = _invoke(
        build_graph(dependencies),
        contract,
        "standard-optuna-v2",
        "standard-optuna-v2-thread",
    )
    result = final["optuna_study_result"]
    outcomes = dependencies.optuna_backend.trial_outcomes(result.study_name)
    assert result.trials_asked == 10
    assert result.trials_pruned > 0
    assert result.trials_completed > 0
    assert any(not outcome.feasible for outcome in outcomes if outcome.objective_values)
    assert any(outcome.intermediate_values for outcome in outcomes)
    assert final["executed_nodes"].count("evaluate_experiment_reused") > 0
    assert any(outcome.evaluation_reused for outcome in outcomes)
    assert result.diagnostics.sampler == "NSGAIISampler"
    assert result.diagnostics.epistemic_status == "OPERATIONAL_SEARCH_DIAGNOSTIC"


def test_standard_runtime_random_sampler_executes_native_start_to_finish(
    tmp_path,
) -> None:
    contract = default_synthetic_contract(
        search_types=frozenset({SearchType.OPTUNA}),
        maximum_experiments=6,
    )
    context = TaskRuntimeContext(
        run_id="standard-optuna-random-v2",
        output_dir=tmp_path,
        manifest_created_at=NOW,
    )
    dependencies = task_memory_dependencies(
        RandomSamplerSyntheticTask(),
        context,
        contract,
        {"trial_budget": 6},
        search_type=SearchType.OPTUNA,
        clock=lambda: NOW,
    )
    final = _invoke(
        build_graph(dependencies),
        contract,
        "standard-optuna-random-v2",
        "standard-optuna-random-v2-thread",
    )
    result = final["optuna_study_result"]
    outcomes = dependencies.optuna_backend.trial_outcomes(result.study_name)
    assert result.trials_asked == 6
    assert result.trials_pruned > 0
    assert result.trials_completed > 0
    assert any(outcome.intermediate_values for outcome in outcomes)
    assert result.diagnostics.sampler == "RandomSampler"


def test_standard_sqlite_start_resume_retains_native_v2_envelope(tmp_path) -> None:
    contract = default_synthetic_contract(
        search_types=frozenset({SearchType.OPTUNA}),
        maximum_experiments=6,
    )
    context = TaskRuntimeContext(
        run_id="standard-optuna-resume-v2",
        output_dir=tmp_path / "output",
        manifest_created_at=NOW,
    )
    paths = (
        tmp_path / "checkpoints.sqlite",
        tmp_path / "provenance.sqlite",
        tmp_path / "optuna.sqlite",
    )
    with task_sqlite_dependencies(
        FullStrengthSyntheticTask(),
        context,
        contract,
        {"trial_budget": 6},
        *paths,
        search_type=SearchType.OPTUNA,
        clock=lambda: NOW,
    ) as dependencies:
        partial = _invoke(
            build_graph(dependencies, interrupt_after=["optuna_ask_trial"]),
            contract,
            "standard-optuna-resume-v2",
            "standard-optuna-resume-v2-thread",
        )
        first_number = partial["optuna_study_state"].current_trial.trial_number

    with task_sqlite_dependencies(
        FullStrengthSyntheticTask(),
        context,
        contract,
        {"trial_budget": 6},
        *paths,
        search_type=SearchType.OPTUNA,
        clock=lambda: NOW,
    ) as dependencies:
        final = build_graph(dependencies).invoke(
            None,
            {"configurable": {"thread_id": "standard-optuna-resume-v2-thread"}},
        )
        result = final["optuna_study_result"]
        outcomes = dependencies.optuna_backend.trial_outcomes(result.study_name)
        assert outcomes[0].trial_number == first_number
        assert result.trials_asked == 6
        assert result.diagnostics.sampler == "NSGAIISampler"


def test_standard_runtime_multi_objective_exposes_native_pareto_without_winner(
    tmp_path,
) -> None:
    contract = default_synthetic_contract(
        search_types=frozenset({SearchType.OPTUNA}),
        maximum_experiments=6,
    )
    context = TaskRuntimeContext(
        run_id="standard-optuna-pareto-v2",
        output_dir=tmp_path,
        manifest_created_at=NOW,
    )
    dependencies = task_memory_dependencies(
        MultiObjectiveSyntheticTask(),
        context,
        contract,
        {"trial_budget": 6},
        search_type=SearchType.OPTUNA,
        clock=lambda: NOW,
    )
    final = _invoke(
        build_graph(dependencies),
        contract,
        "standard-optuna-pareto-v2",
        "standard-optuna-pareto-v2-thread",
    )
    result = final["optuna_study_result"]
    assert result.directions == (
        OptimisationDirection.MAXIMIZE,
        OptimisationDirection.MINIMIZE,
    )
    assert result.pareto_trial_numbers
    assert result.pareto_trials
    assert result.best_overall_trial_number is None
    assert result.best_feasible_trial_number is None
    assert final["experiment_spec"] is None
    assert final["diagnostic_experiment_spec"] is None
