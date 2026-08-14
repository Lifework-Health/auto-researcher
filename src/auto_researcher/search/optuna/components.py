"""Approved public-API factories for Optuna 4.9 samplers and pruners."""

from __future__ import annotations

import importlib.util
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from auto_researcher.search.optuna.models import (
    CategoricalParameterSpec,
    FloatParameterSpec,
    IntParameterSpec,
    OptunaPrunerSpec,
    OptunaSamplerSpec,
    OptunaStudySpec,
)
from auto_researcher.search.optuna.operational import OptunaOperationalRecordStore
from auto_researcher.search.optuna.space import uses_trial_suggestions


SamplerFactory = Callable[[OptunaSamplerSpec, "SamplerBuildContext"], Any]
PrunerFactory = Callable[[OptunaPrunerSpec], Any]


@dataclass(frozen=True)
class SamplerBuildContext:
    study_spec: OptunaStudySpec
    seed: int
    shared_workers: bool
    operational_store: OptunaOperationalRecordStore

    @property
    def constraints_callback(self) -> Callable[[Any], tuple[float, ...]] | None:
        if not self.study_spec.constraints:
            return None
        return self.operational_store.constraints_for_frozen_trial


@dataclass(frozen=True)
class ApprovedSamplerRegistration:
    """Runtime-reviewed custom sampler contract; configuration cannot assert it."""

    factory: SamplerFactory
    supports_single_objective: bool = True
    supports_multi_objective: bool = False
    supports_constraints: bool = False
    shared_worker_safe: bool = False
    supports_dynamic_space: bool = False
    optional_dependency_identity: str | None = None


@dataclass
class ApprovedOptunaComponentRegistry:
    """Runtime-only extension seam; configuration never imports Python paths."""

    sampler_registrations: dict[str, ApprovedSamplerRegistration] = field(
        default_factory=dict
    )
    pruner_factories: dict[str, PrunerFactory] = field(default_factory=dict)

    def register_sampler(
        self,
        name: str,
        factory: SamplerFactory,
        *,
        supports_single_objective: bool = True,
        supports_multi_objective: bool = False,
        supports_constraints: bool = False,
        shared_worker_safe: bool = False,
        supports_dynamic_space: bool = False,
        optional_dependency_identity: str | None = None,
    ) -> None:
        if name in NATIVE_SAMPLERS or name in self.sampler_registrations:
            raise ValueError("Optuna sampler registry name is reserved or duplicated")
        self.sampler_registrations[name] = ApprovedSamplerRegistration(
            factory=factory,
            supports_single_objective=supports_single_objective,
            supports_multi_objective=supports_multi_objective,
            supports_constraints=supports_constraints,
            shared_worker_safe=shared_worker_safe,
            supports_dynamic_space=supports_dynamic_space,
            optional_dependency_identity=optional_dependency_identity,
        )

    def register_pruner(self, name: str, factory: PrunerFactory) -> None:
        if name in NATIVE_PRUNERS or name in self.pruner_factories:
            raise ValueError("Optuna pruner registry name is reserved or duplicated")
        self.pruner_factories[name] = factory


NATIVE_SAMPLERS = frozenset(
    {
        "native_default",
        "tpe",
        "random",
        "cmaes",
        "gp",
        "nsgaii",
        "nsgaiii",
        "qmc",
        "grid",
        "brute_force",
    }
)

NATIVE_PRUNERS = frozenset(
    {
        "native_default",
        "none",
        "median",
        "patient",
        "percentile",
        "successive_halving",
        "hyperband",
        "threshold",
        "wilcoxon",
    }
)

CONSTRAINT_AWARE_SAMPLERS = frozenset({"tpe", "gp", "nsgaii", "nsgaiii"})
MULTI_OBJECTIVE_SAMPLERS = frozenset(
    {
        "native_default",
        "tpe",
        "random",
        "gp",
        "nsgaii",
        "nsgaiii",
        "qmc",
        "grid",
        "brute_force",
    }
)
SHARED_WORKER_SAMPLERS = frozenset(MULTI_OBJECTIVE_SAMPLERS)


def _imports() -> tuple[Any, Any]:
    try:
        from optuna import pruners, samplers
    except ImportError as exc:
        raise RuntimeError(
            "OPTUNA search requires the HPO dependency. "
            "Install with `pip install -e '.[hpo]'`."
        ) from exc
    return samplers, pruners


def _options(configuration: Any, allowed: set[str]) -> dict[str, Any]:
    options = dict(configuration.options)
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(
            f"unsupported Optuna component options: {', '.join(sorted(unknown))}"
        )
    return options


def _grid_space(spec: OptunaStudySpec) -> dict[str, list[Any]]:
    if any(parameter.condition is not None for parameter in spec.parameters):
        raise ValueError("grid_sampler_requires_static_search_space")
    search_space: dict[str, list[Any]] = {}
    for parameter in spec.parameters:
        if isinstance(parameter, CategoricalParameterSpec):
            search_space[parameter.name] = list(parameter.choices)
        elif isinstance(parameter, IntParameterSpec):
            search_space[parameter.name] = list(
                range(parameter.low, parameter.high + 1, parameter.step)
            )
        elif isinstance(parameter, FloatParameterSpec) and parameter.step is not None:
            count = round((parameter.high - parameter.low) / parameter.step)
            search_space[parameter.name] = [
                parameter.low + parameter.step * index for index in range(count + 1)
            ]
        else:
            raise ValueError("grid_sampler_requires_finite_static_distributions")
        if len(search_space[parameter.name]) > 100_000:
            raise ValueError("grid_sampler_distribution_too_large")
    return search_space


def build_sampler(
    configuration: OptunaSamplerSpec,
    context: SamplerBuildContext,
    registry: ApprovedOptunaComponentRegistry,
) -> Any | None:
    """Construct only public native samplers or runtime-approved extensions."""

    sampler_type = configuration.type
    custom = registry.sampler_registrations.get(sampler_type)
    objective_count = len(context.study_spec.objective_specs)
    if custom is not None:
        if objective_count == 1 and not custom.supports_single_objective:
            raise ValueError("optuna_custom_sampler_single_objective_incompatible")
        if objective_count > 1 and not custom.supports_multi_objective:
            raise ValueError("optuna_custom_sampler_multi_objective_incompatible")
        if context.study_spec.constraints and not custom.supports_constraints:
            raise ValueError("optuna_custom_sampler_constraints_unsupported")
        if context.shared_workers and not custom.shared_worker_safe:
            raise ValueError("optuna_custom_sampler_shared_worker_incompatible")
        if (
            uses_trial_suggestions(context.study_spec)
            and not custom.supports_dynamic_space
        ):
            raise ValueError("optuna_custom_sampler_dynamic_space_incompatible")
        sampler = custom.factory(configuration, context)
        samplers, _ = _imports()
        if not isinstance(sampler, samplers.BaseSampler):
            raise TypeError("approved Optuna sampler factory must return BaseSampler")
        return sampler
    if sampler_type not in NATIVE_SAMPLERS:
        raise ValueError("optuna_sampler_not_approved")
    if objective_count > 1 and sampler_type not in MULTI_OBJECTIVE_SAMPLERS:
        raise ValueError("optuna_sampler_multi_objective_incompatible")
    if context.shared_workers and sampler_type not in SHARED_WORKER_SAMPLERS:
        raise ValueError("optuna_sampler_shared_worker_incompatible")
    if context.study_spec.constraints and sampler_type not in CONSTRAINT_AWARE_SAMPLERS:
        raise ValueError("optuna_sampler_native_constraints_unsupported")

    samplers, _ = _imports()
    constraint_callback = (
        context.operational_store.constraints_for_frozen_trial
        if context.study_spec.constraints
        else None
    )
    seed = context.seed
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=Warning, module=r"optuna\..*")
        if sampler_type == "native_default":
            if configuration.options:
                raise ValueError("native_default_sampler_accepts_no_options")
            return None
        if sampler_type == "tpe":
            options = _options(
                configuration,
                {
                    "n_startup_trials",
                    "n_ei_candidates",
                    "multivariate",
                    "group",
                    "warn_independent_sampling",
                },
            )
            options.setdefault("n_startup_trials", context.study_spec.n_startup_trials)
            return samplers.TPESampler(
                seed=seed,
                constant_liar=context.shared_workers,
                constraints_func=constraint_callback,
                **options,
            )
        if sampler_type == "random":
            _options(configuration, set())
            return samplers.RandomSampler(seed=seed)
        if sampler_type == "cmaes":
            if importlib.util.find_spec("cmaes") is None:
                raise RuntimeError(
                    "cmaes_sampler_optional_dependency_missing: install '.[hpo-cmaes]'"
                )
            options = _options(
                configuration,
                {
                    "sigma0",
                    "n_startup_trials",
                    "warn_independent_sampling",
                    "consider_pruned_trials",
                    "popsize",
                    "use_separable_cma",
                    "with_margin",
                    "lr_adapt",
                },
            )
            return samplers.CmaEsSampler(seed=seed, **options)
        if sampler_type == "gp":
            if (
                importlib.util.find_spec("scipy") is None
                or importlib.util.find_spec("torch") is None
            ):
                raise RuntimeError(
                    "gp_sampler_optional_dependency_missing: install '.[hpo-gp]'"
                )
            options = _options(
                configuration,
                {
                    "n_startup_trials",
                    "deterministic_objective",
                    "warn_independent_sampling",
                },
            )
            return samplers.GPSampler(
                seed=seed,
                constraints_func=constraint_callback,
                **options,
            )
        if sampler_type == "nsgaii":
            options = _options(
                configuration,
                {
                    "population_size",
                    "mutation_prob",
                    "crossover_prob",
                    "swapping_prob",
                },
            )
            return samplers.NSGAIISampler(
                seed=seed,
                constraints_func=constraint_callback,
                **options,
            )
        if sampler_type == "nsgaiii":
            options = _options(
                configuration,
                {
                    "population_size",
                    "mutation_prob",
                    "crossover_prob",
                    "swapping_prob",
                    "dividing_parameter",
                },
            )
            return samplers.NSGAIIISampler(
                seed=seed,
                constraints_func=constraint_callback,
                **options,
            )
        if sampler_type == "qmc":
            if importlib.util.find_spec("scipy") is None:
                raise RuntimeError(
                    "qmc_sampler_optional_dependency_missing: install '.[hpo-qmc]'"
                )
            options = _options(
                configuration,
                {
                    "qmc_type",
                    "scramble",
                    "warn_asynchronous_seeding",
                    "warn_independent_sampling",
                },
            )
            return samplers.QMCSampler(seed=seed, **options)
        if sampler_type == "grid":
            _options(configuration, set())
            return samplers.GridSampler(_grid_space(context.study_spec), seed=seed)
        if sampler_type == "brute_force":
            options = _options(configuration, {"avoid_premature_stop"})
            return samplers.BruteForceSampler(seed=seed, **options)
    raise AssertionError("unreachable approved sampler")


def build_pruner(
    configuration: OptunaPrunerSpec,
    registry: ApprovedOptunaComponentRegistry,
) -> Any | None:
    """Construct only public native pruners or runtime-approved extensions."""

    custom = registry.pruner_factories.get(configuration.type)
    if custom is not None:
        pruner = custom(configuration)
        _, pruners = _imports()
        if not isinstance(pruner, pruners.BasePruner):
            raise TypeError("approved Optuna pruner factory must return BasePruner")
        return pruner
    if configuration.type not in NATIVE_PRUNERS:
        raise ValueError("optuna_pruner_not_approved")
    _, pruners = _imports()
    kind = configuration.type
    if kind == "native_default":
        if configuration.options:
            raise ValueError("native_default_pruner_accepts_no_options")
        return None
    if kind == "none":
        _options(configuration, set())
        return pruners.NopPruner()
    if kind == "median":
        return pruners.MedianPruner(
            **_options(
                configuration,
                {
                    "n_startup_trials",
                    "n_warmup_steps",
                    "interval_steps",
                    "n_min_trials",
                },
            )
        )
    if kind == "percentile":
        return pruners.PercentilePruner(
            **_options(
                configuration,
                {
                    "percentile",
                    "n_startup_trials",
                    "n_warmup_steps",
                    "interval_steps",
                    "n_min_trials",
                },
            )
        )
    if kind == "successive_halving":
        return pruners.SuccessiveHalvingPruner(
            **_options(
                configuration,
                {
                    "min_resource",
                    "reduction_factor",
                    "min_early_stopping_rate",
                    "bootstrap_count",
                },
            )
        )
    if kind == "hyperband":
        return pruners.HyperbandPruner(
            **_options(
                configuration,
                {"min_resource", "max_resource", "reduction_factor", "bootstrap_count"},
            )
        )
    if kind == "threshold":
        return pruners.ThresholdPruner(
            **_options(
                configuration,
                {"lower", "upper", "n_warmup_steps", "interval_steps"},
            )
        )
    if kind == "wilcoxon":
        return pruners.WilcoxonPruner(
            **_options(configuration, {"p_threshold", "n_startup_steps"})
        )
    if kind == "patient":
        options = _options(
            configuration,
            {"wrapped_pruner", "patience", "min_delta"},
        )
        raw_wrapped = options.pop("wrapped_pruner", {"type": "median"})
        wrapped = build_pruner(OptunaPrunerSpec.model_validate(raw_wrapped), registry)
        return pruners.PatientPruner(wrapped_pruner=wrapped, **options)
    raise AssertionError("unreachable approved pruner")
