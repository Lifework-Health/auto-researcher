from __future__ import annotations

import inspect
from pathlib import Path

import optuna

from auto_researcher.search.optuna.capabilities import (
    CAPABILITY_MANIFEST_VERSION,
    CapabilityClassification,
    verify_capability_manifest,
)
from auto_researcher.search.optuna.components import NATIVE_PRUNERS, NATIVE_SAMPLERS


ROOT = Path(__file__).parents[2]
MANIFEST = ROOT / "docs/capabilities/optuna-4.9.0-capability-manifest.yaml"

MATERIAL_CAPABILITIES = {
    "base_pruner_extension",
    "base_sampler_extension",
    "brute_force_sampler",
    "categorical_distribution",
    "cmaes_sampler",
    "complete_state",
    "conditional_search_space",
    "cross_worker_constraints",
    "cross_worker_multi_objective",
    "custom_sampler_distributed_seed_policy",
    "default_parameter_importance",
    "distributed_bruteforce_seed",
    "distributed_gp_sampler_seed",
    "distributed_grid_seed",
    "distributed_nsgaii_seed",
    "distributed_nsgaiii_seed",
    "distributed_qmc_sequence_seed",
    "distributed_random_sampler_seed",
    "distributed_tpe_seed",
    "durable_constraint_record",
    "durable_prune_decision",
    "dynamic_trial_suggest",
    "evaluation_reuse_v2",
    "fail_state",
    "fanova_importance",
    "float_distribution",
    "gp_sampler",
    "grid_sampler",
    "hyperband_pruner",
    "infeasible_trial_remains_complete",
    "integer_distribution",
    "live_trial_reconstruction_after_process_loss",
    "median_pruner",
    "mdi_importance",
    "multi_objective_report_pruning",
    "multiple_objectives",
    "native_default_pruner",
    "native_default_sampler",
    "native_pareto_best_trials",
    "nop_pruner",
    "nsgaii_sampler",
    "nsgaiii_sampler",
    "ped_anova_importance",
    "patient_pruner",
    "percentile_pruner",
    "postgresql_storage",
    "pruned_state",
    "qmc_sampler",
    "random_sampler",
    "resource_broker_placement",
    "scientific_constraint_projection",
    "shared_distributed_workers",
    "single_objective",
    "sqlite_storage",
    "standard_runtime_start_resume",
    "study_ask",
    "study_directions",
    "study_persistence",
    "study_tell",
    "successive_halving_pruner",
    "threshold_pruner",
    "tpe_constraints",
    "tpe_sampler",
    "trial_report",
    "trial_should_prune",
    "trial_vs_scientific_identity",
    "wilcoxon_pruner",
}


def _test_names() -> set[str]:
    names: set[str] = set()
    for path in (ROOT / "tests").rglob("test_*.py"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("def test_"):
                names.add(line.removeprefix("def ").split("(", 1)[0])
    return names


def test_exact_pin_capability_manifest_is_executable_and_complete() -> None:
    manifest = verify_capability_manifest(MANIFEST, repository_root=ROOT)

    assert manifest.manifest_version == CAPABILITY_MANIFEST_VERSION
    assert manifest.upstream_version == optuna.__version__ == "4.9.0"
    capabilities = {item.capability for item in manifest.capabilities}
    assert MATERIAL_CAPABILITIES <= capabilities
    assert len(capabilities) >= 95
    assert sum(manifest.counts().values()) == len(capabilities)
    assert all(manifest.counts().values())


def test_manifest_claims_name_real_runtime_probes_and_adapter_contracts() -> None:
    manifest = verify_capability_manifest(MANIFEST, repository_root=ROOT)
    test_names = _test_names()
    for item in manifest.capabilities:
        if item.classification in {
            CapabilityClassification.PRESERVED_NATIVE,
            CapabilityClassification.PRESERVED_VIA_ADAPTER,
        }:
            assert item.probe in test_names
        assert all(probe in test_names for probe in item.probes)
        if item.classification is CapabilityClassification.PRESERVED_VIA_ADAPTER:
            assert item.adapter_contract
        if item.classification in {
            CapabilityClassification.CURRENTLY_WEAKENED,
            CapabilityClassification.CURRENTLY_DISABLED,
        }:
            assert item.justification


def test_native_factories_match_the_exact_material_public_inventory() -> None:
    public_sampler_algorithms = {
        "BruteForceSampler": "brute_force",
        "CmaEsSampler": "cmaes",
        "GPSampler": "gp",
        "GridSampler": "grid",
        "NSGAIISampler": "nsgaii",
        "NSGAIIISampler": "nsgaiii",
        "QMCSampler": "qmc",
        "RandomSampler": "random",
        "TPESampler": "tpe",
    }
    assert set(public_sampler_algorithms) <= set(optuna.samplers.__all__)
    assert (
        set(public_sampler_algorithms.values()) | {"native_default"} == NATIVE_SAMPLERS
    )

    public_pruner_algorithms = {
        name.removesuffix("Pruner")
        for name in optuna.pruners.__all__
        if name not in {"BasePruner"}
    }
    assert public_pruner_algorithms == {
        "Hyperband",
        "Median",
        "Nop",
        "Patient",
        "Percentile",
        "SuccessiveHalving",
        "Threshold",
        "Wilcoxon",
    }
    assert NATIVE_PRUNERS == {
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


def test_pruning_and_pareto_are_delegated_to_public_optuna_apis() -> None:
    from auto_researcher.search.optuna.backend import OptunaAskTellBackend
    from auto_researcher.search.optuna.pruning import OptunaIntermediateReporter

    reporter_source = inspect.getsource(OptunaIntermediateReporter.report)
    backend_source = inspect.getsource(OptunaAskTellBackend.load_study_summary)
    assert ".report(" in reporter_source
    assert ".should_prune()" in reporter_source
    assert "study.best_trials" in backend_source
    assert "non_dominated" not in backend_source
    assert "scalariz" not in backend_source
